using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Linear;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Profiling;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// The number of iterations used to remove the velocity added for contact penetration correction after
        /// integrating bodies solved iteratively. These pairs involve loose bodies that usually settle within a few
        /// iterations.
        /// </summary>
        const int RelaxIterations = 4;

        /// <summary>
        /// Schedules one step for every island. The jobs determine their work counts when they run, so the main thread
        /// can schedule them without waiting for those counts. Islands assigned to iterative phases are solved phase by
        /// phase, then their bodies are integrated. For direct islands, each substep runs a sequence of jobs that
        /// update Jacobians, build rows in parallel, solve and integrate. The iterative and direct paths can run
        /// concurrently because they modify different bodies. <paramref name="jacobiansBuilt"/> marks completion of
        /// Jacobian construction, before jobs that change impulses or velocities in preparation for solving. Listing
        /// the direct islands' Jacobians depends only on construction, so it waits for that handle alone.
        /// <para>
        /// <paramref name="motionsSolved"/> completes when every job that writes the motions has finished. The jobs
        /// that free the solve's buffers and export the direct islands' events come after it. Work that reads only
        /// the solved motions can wait for this handle and skip that chain of small jobs.
        /// </para>
        /// </summary>
        internal static unsafe SimulationJobHandles ScheduleIslandSolveJobs(in PhysicsWorld world, StepInput stepInput, int directIslandSubsteps,
            in NativeStream jacobians, in NativeStream collisionEvents, in NativeStream triggerEvents, in NativeStream impulseEvents,
            in DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo, StabilizationData solverStabilizationData,
            JobHandle jacobiansBuilt, JobHandle inputDeps, out JobHandle motionsSolved)
        {
            NativeArray<MotionVelocity> motionVelocities = world.MotionVelocities;
            DispatchPairSequencer.IterativeSolverSchedulerInfo iterativePairs = solverSchedulerInfo.IterativePairsIterativeScheduling;

            // Schedule two team jobs: one for the solver iterations and one for relaxation. Each job processes its
            // phases and iterations as team stages. This avoids scheduling a separate job for every phase of every
            // iteration. Both jobs finish quickly if there are no iterative pairs.
            var teams = UnsafeList<SparseSolveTeam>.Create(2, Allocator.TempJob);
            teams->Resize(2, NativeArrayOptions.ClearMemory);
            teams->ElementAt(0).Configure();
            teams->ElementAt(1).Configure();
            int lanes = math.min(JobsUtility.JobWorkerCount + 1, MaxTeamLanes);

            var template = new ParallelIterativeSolverJob
            {
                JacobiansReader = jacobians.AsReader(),
                IterativeSolverSchedulerInfo = iterativePairs,
                MotionVelocities = motionVelocities,
                SolverStabilizationData = solverStabilizationData,
                CollisionEventsWriter = collisionEvents.AsWriter(),
                TriggerEventsWriter = triggerEvents.AsWriter(),
                ImpulseEventsWriter = impulseEvents.AsWriter(),
                stepInput = stepInput,
            };

            JobHandle iterativeHandle = new IterateInTeamJob
            {
                Template = template,
                FirstIteration = 0,
                Iterations = stepInput.NumSolverIterations,
                Team = teams->Ptr,
            }.Schedule(lanes, 1, inputDeps);

            iterativeHandle = Integrator.ScheduleIntegrateBodiesOutsideIslandsJob(world.DynamicsWorld,
                solverSchedulerInfo.DirectPairsDirectScheduling.BodiesInIslands.AsDeferredJobArray(), stepInput.Timestep, iterativeHandle);

            // After integration has applied the contact penetration correction to positions, run a few iterations to
            // remove that correction from velocities. This is similar to PhysX velocity iterations after position
            // iterations. Events have already been exported using the step's impulses. Start the relaxation iteration
            // indices after the solver iterations so none is treated as the first or last solver iteration.
            template.stepInput.Relaxing = true;
            template.stepInput.DisableExportEvents = true;
            iterativeHandle = new IterateInTeamJob
            {
                Template = template,
                FirstIteration = stepInput.NumSolverIterations,
                Iterations = RelaxIterations,
                Team = teams->Ptr + 1,
            }.Schedule(lanes, 1, iterativeHandle);

            JobHandle iterativeSolved = iterativeHandle;
            iterativeHandle = new DestroyTeamsJob { Teams = teams }.Schedule(iterativeHandle);

            JobHandle islandHandle = ScheduleDirectIslandJobs(world, stepInput, directIslandSubsteps, jacobians,
                solverSchedulerInfo.DirectPairsDirectScheduling, jacobiansBuilt, inputDeps, out JobHandle islandsSolved);
            motionsSolved = JobHandle.CombineDependencies(iterativeSolved, islandsSolved);

            islandHandle = new ParallelExportEventsJob
            {
                JacobiansReader = jacobians.AsReader(),
                MotionVelocities = motionVelocities,
                CollisionEventsWriter = collisionEvents.AsWriter(),
                TriggerEventsWriter = triggerEvents.AsWriter(),
                ImpulseEventsWriter = impulseEvents.AsWriter(),
                IterativeSolverSchedulerInfo = solverSchedulerInfo.DirectPairsIterativeScheduling,
                Timestep = stepInput.Timestep
            }.ScheduleUnsafeIndex0(solverSchedulerInfo.DirectPairsIterativeScheduling.NumWorkItems, 1, islandHandle);

            SimulationJobHandles returnHandles = default;
            returnHandles.FinalExecutionHandle = JobHandle.CombineDependencies(iterativeHandle, islandHandle);
            returnHandles.FinalDisposeHandle = JobHandle.CombineDependencies(
                jacobians.Dispose(returnHandles.FinalExecutionHandle),
                solverSchedulerInfo.ScheduleDisposeJob(returnHandles.FinalExecutionHandle));

            return returnHandles;
        }

        /// <summary>
        /// The maximum number of lanes scheduled for a team job. Additional lanes would wait if the stages have too few
        /// tasks.
        /// </summary>
        const int MaxTeamLanes = 8;

        /// <summary>
        /// Solves the iterative pairs one phase at a time in each iteration. The job's lanes share the work items for
        /// each phase. Work items in the same phase do not modify the same body. The leader waits for all tasks in one
        /// phase to finish before starting the next phase.
        /// </summary>
        [BurstCompile]
        [NoAlias]
        unsafe struct IterateInTeamJob : IJobParallelFor
        {
            /// <summary>
            /// The job data copied for each phase, with the phase index and iteration number set before execution.
            /// </summary>
            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public ParallelIterativeSolverJob Template;

            public int FirstIteration;
            public int Iterations;

            [NativeDisableUnsafePtrRestriction]
            public SparseSolveTeam* Team;

            public void Execute(int lane)
            {
                if (!Team->TryLead())
                {
                    Team->Help();
                    return;
                }

                var phases = (DispatchPairSequencer.IterativeSolverSchedulerInfo.SolvePhaseInfo*)
                    NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(Template.IterativeSolverSchedulerInfo.PhaseInfo);
                ParallelIterativeSolverJob job = Template;
                for (int iteration = FirstIteration; iteration < FirstIteration + Iterations; iteration++)
                {
                    job.stepInput.CurrentSolverIteration = iteration;
                    for (int phase = 0; phase < DispatchPairSequencer.NumIslandPhases; phase++)
                    {
                        job.PhaseIndex = phase;
                        Team->Lead(&job, SparseSolveTeam.IterativePhaseKind, phases[phase].NumWorkItems, null);
                    }
                }

                Team->Finish();
            }
        }

        /// <summary>
        /// Executes one work item in an iterative phase on a team lane. Each task copies the phase's job from the
        /// context because its event writers contain state for the current work item and cannot be shared across lanes.
        /// </summary>
        internal static unsafe void IterativePhaseTask(void* context, int task)
        {
            ParallelIterativeSolverJob job = *(ParallelIterativeSolverJob*)context;
            job.Execute(task);
        }

        [BurstCompile]
        unsafe struct DestroyTeamsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public UnsafeList<SparseSolveTeam>* Teams;

            public void Execute()
            {
                UnsafeList<SparseSolveTeam>.Destroy(Teams);
            }
        }

        /// <summary>
        /// Schedules the substeps for all direct islands. Each substep updates Jacobians, counts and builds their rows,
        /// solves the islands, and integrates their bodies. Row counting is serial; the other stages run in parallel.
        /// Small islands are solved independently on worker threads. One team job solves all large islands
        /// sequentially, sharing each solve among its lanes. The first substep's row counts depend only on the
        /// constructed Jacobians, so they can be counted while listing the islands, concurrently with jobs that finish
        /// preparing the solve. <paramref name="solved"/> completes when the islands' bodies are integrated, before
        /// the job that frees the islands' buffers.
        /// </summary>
        static unsafe JobHandle ScheduleDirectIslandJobs(in PhysicsWorld world, StepInput stepInput, int substeps, in NativeStream jacobians,
            in DispatchPairSequencer.DirectSolverSchedulerInfo info, JobHandle jacobiansBuilt, JobHandle inputDeps, out JobHandle solved)
        {
            const int headersPerBatch = 16;
            const int bodiesPerBatch = 32;

            var work = new DirectIslandWork(substeps, Allocator.TempJob);
            int lanes = math.min(JobsUtility.JobWorkerCount + 1, MaxTeamLanes);

            JobHandle handle = new IndexDirectIslandsJob
            {
                Info = info,
                JacobiansReader = jacobians.AsReader(),
                Work = work,
                Cache = stepInput.DirectSolverCache,
                Bodies = world.Bodies,
                MotionCount = world.MotionVelocities.Length,
            }.Schedule(jacobiansBuilt);

            handle = JobHandle.CombineDependencies(handle, inputDeps);

            StepInput substepInput = stepInput;
            substepInput.Timestep /= substeps;
            substepInput.NumSubsteps = substeps;
            substepInput.DisableExportEvents = true;

            // Mark the islands that the fused vessel job solves. The substeps' integration then skips their moving
            // bodies. The fused job runs beside the substeps' jobs, because the two never touch the same bodies. The
            // same job gives every vessel island its forest, with or without the fused job.
            bool fusedVessels = VesselFused;
            JobHandle fused = default;
            handle = new MarkFusedVesselIslandsJob
            {
                Islands = work.Islands,
                IslandVessels = work.IslandVessels,
                Vessels = stepInput.VesselSet,
                Cache = stepInput.DirectSolverCache,
                Fuse = fusedVessels && !LegacyRowBuild,
                MinimumParts = FusedVesselParts,
                Contacts = FusedContacts,
                Buffers = work.Buffers,
            }.Schedule(handle);

            if (fusedVessels)
            {
                fused = new SolveFusedVesselIslandsJob
                {
                    Islands = work.Islands.AsDeferredJobArray(),
                    IslandVessels = work.IslandVessels.AsDeferredJobArray(),
                    Headers = work.Headers.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    Bodies = world.Bodies,
                    StepInput = substepInput,
                    Teams = work.Buffers.FusedTeams,
                    Lanes = lanes,
                }.Schedule(lanes, 1, handle);
            }

            for (int substep = 0; substep < substeps; substep++)
            {
                substepInput.CurrentSubstep = substep;
                if (substep > 0)
                {
                    // Normally, update only contacts before counting rows, because only contact row counts can change.
                    // Update each joint while building its rows to reuse the joint data already being read.
                    // LegacyRowBuild instead updates all Jacobians here.
                    handle = new UpdateIslandJacobiansJob
                    {
                        Headers = work.Headers.AsDeferredJobArray(),
                        Islands = work.Islands.AsDeferredJobArray(),
                        MotionVelocities = world.MotionVelocities,
                        MotionDatas = world.MotionDatas,
                        Bodies = world.Bodies,
                        StepInput = substepInput,
                        ContactsOnly = !LegacyRowBuild,
                    }.Schedule(work.Headers, headersPerBatch, handle);

                    handle = new CountIslandRowsJob { Work = work }.Schedule(handle);
                }

                handle = new BuildIslandRowsJob
                {
                    Headers = work.Headers.AsDeferredJobArray(),
                    Islands = work.Islands.AsDeferredJobArray(),
                    Work = work.Buffers,
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    Bodies = world.Bodies,
                    StepInput = substepInput,
                    UpdateJoints = substep > 0 && !LegacyRowBuild,
                    ShareRegularization = !LegacyRowBuild,
                }.Schedule(work.Headers, headersPerBatch, handle);

                JobHandle alone = new SolveDirectIslandJob
                {
                    Info = info,
                    JacobiansReader = jacobians.AsReader(),
                    Headers = work.Headers.AsDeferredJobArray(),
                    Islands = work.Islands.AsDeferredJobArray(),
                    Work = work.Buffers,
                    IslandVessels = work.IslandVessels.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    Bodies = world.Bodies,
                    StepInput = substepInput,
                }.Schedule(work.Islands, 1, handle);

                JobHandle shared = new SolveSharedDirectIslandsJob
                {
                    Info = info,
                    JacobiansReader = jacobians.AsReader(),
                    Headers = work.Headers.AsDeferredJobArray(),
                    Islands = work.Islands.AsDeferredJobArray(),
                    Work = work.Buffers,
                    IslandVessels = work.IslandVessels.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    Bodies = world.Bodies,
                    StepInput = substepInput,
                    Team = work.Buffers.Teams->Ptr + substep,
                }.Schedule(lanes, 1, handle);

                handle = JobHandle.CombineDependencies(alone, shared);

                handle = new IntegrateIslandBodiesJob
                {
                    IslandBodies = info.IslandBodies.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    FirstIslandBodies = info.FirstIslandBodyIndices.AsDeferredJobArray(),
                    Islands = work.Islands.AsDeferredJobArray(),
                    Timestep = substepInput.Timestep,
                }.Schedule(info.IslandBodies, bodiesPerBatch, handle);
            }

            handle = JobHandle.CombineDependencies(handle, fused);
            solved = handle;
            return work.Dispose(handle);
        }

        /// <summary>
        /// Describes the Jacobians for one direct island in <see cref="DirectIslandWork.Headers"/> and its rows for the
        /// current substep.
        /// </summary>
        struct DirectIsland
        {
            public int FirstHeader;

            /// <summary>
            /// The exclusive end index of the joint Jacobians, which precede the contact Jacobians.
            /// </summary>
            public int JointEnd;

            /// <summary>
            /// The exclusive end index of the Jacobians included in the solve. Contact Jacobians after this index are
            /// updated but not solved.
            /// </summary>
            public int SolveEnd;

            public int HeaderEnd;

            /// <summary>
            /// The number of contact pairs included with the joints in the sparse solve.
            /// </summary>
            public int ContactPairs;

            public int FirstRow;
            public int Rows;

            /// <summary>
            /// The number of rows in the joint Jacobians, which remains the same across substeps.
            /// </summary>
            public int JointRows;

            /// <summary>
            /// The range of this island's vessels in <see cref="DirectIslandWork.IslandVessels"/> and their total
            /// number of parts. An island with a vessel is solved even if it has no Jacobian rows.
            /// </summary>
            public int FirstVessel;
            public int VesselCount;
            public int VesselParts;

            /// <summary>The number of the island's bodies that are not parts of its vessels.</summary>
            public int OtherBodies;

            /// <summary>
            /// Whether the island has enough rows or vessel parts to benefit from sharing the solve among lanes.
            /// Smaller islands are solved independently in parallel.
            /// </summary>
            public bool SolvesInTeam => Rows >= SharedSolveRows || VesselParts >= SharedSolveParts;

            /// <summary>
            /// Whether <see cref="SolveFusedVesselIslandsJob"/> solves this island for the whole step. The jobs of each
            /// substep then skip it.
            /// </summary>
            public bool Fused;
        }

        /// <summary>
        /// Whether an island whose bodies are all vessel parts is solved by one fused job for the whole step. In
        /// every substep, that job updates the island's contacts and joints and builds their rows, fills the parts,
        /// solves the vessels with the contacts and joints as their boundary block, writes the loads and integrates
        /// the parts' bodies. It does this without the per-body setup of the sparse solve, without the jobs and waits
        /// of each substep, and without the separate integration job. It is on by default. The backend reads RETOOLED_VESSEL_FUSED when the first
        /// world attaches, and the value 0 turns it off. The flag is stored in a SharedStatic, because the jobs are
        /// scheduled from Burst code.
        /// </summary>
        public static bool VesselFused
        {
            get => !VesselFusedOff.Data;
            set => VesselFusedOff.Data = !value;
        }

        struct VesselFusedKey { }

        /// <summary>
        /// Whether the fused vessel job also takes the vessel islands that have contacts or joints with rows. When this
        /// is false, it takes only the islands that hold vessels and nothing else, as it did before it could build a
        /// boundary block. It is on by default, and the bench turns it off to compare the two.
        /// </summary>
        internal static bool FusedContacts
        {
            get => !FusedContactsOff.Data;
            set => FusedContactsOff.Data = !value;
        }

        struct FusedContactsKey { }

        static readonly SharedStatic<bool> FusedContactsOff = SharedStatic<bool>.GetOrCreate<FusedContactsKey>();

        /// <summary>
        /// The number of island solves, each one for a whole step, that the fused vessel job has done since the
        /// program started. Tests read it to check that an island went to the fused job.
        /// </summary>
        internal static long FusedIslandSolves => FusedIslandSolveCount.Data;

        struct FusedIslandSolveKey { }

        static readonly SharedStatic<long> FusedIslandSolveCount = SharedStatic<long>.GetOrCreate<FusedIslandSolveKey>();

        static readonly SharedStatic<bool> VesselFusedOff = SharedStatic<bool>.GetOrCreate<VesselFusedKey>();

        /// <summary>
        /// The number of vessel parts at which an island whose bodies are all vessel parts goes to the fused vessel
        /// job. A smaller island stays with the substeps' jobs. By default, every such island goes to the fused job,
        /// however small it is and however many there are. The fused job gives each island a team of its own and
        /// solves several islands at once, so on the bench (whole steps, the minimum of three rounds) small craft are
        /// as fast or faster there than in the substeps' jobs. Floating, sixteen craft of 128 parts take 0.78 ms
        /// instead of 1.10 ms on eight threads and 1.94 ms instead of 2.26 ms on one. Thirty-two craft of 32 parts take
        /// 0.48 ms instead of 0.53 ms on eight threads, and 64 craft of 8 parts take 0.37 ms instead of 0.46 ms. On
        /// the ground the times are about the same: sixteen craft of 128 parts take 2.17 ms instead of 2.24 ms on eight
        /// threads and 6.08 ms instead of 6.14 ms on one. Before round 4, the fused job solved one island after
        /// another, and sixteen craft of 128 parts took 0.8 ms more floating and 2 ms more on the ground there. The backend reads RETOOLED_VESSEL_FUSED_PARTS when the first world attaches,
        /// and tests set this to keep small vessels in the substeps' jobs. The value is stored in a SharedStatic,
        /// because the jobs are scheduled from Burst code.
        /// </summary>
        public static int FusedVesselParts
        {
            get => FusedVesselPartsValue.Data > 0 ? FusedVesselPartsValue.Data : DefaultFusedVesselParts;
            set => FusedVesselPartsValue.Data = value;
        }

        /// <summary>The value of <see cref="FusedVesselParts"/> until something sets it.</summary>
        public const int DefaultFusedVesselParts = 1;

        struct FusedVesselPartsKey { }

        static readonly SharedStatic<int> FusedVesselPartsValue = SharedStatic<int>.GetOrCreate<FusedVesselPartsKey>();

        /// <summary> The number of vessel parts at which an island's solve is shared among lanes. </summary>
        const int SharedSolveParts = 256;

        const int SharedSolveRows = 512;

        /// <summary>
        /// Selects the earlier method of building direct island rows, which updates joints separately and computes
        /// regularization for each Jacobian. Tests use this to compare against updating joints during row construction
        /// and sharing regularization. Read when scheduling the jobs.
        /// </summary>
        internal static bool LegacyRowBuild
        {
            get => LegacyRowBuildFlag.Data;
            set => LegacyRowBuildFlag.Data = value;
        }

        struct LegacyRowBuildKey { }

        static readonly SharedStatic<bool> LegacyRowBuildFlag = SharedStatic<bool>.GetOrCreate<LegacyRowBuildKey>();

        unsafe struct IslandHeader
        {
            public JacobianHeader* Header;
            public int Island;
        }

        /// <summary>
        /// Stores rows for all direct islands in the current substep, with each island's rows in a contiguous range.
        /// Jobs can resize these lists, so they store pointers to the list structures instead of pointers to the data
        /// that may move.
        /// </summary>
        unsafe struct DirectIslandBuffers
        {
            /// <summary>
            /// The starting row offset of each Jacobian within its island, or -1 if its rows are not built in this
            /// substep.
            /// </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<int>* HeaderRows;

            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Jacobian;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Epsilon;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Lower;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Upper;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* RightHandSide;

            /// <summary>
            /// The position correction for each contact row, used to move bodies without changing their velocities.
            /// Contains -1 for rows that are not contacts.
            /// </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Recovery;

            [NativeDisableUnsafePtrRestriction] public UnsafeList<LCP.MLCPIndexFlag>* IndexFlags;

            /// <summary>
            /// For each island, contains 1 if any row requires a bounded solve; otherwise contains 0.
            /// </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<int>* Bounded;

            /// <summary>
            /// One team for each substep. A team cannot be reused after its leader finishes and releases the helpers.
            /// </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<SparseSolveTeam>* Teams;

            /// <summary>
            /// One team for each island that the fused vessel job solves, the island with the most parts first.
            /// <see cref="MarkFusedVesselIslandsJob"/> fills this list.
            /// </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<FusedTeam>* FusedTeams;

            public DirectIslandBuffers(int substeps, Allocator allocator)
            {
                // The team after the substeps' teams never runs. The main thread configures it here, because only the
                // main thread can read the clock's conversion rate, and each fused island's team starts as a copy of it.
                Teams = UnsafeList<SparseSolveTeam>.Create(substeps + 1, allocator);
                FusedTeams = UnsafeList<FusedTeam>.Create(4, allocator);
                Teams->Resize(substeps + 1, NativeArrayOptions.ClearMemory);
                for (int substep = 0; substep <= substeps; substep++)
                {
                    Teams->ElementAt(substep).Configure();
                }

                HeaderRows = UnsafeList<int>.Create(0, allocator);
                Jacobian = UnsafeList<float>.Create(0, allocator);
                Epsilon = UnsafeList<float>.Create(0, allocator);
                Lower = UnsafeList<float>.Create(0, allocator);
                Upper = UnsafeList<float>.Create(0, allocator);
                RightHandSide = UnsafeList<float>.Create(0, allocator);
                Recovery = UnsafeList<float>.Create(0, allocator);
                IndexFlags = UnsafeList<LCP.MLCPIndexFlag>.Create(0, allocator);
                Bounded = UnsafeList<int>.Create(0, allocator);
            }

            public void Resize(int rows)
            {
                Jacobian->Resize(rows * SparseConstraintSolver.ColumnsPerRow);
                Epsilon->Resize(rows);
                Lower->Resize(rows);
                Upper->Resize(rows);
                RightHandSide->Resize(rows);
                Recovery->Resize(rows);
                IndexFlags->Resize(rows);
            }

            public JacobianRows Rows(in DirectIsland island)
            {
                return new JacobianRows(Jacobian->Ptr + island.FirstRow * (long)SparseConstraintSolver.ColumnsPerRow, island.Rows);
            }

            public static Vector View(UnsafeList<float>* list, in DirectIsland island)
            {
                return Vector.Create(default, list->Ptr + island.FirstRow, 1, island.Rows);
            }

            /// <summary>
            /// Provides the island's index flags in the array format used by row construction and solving.
            /// </summary>
            public NativeArray<LCP.MLCPIndexFlag> Flags(in DirectIsland island)
            {
                var flags = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<LCP.MLCPIndexFlag>(
                    IndexFlags->Ptr + island.FirstRow, island.Rows, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref flags, AtomicSafetyHandle.GetTempMemoryHandle());
#endif
                return flags;
            }

            public void Destroy()
            {
                UnsafeList<int>.Destroy(HeaderRows);
                UnsafeList<float>.Destroy(Jacobian);
                UnsafeList<float>.Destroy(Epsilon);
                UnsafeList<float>.Destroy(Lower);
                UnsafeList<float>.Destroy(Upper);
                UnsafeList<float>.Destroy(RightHandSide);
                UnsafeList<float>.Destroy(Recovery);
                UnsafeList<LCP.MLCPIndexFlag>.Destroy(IndexFlags);
                UnsafeList<int>.Destroy(Bounded);
                UnsafeList<SparseSolveTeam>.Destroy(Teams);
                UnsafeList<FusedTeam>.Destroy(FusedTeams);
            }
        }

        /// <summary>
        /// The team that solves one fused island, with the island's index, its number of vessel parts, and the number
        /// of lanes that help the team at the moment. A lane that has no island of its own to lead helps the team that
        /// has the most parts for each lane that works on it.
        /// </summary>
        unsafe struct FusedTeam
        {
            public SparseSolveTeam Team;
            public int Island;
            public int Parts;
            public int Helpers;

            // Keep the next team's flags out of the cache line that holds this team's count of helpers, which the lanes
            // change while they pick a team.
            fixed byte separation[64];
        }

        struct DirectIslandWork
        {
            public NativeList<IslandHeader> Headers;
            public NativeList<DirectIsland> Islands;
            public DirectIslandBuffers Buffers;

            /// <summary>
            /// The indices of each island's vessels in the step's vessel set.
            /// </summary>
            public NativeList<int> IslandVessels;

            public DirectIslandWork(int substeps, Allocator allocator)
            {
                Headers = new NativeList<IslandHeader>(allocator);
                Islands = new NativeList<DirectIsland>(allocator);
                Buffers = new DirectIslandBuffers(substeps, allocator);
                IslandVessels = new NativeList<int>(allocator);
            }

            public JobHandle Dispose(JobHandle inputDeps)
            {
                JobHandle buffers = new DestroyDirectIslandBuffersJob { Buffers = Buffers }.Schedule(inputDeps);
                return JobHandle.CombineDependencies(
                    JobHandle.CombineDependencies(Headers.Dispose(inputDeps), Islands.Dispose(inputDeps), buffers),
                    IslandVessels.Dispose(inputDeps));
            }
        }

        [BurstCompile]
        struct DestroyDirectIslandBuffersJob : IJob
        {
            public DirectIslandBuffers Buffers;

            public void Execute()
            {
                Buffers.Destroy();
            }
        }

        /// <summary>
        /// Lists each direct island's Jacobians in iterator order, with joints before contacts. Decides once per step
        /// whether to include the island's contacts in its solve, then counts the rows for the first substep.
        /// </summary>
        [BurstCompile]
        unsafe struct IndexDirectIslandsJob : IJob
        {
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            public NativeStream.Reader JacobiansReader;

            public DirectIslandWork Work;

            public DirectSolverCache Cache;

            /// <summary>The world's bodies, which tell the cache who each island's first body is.</summary>
            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            /// <summary>The number of moving bodies in the world, which the islands' shared index of bodies covers.</summary>
            public int MotionCount;

            public void Execute()
            {
                int islandCount = Info.NumIslands;

                // The job that found the islands also listed each island's vessels. Without that list, the islands have
                // no vessels.
                bool vesselsKnown = Info.IslandVesselInfos.Length == islandCount;
                Work.IslandVessels.Clear();
                if (Cache.IsCreated)
                {
                    var firstBodies = new NativeArray<int>(islandCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    var firstIdentities = new NativeArray<ulong>(islandCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    var bodies = (RigidBody*)Bodies.GetUnsafeReadOnlyPtr();
                    for (int island = 0; island < islandCount; island++)
                    {
                        firstBodies[island] = Info.IslandBodies[Info.FirstIslandBodyIndices[island]];
                        firstIdentities[island] = DirectSolverCache.BodyIdentity(bodies, Bodies.Length, firstBodies[island]);
                    }

                    Cache.Assign(firstBodies, firstIdentities);
                    Cache.ClearCompactIndex(MotionCount);
                }

                Work.Islands.ResizeUninitialized(islandCount);
                Work.Buffers.Bounded->Resize(islandCount);

                // Count each header's rows while reading it, then convert the counts to starting row offsets after
                // listing the islands. This avoids another pass over headers that would mostly no longer be in the CPU
                // cache.
                UnsafeList<int>* headerRows = Work.Buffers.HeaderRows;
                headerRows->Clear();
                for (int island = 0; island < islandCount; island++)
                {
                    int numContacts = Info.IslandContactCounts[island];
                    int firstHeader = Work.Headers.Length;
                    bool touchesSomething = false;

                    // The iterator reads joint pairs before contact pairs. Identify joint Jacobians by excluding the
                    // Contact and Trigger types, which are produced only by contact pairs.
                    int jointHeaders = 0;
                    var iterator = new IslandJacobianIterator(JacobiansReader, Info, island, numContacts);
                    while (iterator.HasJacobiansLeft())
                    {
                        ref JacobianHeader header = ref iterator.ReadJacobianHeader();
                        touchesSomething |= header.Type == JacobianType.Contact && header.Enabled;
                        jointHeaders += header.Type != JacobianType.Contact && header.Type != JacobianType.Trigger ? 1 : 0;
                        Work.Headers.Add(new IslandHeader
                        {
                            Header = (JacobianHeader*)UnsafeUtility.AddressOf(ref header),
                            Island = island,
                        });
                        headerRows->Add(GetConstraintRows(ref header));
                    }

                    int firstVessel = Work.IslandVessels.Length, vesselParts = 0, otherBodies = 0;
                    if (vesselsKnown)
                    {
                        DispatchPairSequencer.IslandVesselInfo vessels = Info.IslandVesselInfos[island];
                        Work.IslandVessels.AddRange((int*)Info.IslandVessels.GetUnsafeReadOnlyPtr() + vessels.FirstVessel, vessels.VesselCount);
                        vesselParts = vessels.VesselParts;
                        otherBodies = vessels.OtherBodies;
                    }

                    Work.Islands[island] = new DirectIsland
                    {
                        FirstHeader = firstHeader,
                        JointEnd = firstHeader + jointHeaders,
                        SolveEnd = touchesSomething ? Work.Headers.Length : firstHeader + jointHeaders,
                        HeaderEnd = Work.Headers.Length,
                        ContactPairs = touchesSomething ? numContacts : 0,
                        FirstVessel = firstVessel,
                        VesselCount = Work.IslandVessels.Length - firstVessel,
                        VesselParts = vesselParts,
                        OtherBodies = otherBodies,
                    };
                }

                CountIslandRowsJob.CountRows(Work, true);
            }
        }

        [BurstCompile]
        [NoAlias]
        unsafe struct UpdateIslandJacobiansJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeArray<IslandHeader> Headers;

            [ReadOnly]
            public NativeArray<DirectIsland> Islands;

            /// <summary>
            /// When true, skips joint Jacobians so <see cref="BuildIslandRowsJob"/> updates them. Joints precede
            /// contacts in each island.
            /// </summary>
            public bool ContactsOnly;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            public void Execute(int index)
            {
                // The fused vessel job updates the Jacobians of its own islands, at the same time as this job runs, so
                // skip them here.
                DirectIsland island = Islands[Headers[index].Island];
                if (island.Fused || (ContactsOnly && index < island.JointEnd))
                {
                    return;
                }

                UpdateJacobian(ref *Headers[index].Header, MotionDatas, MotionVelocities, Bodies, StepInput);
            }
        }

        /// <summary>
        /// Counts the Jacobian rows for the current substep and assigns a contiguous range of rows to each island.
        /// Contact row counts can change as contact points appear or disappear. Joint row counts are fixed by
        /// construction, so only contacts need to be counted again after the first substep. For the first substep,
        /// <see cref="IndexDirectIslandsJob"/> has already stored each header's row count in HeaderRows.
        /// </summary>
        [BurstCompile]
        unsafe struct CountIslandRowsJob : IJob
        {
            public DirectIslandWork Work;

            public void Execute()
            {
                CountRows(Work, false);
            }

            internal static void CountRows(DirectIslandWork work, bool first)
            {
                UnsafeList<int>* headerRows = work.Buffers.HeaderRows;
                int totalRows = 0;
                for (int i = 0; i < work.Islands.Length; i++)
                {
                    ref DirectIsland island = ref work.Islands.ElementAt(i);
                    int rows = 0;
                    if (first)
                    {
                        for (int h = island.FirstHeader; h < island.SolveEnd; h++)
                        {
                            int count = (*headerRows)[h];
                            (*headerRows)[h] = rows;
                            rows += count;
                            island.JointRows = h < island.JointEnd ? rows : island.JointRows;
                        }

                        for (int h = island.SolveEnd; h < island.HeaderEnd; h++)
                        {
                            (*headerRows)[h] = -1;
                        }
                    }
                    else if (island.Fused)
                    {
                        // The fused vessel job counts the rows of its own islands, and it changes their Jacobians while
                        // this job runs.
                        rows = 0;
                    }
                    else
                    {
                        rows = island.JointRows;
                        for (int h = island.JointEnd; h < island.SolveEnd; h++)
                        {
                            (*headerRows)[h] = rows;
                            rows += GetConstraintRows(ref *work.Headers[h].Header);
                        }
                    }

                    island.FirstRow = totalRows;
                    island.Rows = rows;
                    (*work.Buffers.Bounded)[i] = 0;
                    totalRows += rows;
                }

                work.Buffers.Resize(totalRows);
            }
        }

        /// <summary>
        /// Builds rows for one Jacobian in a sparse island. Each invocation writes only the rows assigned to that
        /// Jacobian.
        /// </summary>
        [BurstCompile]
        [NoAlias]
        unsafe struct BuildIslandRowsJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeArray<IslandHeader> Headers;

            [ReadOnly]
            public NativeArray<DirectIsland> Islands;

            public DirectIslandBuffers Work;

            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            /// <summary>
            /// Updates each joint Jacobian from the current substep's poses before building its rows. Updating and
            /// building a Jacobian do not read other Jacobians, so this gives the same result as updating all Jacobians
            /// first.
            /// </summary>
            public bool UpdateJoints;

            /// <summary>
            /// Computes the substep's default regularization once per worker instead of once per Jacobian.
            /// </summary>
            public bool ShareRegularization;

            // Compute this when a worker builds its first Jacobian, and keep it in that worker's copy of the job.
            RowRegularization regularization;
            bool hasRegularization;

            public void Execute(int index)
            {
                int islandIndex = Headers[index].Island;
                DirectIsland island = Islands[islandIndex];
                ref JacobianHeader header = ref *Headers[index].Header;

                // The fused vessel job builds the rows of its own islands and moves their bodies at the same time as this
                // job runs, so this job must not read those Jacobians or bodies.
                if (island.Fused)
                {
                    return;
                }

                if (UpdateJoints && index < island.JointEnd)
                {
                    UpdateJacobian(ref header, MotionDatas, MotionVelocities, Bodies, StepInput);
                }

                int startRow = (*Work.HeaderRows)[index];
                if (startRow < 0)
                {
                    return;
                }

                int jacRows = GetConstraintRows(ref header);
                if (jacRows == 0)
                {
                    return;
                }

                RowRegularization* shared = null;
                if (ShareRegularization)
                {
                    if (!hasRegularization)
                    {
                        regularization = RowRegularization.Of(StepInput);
                        hasRegularization = true;
                    }

                    shared = (RowRegularization*)UnsafeUtility.AddressOf(ref regularization);
                }

                bool lcp = BuildHeaderRows(ref header, startRow, jacRows, IslandRowBuffers.Of(Work, island), MotionVelocities, MotionDatas,
                    StepInput, island.ContactPairs > 0, shared);

                // All workers that update this flag store 1, and the solver reads it only after this job finishes, so a
                // plain store is enough. Check the flag first to avoid repeated writes to the same cache line by
                // workers building rows for one island.
                int* bounded = Work.Bounded->Ptr + islandIndex;
                if (lcp && *bounded == 0)
                {
                    *bounded = 1;
                }
            }
        }

        /// <summary>
        /// Where one island's rows are kept for a substep: the Jacobian, epsilon, bounds, right-hand side, push-out
        /// and index flag of each row, from the island's first row on.
        /// </summary>
        unsafe struct IslandRowBuffers
        {
            public float* Jacobian;
            public float* Epsilon;
            public float* Lower;
            public float* Upper;
            public float* RightHandSide;
            public float* Recovery;
            public LCP.MLCPIndexFlag* Flags;
            public int Rows;

            public static IslandRowBuffers Of(in DirectIslandBuffers work, in DirectIsland island)
            {
                return new IslandRowBuffers
                {
                    Jacobian = work.Jacobian->Ptr + island.FirstRow * (long)SparseConstraintSolver.ColumnsPerRow,
                    Epsilon = work.Epsilon->Ptr + island.FirstRow,
                    Lower = work.Lower->Ptr + island.FirstRow,
                    Upper = work.Upper->Ptr + island.FirstRow,
                    RightHandSide = work.RightHandSide->Ptr + island.FirstRow,
                    Recovery = work.Recovery->Ptr + island.FirstRow,
                    Flags = work.IndexFlags->Ptr + island.FirstRow,
                    Rows = island.Rows,
                };
            }
        }

        /// <summary>
        /// Builds the rows of one Jacobian of an island, from its first row <paramref name="startRow"/> among the
        /// island's rows. Every row starts free and unbounded, and the row build sets the bounds that the row needs.
        /// Returns whether a row needs a bounded solve. A Jacobian writes only its own rows, so the lanes can build
        /// the Jacobians of one island at the same time.
        /// </summary>
        static unsafe bool BuildHeaderRows(ref JacobianHeader header, int startRow, int jacRows, in IslandRowBuffers buffers,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, in StepInput stepInput,
            bool sparseContacts, RowRegularization* regularization)
        {
            for (int row = startRow; row < startRow + jacRows; row++)
            {
                buffers.Lower[row] = float.MinValue;
                buffers.Upper[row] = float.MaxValue;
                buffers.Flags[row] = LCP.MLCPIndexFlag.Free;
                buffers.Recovery[row] = -1.0f;
            }

            var flags = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<LCP.MLCPIndexFlag>(buffers.Flags, buffers.Rows, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref flags, AtomicSafetyHandle.GetTempMemoryHandle());
#endif
            var context = new RowBuildContext
            {
                MotionVelocities = motionVelocities,
                MotionDatas = motionDatas,
                StepInput = stepInput,
                JacobianRows = new JacobianRows(buffers.Jacobian, buffers.Rows),
                Eps = Vector.Create(default, buffers.Epsilon, 1, buffers.Rows),
                L = Vector.Create(default, buffers.Lower, 1, buffers.Rows),
                U = Vector.Create(default, buffers.Upper, 1, buffers.Rows),
                RightHandSide = Vector.Create(default, buffers.RightHandSide, 1, buffers.Rows),
                Recovery = buffers.Recovery,
                IndexSetArray = flags,
                SparseContacts = sparseContacts,
                SolveSparsely = true,
                Regularization = regularization,
            };

            Vector* JBlocks = stackalloc Vector[jacRows];
            BuildJacobianRows(ref header, startRow, jacRows, ref context, JBlocks, null);
            return context.IsLCP;
        }

        /// <summary>
        /// Solves one direct island using its constructed rows and applies the resulting impulses. Solves joints and
        /// contacts, including friction, together. This accounts for the combined constraints when supporting a heavy
        /// stack on a light part or transmitting friction forces through a craft's joints.
        /// </summary>
        [BurstCompile]
        [NoAlias]
        unsafe struct SolveDirectIslandJob : IJobParallelForDefer
        {
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            public NativeStream.Reader JacobiansReader;

            [ReadOnly]
            public NativeArray<IslandHeader> Headers;

            [ReadOnly]
            public NativeArray<DirectIsland> Islands;

            public DirectIslandBuffers Work;

            [ReadOnly]
            public NativeArray<int> IslandVessels;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            /// <summary>The world's bodies, which tell an island's forest who its bodies are.</summary>
            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            public void Execute(int islandIndex)
            {
                if (!Islands[islandIndex].SolvesInTeam)
                {
                    SolveDirectIsland(Info, JacobiansReader, Headers, Islands, Work, IslandVessels, ref MotionVelocities, MotionDatas, Bodies, StepInput, islandIndex, null);
                }
            }
        }

        /// <summary>
        /// Solves large islands sequentially, using the team's available lanes for each solve. Runs concurrently with
        /// <see cref="SolveDirectIslandJob"/>, which solves the smaller islands.
        /// </summary>
        [BurstCompile]
        [NoAlias]
        unsafe struct SolveSharedDirectIslandsJob : IJobParallelFor
        {
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            public NativeStream.Reader JacobiansReader;

            [ReadOnly]
            public NativeArray<IslandHeader> Headers;

            [ReadOnly]
            public NativeArray<DirectIsland> Islands;

            public DirectIslandBuffers Work;

            [ReadOnly]
            public NativeArray<int> IslandVessels;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            /// <summary>The world's bodies, which tell an island's forest who its bodies are.</summary>
            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            [NativeDisableUnsafePtrRestriction]
            public SparseSolveTeam* Team;

            static readonly ProfilerMarker HelpMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Island.HelpSolve");

            public void Execute(int lane)
            {
                if (!Team->TryLead())
                {
                    using (HelpMarker.Auto())
                    {
                        Team->Help();
                    }

                    return;
                }

                for (int islandIndex = 0; islandIndex < Islands.Length; islandIndex++)
                {
                    if (Islands[islandIndex].SolvesInTeam)
                    {
                        SolveDirectIsland(Info, JacobiansReader, Headers, Islands, Work, IslandVessels, ref MotionVelocities, MotionDatas, Bodies, StepInput, islandIndex, Team);
                    }
                }

                Team->Finish();
            }
        }

        static readonly ProfilerMarker DirectSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Island.DirectSolve");

        static unsafe void SolveDirectIsland(in DispatchPairSequencer.DirectSolverSchedulerInfo info, in NativeStream.Reader jacobiansReader,
            in NativeArray<IslandHeader> headers, in NativeArray<DirectIsland> islands, in DirectIslandBuffers work,
            in NativeArray<int> islandVessels, ref NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, in NativeArray<RigidBody> bodies, in StepInput stepInput, int islandIndex,
            SparseSolveTeam* team)
        {
            DirectIsland island = islands[islandIndex];
            using (DirectSolveMarker.Auto())
            {
                if (!island.Fused && (island.Rows > 0 || island.VesselCount > 0))
                {
                    JacobianRows rows = work.Rows(island);
                    Vector rightHandSide = DirectIslandBuffers.View(work.RightHandSide, island);
                    SolveSparsely(info, islandIndex, jacobiansReader, motionVelocities, motionDatas, stepInput, island.Rows, rows,
                        DirectIslandBuffers.View(work.Epsilon, island), DirectIslandBuffers.View(work.Lower, island),
                        DirectIslandBuffers.View(work.Upper, island), rightHandSide, work.Flags(island),
                        (*work.Bounded)[islandIndex] != 0, island.ContactPairs, team, true,
                        (IslandHeader*)headers.GetUnsafeReadOnlyPtr() + island.FirstHeader, work.HeaderRows->Ptr + island.FirstHeader,
                        island.SolveEnd - island.FirstHeader, work.Recovery->Ptr + island.FirstRow,
                        (int*)islandVessels.GetUnsafeReadOnlyPtr() + island.FirstVessel, island.VesselCount,
                        (RigidBody*)bodies.GetUnsafeReadOnlyPtr(), bodies.Length);
                }
            }
        }

        [BurstCompile]
        [NoAlias]
        struct IntegrateIslandBodiesJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeArray<int> IslandBodies;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            /// <summary>The index of each island's first body in <see cref="IslandBodies"/>.</summary>
            [ReadOnly]
            public NativeArray<int> FirstIslandBodies;

            // The fused vessel job integrates the moving bodies of the fused islands itself. This job reads only which
            // islands are fused, and the substeps' jobs don't change that field.
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<DirectIsland> Islands;

            public float Timestep;

            public void Execute(int index)
            {
                int body = IslandBodies[index];
                if (!Islands[IslandOf(index)].Fused || MotionVelocities[body].IsKinematic)
                {
                    Integrator.IntegrateMotion(body, MotionDatas, MotionVelocities, Timestep);
                }
            }

            /// <summary>Finds the island whose bodies include the given entry of <see cref="IslandBodies"/>.</summary>
            int IslandOf(int index)
            {
                int low = 0, high = FirstIslandBodies.Length - 1;
                while (low < high)
                {
                    int middle = (low + high + 1) >> 1;
                    if (FirstIslandBodies[middle] <= index)
                    {
                        low = middle;
                    }
                    else
                    {
                        high = middle - 1;
                    }
                }

                return low;
            }
        }

        /// <summary>
        /// Marks each island whose bodies are all vessel parts, and that has at least <see cref="FusedVesselParts"/>
        /// vessel parts, for the fused vessel job. The island's contacts and joints go into the fused job's boundary
        /// block. Then gives every vessel island the forest that will solve it, see <see cref="AssignVesselForests"/>.
        /// The job works per island and per vessel, and never per part.
        /// </summary>
        [BurstCompile]
        unsafe struct MarkFusedVesselIslandsJob : IJob
        {
            public NativeList<DirectIsland> Islands;

            [ReadOnly]
            public NativeList<int> IslandVessels;

            public VesselSet Vessels;

            public DirectSolverCache Cache;

            /// <summary>
            /// Whether the fused job is on, and the value of <see cref="FusedVesselParts"/>, when the job was scheduled.
            /// </summary>
            public bool Fuse;

            public int MinimumParts;

            /// <summary>The value of <see cref="FusedContacts"/> when the job was scheduled.</summary>
            public bool Contacts;

            /// <summary>The buffers that hold a team for each fused island and the configured team that they copy.</summary>
            public DirectIslandBuffers Buffers;

            public void Execute()
            {
                for (int i = 0; Fuse && i < Islands.Length; i++)
                {
                    DirectIsland island = Islands[i];
                    if (!Fusable(island, MinimumParts))
                    {
                        continue;
                    }

                    island.Fused = true;
                    Islands[i] = island;
                }

                ListFusedTeams();
                AssignVesselForests(Cache, Islands.AsArray(), IslandVessels.AsArray(), Vessels);
            }

            /// <summary>
            /// Gives each fused island a team of its own. The teams are sorted by the islands' part counts, the most
            /// parts first and then by island index, so that the first lanes of the fused job lead the largest islands.
            /// Each team is a copy of the unused team after the substeps' teams, which the main thread configured.
            /// </summary>
            void ListFusedTeams()
            {
                UnsafeList<FusedTeam>* teams = Buffers.FusedTeams;
                SparseSolveTeam configured = Buffers.Teams->Ptr[Buffers.Teams->Length - 1];
                teams->Clear();
                for (int i = 0; i < Islands.Length; i++)
                {
                    if (!Islands[i].Fused)
                    {
                        continue;
                    }

                    var team = new FusedTeam { Team = configured, Island = i, Parts = Islands[i].VesselParts };
                    int at = teams->Length;
                    teams->Add(team);

                    // The islands come in index order, so an island moves ahead only of islands with fewer parts.
                    while (at > 0 && teams->Ptr[at - 1].Parts < team.Parts)
                    {
                        teams->Ptr[at] = teams->Ptr[at - 1];
                        at--;
                    }

                    teams->Ptr[at] = team;
                }
            }

            /// <summary>
            /// Whether the fused job can take an island of at least <paramref name="minimumParts"/> vessel parts: all
            /// its bodies must be vessel parts.
            /// </summary>
            bool Fusable(in DirectIsland island, int minimumParts)
            {
                return island.VesselCount > 0 && island.VesselParts >= minimumParts && island.OtherBodies == 0
                    && (Contacts || (island.JointRows == 0 && island.Rows == 0 && island.JointEnd == island.HeaderEnd));
            }
        }

        /// <summary>
        /// Solves every island that <see cref="MarkFusedVesselIslandsJob"/> marked, for all the substeps of the step.
        /// Each island has a team of its own, so that several islands can be solved at the same time. Each lane first
        /// leads every island that has no leader yet, the largest island first. A lane that finds no island to lead
        /// helps the stages of the islands that are still being solved, which split the parts into ranges. With one
        /// thread, the islands are solved one after another, as before. Each island's results do not depend on which
        /// lanes helped it, because each stage combines its tasks' results in the same order.
        /// </summary>
        [BurstCompile]
        unsafe struct SolveFusedVesselIslandsJob : IJobParallelFor
        {
            // The substeps' jobs change the row counts of the islands while this job runs. This job reads only the
            // fields that say which islands are fused and which vessels they hold, and those fields do not change.
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<DirectIsland> Islands;

            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<int> IslandVessels;

            /// <summary>
            /// The Jacobians of every island. This job updates and reads only the Jacobians of the fused islands, which
            /// the substeps' jobs skip.
            /// </summary>
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<IslandHeader> Headers;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            /// <summary>A team for each fused island, the island with the most parts first.</summary>
            [NativeDisableUnsafePtrRestriction]
            public UnsafeList<FusedTeam>* Teams;

            /// <summary>
            /// The number of lanes that the job was scheduled with. With one lane, no other lane can help an island,
            /// so the lane solves each island without a team. Publishing the stages to a team cost 0.05 to 0.1 ms a
            /// step on one thread for sixteen craft of 128 parts.
            /// </summary>
            public int Lanes;

            static readonly ProfilerMarker HelpMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Vessel.HelpFused");

            public void Execute(int lane)
            {
                FusedTeam* teams = Teams->Ptr;
                int count = Teams->Length;

                // Lead each island that no lane leads yet. A lane that finishes an island looks for another one.
                for (int t = 0; t < count; t++)
                {
                    if (!teams[t].Team.TryLead())
                    {
                        continue;
                    }

                    DirectIsland island = Islands[teams[t].Island];
                    System.Threading.Interlocked.Increment(ref FusedIslandSolveCount.Data);
                    SolveFusedVesselIsland(island, (IslandHeader*)Headers.GetUnsafeReadOnlyPtr() + island.FirstHeader,
                        (int*)IslandVessels.GetUnsafeReadOnlyPtr() + island.FirstVessel, MotionVelocities, MotionDatas, Bodies,
                        StepInput, Lanes > 1 ? &teams[t].Team : null);
                    teams[t].Team.Finish();
                }

                // Every island has a leader now. With several islands, work on the open stages of every island, the
                // largest island first, for as long as some stage is open. A lane that has found no open stage for the
                // spin time waits on the island that has the most parts for each lane that works on it, counting its
                // leader. It works on that island's stages until the island is solved, and then looks at all the
                // islands again. With one island, the lane waits on it at once, as before.
                using (HelpMarker.Auto())
                {
                    while (true)
                    {
                        if (count > 1 && !SweepOpenStages(teams, count))
                        {
                            return;
                        }

                        int best = -1;
                        long bestParts = 0, bestLanes = 1;
                        for (int t = 0; t < count; t++)
                        {
                            if (teams[t].Team.Finished)
                            {
                                continue;
                            }

                            // This island has more parts for each lane than the best so far when its parts times the best
                            // island's lanes are more than the best island's parts times its lanes.
                            long lanes = System.Threading.Volatile.Read(ref teams[t].Helpers) + 1;
                            if (best < 0 || teams[t].Parts * bestLanes > bestParts * lanes)
                            {
                                best = t;
                                bestParts = teams[t].Parts;
                                bestLanes = lanes;
                            }
                        }

                        if (best < 0)
                        {
                            return;
                        }

                        System.Threading.Interlocked.Increment(ref teams[best].Helpers);
                        teams[best].Team.Help();
                        System.Threading.Interlocked.Decrement(ref teams[best].Helpers);

                        // A helper that stayed idle past its patience gives its thread back to the job system rather than
                        // wait for another island.
                        if (!teams[best].Team.Finished)
                        {
                            return;
                        }
                    }
                }
            }

            /// <summary>
            /// Runs the tasks of every open stage of the islands, the largest island first, until no stage has been
            /// open for the teams' spin time. Returns false if every island is solved.
            /// </summary>
            static bool SweepOpenStages(FusedTeam* teams, int count)
            {
                long spinTicks = teams[0].Team.SpinTicks;
                long idleSince = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                while (true)
                {
                    bool unsolved = false, worked = false;
                    for (int t = 0; t < count; t++)
                    {
                        if (!teams[t].Team.Finished)
                        {
                            unsolved = true;
                            worked |= teams[t].Team.TryWork();
                        }
                    }

                    if (!unsolved)
                    {
                        return false;
                    }

                    long now = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    if (worked)
                    {
                        idleSince = now;
                    }
                    else if (now - idleSince > spinTicks)
                    {
                        return true;
                    }
                    else
                    {
                        Unity.Burst.Intrinsics.Common.Pause();
                    }
                }
            }
        }
    }
}
