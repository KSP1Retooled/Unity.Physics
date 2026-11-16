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
        /// </summary>
        internal static unsafe SimulationJobHandles ScheduleIslandSolveJobs(in PhysicsWorld world, StepInput stepInput, int directIslandSubsteps,
            in NativeStream jacobians, in NativeStream collisionEvents, in NativeStream triggerEvents, in NativeStream impulseEvents,
            in DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo, StabilizationData solverStabilizationData,
            JobHandle jacobiansBuilt, JobHandle inputDeps)
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

            iterativeHandle = new DestroyTeamsJob { Teams = teams }.Schedule(iterativeHandle);

            JobHandle islandHandle = ScheduleDirectIslandJobs(world, stepInput, directIslandSubsteps, jacobians,
                solverSchedulerInfo.DirectPairsDirectScheduling, jacobiansBuilt, inputDeps);

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
        /// preparing the solve.
        /// </summary>
        static unsafe JobHandle ScheduleDirectIslandJobs(in PhysicsWorld world, StepInput stepInput, int substeps, in NativeStream jacobians,
            in DispatchPairSequencer.DirectSolverSchedulerInfo info, JobHandle jacobiansBuilt, JobHandle inputDeps)
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
                Vessels = stepInput.VesselSet,
            }.Schedule(jacobiansBuilt);

            handle = JobHandle.CombineDependencies(handle, inputDeps);

            StepInput substepInput = stepInput;
            substepInput.Timestep /= substeps;
            substepInput.NumSubsteps = substeps;
            substepInput.DisableExportEvents = true;

            // Mark the islands that the fused vessel job solves, and their bodies, which the substeps' integration
            // then skips. The fused job runs beside the substeps' jobs, because the two never touch the same bodies.
            bool fusedVessels = VesselFused;
            var fusedBodies = new NativeArray<byte>(math.max(1, world.MotionVelocities.Length), Allocator.TempJob);
            JobHandle fused = default;
            if (fusedVessels)
            {
                handle = new MarkFusedVesselIslandsJob
                {
                    Islands = work.Islands,
                    IslandVessels = work.IslandVessels,
                    Vessels = stepInput.VesselSet,
                    MotionVelocities = world.MotionVelocities,
                    FusedBodies = fusedBodies,
                }.Schedule(handle);

                fused = new SolveFusedVesselIslandsJob
                {
                    Islands = work.Islands.AsDeferredJobArray(),
                    IslandVessels = work.IslandVessels.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    StepInput = substepInput,
                    Team = work.Buffers.Teams->Ptr + substeps,
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
                    StepInput = substepInput,
                    Team = work.Buffers.Teams->Ptr + substep,
                }.Schedule(lanes, 1, handle);

                handle = JobHandle.CombineDependencies(alone, shared);

                handle = new IntegrateIslandBodiesJob
                {
                    IslandBodies = info.IslandBodies.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    FusedBodies = fusedBodies,
                    Timestep = substepInput.Timestep,
                }.Schedule(info.IslandBodies, bodiesPerBatch, handle);
            }

            handle = JobHandle.CombineDependencies(handle, fused);
            handle = fusedBodies.Dispose(handle);
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
        /// Whether a large island that holds only vessels, with no contacts and no other joints that have rows, is
        /// solved by one fused job for the whole step. That job fills the parts, solves, writes the loads and
        /// integrates the parts' bodies in every substep, without the per-body setup of the sparse solve and without
        /// the separate integration job. It is on by default. The backend reads RETOOLED_VESSEL_FUSED when the first
        /// world attaches, and the value 0 turns it off. The flag is stored in a SharedStatic, because the jobs are
        /// scheduled from Burst code.
        /// </summary>
        public static bool VesselFused
        {
            get => !VesselFusedOff.Data;
            set => VesselFusedOff.Data = !value;
        }

        struct VesselFusedKey { }

        static readonly SharedStatic<bool> VesselFusedOff = SharedStatic<bool>.GetOrCreate<VesselFusedKey>();

        /// <summary>
        /// The number of vessel parts at which an island with only vessels goes to the fused vessel job. A smaller
        /// island stays with the substeps' jobs, which solve the small islands in parallel with each other.
        /// </summary>
        const int FusedVesselParts = 256;

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

            public DirectIslandBuffers(int substeps, Allocator allocator)
            {
                // The team after the substeps' teams belongs to the fused vessel job.
                Teams = UnsafeList<SparseSolveTeam>.Create(substeps + 1, allocator);
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
            }
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

            public VesselSet Vessels;

            public void Execute()
            {
                int islandCount = Info.NumIslands;

                // Map each dynamic body to the index of its vessel, or -1 if it is not part of a vessel.
                int bodyCount = Info.BodiesInIslands.Length;
                var bodyVessel = new NativeArray<int>(Vessels.IsCreated ? bodyCount : 0, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < bodyVessel.Length; i++)
                {
                    bodyVessel[i] = -1;
                }

                for (int v = 0; v < Vessels.VesselCount && bodyVessel.Length > 0; v++)
                {
                    VesselRecord vessel = Vessels.Vessels[v];
                    for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                    {
                        int body = Vessels.PartBodies[part];
                        if (body >= 0 && body < bodyCount)
                        {
                            bodyVessel[body] = v;
                        }
                    }
                }

                var vesselSeen = new NativeArray<int>(Vessels.IsCreated ? Vessels.VesselCount : 0, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int v = 0; v < vesselSeen.Length; v++)
                {
                    vesselSeen[v] = -1;
                }

                Work.IslandVessels.Clear();
                if (Cache.IsCreated)
                {
                    var firstBodies = new NativeArray<int>(islandCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    for (int island = 0; island < islandCount; island++)
                    {
                        firstBodies[island] = Info.IslandBodies[Info.FirstIslandBodyIndices[island]];
                    }

                    Cache.Assign(firstBodies);
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

                    int firstVessel = Work.IslandVessels.Length, vesselParts = 0;
                    if (bodyVessel.Length > 0)
                    {
                        int bodiesEnd = island + 1 < islandCount ? Info.FirstIslandBodyIndices[island + 1] : Info.IslandBodies.Length;
                        for (int b = Info.FirstIslandBodyIndices[island]; b < bodiesEnd; b++)
                        {
                            int v = bodyVessel[Info.IslandBodies[b]];
                            if (v >= 0 && vesselSeen[v] != island)
                            {
                                vesselSeen[v] = island;
                                Work.IslandVessels.Add(v);
                                vesselParts += Vessels.Vessels[v].PartCount;
                            }
                        }
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
                if (ContactsOnly && index < Islands[Headers[index].Island].JointEnd)
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

                // Initialize every row as free and unbounded. Row construction then sets bounds where required.
                int firstRow = island.FirstRow + startRow;
                for (int row = firstRow; row < firstRow + jacRows; row++)
                {
                    (*Work.Lower)[row] = float.MinValue;
                    (*Work.Upper)[row] = float.MaxValue;
                    (*Work.IndexFlags)[row] = LCP.MLCPIndexFlag.Free;
                    (*Work.Recovery)[row] = -1.0f;
                }

                var context = new RowBuildContext
                {
                    MotionVelocities = MotionVelocities,
                    MotionDatas = MotionDatas,
                    StepInput = StepInput,
                    JacobianRows = Work.Rows(island),
                    Eps = DirectIslandBuffers.View(Work.Epsilon, island),
                    L = DirectIslandBuffers.View(Work.Lower, island),
                    U = DirectIslandBuffers.View(Work.Upper, island),
                    RightHandSide = DirectIslandBuffers.View(Work.RightHandSide, island),
                    Recovery = Work.Recovery->Ptr + island.FirstRow,
                    IndexSetArray = Work.Flags(island),
                    SparseContacts = island.ContactPairs > 0,
                    SolveSparsely = true,
                };

                if (ShareRegularization)
                {
                    if (!hasRegularization)
                    {
                        regularization = RowRegularization.Of(StepInput);
                        hasRegularization = true;
                    }

                    context.Regularization = (RowRegularization*)UnsafeUtility.AddressOf(ref regularization);
                }

                Vector* JBlocks = stackalloc Vector[jacRows];
                BuildJacobianRows(ref header, startRow, jacRows, ref context, JBlocks, null);

                // All workers that update this flag store 1, and the solver reads it only after this job finishes, so a
                // plain store is enough. Check the flag first to avoid repeated writes to the same cache line by
                // workers building rows for one island.
                int* bounded = Work.Bounded->Ptr + islandIndex;
                if (context.IsLCP && *bounded == 0)
                {
                    *bounded = 1;
                }
            }
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

            public StepInput StepInput;

            public void Execute(int islandIndex)
            {
                if (!Islands[islandIndex].SolvesInTeam)
                {
                    SolveDirectIsland(Info, JacobiansReader, Headers, Islands, Work, IslandVessels, ref MotionVelocities, MotionDatas, StepInput, islandIndex, null);
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
                        SolveDirectIsland(Info, JacobiansReader, Headers, Islands, Work, IslandVessels, ref MotionVelocities, MotionDatas, StepInput, islandIndex, Team);
                    }
                }

                Team->Finish();
            }
        }

        static readonly ProfilerMarker DirectSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Island.DirectSolve");

        static unsafe void SolveDirectIsland(in DispatchPairSequencer.DirectSolverSchedulerInfo info, in NativeStream.Reader jacobiansReader,
            in NativeArray<IslandHeader> headers, in NativeArray<DirectIsland> islands, in DirectIslandBuffers work,
            in NativeArray<int> islandVessels, ref NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, in StepInput stepInput, int islandIndex, SparseSolveTeam* team)
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
                        (int*)islandVessels.GetUnsafeReadOnlyPtr() + island.FirstVessel, island.VesselCount);
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

            /// <summary>Whether the fused vessel job integrates each body itself.</summary>
            [ReadOnly]
            public NativeArray<byte> FusedBodies;

            public float Timestep;

            public void Execute(int index)
            {
                int body = IslandBodies[index];
                if (FusedBodies[body] == 0)
                {
                    Integrator.IntegrateMotion(body, MotionDatas, MotionVelocities, Timestep);
                }
            }
        }

        /// <summary>
        /// Marks each large island that holds only vessels, with no contacts and no joint rows, for the fused vessel
        /// job, and marks the moving bodies of those islands' vessels.
        /// </summary>
        [BurstCompile]
        unsafe struct MarkFusedVesselIslandsJob : IJob
        {
            public NativeList<DirectIsland> Islands;

            [ReadOnly]
            public NativeList<int> IslandVessels;

            public VesselSet Vessels;

            // The job reads only whether each body is kinematic. The integration of the bodies outside islands can run
            // at the same time, and it changes only their velocities and poses.
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            public NativeArray<byte> FusedBodies;

            public void Execute()
            {
                var motions = (MotionVelocity*)MotionVelocities.GetUnsafeReadOnlyPtr();
                for (int i = 0; i < Islands.Length; i++)
                {
                    DirectIsland island = Islands[i];
                    if (island.VesselCount == 0 || island.VesselParts < FusedVesselParts || island.JointRows > 0 || island.Rows > 0
                        || island.JointEnd != island.HeaderEnd)
                    {
                        continue;
                    }

                    island.Fused = true;
                    Islands[i] = island;
                    for (int v = island.FirstVessel; v < island.FirstVessel + island.VesselCount; v++)
                    {
                        VesselRecord vessel = Vessels.Vessels[IslandVessels[v]];
                        for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                        {
                            int body = Moving(Vessels.PartBodies[part], motions, MotionVelocities.Length);
                            if (body >= 0)
                            {
                                FusedBodies[body] = 1;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Solves every island that <see cref="MarkFusedVesselIslandsJob"/> marked, for all the substeps of the step.
        /// The first lane leads, and the others help with the stages that split the parts into ranges.
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

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            public StepInput StepInput;

            [NativeDisableUnsafePtrRestriction]
            public SparseSolveTeam* Team;

            static readonly ProfilerMarker HelpMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Vessel.HelpFused");

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
                    DirectIsland island = Islands[islandIndex];
                    if (island.Fused)
                    {
                        SolveFusedVesselIsland(islandIndex, (int*)IslandVessels.GetUnsafeReadOnlyPtr() + island.FirstVessel,
                            island.VesselCount, MotionVelocities, MotionDatas, StepInput, Team);
                    }
                }

                Team->Finish();
            }
        }
    }
}
