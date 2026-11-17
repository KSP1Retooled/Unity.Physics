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
        /// Schedule iterative and direct island solving on disjoint bodies.
        /// jacobiansBuilt precedes impulse/velocity preparation; motionsSolved precedes buffer disposal and event
        /// export.
        /// Motion-only consumers may wait for motionsSolved, while event readers must wait for final completion.
        /// </summary>
        internal static unsafe SimulationJobHandles ScheduleIslandSolveJobs(in PhysicsWorld world, StepInput stepInput, int directIslandSubsteps,
            in NativeStream jacobians, in NativeStream collisionEvents, in NativeStream triggerEvents, in NativeStream impulseEvents,
            in DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo, StabilizationData solverStabilizationData,
            JobHandle jacobiansBuilt, JobHandle inputDeps, out JobHandle motionsSolved)
        {
            NativeArray<MotionVelocity> motionVelocities = world.MotionVelocities;
            DispatchPairSequencer.IterativeSolverSchedulerInfo iterativePairs = solverSchedulerInfo.IterativePairsIterativeScheduling;

            // Use team stages for iterations and relaxation to avoid separate jobs for each phase.
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

            // After integration, remove penetration correction velocities. Keep iteration indices past the
            // solver's event-export range.
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
        /// Solve iterative phases in order, sharing each phase's disjoint work items among team lanes.
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
        /// Execute one iterative work item using a private copy of the phase job, since event writers have per-item
        /// state.
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
        /// Schedule direct island Jacobian updates, row construction, solves and integration for each substep.
        /// Count rows serially; run other stages in parallel. Return solved before buffer disposal.
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

            // Fused islands own their bodies for the entire step; substep jobs must skip them.
            bool fusedVessels = VesselFused;
            JobHandle fused = default;
            handle = new MarkFusedVesselIslandsJob
            {
                Islands = work.Islands,
                IslandVessels = work.IslandVessels,
                Info = info,
                UnfusedBodies = work.UnfusedBodies,
                Vessels = stepInput.VesselSet,
                Cache = stepInput.DirectSolverCache,
                Fuse = fusedVessels && !LegacyRowBuild,
                MinimumParts = FusedVesselParts,
                Contacts = FusedContacts,
                Buffers = work.Buffers,
            }.Schedule(handle);

            // Choose serial unfused work from the previous step's counts; current counts are not known at
            // scheduling time.
            bool fold = FoldsUnfusedSubsteps(stepInput.DirectSolverCache);
            var unfused = new UnfusedSubsteps
            {
                Info = info,
                JacobiansReader = jacobians.AsReader(),
                Buffers = work.Buffers,
                StepInput = substepInput,
                LegacyRowBuild = LegacyRowBuild,
            };

            if (fusedVessels)
            {
                fused = new SolveFusedVesselIslandsJob
                {
                    Islands = work.Islands.AsDeferredJobArray(),
                    IslandVessels = work.IslandVessels.AsDeferredJobArray(),
                    Headers = work.Headers.AsDeferredJobArray(),
                    UnfusedBodies = work.UnfusedBodies.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    Bodies = world.Bodies,
                    StepInput = substepInput,
                    Teams = work.Buffers.FusedTeams,
                    Lanes = lanes,
                    Unfused = unfused,
                    UnfusedClaim = fold ? work.Buffers.UnfusedClaim : null,
                }.Schedule(lanes, 1, handle);
            }
            else if (fold)
            {
                handle = new SolveUnfusedSubstepsJob
                {
                    Islands = work.Islands.AsDeferredJobArray(),
                    IslandVessels = work.IslandVessels.AsDeferredJobArray(),
                    Headers = work.Headers.AsDeferredJobArray(),
                    UnfusedBodies = work.UnfusedBodies.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    Bodies = world.Bodies,
                    Unfused = unfused,
                }.Schedule(handle);
            }

            for (int substep = 0; !fold && substep < substeps; substep++)
            {
                substepInput.CurrentSubstep = substep;
                if (substep > 0)
                {
                    // Recount contacts before building rows. Update fixed-size joint Jacobians during row
                    // construction unless using LegacyRowBuild.
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
                    UnfusedBodies = work.UnfusedBodies.AsDeferredJobArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    Timestep = substepInput.Timestep,
                }.Schedule(work.UnfusedBodies, bodiesPerBatch, handle);
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
        /// Solve vessel-only islands in one fused job across all substeps, including their contact/joint boundary.
        /// Enabled by default; RETOOLED_VESSEL_FUSED=0 disables it at first attach.
        /// </summary>
        public static bool VesselFused
        {
            get => !VesselFusedOff.Data;
            set => VesselFusedOff.Data = !value;
        }

        struct VesselFusedKey { }

        /// <summary>
        /// Allow fused vessel islands with contact or joint boundary rows. Enabled by default; benchmarks may
        /// disable it.
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
        /// Minimum vessel part count for fused solving. By default every eligible island is fused.
        /// Read RETOOLED_VESSEL_FUSED_PARTS at first attach; tests may override it.
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
        /// Use the reference row builder with separate joint updates and per-Jacobian regularization.
        /// Test/benchmark option.
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
        /// Contiguous row ranges for direct islands. Store pointers to list headers because jobs may resize their
        /// data.
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

            // First body and stable identity per island, used to assign cache slots to unfused islands.
            [NativeDisableUnsafePtrRestriction] public UnsafeList<int>* FirstBodies;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<ulong>* FirstIdentities;

            // Atomic claim for the one fused lane that executes unfused substeps.
            [NativeDisableUnsafePtrRestriction] public int* UnfusedClaim;

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
                FirstBodies = UnsafeList<int>.Create(0, allocator);
                FirstIdentities = UnsafeList<ulong>.Create(0, allocator);
                UnfusedClaim = (int*)UnsafeUtility.MallocTracked(sizeof(int), UnsafeUtility.AlignOf<int>(), allocator, 0);
                *UnfusedClaim = 0;
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
                UnsafeList<int>.Destroy(FirstBodies);
                UnsafeList<ulong>.Destroy(FirstIdentities);
                UnsafeUtility.FreeTracked(UnfusedClaim, Allocator.TempJob);
            }
        }

        /// <summary>
        /// Team for one fused island, with its part count and active lane count.
        /// Idle lanes prefer the team with the most parts per lane.
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

            /// <summary>
            /// The bodies of the islands that are not fused, in the order of the islands. The substeps integrate these
            /// bodies. <see cref="MarkFusedVesselIslandsJob"/> lists them.
            /// </summary>
            public NativeList<int> UnfusedBodies;

            public DirectIslandWork(int substeps, Allocator allocator)
            {
                Headers = new NativeList<IslandHeader>(allocator);
                Islands = new NativeList<DirectIsland>(allocator);
                Buffers = new DirectIslandBuffers(substeps, allocator);
                IslandVessels = new NativeList<int>(allocator);
                UnfusedBodies = new NativeList<int>(allocator);
            }

            public JobHandle Dispose(JobHandle inputDeps)
            {
                JobHandle buffers = new DestroyDirectIslandBuffersJob { Buffers = Buffers }.Schedule(inputDeps);
                return JobHandle.CombineDependencies(
                    JobHandle.CombineDependencies(Headers.Dispose(inputDeps), Islands.Dispose(inputDeps), buffers),
                    JobHandle.CombineDependencies(IslandVessels.Dispose(inputDeps), UnfusedBodies.Dispose(inputDeps)));
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
        /// List direct-island Jacobians in iterator order, joints before contacts, and count first-substep rows.
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
                // The cache gives its slots to the islands after MarkFusedVesselIslandsJob has found the fused islands,
                // which keep their state in slots of their own.
                if (Cache.IsCreated)
                {
                    UnsafeList<int>* firstBodies = Work.Buffers.FirstBodies;
                    UnsafeList<ulong>* firstIdentities = Work.Buffers.FirstIdentities;
                    firstBodies->Resize(islandCount);
                    firstIdentities->Resize(islandCount);
                    var bodies = (RigidBody*)Bodies.GetUnsafeReadOnlyPtr();
                    for (int island = 0; island < islandCount; island++)
                    {
                        int first = Info.IslandBodies[Info.FirstIslandBodyIndices[island]];
                        firstBodies->Ptr[island] = first;
                        firstIdentities->Ptr[island] = DirectSolverCache.BodyIdentity(bodies, Bodies.Length, first);
                    }

                    Cache.ClearCompactIndex(MotionCount);
                }

                Work.Islands.ResizeUninitialized(islandCount);
                Work.Buffers.Bounded->Resize(islandCount);

                // Count rows while listing headers, then convert counts to offsets to avoid rereading cold
                // headers.
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
        /// Count Jacobian rows and assign contiguous island ranges.
        /// Reuse first-substep counts from IndexDirectIslandsJob; later recount only contacts.
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
                CountRows((DirectIsland*)work.Islands.GetUnsafePtr(), work.Islands.Length, (IslandHeader*)work.Headers.GetUnsafeReadOnlyPtr(),
                    work.Buffers, first);
            }

            internal static void CountRows(DirectIsland* islands, int islandCount, IslandHeader* headers, in DirectIslandBuffers buffers,
                bool first)
            {
                UnsafeList<int>* headerRows = buffers.HeaderRows;
                int totalRows = 0;
                for (int i = 0; i < islandCount; i++)
                {
                    ref DirectIsland island = ref islands[i];
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
                            rows += GetConstraintRows(ref *headers[h].Header);
                        }
                    }

                    island.FirstRow = totalRows;
                    island.Rows = rows;
                    (*buffers.Bounded)[i] = 0;
                    totalRows += rows;
                }

                buffers.Resize(totalRows);
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
            /// Update a joint Jacobian from current poses immediately before building its rows.
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
                // The fused vessel job builds the rows of its own islands and moves their bodies at the same time as this
                // job runs, so this job must not read those Jacobians or bodies.
                IslandHeader entry = Headers[index];
                if (Islands[entry.Island].Fused)
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

                BuildIslandHeader(index, entry, Islands[entry.Island], Work, MotionVelocities, MotionDatas, Bodies, StepInput, UpdateJoints, shared);
            }
        }

        /// <summary>
        /// Build a Jacobian's disjoint row range, optionally updating joint poses, and mark the island bounded when
        /// needed.
        /// </summary>
        static unsafe void BuildIslandHeader(int index, in IslandHeader entry, in DirectIsland island, in DirectIslandBuffers work,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, in NativeArray<RigidBody> bodies,
            in StepInput stepInput, bool updateJoints, RowRegularization* regularization)
        {
            ref JacobianHeader header = ref *entry.Header;
            if (updateJoints && index < island.JointEnd)
            {
                UpdateJacobian(ref header, motionDatas, motionVelocities, bodies, stepInput);
            }

            int startRow = (*work.HeaderRows)[index];
            if (startRow < 0)
            {
                return;
            }

            int jacRows = GetConstraintRows(ref header);
            if (jacRows == 0)
            {
                return;
            }

            bool lcp = BuildHeaderRows(ref header, startRow, jacRows, IslandRowBuffers.Of(work, island), motionVelocities, motionDatas,
                stepInput, island.ContactPairs > 0, regularization);

            // Workers only set this flag to one; the solver reads it after all row builds complete.
            int* bounded = work.Bounded->Ptr + entry.Island;
            if (lcp && *bounded == 0)
            {
                *bounded = 1;
            }
        }

        // Per-island Jacobian rows, regularization, bounds, targets, push-out and states for one substep.
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
        /// Build a Jacobian's rows starting at startRow, initially free and unbounded.
        /// Return whether any bound requires a bounded solve. Parallel Jacobians write disjoint ranges.
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
        /// Solve a direct island's joints and contacts together and apply impulses.
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

        /// <summary>
        /// Integrates the bodies of the islands that are not fused over the substep. The fused vessel job integrates
        /// the bodies of the fused islands itself, the kinematic ones too.
        /// </summary>
        [BurstCompile]
        [NoAlias]
        struct IntegrateIslandBodiesJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeArray<int> UnfusedBodies;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            public float Timestep;

            public void Execute(int index)
            {
                Integrator.IntegrateMotion(UnfusedBodies[index], MotionDatas, MotionVelocities, Timestep);
            }
        }

        /// <summary>
        /// Inputs and row buffers for serial execution of unfused island substeps.
        /// </summary>
        unsafe struct UnfusedSubsteps
        {
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            [ReadOnly]
            public NativeStream.Reader JacobiansReader;

            public DirectIslandBuffers Buffers;

            /// <summary>The input of the first substep. Its time step is the substep's.</summary>
            public StepInput StepInput;

            /// <summary>The value of <see cref="LegacyRowBuild"/> when the jobs were scheduled.</summary>
            public bool LegacyRowBuild;
        }

        /// <summary>
        /// Run unfused islands' Jacobian updates, row builds, solves and integration serially.
        /// Preserve the chained jobs' operation order and results.
        /// </summary>
        static unsafe void RunUnfusedSubsteps(in UnfusedSubsteps unfused, NativeArray<IslandHeader> headers, NativeArray<DirectIsland> islands,
            NativeArray<int> islandVessels, NativeArray<int> unfusedBodies, NativeArray<MotionVelocity> motionVelocities,
            NativeArray<MotionData> motionDatas, NativeArray<RigidBody> bodies)
        {
            using ProfilerMarker.AutoScope span = UnfusedSubstepsMarker.Auto();

            // Only the substeps' work changes the islands' row ranges, and this thread does all of it.
            var islandPointer = (DirectIsland*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(islands);
            var headerPointer = (IslandHeader*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(headers);
            int islandCount = islands.Length;
            if (unfusedBodies.Length == 0)
            {
                return;
            }

            System.Threading.Interlocked.Increment(ref UnfusedFoldedStepCount.Data);

            StepInput input = unfused.StepInput;
            for (int substep = 0; substep < input.NumSubsteps; substep++)
            {
                input.CurrentSubstep = substep;
                if (substep > 0)
                {
                    for (int i = 0; i < islandCount; i++)
                    {
                        DirectIsland island = islandPointer[i];
                        if (island.Fused)
                        {
                            continue;
                        }

                        // As UpdateIslandJacobiansJob does: contacts only, unless the rows are built the earlier way.
                        int from = unfused.LegacyRowBuild ? island.FirstHeader : island.JointEnd;
                        for (int h = from; h < island.HeaderEnd; h++)
                        {
                            UpdateJacobian(ref *headerPointer[h].Header, motionDatas, motionVelocities, bodies, input);
                        }
                    }

                    CountIslandRowsJob.CountRows(islandPointer, islandCount, headerPointer, unfused.Buffers, false);
                }

                RowRegularization regularization = RowRegularization.Of(input);
                RowRegularization* shared = unfused.LegacyRowBuild ? null : &regularization;
                bool updateJoints = substep > 0 && !unfused.LegacyRowBuild;
                for (int i = 0; i < islandCount; i++)
                {
                    DirectIsland island = islandPointer[i];
                    if (island.Fused)
                    {
                        continue;
                    }

                    for (int h = island.FirstHeader; h < island.HeaderEnd; h++)
                    {
                        BuildIslandHeader(h, headerPointer[h], island, unfused.Buffers, motionVelocities, motionDatas, bodies, input,
                            updateJoints, shared);
                    }
                }

                for (int i = 0; i < islandCount; i++)
                {
                    if (!islandPointer[i].Fused)
                    {
                        SolveDirectIsland(unfused.Info, unfused.JacobiansReader, headers, islands, unfused.Buffers, islandVessels,
                            ref motionVelocities, motionDatas, bodies, input, i, null);
                    }
                }

                for (int b = 0; b < unfusedBodies.Length; b++)
                {
                    Integrator.IntegrateMotion(unfusedBodies[b], motionDatas, motionVelocities, input.Timestep);
                }
            }
        }

        /// <summary>
        /// Number of steps using serial unfused substeps, for tests of scheduling choices.
        /// </summary>
        internal static long UnfusedFoldedSteps => UnfusedFoldedStepCount.Data;

        struct UnfusedFoldedStepKey { }

        static readonly SharedStatic<long> UnfusedFoldedStepCount = SharedStatic<long>.GetOrCreate<UnfusedFoldedStepKey>();

        static readonly ProfilerMarker UnfusedSubstepsMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Island.UnfusedSubsteps");

        /// <summary>
        /// Run unfused substeps on one thread when fused solving is disabled and folding is selected.
        /// </summary>
        [BurstCompile]
        unsafe struct SolveUnfusedSubstepsJob : IJob
        {
            [ReadOnly]
            public NativeArray<IslandHeader> Headers;

            public NativeArray<DirectIsland> Islands;

            [ReadOnly]
            public NativeArray<int> IslandVessels;

            [ReadOnly]
            public NativeArray<int> UnfusedBodies;

            // The iterative solve changes the motions of other bodies at the same time.
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public UnfusedSubsteps Unfused;

            public void Execute()
            {
                RunUnfusedSubsteps(Unfused, Headers, Islands, IslandVessels, UnfusedBodies, MotionVelocities, MotionDatas, Bodies);
            }
        }

        /// <summary>
        /// Select chained or serial unfused substeps. Auto uses previous-step work counts.
        /// RETOOLED_SUBSTEP_FOLD=0 forces Chain; 1 forces Fold.
        /// </summary>
        internal enum SubstepFolding
        {
            Auto = 0,
            Fold = 1,
            Chain = 2,
        }

        internal static SubstepFolding UnfusedFolding
        {
            get => (SubstepFolding)UnfusedFoldingValue.Data;
            set => UnfusedFoldingValue.Data = (int)value;
        }

        struct UnfusedFoldingKey { }

        static readonly SharedStatic<int> UnfusedFoldingValue = SharedStatic<int>.GetOrCreate<UnfusedFoldingKey>();

        /// <summary>
        /// The most rows that the islands that are not fused can have, at the first substep of the last step, for this
        /// step to run their substeps on one thread.
        /// </summary>
        const int FoldedRows = 256;

        /// <summary>The most bodies that those islands can have, for the same.</summary>
        const int FoldedBodies = 64;

        /// <summary>
        /// Minimum fused vessel parts per unfused body required for automatic substep folding.
        /// </summary>
        const int FoldedPartsPerBody = 8;

        /// <summary>
        /// Fold unfused substeps onto one thread when prior-step work is small enough to finish alongside fused
        /// solving.
        /// Require no shareable large islands, bounded row/body counts and enough fused parts per unfused body.
        /// </summary>
        static bool FoldsUnfusedSubsteps(in DirectSolverCache cache)
        {
            SubstepFolding folding = UnfusedFolding;
            if (folding != SubstepFolding.Auto)
            {
                return folding == SubstepFolding.Fold;
            }

            if (!cache.IsCreated || !cache.LastUnfusedWork(out DirectSolverCache.UnfusedWork work))
            {
                return false;
            }

            return work.Islands == 0 || (work.TeamIslands == 0 && work.Rows <= FoldedRows && work.Bodies <= FoldedBodies
                && work.FusedParts >= FoldedPartsPerBody * work.Bodies);
        }

        /// <summary>
        /// Mark eligible vessel islands for fused solving, assign other cache slots and list unfused bodies for
        /// integration.
        /// </summary>
        [BurstCompile]
        unsafe struct MarkFusedVesselIslandsJob : IJob
        {
            public NativeList<DirectIsland> Islands;

            [ReadOnly]
            public NativeList<int> IslandVessels;

            /// <summary>The islands' bodies, in island order, and where each island's bodies start.</summary>
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            public NativeList<int> UnfusedBodies;

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

                AssignSlots();
                ListUnfusedBodies();
                ListFusedTeams();
                AssignVesselForests(Cache, Islands.AsArray(), IslandVessels.AsArray(), Vessels);
            }

            /// <summary>
            /// Assign substep cache slots by first-body identity. Fused islands use their own slots.
            /// </summary>
            void AssignSlots()
            {
                if (!Cache.IsCreated)
                {
                    return;
                }

                UnsafeList<ulong>* identities = Buffers.FirstIdentities;
                for (int i = 0; i < Islands.Length; i++)
                {
                    if (Islands[i].Fused)
                    {
                        identities->Ptr[i] = DirectSolverCache.NoSlot;
                    }
                }

                Cache.Assign(View(Buffers.FirstBodies->Ptr, Islands.Length), View(identities->Ptr, Islands.Length));
            }

            /// <summary>
            /// List unfused bodies and record their workload for the next step's scheduling decision.
            /// </summary>
            void ListUnfusedBodies()
            {
                UnfusedBodies.Clear();
                int islandCount = Islands.Length;
                var work = new DirectSolverCache.UnfusedWork();
                for (int i = 0; i < islandCount; i++)
                {
                    DirectIsland island = Islands[i];
                    if (island.Fused)
                    {
                        work.FusedParts += island.VesselParts;
                        continue;
                    }

                    int first = Info.FirstIslandBodyIndices[i];
                    int end = i + 1 < islandCount ? Info.FirstIslandBodyIndices[i + 1] : Info.IslandBodies.Length;
                    UnfusedBodies.AddRange((int*)Info.IslandBodies.GetUnsafeReadOnlyPtr() + first, end - first);
                    work.Islands++;
                    work.Rows += island.Rows;
                    work.TeamIslands += island.SolvesInTeam ? 1 : 0;
                }

                work.Bodies = UnfusedBodies.Length;
                Cache.NoteUnfusedWork(work);
            }

            /// <summary>
            /// Create a team per fused island, ordered by descending part count then island index.
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
        /// Solve marked vessel islands across all substeps with a team per island.
        /// Idle lanes help active teams. Combine task results in fixed order for deterministic results.
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
            /// The bodies of the islands that are not fused, which the lane that does the substeps' work of those
            /// islands integrates.
            /// </summary>
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<int> UnfusedBodies;

            // Unfused substep inputs and optional atomic lane claim. Without the claim word, chained jobs own
            // the work.
            public UnfusedSubsteps Unfused;

            [NativeDisableUnsafePtrRestriction]
            public int* UnfusedClaim;

            /// <summary>
            /// Scheduled lane count. With one lane, skip team stage publication and solve islands directly.
            /// </summary>
            public int Lanes;

            static readonly ProfilerMarker HelpMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Vessel.HelpFused");

            public void Execute(int lane)
            {
                FusedTeam* teams = Teams->Ptr;
                int count = Teams->Length;

                // Claim unled islands. Give unfused substeps to the first lane not leading the largest island.
                for (int t = 0; t < count; t++)
                {
                    if (teams[t].Team.TryLead())
                    {
                        DirectIsland island = Islands[teams[t].Island];
                        System.Threading.Interlocked.Increment(ref FusedIslandSolveCount.Data);
                        SolveFusedVesselIsland(island, (IslandHeader*)Headers.GetUnsafeReadOnlyPtr() + island.FirstHeader,
                            (int*)IslandVessels.GetUnsafeReadOnlyPtr() + island.FirstVessel, MotionVelocities, MotionDatas, Bodies,
                            StepInput, Lanes > 1 ? &teams[t].Team : null);
                        teams[t].Team.Finish();
                    }

                    if (t == 0)
                    {
                        RunUnfusedOnce();
                    }
                }

                if (count == 0)
                {
                    RunUnfusedOnce();
                }

                // Help open stages, largest island first. After the spin timeout, wait on the team with the
                // most parts per lane.
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
            /// Does the substeps' work of the islands that are not fused, if the step gave that work to this job and no
            /// other lane has taken it.
            /// </summary>
            void RunUnfusedOnce()
            {
                if (UnfusedClaim != null && System.Threading.Interlocked.CompareExchange(ref *UnfusedClaim, 1, 0) == 0)
                {
                    RunUnfusedSubsteps(Unfused, Headers, Islands, IslandVessels, UnfusedBodies, MotionVelocities, MotionDatas, Bodies);
                }
            }

            /// <summary>
            /// Help open island stages, largest first, until the spin timeout. Return false when all islands are
            /// solved.
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
