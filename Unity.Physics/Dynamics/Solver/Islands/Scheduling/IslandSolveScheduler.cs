using Unity.Physics.Dynamics.Solver.Vessels;
using static Unity.Physics.Solver;
using static Unity.Physics.Dynamics.Solver.Vessels.VesselIslandSolver;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Configuration;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Linear;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Profiling;

namespace Unity.Physics.Dynamics.Solver.Islands.Scheduling
{
    /// <summary>Schedule island solving, integration and buffer disposal using the direct and vessel solvers.</summary>
    internal static partial class IslandSolveScheduler
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
        internal static unsafe SimulationJobHandles ScheduleIslandSolveJobs(in PhysicsWorld world, StepInput stepInput,
            in NativeStream jacobians, in NativeStream collisionEvents, in NativeStream triggerEvents, in NativeStream impulseEvents,
            in DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo, StabilizationData solverStabilizationData,
            JobHandle jacobiansBuilt, JobHandle inputDeps, out JobHandle motionsSolved, bool multiThreaded)
        {
            if (!multiThreaded)
            {
                motionsSolved = new SolveIslandsJob
                {
                    World = world,
                    Input = stepInput,
                    Jacobians = jacobians,
                    CollisionEvents = collisionEvents,
                    TriggerEvents = triggerEvents,
                    ImpulseEvents = impulseEvents,
                    Info = solverSchedulerInfo,
                    Stabilization = solverStabilizationData,
                }.Schedule(JobHandle.CombineDependencies(jacobiansBuilt, inputDeps));

                return new SimulationJobHandles(motionsSolved)
                {
                    FinalDisposeHandle = JobHandle.CombineDependencies(jacobians.Dispose(motionsSolved),
                        solverSchedulerInfo.ScheduleDisposeJob(motionsSolved)),
                };
            }

            NativeArray<MotionVelocity> motionVelocities = world.MotionVelocities;
            DispatchPairSequencer.IterativeSolverSchedulerInfo iterativePairs = solverSchedulerInfo.IterativePairsIterativeScheduling;

            // Use team stages for iterations and relaxation to avoid separate jobs for each phase.
            var teams = UnsafeList<WorkTeam>.Create(2, Allocator.TempJob);
            teams->Resize(2, NativeArrayOptions.ClearMemory);
            teams->ElementAt(0).Configure();
            teams->ElementAt(1).Configure();
            int lanes = math.min(JobsUtility.JobWorkerCount + 1, MaxTeamLanes);

            var template = new IterativeSolveWork
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

            JobHandle islandHandle = ScheduleDirectIslandJobs(world, stepInput, jacobians,
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
            public IterativeSolveWork Template;

            public int FirstIteration;
            public int Iterations;

            [NativeDisableUnsafePtrRestriction]
            public WorkTeam* Team;

            public void Execute(int lane)
            {
                if (!Team->TryLead())
                {
                    Team->Help<IslandTaskDispatch>();
                    return;
                }

                var phases = (DispatchPairSequencer.IterativeSolverSchedulerInfo.SolvePhaseInfo*)
                    NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(Template.IterativeSolverSchedulerInfo.PhaseInfo);
                IterativeSolveWork job = Template;
                for (int iteration = FirstIteration; iteration < FirstIteration + Iterations; iteration++)
                {
                    job.stepInput.CurrentSolverIteration = iteration;
                    for (int phase = 0; phase < DispatchPairSequencer.NumIslandPhases; phase++)
                    {
                        job.PhaseIndex = phase;
                        Team->Lead<IslandTaskDispatch>(&job, IslandTaskKind.IterativePhaseKind, phases[phase].NumWorkItems, null);
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
            IterativeSolveWork job = *(IterativeSolveWork*)context;
            job.Execute(task);
        }

        [BurstCompile]
        unsafe struct DestroyTeamsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public UnsafeList<WorkTeam>* Teams;

            public void Execute()
            {
                UnsafeList<WorkTeam>.Destroy(Teams);
            }
        }

        /// <summary>
        /// Schedule direct island Jacobian updates, row construction, solves and integration for each substep.
        /// Count rows serially; run other stages in parallel. Return solved before buffer disposal.
        /// </summary>
        static unsafe JobHandle ScheduleDirectIslandJobs(in PhysicsWorld world, StepInput stepInput, in NativeStream jacobians,
            in DispatchPairSequencer.DirectSolverSchedulerInfo info, JobHandle jacobiansBuilt, JobHandle inputDeps, out JobHandle solved)
        {
            const int substeps = SolverConstants.DirectIslandSubsteps;
            const int headersPerBatch = 16;
            const int bodiesPerBatch = 32;

            var work = new DirectIslandWork(substeps, Allocator.TempJob);
            int lanes = math.min(JobsUtility.JobWorkerCount + 1, MaxTeamLanes);

            JobHandle handle = new IndexDirectIslandsJob
            {
                Info = info,
                JacobiansReader = jacobians.AsReader(),
                Work = work,
                Cache = stepInput.IslandSolverCache,
                Bodies = world.Bodies,
                MotionCount = world.MotionVelocities.Length,
            }.Schedule(jacobiansBuilt);

            handle = JobHandle.CombineDependencies(handle, inputDeps);

            StepInput substepInput = stepInput;
            substepInput.Timestep /= substeps;
            substepInput.NumSubsteps = substeps;
            substepInput.DisableExportEvents = true;

            // Fused islands own their bodies for the entire step; substep jobs must skip them.
            bool fusedVessels = SolverSwitches.VesselFused;
            JobHandle fused = default;
            handle = new MarkFusedVesselIslandsJob
            {
                Islands = work.Islands,
                IslandVessels = work.IslandVessels,
                Info = info,
                UnfusedBodies = work.UnfusedBodies,
                Vessels = stepInput.VesselSet,
                Cache = stepInput.IslandSolverCache,
                Fuse = fusedVessels,
                MinimumParts = SolverSwitches.FusedVesselParts,
                Contacts = SolverSwitches.FusedContacts,
                Buffers = work.Buffers,
            }.Schedule(handle);

            // Choose serial unfused work from the previous step's counts; current counts are not known at
            // scheduling time.
            bool fold = FoldsUnfusedSubsteps(stepInput.IslandSolverCache);
            var unfused = new UnfusedSubsteps
            {
                Buffers = work.Buffers,
                StepInput = substepInput,
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
                    // Update contacts here. The row builder updates joints before building their rows.
                    handle = new UpdateIslandJacobiansJob
                    {
                        Headers = work.Headers.AsDeferredJobArray(),
                        Islands = work.Islands.AsDeferredJobArray(),
                        MotionVelocities = world.MotionVelocities,
                        MotionDatas = world.MotionDatas,
                        Bodies = world.Bodies,
                        StepInput = substepInput,
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
                    UpdateJoints = substep > 0,
                }.Schedule(work.Headers, headersPerBatch, handle);

                JobHandle alone = new SolveDirectIslandJob
                {
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
            [NativeDisableUnsafePtrRestriction] public UnsafeList<WorkTeam>* Teams;

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

            public DirectIslandBuffers(int substeps, Allocator allocator, bool configureTeams)
            {
                // The team after the substeps' teams never runs. The main thread configures it here, because only the
                // main thread can read the clock's conversion rate, and each fused island's team starts as a copy of it.
                Teams = UnsafeList<WorkTeam>.Create(substeps + 1, allocator);
                FusedTeams = UnsafeList<FusedTeam>.Create(4, allocator);
                Teams->Resize(substeps + 1, NativeArrayOptions.ClearMemory);
                for (int substep = 0; configureTeams && substep <= substeps; substep++)
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
                UnsafeList<WorkTeam>.Destroy(Teams);
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
            public WorkTeam Team;
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

            public DirectIslandWork(int substeps, Allocator allocator, bool configureTeams = true)
            {
                Headers = new NativeList<IslandHeader>(allocator);
                Islands = new NativeList<DirectIsland>(allocator);
                Buffers = new DirectIslandBuffers(substeps, allocator, configureTeams);
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

        [BurstCompile]
        [NoAlias]
        unsafe struct UpdateIslandJacobiansJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeArray<IslandHeader> Headers;

            [ReadOnly]
            public NativeArray<DirectIsland> Islands;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            public void Execute(int index)
            {
                // The fused job updates its own islands. BuildIslandRowsJob updates the joints of other islands.
                DirectIsland island = Islands[Headers[index].Island];
                if (island.Fused || index < island.JointEnd)
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

                if (!hasRegularization)
                {
                    regularization = RowRegularization.Of(StepInput);
                    hasRegularization = true;
                }

                BuildIslandHeader(index, entry, Islands[entry.Island], Work, MotionVelocities, MotionDatas, Bodies, StepInput,
                    UpdateJoints, (RowRegularization*)UnsafeUtility.AddressOf(ref regularization));
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

            bool lcp = BuildHeaderRows(ref header, startRow, jacRows, GetIslandRows(work, island), motionVelocities, motionDatas,
                stepInput, island.ContactPairs > 0, regularization);

            // Workers only set this flag to one; the solver reads it after all row builds complete.
            int* bounded = work.Bounded->Ptr + entry.Island;
            if (lcp && *bounded == 0)
            {
                *bounded = 1;
            }
        }

        // Per-island Jacobian rows, regularization, bounds, targets, push-out and states for one substep.
        static unsafe IslandRowBuffers GetIslandRows(in DirectIslandBuffers work, in DirectIsland island)
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

        /// <summary>
        /// Solve a direct island's joints and contacts together and apply impulses.
        /// </summary>
        [BurstCompile]
        [NoAlias]
        unsafe struct SolveDirectIslandJob : IJobParallelForDefer
        {
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
                    SolveDirectIsland(Headers, Islands, Work, IslandVessels, ref MotionVelocities, MotionDatas, Bodies, StepInput, islandIndex, null);
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
            public WorkTeam* Team;

            static readonly ProfilerMarker HelpMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Island.HelpSolve");

            public void Execute(int lane)
            {
                if (!Team->TryLead())
                {
                    using (HelpMarker.Auto())
                    {
                        Team->Help<IslandTaskDispatch>();
                    }

                    return;
                }

                for (int islandIndex = 0; islandIndex < Islands.Length; islandIndex++)
                {
                    if (Islands[islandIndex].SolvesInTeam)
                    {
                        SolveDirectIsland(Headers, Islands, Work, IslandVessels, ref MotionVelocities, MotionDatas, Bodies, StepInput, islandIndex, Team);
                    }
                }

                Team->Finish();
            }
        }

        static readonly ProfilerMarker DirectSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Island.DirectSolve");

        static unsafe void SolveDirectIsland(in NativeArray<IslandHeader> headers, in NativeArray<DirectIsland> islands, in DirectIslandBuffers work,
            in NativeArray<int> islandVessels, ref NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, in NativeArray<RigidBody> bodies, in StepInput stepInput, int islandIndex,
            WorkTeam* team)
        {
            DirectIsland island = islands[islandIndex];
            using (DirectSolveMarker.Auto())
            {
                if (!island.Fused && (island.Rows > 0 || island.VesselCount > 0))
                {
                    JacobianRows rows = work.Rows(island);
                    Vector rightHandSide = DirectIslandBuffers.View(work.RightHandSide, island);
                    SolveSparsely(islandIndex, motionVelocities, motionDatas, stepInput, island.Rows, rows,
                        DirectIslandBuffers.View(work.Epsilon, island), DirectIslandBuffers.View(work.Lower, island),
                        DirectIslandBuffers.View(work.Upper, island), rightHandSide, work.Flags(island),
                        (*work.Bounded)[islandIndex] != 0, island.ContactPairs, team,
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
            public DirectIslandBuffers Buffers;

            /// <summary>The input of the first substep. Its time step is the substep's.</summary>
            public StepInput StepInput;
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

            IslandWorkCounters.CountUnfusedFoldedStep();

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

                        // The row builder updates the joints. Update the contacts before counting their rows.
                        for (int h = island.JointEnd; h < island.HeaderEnd; h++)
                        {
                            UpdateJacobian(ref *headerPointer[h].Header, motionDatas, motionVelocities, bodies, input);
                        }
                    }

                    CountIslandRowsJob.CountRows(islandPointer, islandCount, headerPointer, unfused.Buffers, false);
                }

                RowRegularization regularization = RowRegularization.Of(input);
                bool updateJoints = substep > 0;
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
                            updateJoints, &regularization);
                    }
                }

                for (int i = 0; i < islandCount; i++)
                {
                    if (!islandPointer[i].Fused)
                    {
                        SolveDirectIsland(headers, islands, unfused.Buffers, islandVessels,
                            ref motionVelocities, motionDatas, bodies, input, i, null);
                    }
                }

                for (int b = 0; b < unfusedBodies.Length; b++)
                {
                    Integrator.IntegrateMotion(unfusedBodies[b], motionDatas, motionVelocities, input.Timestep);
                }
            }
        }

        static readonly ProfilerMarker UnfusedSubstepsMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Island.UnfusedSubsteps");

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
        static bool FoldsUnfusedSubsteps(in IslandSolverCache cache)
        {
            SolverSwitches.SubstepFolding folding = SolverSwitches.UnfusedFolding;
            if (folding != SolverSwitches.SubstepFolding.Auto)
            {
                return folding == SolverSwitches.SubstepFolding.Fold;
            }

            if (!cache.IsCreated || !cache.LastUnfusedWork(out IslandWorkCounters work))
            {
                return false;
            }

            return work.Islands == 0 || (work.TeamIslands == 0 && work.Rows <= FoldedRows && work.Bodies <= FoldedBodies
                && work.FusedParts >= FoldedPartsPerBody * work.Bodies);
        }

        /// <summary>
        /// Assign forests and transfer each vessel's newest lock states before solving.
        /// Copy all states before handing out forests, because slot assignment may destroy a previous owner.
        /// Run serially after cache slot assignment.
        /// </summary>
        static unsafe void AssignVesselForests(IslandSolverCache cache, NativeArray<DirectIsland> islands, NativeArray<int> islandVessels,
            VesselSet set)
        {
            if (!cache.IsCreated || !set.IsCreated)
            {
                return;
            }

            // For each of the island vessels, where its copied states start in the lists below, or -1 if the vessel
            // keeps its states in the forest that will solve it.
            var copied = new NativeArray<int>(math.max(1, islandVessels.Length), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var locked = new NativeList<byte>(Allocator.Temp);
            var held = new NativeList<int>(Allocator.Temp);
            var ratio = new NativeList<float>(Allocator.Temp);
            for (int i = 0; i < islands.Length; i++)
            {
                DirectIsland island = islands[i];
                if (island.VesselCount == 0)
                {
                    continue;
                }

                ForestIsland* planned = island.Fused ? cache.FindFusedForest(set.Vessels[islandVessels[island.FirstVessel]].Id) : cache.ForestOf(i);
                for (int k = island.FirstVessel; k < island.FirstVessel + island.VesselCount; k++)
                {
                    copied[k] = -1;
                    VesselRecord vessel = set.Vessels[islandVessels[k]];
                    ForestIsland* owner = cache.OwnerOf(vessel.Id);
                    if (owner == null || owner == planned || !FindVesselStates(owner, vessel, out int start))
                    {
                        continue;
                    }

                    copied[k] = locked.Length;
                    locked.AddRange(owner->PartLocked.Ptr + start, vessel.PartCount);
                    held.AddRange(owner->PartHeld.Ptr + start, vessel.PartCount);
                    ratio.AddRange(owner->PartRatio.Ptr + start, vessel.PartCount);
                }
            }

            for (int i = 0; i < islands.Length; i++)
            {
                DirectIsland island = islands[i];
                if (island.VesselCount == 0)
                {
                    continue;
                }

                ForestIsland* forest = island.Fused ? cache.FusedForestOf(set.Vessels[islandVessels[island.FirstVessel]].Id) : cache.ForestOf(i);
                for (int k = island.FirstVessel; k < island.FirstVessel + island.VesselCount; k++)
                {
                    VesselRecord vessel = set.Vessels[islandVessels[k]];
                    cache.SetOwner(vessel.Id, forest);
                    if (forest == null || copied[k] < 0)
                    {
                        continue;
                    }

                    if (!forest->TakesStates)
                    {
                        forest->TakenIds.Clear();
                        forest->TakenVersions.Clear();
                        forest->TakenStarts.Clear();
                        forest->TakenLocked.Clear();
                        forest->TakenHeld.Clear();
                        forest->TakenRatio.Clear();
                        forest->TakesStates = true;
                    }

                    forest->TakenIds.Add(vessel.Id);
                    forest->TakenVersions.Add(vessel.StructureVersion);
                    forest->TakenStarts.Add(forest->TakenLocked.Length);
                    forest->TakenLocked.AddRange(locked.GetUnsafeReadOnlyPtr() + copied[k], vessel.PartCount);
                    forest->TakenHeld.AddRange(held.GetUnsafeReadOnlyPtr() + copied[k], vessel.PartCount);
                    forest->TakenRatio.AddRange(ratio.GetUnsafeReadOnlyPtr() + copied[k], vessel.PartCount);
                }
            }

            cache.PruneOwners();
            copied.Dispose();
            locked.Dispose();
            held.Dispose();
            ratio.Dispose();
        }

    }
}
