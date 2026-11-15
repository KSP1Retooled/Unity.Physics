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
        /// How many iterations take the push back out of the iteratively solved pairs' velocities. Those pairs are loose bodies,
        /// which a few iterations settle.
        /// </summary>
        const int RelaxIterations = 4;

        /// <summary>
        /// One step of every island, in a fixed set of jobs that find out how much work they have when they start, which leaves
        /// nothing for the main thread to wait for. The islands that fit in phases run their iterations phase by phase, and the
        /// bodies they hold integrate after them. The direct islands run each substep as a chain of jobs, with the Jacobians
        /// updated and their rows built in parallel. An island's bodies are its own, so the two sides run side by side.
        /// <paramref name="jacobiansBuilt"/> is the end of the Jacobian build alone, without the jobs that change the Jacobians'
        /// impulses or the velocities before the solve, which is all that listing the direct islands' Jacobians waits for.
        /// </summary>
        internal static unsafe SimulationJobHandles ScheduleIslandSolveJobs(in PhysicsWorld world, StepInput stepInput, int directIslandSubsteps,
            in NativeStream jacobians, in NativeStream collisionEvents, in NativeStream triggerEvents, in NativeStream impulseEvents,
            in DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo, StabilizationData solverStabilizationData,
            JobHandle jacobiansBuilt, JobHandle inputDeps)
        {
            NativeArray<MotionVelocity> motionVelocities = world.MotionVelocities;
            DispatchPairSequencer.IterativeSolverSchedulerInfo iterativePairs = solverSchedulerInfo.IterativePairsIterativeScheduling;

            // Every phase of every iteration runs as a stage of one team job rather than as a job of its own, which leaves two
            // jobs to schedule instead of one per phase and iteration, and two that finish at once when no pairs iterate.
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

            // With the positions moved by the contacts' push, a few more iterations take the push back out of the velocities,
            // as PhysX's velocity iterations do after its position iterations. Events were exported with the step's impulses.
            // They count past the step's own iterations, so that none of them counts as its first or last.
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

        /// <summary>The most lanes a team job gets. Lanes past what its stages keep busy only wait.</summary>
        const int MaxTeamLanes = 8;

        /// <summary>
        /// Runs iterations of the iteratively solved pairs, phase by phase, with every lane of the job sharing each phase's work
        /// items. A phase's pairs touch no body another of its work items touches, and the leader starts a phase only once
        /// the one before has finished.
        /// </summary>
        [BurstCompile]
        [NoAlias]
        unsafe struct IterateInTeamJob : IJobParallelFor
        {
            /// <summary>The job one phase would run, but for its phase and iteration.</summary>
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
        /// One work item of an iterative phase, for a team. The context is that phase's job, which each task copies: the
        /// event writers in it keep the state of the work item they are writing, so lanes can't share one.
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
        /// The substeps of every direct island. Each substep updates the Jacobians, counts their rows, builds the rows, solves
        /// each island and integrates its bodies. Only the counting runs alone. Small islands solve one per thread, and each big
        /// one shares its solve among the lanes of a job of its own. The first substep's rows depend only on what the Jacobian
        /// build wrote, so they are counted while the islands are listed, beside the jobs that finish preparing the solve.
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
            }.Schedule(jacobiansBuilt);

            handle = JobHandle.CombineDependencies(handle, inputDeps);

            StepInput substepInput = stepInput;
            substepInput.Timestep /= substeps;
            substepInput.NumSubsteps = substeps;
            substepInput.DisableExportEvents = true;

            for (int substep = 0; substep < substeps; substep++)
            {
                substepInput.CurrentSubstep = substep;
                if (substep > 0)
                {
                    handle = new UpdateIslandJacobiansJob
                    {
                        Headers = work.Headers.AsDeferredJobArray(),
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
                    StepInput = substepInput,
                }.Schedule(work.Headers, headersPerBatch, handle);

                JobHandle alone = new SolveDirectIslandJob
                {
                    Info = info,
                    JacobiansReader = jacobians.AsReader(),
                    Headers = work.Headers.AsDeferredJobArray(),
                    Islands = work.Islands.AsDeferredJobArray(),
                    Work = work.Buffers,
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
                    Timestep = substepInput.Timestep,
                }.Schedule(info.IslandBodies, bodiesPerBatch, handle);
            }

            return work.Dispose(handle);
        }

        /// <summary> A direct island's Jacobians among <see cref="DirectIslandWork.Headers"/>, and its rows this substep. </summary>
        struct DirectIsland
        {
            public int FirstHeader;

            /// <summary> The end of the Jacobians the solve takes. The contacts after it are only updated. </summary>
            public int SolveEnd;

            public int HeaderEnd;

            /// <summary> The contacts the solve takes along with the joints, which it solves sparsely. </summary>
            public int ContactPairs;

            public int FirstRow;
            public int Rows;

            /// <summary>
            /// Whether the island's sparse solve is big enough to share among lanes. A smaller one finishes before helpers
            /// would gain anything, and runs beside the others.
            /// </summary>
            public bool SolvesInTeam => Rows >= SharedSolveRows;
        }

        const int SharedSolveRows = 512;

        unsafe struct IslandHeader
        {
            public JacobianHeader* Header;
            public int Island;
        }

        /// <summary>
        /// The rows of every direct island in a substep, one island after another. Each list is resized while the jobs run, so
        /// the jobs hold the lists rather than their contents.
        /// </summary>
        unsafe struct DirectIslandBuffers
        {
            /// <summary> Where each Jacobian's rows start in its island, or -1 if this substep doesn't build them. </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<int>* HeaderRows;

            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Jacobian;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Epsilon;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Lower;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Upper;
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* RightHandSide;

            /// <summary>
            /// The push-out of each contact row that moves bodies rather than speeding them up, and -1 on rows that aren't
            /// contacts.
            /// </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<float>* Recovery;

            [NativeDisableUnsafePtrRestriction] public UnsafeList<LCP.MLCPIndexFlag>* IndexFlags;

            /// <summary> Per island, 1 once any of its rows has a bound, which makes its solve a bounded one. </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<int>* Bounded;

            /// <summary> One team per substep, since a team has one leader and lets its helpers go for good. </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<SparseSolveTeam>* Teams;

            public DirectIslandBuffers(int substeps, Allocator allocator)
            {
                Teams = UnsafeList<SparseSolveTeam>.Create(substeps, allocator);
                Teams->Resize(substeps, NativeArrayOptions.ClearMemory);
                for (int substep = 0; substep < substeps; substep++)
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

            /// <summary> The island's index flags, as the row builder and the solver take them. </summary>
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

            public DirectIslandWork(int substeps, Allocator allocator)
            {
                Headers = new NativeList<IslandHeader>(allocator);
                Islands = new NativeList<DirectIsland>(allocator);
                Buffers = new DirectIslandBuffers(substeps, allocator);
            }

            public JobHandle Dispose(JobHandle inputDeps)
            {
                JobHandle buffers = new DestroyDirectIslandBuffersJob { Buffers = Buffers }.Schedule(inputDeps);
                return JobHandle.CombineDependencies(Headers.Dispose(inputDeps), Islands.Dispose(inputDeps), buffers);
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
        /// Lists every direct island's Jacobians, with the joints before the contacts, in the order the island's iterator reads
        /// them, decides once a step whether an island's contacts take part in its solve, and counts the first substep's rows.
        /// </summary>
        [BurstCompile]
        unsafe struct IndexDirectIslandsJob : IJob
        {
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            public NativeStream.Reader JacobiansReader;

            public DirectIslandWork Work;

            public DirectSolverCache Cache;

            public void Execute()
            {
                int islandCount = Info.NumIslands;
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
                for (int island = 0; island < islandCount; island++)
                {
                    int numContacts = Info.IslandContactCounts[island];
                    int firstHeader = Work.Headers.Length;
                    bool touchesSomething = false;
                    var iterator = new IslandJacobianIterator(JacobiansReader, Info, island, numContacts);
                    while (iterator.HasJacobiansLeft())
                    {
                        ref JacobianHeader header = ref iterator.ReadJacobianHeader();
                        touchesSomething |= header.Type == JacobianType.Contact && header.Enabled;
                        Work.Headers.Add(new IslandHeader
                        {
                            Header = (JacobianHeader*)UnsafeUtility.AddressOf(ref header),
                            Island = island,
                        });
                    }

                    int jointHeaders = 0;
                    var joints = new IslandJacobianIterator(JacobiansReader, Info, island);
                    while (joints.HasJacobiansLeft())
                    {
                        joints.ReadJacobianHeader();
                        jointHeaders++;
                    }

                    Work.Islands[island] = new DirectIsland
                    {
                        FirstHeader = firstHeader,
                        SolveEnd = touchesSomething ? Work.Headers.Length : firstHeader + jointHeaders,
                        HeaderEnd = Work.Headers.Length,
                        ContactPairs = touchesSomething ? numContacts : 0,
                    };
                }

                Work.Buffers.HeaderRows->Resize(Work.Headers.Length);
                CountIslandRowsJob.CountRows(Work);
            }
        }

        [BurstCompile]
        [NoAlias]
        unsafe struct UpdateIslandJacobiansJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeArray<IslandHeader> Headers;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            public void Execute(int index)
            {
                UpdateJacobian(ref *Headers[index].Header, MotionDatas, MotionVelocities, Bodies, StepInput);
            }
        }

        /// <summary>
        /// Counts the rows each Jacobian has this substep, which a contact's points change as they come and go, and lays the
        /// islands' rows out one after another.
        /// </summary>
        [BurstCompile]
        unsafe struct CountIslandRowsJob : IJob
        {
            public DirectIslandWork Work;

            public void Execute()
            {
                CountRows(Work);
            }

            internal static void CountRows(DirectIslandWork work)
            {
                UnsafeList<int>* headerRows = work.Buffers.HeaderRows;
                int totalRows = 0;
                for (int i = 0; i < work.Islands.Length; i++)
                {
                    ref DirectIsland island = ref work.Islands.ElementAt(i);
                    int rows = 0;
                    for (int h = island.FirstHeader; h < island.SolveEnd; h++)
                    {
                        (*headerRows)[h] = rows;
                        rows += GetConstraintRows(ref *work.Headers[h].Header);
                    }

                    for (int h = island.SolveEnd; h < island.HeaderEnd; h++)
                    {
                        (*headerRows)[h] = -1;
                    }

                    island.FirstRow = totalRows;
                    island.Rows = rows;
                    (*work.Buffers.Bounded)[i] = 0;
                    totalRows += rows;
                }

                work.Buffers.Resize(totalRows);
            }
        }

        /// <summary> Builds the rows of one Jacobian of a sparse island, which writes no rows but its own. </summary>
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

            public StepInput StepInput;

            public void Execute(int index)
            {
                int startRow = (*Work.HeaderRows)[index];
                if (startRow < 0)
                {
                    return;
                }

                ref JacobianHeader header = ref *Headers[index].Header;
                int jacRows = GetConstraintRows(ref header);
                if (jacRows == 0)
                {
                    return;
                }

                int islandIndex = Headers[index].Island;
                DirectIsland island = Islands[islandIndex];

                // Every row starts free and unbounded, and the builder bounds the rows that need it.
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

                Vector* JBlocks = stackalloc Vector[jacRows];
                BuildJacobianRows(ref header, startRow, jacRows, ref context, JBlocks, null);

                // Builders only ever store 1, and the solve reads it after this job, so a plain store is enough. Checking
                // first keeps the builders of one island from writing its line over and over.
                int* bounded = Work.Bounded->Ptr + islandIndex;
                if (context.IsLCP && *bounded == 0)
                {
                    *bounded = 1;
                }
            }
        }

        /// <summary>
        /// Solves one direct island from the rows built for it and applies the impulses. The direct solver takes the joints and the contacts, friction included, together: a heavy stack resting
        /// on a light part can't press it into the ground, and the friction under a craft holds all of it through its joints
        /// in one solve.
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

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            public StepInput StepInput;

            public void Execute(int islandIndex)
            {
                if (!Islands[islandIndex].SolvesInTeam)
                {
                    SolveDirectIsland(Info, JacobiansReader, Headers, Islands, Work, ref MotionVelocities, MotionDatas, StepInput, islandIndex, null);
                }
            }
        }

        /// <summary>
        /// Solves the islands big enough to share, one after another, with every lane of this job sharing each solve. It runs
        /// beside <see cref="SolveDirectIslandJob"/>, which takes the rest.
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
                        SolveDirectIsland(Info, JacobiansReader, Headers, Islands, Work, ref MotionVelocities, MotionDatas, StepInput, islandIndex, Team);
                    }
                }

                Team->Finish();
            }
        }

        static readonly ProfilerMarker DirectSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Island.DirectSolve");

        static unsafe void SolveDirectIsland(in DispatchPairSequencer.DirectSolverSchedulerInfo info, in NativeStream.Reader jacobiansReader,
            in NativeArray<IslandHeader> headers, in NativeArray<DirectIsland> islands, in DirectIslandBuffers work,
            ref NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, in StepInput stepInput, int islandIndex, SparseSolveTeam* team)
        {
            DirectIsland island = islands[islandIndex];
            using (DirectSolveMarker.Auto())
            {
                if (island.Rows > 0)
                {
                    JacobianRows rows = work.Rows(island);
                    Vector rightHandSide = DirectIslandBuffers.View(work.RightHandSide, island);
                    SolveSparsely(info, islandIndex, jacobiansReader, motionVelocities, motionDatas, stepInput, island.Rows, rows,
                        DirectIslandBuffers.View(work.Epsilon, island), DirectIslandBuffers.View(work.Lower, island),
                        DirectIslandBuffers.View(work.Upper, island), rightHandSide, work.Flags(island),
                        (*work.Bounded)[islandIndex] != 0, island.ContactPairs, team, true,
                        (IslandHeader*)headers.GetUnsafeReadOnlyPtr() + island.FirstHeader, work.HeaderRows->Ptr + island.FirstHeader,
                        island.SolveEnd - island.FirstHeader, work.Recovery->Ptr + island.FirstRow);
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

            public float Timestep;

            public void Execute(int index)
            {
                Integrator.IntegrateMotion(IslandBodies[index], MotionDatas, MotionVelocities, Timestep);
            }
        }
    }
}
