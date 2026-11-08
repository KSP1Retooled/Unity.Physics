using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Numerics.Linear;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Profiling;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// One step of every island, in a fixed set of jobs that find out how much work they have when they start, which leaves
        /// nothing for the main thread to wait for. The islands that fit in phases run their iterations phase by phase, and the
        /// bodies they hold integrate after them. The direct islands run each substep as a chain of jobs, with the Jacobians
        /// updated and their rows built in parallel. An island's bodies are its own, so the two sides run side by side.
        /// </summary>
        internal static SimulationJobHandles ScheduleIslandSolveJobs(in PhysicsWorld world, StepInput stepInput, int directIslandSubsteps,
            in NativeStream jacobians, in NativeStream collisionEvents, in NativeStream triggerEvents, in NativeStream impulseEvents,
            in DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo, StabilizationData solverStabilizationData, JobHandle inputDeps)
        {
            const int workItemsPerBatch = 4;
            NativeArray<MotionVelocity> motionVelocities = world.MotionVelocities;
            DispatchPairSequencer.IterativeSolverSchedulerInfo iterativePairs = solverSchedulerInfo.IterativePairsIterativeScheduling;

            JobHandle iterativeHandle = inputDeps;
            for (int iteration = 0; iteration < stepInput.NumSolverIterations; iteration++)
            {
                stepInput.CurrentSolverIteration = iteration;
                for (int phase = 0; phase < DispatchPairSequencer.NumIslandPhases; phase++)
                {
                    var job = new ParallelIterativeSolverJob
                    {
                        JacobiansReader = jacobians.AsReader(),
                        PhaseIndex = phase,
                        IterativeSolverSchedulerInfo = iterativePairs,
                        MotionVelocities = motionVelocities,
                        SolverStabilizationData = solverStabilizationData,
                        stepInput = stepInput
                    };

                    if (stepInput.ExportEventsInThisIteration)
                    {
                        job.CollisionEventsWriter = collisionEvents.AsWriter();
                        job.TriggerEventsWriter = triggerEvents.AsWriter();
                        job.ImpulseEventsWriter = impulseEvents.AsWriter();
                    }

                    iterativeHandle = SchedulePhase(job, iterativePairs, phase, workItemsPerBatch, iterativeHandle);
                }
            }

            iterativeHandle = Integrator.ScheduleIntegrateBodiesOutsideIslandsJob(world.DynamicsWorld,
                solverSchedulerInfo.DirectPairsDirectScheduling.BodiesInIslands.AsDeferredJobArray(), stepInput.Timestep, iterativeHandle);

            JobHandle islandHandle = ScheduleDirectIslandJobs(world, stepInput, directIslandSubsteps, jacobians,
                solverSchedulerInfo.DirectPairsDirectScheduling, inputDeps);

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
        /// The substeps of every direct island. Each substep updates the Jacobians, counts their rows, builds the rows, solves
        /// each island and integrates its bodies, and only the counting and the solves run one island at a time.
        /// </summary>
        static JobHandle ScheduleDirectIslandJobs(in PhysicsWorld world, StepInput stepInput, int substeps, in NativeStream jacobians,
            in DispatchPairSequencer.DirectSolverSchedulerInfo info, JobHandle inputDeps)
        {
            const int headersPerBatch = 16;
            const int bodiesPerBatch = 32;

            var work = new DirectIslandWork(Allocator.TempJob);

            JobHandle handle = new IndexDirectIslandsJob
            {
                Info = info,
                JacobiansReader = jacobians.AsReader(),
                Work = work,
            }.Schedule(inputDeps);

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
                }

                handle = new CountIslandRowsJob { Work = work }.Schedule(handle);

                handle = new BuildIslandRowsJob
                {
                    Headers = work.Headers.AsDeferredJobArray(),
                    Islands = work.Islands.AsDeferredJobArray(),
                    Work = work.Buffers,
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    StepInput = substepInput,
                }.Schedule(work.Headers, headersPerBatch, handle);

                handle = new SolveDirectIslandJob
                {
                    Info = info,
                    JacobiansReader = jacobians.AsReader(),
                    Islands = work.Islands.AsDeferredJobArray(),
                    Work = work.Buffers,
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    StepInput = substepInput,
                }.Schedule(work.Islands, 1, handle);

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

            /// <summary> Whether the island's contacts need the dense solve, which builds its own rows. </summary>
            public bool Dense;

            public int FirstRow;
            public int Rows;
        }

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
            [NativeDisableUnsafePtrRestriction] public UnsafeList<LCP.MLCPIndexFlag>* IndexFlags;

            /// <summary> Per island, 1 once any of its rows has a bound, which makes its solve a bounded one. </summary>
            [NativeDisableUnsafePtrRestriction] public UnsafeList<int>* Bounded;

            public DirectIslandBuffers(Allocator allocator)
            {
                HeaderRows = UnsafeList<int>.Create(0, allocator);
                Jacobian = UnsafeList<float>.Create(0, allocator);
                Epsilon = UnsafeList<float>.Create(0, allocator);
                Lower = UnsafeList<float>.Create(0, allocator);
                Upper = UnsafeList<float>.Create(0, allocator);
                RightHandSide = UnsafeList<float>.Create(0, allocator);
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
                UnsafeList<LCP.MLCPIndexFlag>.Destroy(IndexFlags);
                UnsafeList<int>.Destroy(Bounded);
            }
        }

        struct DirectIslandWork
        {
            public NativeList<IslandHeader> Headers;
            public NativeList<DirectIsland> Islands;
            public DirectIslandBuffers Buffers;

            public DirectIslandWork(Allocator allocator)
            {
                Headers = new NativeList<IslandHeader>(allocator);
                Islands = new NativeList<DirectIsland>(allocator);
                Buffers = new DirectIslandBuffers(allocator);
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
        /// them, and decides once a step whether an island's contacts take part in its solve.
        /// </summary>
        [BurstCompile]
        unsafe struct IndexDirectIslandsJob : IJob
        {
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            public NativeStream.Reader JacobiansReader;

            public DirectIslandWork Work;

            public void Execute()
            {
                int islandCount = Info.NumIslands;
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
                UnsafeList<int>* headerRows = Work.Buffers.HeaderRows;
                int totalRows = 0;
                for (int i = 0; i < Work.Islands.Length; i++)
                {
                    ref DirectIsland island = ref Work.Islands.ElementAt(i);
                    int rows = 0;
                    int contactCount = 0;
                    for (int h = island.FirstHeader; h < island.SolveEnd; h++)
                    {
                        ref JacobianHeader header = ref *Work.Headers[h].Header;
                        (*headerRows)[h] = rows;
                        rows += GetConstraintRows(ref header);
                        contactCount += header.Type == JacobianType.Contact ? 1 : 0;
                    }

                    for (int h = island.SolveEnd; h < island.HeaderEnd; h++)
                    {
                        (*headerRows)[h] = -1;
                    }

                    // Only contacts need the bound coupling of the dense solve. Everything else solves sparsely, with no matrix.
                    island.Dense = contactCount > 0 && island.ContactPairs == 0;
                    if (island.Dense)
                    {
                        for (int h = island.FirstHeader; h < island.SolveEnd; h++)
                        {
                            (*headerRows)[h] = -1;
                        }

                        rows = 0;
                    }

                    island.FirstRow = totalRows;
                    island.Rows = rows;
                    (*Work.Buffers.Bounded)[i] = 0;
                    totalRows += rows;
                }

                Work.Buffers.Resize(totalRows);
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

                // Every row starts free and unbounded, as a dense solve's do.
                int firstRow = island.FirstRow + startRow;
                for (int row = firstRow; row < firstRow + jacRows; row++)
                {
                    (*Work.Lower)[row] = float.MinValue;
                    (*Work.Upper)[row] = float.MaxValue;
                    (*Work.IndexFlags)[row] = LCP.MLCPIndexFlag.Free;
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
        /// Solves one direct island from the rows built for it and applies the impulses. A dense island builds and solves its
        /// rows here. The direct solver takes the joints and the contacts, friction included, together: a heavy stack resting
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
            public NativeArray<DirectIsland> Islands;

            public DirectIslandBuffers Work;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            public StepInput StepInput;

            static readonly ProfilerMarker DirectSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Island.DirectSolve");

            public void Execute(int islandIndex)
            {
                DirectIsland island = Islands[islandIndex];
                using (DirectSolveMarker.Auto())
                {
                    if (island.Dense)
                    {
                        DirectSolver(Info, islandIndex, JacobiansReader, ref MotionVelocities, MotionDatas, StepInput, island.ContactPairs);
                    }
                    else if (island.Rows > 0)
                    {
                        JacobianRows rows = Work.Rows(island);
                        Vector rightHandSide = DirectIslandBuffers.View(Work.RightHandSide, island);
                        SolveSparsely(Info, islandIndex, JacobiansReader, MotionVelocities, MotionDatas, StepInput, island.Rows, rows,
                            DirectIslandBuffers.View(Work.Epsilon, island), DirectIslandBuffers.View(Work.Lower, island),
                            DirectIslandBuffers.View(Work.Upper, island), rightHandSide, Work.Flags(island),
                            (*Work.Bounded)[islandIndex] != 0, island.ContactPairs);

                        ApplyImpulses(Info, islandIndex, JacobiansReader, ref MotionVelocities, MotionDatas, island.ContactPairs, rows,
                            Work.RightHandSide->Ptr + island.FirstRow);
                    }
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
