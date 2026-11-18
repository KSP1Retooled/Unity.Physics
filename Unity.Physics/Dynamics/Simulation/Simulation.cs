using Unity.Physics.Dynamics.Solver.Islands.Scheduling;
using Unity.Physics.Dynamics.Solver.Vessels;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Vessels.Model;

namespace Unity.Physics
{
    /// <summary>
    /// Holds temporary data in a storage that lives as long as simulation lives and is only re-
    /// allocated if necessary.
    /// </summary>
    public struct SimulationContext : IDisposable
    {
        private int m_NumDynamicBodies;
        private NativeArray<Velocity> m_InputVelocities;

        // Solver stabilization data (it's completely ok to be unallocated)
        [NativeDisableContainerSafetyRestriction]
        private NativeArray<Solver.StabilizationMotionData> m_SolverStabilizationMotionData;
        internal NativeArray<Solver.StabilizationMotionData> SolverStabilizationMotionData => m_SolverStabilizationMotionData.GetSubArray(0, m_NumDynamicBodies);

        internal float TimeStep;

        internal NativeArray<Velocity> InputVelocities => m_InputVelocities.GetSubArray(0, m_NumDynamicBodies);

        internal NativeStream CollisionEventDataStream;
        internal NativeStream TriggerEventDataStream;
        internal NativeStream ImpulseEventDataStream;

        // Previous step's island pair count, used to choose serial or parallel sorting.
        internal NativeReference<int> IslandPairCount;

        /// <summary>   Gets the collision events. </summary>
        ///
        /// <value> The collision events. </value>
        public CollisionEvents CollisionEvents => new CollisionEvents(CollisionEventDataStream, InputVelocities, TimeStep);

        /// <summary>   Gets the trigger events. </summary>
        ///
        /// <value> The trigger events. </value>
        public TriggerEvents TriggerEvents => new TriggerEvents(TriggerEventDataStream);

        /// <summary>   Gets the impulse events. </summary>
        ///
        /// <value> The impulse events. </value>
        public ImpulseEvents ImpulseEvents => new ImpulseEvents(ImpulseEventDataStream);

        internal bool ReadyForEventScheduling => m_InputVelocities.IsCreated && CollisionEventDataStream.IsCreated && TriggerEventDataStream.IsCreated && ImpulseEventDataStream.IsCreated;

        /// <summary>
        /// Resets the simulation storage
        /// - Reallocates input velocities storage if necessary
        /// - Disposes event streams and allocates new ones with enough work items for a single-threaded step
        /// NOTE: Reset or ScheduleReset needs to be called before passing the SimulationContext to a
        /// simulation step job. If you don't then you may get initialization errors.
        /// </summary>
        ///
        /// <param name="stepInput">    The step input. </param>
        public void Reset(SimulationStepInput stepInput)
        {
            // A context that is only ever reset here, never through ScheduleReset, still needs the pair count: a step
            // job that it is given checks that all of its containers exist.
            if (!IslandPairCount.IsCreated)
            {
                IslandPairCount = new NativeReference<int>(0, Allocator.Persistent);
            }

            m_NumDynamicBodies = stepInput.World.NumDynamicBodies;
            if (!m_InputVelocities.IsCreated || m_InputVelocities.Length < m_NumDynamicBodies)
            {
                if (m_InputVelocities.IsCreated)
                {
                    m_InputVelocities.Dispose();
                }
                m_InputVelocities = new NativeArray<Velocity>(m_NumDynamicBodies, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            }

            // Solver stabilization data
            if (stepInput.SolverStabilizationHeuristicSettings.EnableSolverStabilization)
            {
                if (!m_SolverStabilizationMotionData.IsCreated || m_SolverStabilizationMotionData.Length < m_NumDynamicBodies)
                {
                    if (m_SolverStabilizationMotionData.IsCreated)
                    {
                        m_SolverStabilizationMotionData.Dispose();
                    }
                    m_SolverStabilizationMotionData = new NativeArray<Solver.StabilizationMotionData>(m_NumDynamicBodies, Allocator.Persistent, NativeArrayOptions.ClearMemory);
                }
                else if (m_NumDynamicBodies > 0)
                {
                    unsafe
                    {
                        UnsafeUtility.MemClear(m_SolverStabilizationMotionData.GetUnsafePtr(), m_NumDynamicBodies * UnsafeUtility.SizeOf<Solver.StabilizationMotionData>());
                    }
                }
            }

            if (CollisionEventDataStream.IsCreated)
            {
                CollisionEventDataStream.Dispose();
            }
            if (TriggerEventDataStream.IsCreated)
            {
                TriggerEventDataStream.Dispose();
            }
            if (ImpulseEventDataStream.IsCreated)
            {
                ImpulseEventDataStream.Dispose();
            }

            {
                const int k_NumWorkItems = DispatchPairSequencer.SolverSchedulerInfo.k_MaxSingleThreadedWorkItems;
                CollisionEventDataStream = new NativeStream(k_NumWorkItems, Allocator.Persistent);
                TriggerEventDataStream = new NativeStream(k_NumWorkItems, Allocator.Persistent);
                ImpulseEventDataStream = new NativeStream(k_NumWorkItems, Allocator.Persistent);
            }
        }

        internal JobHandle ScheduleReset(SimulationStepInput stepInput, JobHandle inputDeps)
        {
            if (!IslandPairCount.IsCreated)
            {
                IslandPairCount = new NativeReference<int>(0, Allocator.Persistent);
            }

            m_NumDynamicBodies = stepInput.World.NumDynamicBodies;
            if (!m_InputVelocities.IsCreated || m_InputVelocities.Length < m_NumDynamicBodies)
            {
                if (m_InputVelocities.IsCreated)
                {
                    m_InputVelocities.Dispose();
                }
                m_InputVelocities = new NativeArray<Velocity>(m_NumDynamicBodies, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            }

            // Solver stabilization data
            if (stepInput.SolverStabilizationHeuristicSettings.EnableSolverStabilization)
            {
                if (!m_SolverStabilizationMotionData.IsCreated || m_SolverStabilizationMotionData.Length < m_NumDynamicBodies)
                {
                    if (m_SolverStabilizationMotionData.IsCreated)
                    {
                        m_SolverStabilizationMotionData.Dispose();
                    }
                    m_SolverStabilizationMotionData = new NativeArray<Solver.StabilizationMotionData>(m_NumDynamicBodies, Allocator.Persistent, NativeArrayOptions.ClearMemory);
                }
                else if (m_NumDynamicBodies > 0)
                {
                    unsafe
                    {
                        UnsafeUtility.MemClear(m_SolverStabilizationMotionData.GetUnsafePtr(), m_NumDynamicBodies * UnsafeUtility.SizeOf<Solver.StabilizationMotionData>());
                    }
                }
            }

            var handle = inputDeps;
            if (CollisionEventDataStream.IsCreated)
            {
                handle = CollisionEventDataStream.Dispose(handle);
            }
            if (TriggerEventDataStream.IsCreated)
            {
                handle = TriggerEventDataStream.Dispose(handle);
            }
            if (ImpulseEventDataStream.IsCreated)
            {
                handle = ImpulseEventDataStream.Dispose(handle);
            }

            return handle;
        }

        /// <summary>
        /// Disposes the simulation context.
        /// </summary>
        public void Dispose()
        {
            if (m_InputVelocities.IsCreated)
            {
                m_InputVelocities.Dispose();
            }

            if (IslandPairCount.IsCreated)
            {
                IslandPairCount.Dispose();
            }

            if (m_SolverStabilizationMotionData.IsCreated)
            {
                m_SolverStabilizationMotionData.Dispose();
            }

            if (CollisionEventDataStream.IsCreated)
            {
                CollisionEventDataStream.Dispose();
            }

            if (TriggerEventDataStream.IsCreated)
            {
                TriggerEventDataStream.Dispose();
            }

            if (ImpulseEventDataStream.IsCreated)
            {
                ImpulseEventDataStream.Dispose();
            }
        }
    }

    // Temporary data created and destroyed during the step
    struct StepContext
    {
        // Built by the scheduler. Groups body pairs into phases in which each
        // body appears at most once, so that the interactions within each phase can be solved
        // in parallel with each other but not with other phases. This is consumed by the
        // ProcessBodyPairsJob, which outputs contact and joint Jacobians.
        public NativeList<DispatchPairSequencer.DispatchPair> PhasedDispatchPairs;

        // The sorted pairs available after the broadphase and before Simulation.ScheduleIslandJobs runs.
        public NativeList<DispatchPairSequencer.DispatchPair> PairsToSortByIsland;
        public bool IslandsSorted;

        // Built by the scheduler. Describes the grouping of phased dispatch pairs for parallel processing
        // of joints and contacts in the solver.
        // Informs how we can schedule the solver jobs and what data locations they read info from.
        public DispatchPairSequencer.SolverSchedulerInfo SolverSchedulerInfo;

        public NativeStream Contacts;
        public NativeStream Jacobians;

        // This handle completes when the solver has written the final motions. Buffer disposal and direct island
        // event export can finish later.
        public JobHandle MotionsSolved;
        public bool HasMotionsSolved;

        // When the step solves by island, gravity is applied while the pairs are found and sorted. The narrow phase
        // waits for this handle instead of scheduling gravity itself.
        public JobHandle GravityApplied;
        public bool HasGravityApplied;
    }

    /// <summary>   Steps a physics world. </summary>
    public partial struct Simulation : ISimulation
    {
        /// <summary>   Gets the simulation type. </summary>
        ///
        /// <value> <see cref="SimulationType.UnityPhysics"/>. </value>
        public SimulationType Type => SimulationType.UnityPhysics;

        /// <summary>   Gets the handle of the final simulation job (not including dispose jobs). </summary>
        ///
        /// <value> The final simulation job handle. </value>
        public JobHandle FinalSimulationJobHandle => m_StepHandles.FinalExecutionHandle;

        /// <summary>   Gets the handle of the final job. </summary>
        ///
        /// <value> The final job handle. </value>
        public JobHandle FinalJobHandle => JobHandle.CombineDependencies(FinalSimulationJobHandle, m_StepHandles.FinalDisposeHandle);

        internal SimulationScheduleStage m_SimulationScheduleStage;

        internal StepContext StepContext;

        /// <summary>
        /// Gets the contacts stream.
        ///
        /// This value is only valid after the CreateContactsJob (Narrowphase System),
        /// and before BuildJacobiansJob (CreateJacobiansSystem).
        /// </summary>
        /// <value> The contacts stream. </value>
        public readonly NativeStream Contacts => StepContext.Contacts;

        /// <summary>   Gets the collision events. </summary>
        ///
        /// <value> The collision events. </value>
        public CollisionEvents CollisionEvents => SimulationContext.CollisionEvents;

        /// <summary>   Gets the trigger events. </summary>
        ///
        /// <value> The trigger events. </value>
        public TriggerEvents TriggerEvents => SimulationContext.TriggerEvents;

        /// <summary>   Gets the impulse events. </summary>
        ///
        /// <value> The impulse events. </value>
        public ImpulseEvents ImpulseEvents => SimulationContext.ImpulseEvents;

        internal SimulationContext SimulationContext;

        internal SimulationJobHandles m_StepHandles;

        internal bool ReadyForEventScheduling => SimulationContext.ReadyForEventScheduling;

        /// <summary>   Creates a new Simulation. </summary>
        ///
        /// <returns>   A Simulation. </returns>
        public static Simulation Create()
        {
            Simulation sim = new Simulation();
            sim.Init();
            return sim;
        }

        /// <summary>
        /// Disposes the simulation.
        /// </summary>
        public void Dispose()
        {
            SimulationContext.Dispose();
        }

        private void Init()
        {
            StepContext = new StepContext();
            SimulationContext = new SimulationContext();
            m_StepHandles = new SimulationJobHandles(new JobHandle());
            m_SimulationScheduleStage = SimulationScheduleStage.Idle;
        }

        /// <summary>
        /// Steps the simulation immediately on a single thread without spawning any jobs.
        /// </summary>
        ///
        /// <param name="input">                The input. </param>
        /// <param name="simulationContext">    [in,out] Context for the simulation. </param>
        public static void StepImmediate(SimulationStepInput input, ref SimulationContext simulationContext)
        {
            SafetyChecks.CheckFiniteAndPositiveAndThrow(input.TimeStep, nameof(input.TimeStep));
            SafetyChecks.CheckInRangeAndThrow(input.NumSolverIterations, new int2(1, int.MaxValue), nameof(input.NumSolverIterations));

            if (input.World.NumDynamicBodies == 0)
            {
                // No need to do anything, since nothing can move
                return;
            }

            simulationContext.TimeStep = input.TimeStep;

            using var dynamicPairs = new NativeStream(1, Allocator.Temp);
            using var staticPairs = new NativeStream(1, Allocator.Temp);
            var dynamicWriter = dynamicPairs.AsWriter();
            var staticWriter = staticPairs.AsWriter();
            input.World.CollisionWorld.FindOverlaps(ref dynamicWriter, ref staticWriter);

            using var sortedPairs = new NativeList<DispatchPairSequencer.DispatchPair>(Allocator.Temp);
            using var phasedPairs = new NativeList<DispatchPairSequencer.DispatchPair>(Allocator.Temp);
            using var info = new DispatchPairSequencer.SolverSchedulerInfo(DispatchPairSequencer.NumIslandPhases, Allocator.Temp);
            DispatchPairSequencer.SortPairsForIslands(input.World, dynamicPairs.AsReader(), staticPairs.AsReader(),
                phasedPairs, sortedPairs);
            DispatchPairSequencer.SortPairsByIsland(input, sortedPairs, phasedPairs, info);

            Solver.ApplyGravityAndUpdateInputVelocities(input.World.MotionVelocities,
                simulationContext.InputVelocities, input.Gravity, input.TimeStep, input.EnableGyroscopicTorque);

            int workItems = info.NumIterativeWorkItems.Value;
            using var contacts = new NativeStream(math.max(1, workItems), Allocator.Temp);
            using var jacobians = new NativeStream(math.max(1, workItems), Allocator.Temp);
            var contactsWriter = contacts.AsWriter();
            var world = input.World;
            NarrowPhase.CreateContacts(ref world, simulationContext.InputVelocities, phasedPairs.AsArray(),
                info, input.TimeStep, ref contactsWriter);
            Solver.BuildIslandJacobians(input, phasedPairs.AsArray(), info, contacts.AsReader(), jacobians.AsWriter());
            var schedulerInfo = info;
            Solver.BuildDirectSolverJacobiansMap(ref schedulerInfo, jacobians.AsReader(), phasedPairs.AsArray());
            IslandSolveScheduler.SolveIslands(input.World, CreateSolverInput(input), jacobians,
                simulationContext.CollisionEventDataStream, simulationContext.TriggerEventDataStream,
                simulationContext.ImpulseEventDataStream, info, new Solver.StabilizationData(input, simulationContext));

            if (input.SynchronizeCollisionWorld)
            {
                input.World.CollisionWorld.UpdateDynamicTree(ref input.World, input.TimeStep, input.Gravity);
            }
        }

        /// <summary>   Schedule broadphase jobs. </summary>
        ///
        /// <param name="input">            The input. </param>
        /// <param name="inputDeps">        The input deps. </param>
        /// <param name="multiThreaded">    (Optional) True if multi threaded. </param>
        ///
        /// <returns>   The SimulationJobHandles. </returns>
        public SimulationJobHandles ScheduleBroadphaseJobs(SimulationStepInput input, JobHandle inputDeps, bool multiThreaded = true)
        {
            return ScheduleBroadphaseJobsInternal(input, inputDeps, multiThreaded, incrementalDynamicBroadphase: false, incrementalStaticBroadphase: false);
        }

        internal SimulationJobHandles ScheduleBroadphaseJobsInternal(SimulationStepInput input, JobHandle inputDeps, bool multiThreaded, bool incrementalDynamicBroadphase, bool incrementalStaticBroadphase)
        {
            SafetyChecks.CheckFiniteAndPositiveAndThrow(input.TimeStep, nameof(input.TimeStep));
            SafetyChecks.CheckInRangeAndThrow(input.NumSolverIterations, new int2(1, int.MaxValue), nameof(input.NumSolverIterations));
            SafetyChecks.CheckSimulationStageAndThrow(m_SimulationScheduleStage, SimulationScheduleStage.Idle);
            m_SimulationScheduleStage = SimulationScheduleStage.PostCreateBodyPairs;

            // Dispose and reallocate input velocity buffer, if dynamic body count has increased.
            // Dispose previous collision, trigger and impulse event data streams.
            // New event streams are reallocated later when the work item count is known.
            JobHandle handle = SimulationContext.ScheduleReset(input, inputDeps);
            SimulationContext.TimeStep = input.TimeStep;

            StepContext = new StepContext();

            JobHandle joints = input.World.DynamicsWorld.JointsBuilt;
            if (input.World.NumDynamicBodies == 0)
            {
                // No need to do anything, since nothing can move
                m_StepHandles = new SimulationJobHandles(JobHandle.CombineDependencies(handle, joints));
                return m_StepHandles;
            }

            // Apply gravity alongside island pair sorting; the narrow phase waits for this handle.
            if (multiThreaded)
            {
                StepContext.GravityApplied = Solver.ScheduleApplyGravityAndUpdateInputVelocitiesJob(ref input.World.DynamicsWorld,
                    SimulationContext.InputVelocities, input.Gravity, input.TimeStep, input.EnableGyroscopicTorque, handle, multiThreaded);
                StepContext.HasGravityApplied = true;
            }

            // Find all body pairs that overlap in the broadphase
            var handles = input.World.CollisionWorld.ScheduleFindOverlapsJobsInternal(
                out NativeStream dynamicVsDynamicBodyPairs, out NativeStream dynamicVsStaticBodyPairs,
                handle, multiThreaded, incrementalDynamicBroadphase, incrementalStaticBroadphase);
            handle = handles.FinalExecutionHandle;
            var disposeHandle = handles.FinalDisposeHandle;
            var postOverlapsHandle = handle;

            // Sort all overlapping and jointed body pairs into phases
            handles = DispatchPairSequencer.ScheduleCreatePhasedDispatchPairsJob(input,
                ref dynamicVsDynamicBodyPairs, ref dynamicVsStaticBodyPairs, JobHandle.CombineDependencies(handle, joints),
                ref StepContext.PhasedDispatchPairs, out StepContext.SolverSchedulerInfo, out StepContext.PairsToSortByIsland,
                multiThreaded, SimulationContext.IslandPairCount);

            m_StepHandles.FinalDisposeHandle = JobHandle.CombineDependencies(handles.FinalDisposeHandle, disposeHandle);
            m_StepHandles.FinalExecutionHandle = multiThreaded ?
                JobHandle.CombineDependencies(handles.FinalExecutionHandle, postOverlapsHandle) : handles.FinalExecutionHandle;

            return m_StepHandles;
        }

        /// <summary>   Schedule narrowphase jobs. </summary>
        ///
        /// <param name="input">            The input. </param>
        /// <param name="inputDeps">        The input deps. </param>
        /// <param name="multiThreaded">    (Optional) True if multi threaded. </param>
        ///
        /// <returns>   The SimulationJobHandles. </returns>
        public SimulationJobHandles ScheduleNarrowphaseJobs(SimulationStepInput input, JobHandle inputDeps, bool multiThreaded = true)
        {
            SafetyChecks.CheckSimulationStageAndThrow(m_SimulationScheduleStage, SimulationScheduleStage.PostCreateBodyPairs);
            m_SimulationScheduleStage = SimulationScheduleStage.PostCreateContacts;

            if (input.World.NumDynamicBodies == 0)
            {
                // No need to do anything, since nothing can move
                m_StepHandles = new SimulationJobHandles(inputDeps);
                return m_StepHandles;
            }

            if (!StepContext.IslandsSorted)
            {
                inputDeps = ScheduleIslandJobs(input, inputDeps, multiThreaded);
            }

            var disposeHandle = m_StepHandles.FinalDisposeHandle;
            var handle = StepContext.HasGravityApplied ? JobHandle.CombineDependencies(inputDeps, StepContext.GravityApplied)
                : Solver.ScheduleApplyGravityAndUpdateInputVelocitiesJob(ref input.World.DynamicsWorld,
                    SimulationContext.InputVelocities, input.Gravity, input.TimeStep, input.EnableGyroscopicTorque, inputDeps, multiThreaded);

            m_StepHandles = NarrowPhase.ScheduleCreateContactsJobs(ref input.World, input.TimeStep,
                SimulationContext.InputVelocities, ref StepContext.Contacts, ref StepContext.Jacobians,
                ref StepContext.PhasedDispatchPairs, handle, ref StepContext.SolverSchedulerInfo, multiThreaded);

            m_StepHandles.FinalDisposeHandle = JobHandle.CombineDependencies(disposeHandle, m_StepHandles.FinalDisposeHandle);

            return m_StepHandles;
        }

        /// <summary>   Schedule create jacobians jobs. </summary>
        ///
        /// <param name="input">            The input. </param>
        /// <param name="inputDeps">        The input deps. </param>
        /// <param name="multiThreaded">    (Optional) True if multi threaded. </param>
        ///
        /// <returns>   The SimulationJobHandles. </returns>
        public SimulationJobHandles ScheduleCreateJacobiansJobs(SimulationStepInput input, JobHandle inputDeps, bool multiThreaded = true)
        {
            SafetyChecks.CheckSimulationStageAndThrow(m_SimulationScheduleStage, SimulationScheduleStage.PostCreateContacts);
            m_SimulationScheduleStage = SimulationScheduleStage.PostCreateJacobians;

            if (input.World.NumDynamicBodies == 0)
            {
                // No need to do anything, since nothing can move
                m_StepHandles = new SimulationJobHandles(inputDeps);
                return m_StepHandles;
            }

            // Build Jacobians for the island solve.
            var disposeHandle = m_StepHandles.FinalDisposeHandle;
            m_StepHandles = Solver.ScheduleBuildJacobiansJobs(input,
                inputDeps, ref StepContext.PhasedDispatchPairs,
                ref StepContext.SolverSchedulerInfo, ref StepContext.Contacts, ref StepContext.Jacobians, multiThreaded);
            m_StepHandles.FinalDisposeHandle = JobHandle.CombineDependencies(disposeHandle, m_StepHandles.FinalDisposeHandle);

            return m_StepHandles;
        }

        /// <summary>   Schedule solve and integrate jobs. </summary>
        ///
        /// <param name="input">            The input. </param>
        /// <param name="inputDeps">        The input deps. </param>
        /// <param name="multiThreaded">    (Optional) True if multi threaded. </param>
        ///
        /// <returns>   The SimulationJobHandles. </returns>
        public SimulationJobHandles ScheduleSolveAndIntegrateJobs(SimulationStepInput input, JobHandle inputDeps, bool multiThreaded = true)
        {
            SafetyChecks.CheckSimulationStageAndThrow(m_SimulationScheduleStage, SimulationScheduleStage.PostCreateJacobians);
            m_SimulationScheduleStage = SimulationScheduleStage.Idle;

            if (input.World.NumDynamicBodies == 0)
            {
                // No need to do anything, since nothing can move
                m_StepHandles = new SimulationJobHandles(inputDeps);
                return m_StepHandles;
            }

            var executionHandle = m_StepHandles.FinalExecutionHandle;
            var disposeHandle = m_StepHandles.FinalDisposeHandle;
            var jobHandle = inputDeps;
            Solver.StabilizationData solverStabilizationData = new Solver.StabilizationData(input, SimulationContext);

            var numWorkItems = StepContext.SolverSchedulerInfo.NumIterativeWorkItems;

            JobHandle collisionEventStreamHandle = NativeStream.ScheduleConstruct(
            out SimulationContext.CollisionEventDataStream, numWorkItems, inputDeps, Allocator.Persistent);
            JobHandle triggerEventStreamHandle = NativeStream.ScheduleConstruct(
            out SimulationContext.TriggerEventDataStream, numWorkItems, inputDeps, Allocator.Persistent);
            JobHandle impulseEventStreamHandle = NativeStream.ScheduleConstruct(
            out SimulationContext.ImpulseEventDataStream, numWorkItems, inputDeps, Allocator.Persistent);

            var streamJobHandle = JobHandle.CombineDependencies(
                collisionEventStreamHandle, triggerEventStreamHandle, impulseEventStreamHandle);
            jobHandle = JobHandle.CombineDependencies(jobHandle, streamJobHandle);

            var stepInput = CreateSolverInput(input);
            m_StepHandles = IslandSolveScheduler.ScheduleIslandSolveJobs(input.World, stepInput,
                StepContext.Jacobians, SimulationContext.CollisionEventDataStream,
                SimulationContext.TriggerEventDataStream, SimulationContext.ImpulseEventDataStream,
                StepContext.SolverSchedulerInfo, solverStabilizationData, executionHandle, jobHandle, out JobHandle motionsSolved,
                multiThreaded);

            StepContext.MotionsSolved = JobHandle.CombineDependencies(executionHandle, motionsSolved);
            StepContext.HasMotionsSolved = true;
            m_StepHandles.FinalDisposeHandle = JobHandle.CombineDependencies(disposeHandle, m_StepHandles.FinalDisposeHandle);
            jobHandle = JobHandle.CombineDependencies(executionHandle, m_StepHandles.FinalExecutionHandle);
            m_StepHandles.FinalExecutionHandle = jobHandle;
            JobHandle.ScheduleBatchedJobs();

            m_StepHandles.FinalExecutionHandle = JobHandle.CombineDependencies(m_StepHandles.FinalExecutionHandle, jobHandle);

            // Synchronize the collision world
            if (input.SynchronizeCollisionWorld)
            {
                m_StepHandles.FinalExecutionHandle = input.World.CollisionWorld.ScheduleUpdateDynamicTree(
                    ref input.World, input.TimeStep, input.Gravity, m_StepHandles.FinalExecutionHandle, multiThreaded);
            }

            return m_StepHandles;
        }

        static Solver.StepInput CreateSolverInput(in SimulationStepInput input)
        {
            var stepInput = new Solver.StepInput()
            {
                Gravity = input.Gravity,
                Timestep = input.TimeStep,
                EnableGyroscopicTorque = input.EnableGyroscopicTorque,
                NumSubsteps = 1,
                NumSolverIterations = input.NumSolverIterations,
                DirectSolverSettings = input.DirectSolverSettings,
                IslandSolverCache = input.IslandSolverCache,
                VesselSet = VesselSet.OrTestSet(input.VesselSet),
                CurrentSubstep = 0,
                CurrentSolverIteration = -1,
                MaxStaticDepenetrationVelocity = input.MaxStaticDepenetrationVelocity > 0
                    ? input.MaxStaticDepenetrationVelocity : SimulationStepInput.DefaultMaxStaticDepenetrationVelocity,
                MaxDynamicDepenetrationVelocity = input.MaxDynamicDepenetrationVelocity > 0
                    ? input.MaxDynamicDepenetrationVelocity : SimulationStepInput.DefaultMaxDynamicDepenetrationVelocity,
            };
            VesselIslandSolver.SetBodies(ref stepInput, input.World);

            return stepInput;
        }

        /// <summary>
        /// Resets the simulation storage
        /// - Reallocates input velocities storage if necessary
        /// - Disposes event streams and allocates new ones with enough work items for a single-threaded step
        /// </summary>
        /// <param name="input">    The input. </param>
        public void ResetSimulationContext(SimulationStepInput input)
        {
            SimulationContext.Reset(input);
        }

        /// <summary>   Steps the world immediately. </summary>
        ///
        /// <param name="input">    The input. </param>
        public void Step(SimulationStepInput input)
        {
            StepImmediate(input, ref SimulationContext);
        }

        /// <summary>
        /// Schedule an island step. Use the staged methods to run callbacks between stages.
        /// Set multiThreaded to false to run each stage on one thread.
        /// </summary>
        ///
        /// <param name="input">            The input. </param>
        /// <param name="inputDeps">        The input deps. </param>
        /// <param name="multiThreaded">    (Optional) True if multi threaded. </param>
        ///
        /// <returns>   The SimulationJobHandles. </returns>
        public SimulationJobHandles ScheduleStepJobs(SimulationStepInput input, JobHandle inputDeps, bool multiThreaded = true)
        {
            ScheduleBroadphaseJobs(input, inputDeps, multiThreaded);
            ScheduleNarrowphaseJobs(input, m_StepHandles.FinalExecutionHandle, multiThreaded);
            ScheduleCreateJacobiansJobs(input, m_StepHandles.FinalExecutionHandle, multiThreaded);
            ScheduleSolveAndIntegrateJobs(input, m_StepHandles.FinalExecutionHandle, multiThreaded);

            return m_StepHandles;
        }
    }
}
