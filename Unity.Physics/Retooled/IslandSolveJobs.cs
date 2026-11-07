using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// One step of every island, in a fixed set of jobs that find out how much work they have when they start, which leaves
        /// nothing for the main thread to wait for. The islands that fit in phases run their iterations phase by phase, and the
        /// bodies they hold integrate after them. A direct island does all of its substeps inside one work item of one job.
        /// An island's bodies are its own, so the two sides run side by side.
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

            StepInput islandStepInput = stepInput;
            islandStepInput.Timestep /= directIslandSubsteps;
            islandStepInput.NumSubsteps = directIslandSubsteps;

            JobHandle islandHandle = new DirectIslandJob
            {
                DirectSolverSchedulerInfo = solverSchedulerInfo.DirectPairsDirectScheduling,
                JacobiansReader = jacobians.AsReader(),
                MotionVelocities = motionVelocities,
                MotionDatas = world.MotionDatas,
                Bodies = world.Bodies,
                StepInput = islandStepInput,
            }.Schedule(solverSchedulerInfo.DirectPairsDirectScheduling.DispatchPairIslandInfoCounts, 1, inputDeps);

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
        /// Solves and integrates one island of the direct solver through all of its substeps. The direct solver takes the joints.
        /// Contacts are outside what it can solve, so the iterative solver takes everything before it, to find what the contacts
        /// ask of the joints, and again after it, to reconcile the two.
        /// </summary>
        [BurstCompile]
        [NoAlias]
        struct DirectIslandJob : IJobParallelForDefer
        {
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo DirectSolverSchedulerInfo;

            public NativeStream.Reader JacobiansReader;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            public void Execute(int island)
            {
                // The job's own copy is shared with every thread running it.
                StepInput stepInput = StepInput;
                stepInput.DisableExportEvents = true;
                int numContacts = DirectSolverSchedulerInfo.IslandContactCounts[island];
                bool touchesSomething = HasSolidContact(island, numContacts);
                int firstBody = DirectSolverSchedulerInfo.FirstIslandBodyIndices[island];
                int bodyEnd = island + 1 < DirectSolverSchedulerInfo.FirstIslandBodyIndices.Length
                    ? DirectSolverSchedulerInfo.FirstIslandBodyIndices[island + 1]
                    : DirectSolverSchedulerInfo.IslandBodies.Length;

                for (int substep = 0; substep < stepInput.NumSubsteps; substep++)
                {
                    stepInput.CurrentSubstep = substep;
                    if (substep > 0)
                    {
                        var iterator = new IslandJacobianIterator(JacobiansReader, DirectSolverSchedulerInfo, island, numContacts);
                        while (iterator.HasJacobiansLeft())
                        {
                            UpdateJacobian(ref iterator.ReadJacobianHeader(), MotionDatas, MotionVelocities, Bodies, stepInput);
                        }
                    }

                    if (touchesSomething)
                    {
                        // One iteration short of last, as a contact counts its impulse towards events in the last one.
                        StepInput beforeDirect = stepInput;
                        beforeDirect.NumSolverIterations++;
                        Sweep(island, numContacts, stepInput.NumSolverIterations, beforeDirect);
                    }

                    DirectSolver(DirectSolverSchedulerInfo, island, JacobiansReader, ref MotionVelocities, MotionDatas, stepInput);

                    if (touchesSomething)
                    {
                        Sweep(island, numContacts, stepInput.NumSolverIterations, stepInput);
                    }

                    for (int i = firstBody; i < bodyEnd; i++)
                    {
                        Integrator.IntegrateMotion(DirectSolverSchedulerInfo.IslandBodies[i], MotionDatas, MotionVelocities, stepInput.Timestep);
                    }
                }
            }

            /// <summary>
            /// Whether any of the island's contacts can push on it. A craft's parts sit in each other's bounds all the time, and
            /// the jobs that read contacts disable most of what that produces, which isn't worth sweeping the joints for.
            /// </summary>
            bool HasSolidContact(int island, int numContacts)
            {
                var iterator = new IslandJacobianIterator(JacobiansReader, DirectSolverSchedulerInfo, island, numContacts);
                while (iterator.HasJacobiansLeft())
                {
                    ref JacobianHeader header = ref iterator.ReadJacobianHeader();
                    if (header.Type == JacobianType.Contact && header.Enabled)
                    {
                        return true;
                    }
                }

                return false;
            }

            void Sweep(int island, int numContacts, int iterations, StepInput stepInput)
            {
                // Nothing is exported here, so the writers stay unused.
                NativeStream.Writer noEvents = default;
                for (int iteration = 0; iteration < iterations; iteration++)
                {
                    stepInput.CurrentSolverIteration = iteration;
                    var iterator = new IslandJacobianIterator(JacobiansReader, DirectSolverSchedulerInfo, island, numContacts);
                    while (iterator.HasJacobiansLeft())
                    {
                        ref JacobianHeader header = ref iterator.ReadJacobianHeader();
                        int bodyA = header.BodyPair.BodyIndexA;
                        int bodyB = header.BodyPair.BodyIndexB;
                        MotionVelocity velocityA = bodyA < MotionVelocities.Length ? MotionVelocities[bodyA] : MotionVelocity.Zero;
                        MotionVelocity velocityB = bodyB < MotionVelocities.Length ? MotionVelocities[bodyB] : MotionVelocity.Zero;

                        header.Solve(ref velocityA, ref velocityB, stepInput, ref noEvents, ref noEvents, ref noEvents, false,
                            MotionStabilizationInput.Default, MotionStabilizationInput.Default);

                        if (bodyA < MotionVelocities.Length)
                        {
                            MotionVelocities[bodyA] = velocityA;
                        }

                        if (bodyB < MotionVelocities.Length)
                        {
                            MotionVelocities[bodyB] = velocityB;
                        }
                    }
                }
            }
        }
    }
}
