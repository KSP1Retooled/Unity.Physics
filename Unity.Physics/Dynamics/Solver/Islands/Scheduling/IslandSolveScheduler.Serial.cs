using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Configuration;
using static Unity.Physics.Solver;

namespace Unity.Physics.Dynamics.Solver.Islands.Scheduling
{
    internal static partial class IslandSolveScheduler
    {
        [BurstCompile]
        struct SolveIslandsJob : IJob
        {
            public PhysicsWorld World;
            public StepInput Input;
            public NativeStream Jacobians;
            public NativeStream CollisionEvents;
            public NativeStream TriggerEvents;
            public NativeStream ImpulseEvents;
            public DispatchPairSequencer.SolverSchedulerInfo Info;
            public StabilizationData Stabilization;

            public void Execute()
            {
                SolveIslands(World, Input, Jacobians, CollisionEvents, TriggerEvents, ImpulseEvents, Info, Stabilization);
            }
        }

        /// <summary>Run the island solve on one thread, including when a caller steps from inside a job.</summary>
        internal static unsafe void SolveIslands(in PhysicsWorld world, StepInput stepInput, in NativeStream jacobians,
            in NativeStream collisionEvents, in NativeStream triggerEvents, in NativeStream impulseEvents,
            in DispatchPairSequencer.SolverSchedulerInfo info, StabilizationData stabilization)
        {
            var iterative = new IterativeSolveWork
            {
                JacobiansReader = jacobians.AsReader(),
                IterativeSolverSchedulerInfo = info.IterativePairsIterativeScheduling,
                MotionVelocities = world.MotionVelocities,
                SolverStabilizationData = stabilization,
                CollisionEventsWriter = collisionEvents.AsWriter(),
                TriggerEventsWriter = triggerEvents.AsWriter(),
                ImpulseEventsWriter = impulseEvents.AsWriter(),
                stepInput = stepInput,
            };
            Iterate(iterative, 0, stepInput.NumSolverIterations);
            var inIslands = info.DirectPairsDirectScheduling.BodiesInIslands;
            for (int i = 0; i < world.NumDynamicBodies; i++)
            {
                if (inIslands.Length == 0 || !inIslands[i])
                {
                    Integrator.IntegrateMotion(i, world.MotionDatas, world.MotionVelocities, stepInput.Timestep);
                }
            }

            iterative.stepInput.Relaxing = true;
            iterative.stepInput.DisableExportEvents = true;
            Iterate(iterative, stepInput.NumSolverIterations, RelaxIterations);

            const int substeps = SolverConstants.DirectIslandSubsteps;
            var work = new DirectIslandWork(substeps, Allocator.TempJob, configureTeams: false);
            new IndexDirectIslandsJob
            {
                Info = info.DirectPairsDirectScheduling,
                JacobiansReader = jacobians.AsReader(),
                Work = work,
                Cache = stepInput.IslandSolverCache,
                Bodies = world.Bodies,
                MotionCount = world.NumDynamicBodies,
            }.Execute();

            new MarkFusedVesselIslandsJob
            {
                Islands = work.Islands,
                IslandVessels = work.IslandVessels,
                Info = info.DirectPairsDirectScheduling,
                UnfusedBodies = work.UnfusedBodies,
                Vessels = stepInput.VesselSet,
                Cache = stepInput.IslandSolverCache,
                Fuse = SolverSwitches.VesselFused,
                MinimumParts = SolverSwitches.FusedVesselParts,
                Contacts = SolverSwitches.FusedContacts,
                Buffers = work.Buffers,
            }.Execute();

            StepInput substepInput = stepInput;
            substepInput.Timestep /= substeps;
            substepInput.NumSubsteps = substeps;
            substepInput.DisableExportEvents = true;
            var unfused = new UnfusedSubsteps
            {
                Buffers = work.Buffers,
                StepInput = substepInput,
            };
            if (SolverSwitches.VesselFused)
            {
                new SolveFusedVesselIslandsJob
                {
                    Islands = work.Islands.AsArray(),
                    IslandVessels = work.IslandVessels.AsArray(),
                    Headers = work.Headers.AsArray(),
                    UnfusedBodies = work.UnfusedBodies.AsArray(),
                    MotionVelocities = world.MotionVelocities,
                    MotionDatas = world.MotionDatas,
                    Bodies = world.Bodies,
                    StepInput = substepInput,
                    Teams = work.Buffers.FusedTeams,
                    Lanes = 1,
                    Unfused = unfused,
                    UnfusedClaim = work.Buffers.UnfusedClaim,
                }.Execute(0);
            }
            else
            {
                RunUnfusedSubsteps(unfused, work.Headers.AsArray(), work.Islands.AsArray(), work.IslandVessels.AsArray(),
                    work.UnfusedBodies.AsArray(), world.MotionVelocities, world.MotionDatas, world.Bodies);
            }

            var export = new ParallelExportEventsJob
            {
                JacobiansReader = jacobians.AsReader(),
                MotionVelocities = world.MotionVelocities,
                MotionDatas = world.MotionDatas,
                CollisionEventsWriter = collisionEvents.AsWriter(),
                TriggerEventsWriter = triggerEvents.AsWriter(),
                ImpulseEventsWriter = impulseEvents.AsWriter(),
                IterativeSolverSchedulerInfo = info.DirectPairsIterativeScheduling,
                Timestep = stepInput.Timestep,
            };
            for (int i = 0; i < export.IterativeSolverSchedulerInfo.NumWorkItems[0]; i++)
            {
                export.Execute(i);
            }

            work.Headers.Dispose();
            work.Islands.Dispose();
            work.IslandVessels.Dispose();
            work.UnfusedBodies.Dispose();
            work.Buffers.Destroy();
        }

        static void Iterate(IterativeSolveWork job, int firstIteration, int iterations)
        {
            for (int i = firstIteration; i < firstIteration + iterations; i++)
            {
                job.stepInput.CurrentSolverIteration = i;
                for (int phase = 0; phase < DispatchPairSequencer.NumIslandPhases; phase++)
                {
                    job.PhaseIndex = phase;
                    int workItems = job.IterativeSolverSchedulerInfo.PhaseInfo[phase].NumWorkItems;
                    for (int item = 0; item < workItems; item++)
                    {
                        var work = job;
                        work.Execute(item);
                    }
                }
            }
        }
    }
}
