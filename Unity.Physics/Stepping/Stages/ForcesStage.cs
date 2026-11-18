using Unity.Jobs;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Dynamics.Sleep;
using Unity.Physics.Dynamics.Vehicles;
using Unity.Physics.Stepping.State;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using Unity.Profiling;

namespace Unity.Physics.Stepping.Stages
{
    /// <summary>
    /// Waits for earlier work, invalidates queries, and schedules wake propagation and force integration.
    /// </summary>
    internal static unsafe class ForcesStage
    {
        private static readonly ProfilerMarker StepApplyForcesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ApplyForces");

        private const int ParallelBatchSize = 64;

        /// <summary>
        /// Complete outstanding work and invalidate queries before the caller prepares body pointers for jobs.
        /// </summary>
        internal static void BeginForces(WorldData* store, ref TreeState trees, ref JobHandle forcesApplied)
        {
            trees.FinishPendingRefit(store);
            StoreStepper.ThrowBeforeApplyForcesIfRequested();

            // A step that threw after this stage last time can have left the job running.
            forcesApplied.Complete();
            store->queryWorld = null;
            store->queryDirty = StoreMemory.QueryStale;
        }

        /// <summary>
        /// Wake bodies through joints, then schedule force integration.
        /// BuildWorld waits for the force jobs. The main thread may read flags while they run.
        /// </summary>
        internal static void ApplyForces(WorldData* store, ref ContactState contactState,
            ref JobHandle forcesApplied, float deltaTime)
        {
            WheelSolver.RecordStartVelocities(store);

            using (StepApplyForcesMarker.Auto())
            {
                contactState.WakeRefilteredPairs(store);

                new WakePropagation.WakeJointedBodiesJob { World = store }.Run();

                forcesApplied = new ExternalForces.ApplyForcesJob
                {
                    Bodies = store->bodies,
                    Gravity = store->gravity,
                    DeltaTime = deltaTime,
                }.Schedule((int)store->bodyCount, ParallelBatchSize);
                JobHandle.ScheduleBatchedJobs();
            }
        }
    }
}
