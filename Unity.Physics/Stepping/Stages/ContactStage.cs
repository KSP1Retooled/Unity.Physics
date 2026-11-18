using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Dynamics.Sleep;
using Unity.Physics.Stepping.State;
using Unity.Physics.Store.World;
using Unity.Profiling;

namespace Unity.Physics.Stepping.Stages
{
    /// <summary>
    /// Reads narrow-phase contacts, applies materials, and schedules manifold merge and wake work.
    /// </summary>
    internal static unsafe class ContactStage
    {
        private static readonly ProfilerMarker StepScheduleMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Schedule");

        private const int ParallelBatchSize = 64;

        /// <summary>Reads the contacts produced by the narrow phase before the Jacobians are built from them.</summary>
        internal static JobHandle ReadContacts(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref BodyPartition partition, ref ContactState contactState, ref JobHandle warmStartsIndexed,
            SimulationSingleton singleton, float deltaTime, JobHandle dependency)
        {
            using var span = StepScheduleMarker.Auto();

            // Submit the batched jobs so that workers can run them while the main thread schedules the jobs below.
            JobHandle.ScheduleBatchedJobs();

            contactState.ForgetRemovedImpulses(store);

            contactState.manifolds.Clear();
            contactState.manifoldPoints.Clear();
            JobHandle read = dependency;
            JobHandle merged = dependency;
            NativeStream contacts = singleton.AsSimulationPtr()->StepContext.Contacts;
            if (contacts.IsCreated)
            {
                NativeStream.Reader reader = contacts.AsReader();
                read = new ContactBuckets.PrepareContactBucketsJob
                {
                    Contacts = reader,
                    Buckets = contactState.contactBuckets,
                    Filled = contactState.filledContactBuckets,
                }.Schedule(read);
                read = new ContactMaterials.ReadContactsJob
                {
                    Contacts = reader,
                    Buckets = contactState.contactBuckets,
                    Store = store->bodies,
                    Globals = globals,
                    Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
                    LastImpulses = contactState.lastImpulses,
                    DeltaTime = deltaTime,
                    BounceThreshold = store->bounceThreshold,
                }.ScheduleUnsafe(contacts, 1, read);

                merged = new ContactBuckets.MergeContactBucketsJob
                {
                    Filled = contactState.filledContactBuckets,
                    Buckets = contactState.contactBuckets,
                    Records = contactState.manifolds,
                    Points = contactState.manifoldPoints,
                }.Schedule(read);
            }

            // Only warm starting requires the merged manifolds before solving. Merging the manifolds can therefore run
            // in parallel with building the Jacobians.
            warmStartsIndexed = new WarmStart.IndexWarmStartsJob { Records = contactState.manifolds, WarmStarts = contactState.warmStarts }.Schedule(merged);

            // If there are no contacts, no more bodies need to be woken. Bodies connected by joints were already woken
            // at the start of the step.
            if (!contacts.IsCreated)
            {
                return read;
            }

            return new WakePropagation.WakeTouchedBodiesJob
            {
                World = store,
                Filled = contactState.filledContactBuckets,
                Buckets = contactState.contactBuckets,
                DynamicIndices = partition.dynamicIndices.AsArray(),
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
            }.Schedule(read);
        }
    }
}
