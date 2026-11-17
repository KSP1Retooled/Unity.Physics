using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Joint.Translation;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Wake sleeping bodies reached through contacts or joints from awake bodies before solving.
    /// Read contact buckets directly so that Jacobian construction need not wait for merging.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WakeTouchedBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        /// <summary>Counts the buckets filled during this step.</summary>
        [ReadOnly]
        public NativeReference<int> Filled;

        [ReadOnly]
        [NativeDisableContainerSafetyRestriction]
        public NativeList<ContactBucket> Buckets;

        [ReadOnly]
        public NativeArray<int> DynamicIndices;

        [NativeDisableUnsafePtrRestriction]
        public MotionVelocity* Motions;

        public void Execute()
        {
            // Bodies connected by joints were already woken at the start of the step. Here, only a contact can cause
            // another body to start waking.
            bool wakes = false;
            for (int bucket = 0; bucket < Filled.Value; bucket++)
            {
                wakes |= Buckets[bucket].Wakes;
            }

            if (!wakes)
            {
                return;
            }

            bool woke = true;
            bool any = false;
            while (woke)
            {
                woke = WakeThroughContacts();
                woke |= WakeJointedBodiesJob.WakeThroughJoints(World);
                any |= woke;
            }

            if (any)
            {
                ReleaseWokenMotions();
            }
        }

        private bool WakeThroughContacts()
        {
            bool woke = false;
            for (int bucket = 0; bucket < Filled.Value; bucket++)
            {
                UnsafeList<ManifoldRecord> records = Buckets[bucket].Records;
                for (int i = 0; i < records.Length; i++)
                {
                    ManifoldRecord record = records[i];
                    if (record.Touching && !record.Trigger)
                    {
                        woke |= WakeFrom(World->bodies + record.RowA, World->bodies + record.RowB);
                        woke |= WakeFrom(World->bodies + record.RowB, World->bodies + record.RowA);
                    }
                }
            }

            return woke;
        }

        /// <summary>
        /// Returns whether a contact can wake either body, using their states when the contacts were read.
        /// </summary>
        internal static bool Wakes(BodyData* a, BodyData* b)
        {
            bool sleepingA = (a->flags & BodyFlags.Sleeping) != 0;
            bool sleepingB = (b->flags & BodyFlags.Sleeping) != 0;
            return (sleepingB && Moves(a)) || (sleepingA && Moves(b));
        }

        private static bool WakeFrom(BodyData* source, BodyData* target)
        {
            if ((target->flags & BodyFlags.Sleeping) == 0 || !Moves(source))
            {
                return false;
            }

            JointKernel.WakeRow(target);
            return true;
        }

        private static bool Moves(BodyData* body)
        {
            const BodyFlags resting = BodyFlags.Static | BodyFlags.Sleeping;
            if ((body->flags & resting) != 0)
            {
                return false;
            }

            return (body->flags & BodyFlags.Kinematic) == 0 || (body->flags & BodyFlags.HasKinematicTarget) != 0;
        }

        /// <summary>
        /// Restores mass and inertia for bodies that have woken. Sleeping bodies were built with infinite mass. The
        /// solver does not read their motion masses until after this job.
        /// </summary>
        private void ReleaseWokenMotions()
        {
            for (int i = 0; i < DynamicIndices.Length; i++)
            {
                BodyData* body = World->bodies + DynamicIndices[i];
                const BodyFlags held = BodyFlags.Kinematic | BodyFlags.Sleeping;
                if ((body->flags & held) == 0)
                {
                    Motions[i].InverseMass = (*body).Mass()->InverseMass;
                    Motions[i].InverseInertia = (*body).Mass()->InverseInertia;
                }
            }
        }
    }

    /// <summary>
    /// Apply the PhysX island sleep rule after solving: all connected bodies must have zero wake counters.
    /// Static and kinematic bodies do not connect islands. Defer sleeping until contact reporting finishes.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SleepJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        public NativeList<int> Parents;

        public NativeList<int> Falling;

        /// <summary>
        /// Has a non-zero value once a solved body's counter reaches zero during this step. This job clears the value
        /// for the next step.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        public int* RunOut;

        public void Execute()
        {
            Falling.Clear();
            if (*RunOut == 0)
            {
                return;
            }

            *RunOut = 0;
            int count = (int)World->bodyCount;
            NativeArray<int> parents = Parents.AsArray();
            for (uint i = 0; i < World->jointCount; i++)
            {
                JointData* joint = World->joints[i];
                if (joint->body0 != null && joint->body1 != null && (joint->state & JointConstraints.Inactive) == 0
                    && Solved(StoreOps.Hot(joint->body0)) && Solved(StoreOps.Hot(joint->body1)))
                {
                    Union(parents, (int)joint->body0->index, (int)joint->body1->index);
                }
            }

            var islandAwake = new NativeArray<bool>(count, Allocator.Temp);
            for (int i = 0; i < count; i++)
            {
                BodyData* body = World->bodies + i;
                if (Solved(body) && (body->flags & BodyFlags.Sleeping) == 0)
                {
                    islandAwake[Find(parents, i)] |= (*body).Sleep()->WakeCounter > 0.0f;
                }
            }

            for (int i = 0; i < count; i++)
            {
                BodyData* body = World->bodies + i;
                if (Solved(body) && (body->flags & BodyFlags.Sleeping) == 0 && !islandAwake[Find(parents, i)])
                {
                    Falling.Add(i);
                }
            }
        }

        internal static bool Solved(BodyData* body)
        {
            const BodyFlags immobile = BodyFlags.Static | BodyFlags.Kinematic;
            return (body->flags & immobile) == 0 && (body->flags & BodyFlags.Simulated) != 0;
        }

        internal static int Find(NativeArray<int> parents, int index)
        {
            while (parents[index] != index)
            {
                parents[index] = parents[parents[index]];
                index = parents[index];
            }

            return index;
        }

        internal static void Union(NativeArray<int> parents, int a, int b)
        {
            parents[Find(parents, a)] = Find(parents, b);
        }
    }

    /// <summary>
    /// Updates the wake counter for each body as part of <see cref="SleepJob"/>. It decreases or increases the counter
    /// and records whether any counter reaches zero. An island can fall asleep only if a counter has reached zero.
    /// </summary>
    [BurstCompile]
    internal unsafe struct UpdateWakeCountersJob : IJobParallelFor
    {
        private const int MaximumCountedContacts = 10;

        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        [ReadOnly]
        public NativeList<int> Contacts;

        [ReadOnly]
        public NativeList<bool> TouchesStill;

        [NativeDisableUnsafePtrRestriction]
        public int* RunOut;

        public float DeltaTime;

        public void Execute(int i)
        {
            BodyData* body = World->bodies + i;
            if (SleepJob.Solved(body) && (body->flags & BodyFlags.Sleeping) == 0)
            {
                UpdateWakeCounter(body, TouchesStill[i] ? Contacts[i] : 0);
                if (!((*body).Sleep()->WakeCounter > 0.0f))
                {
                    *RunOut = 1;
                }
            }
        }

        /// <summary>
        /// Update the wake counter using velocities accumulated across steps, so that jitter cancels but drift
        /// keeps the body awake.
        /// Delay energy checks after wakeup and raise the sleep threshold for contacts with immovable bodies,
        /// matching PhysX.
        /// </summary>
        private void UpdateWakeCounter(BodyData* body, int contacts)
        {
            float counter = (*body).Sleep()->WakeCounter;
            if (counter < StoreMemory.WakeCounterReset * 0.5f || counter < DeltaTime)
            {
                (*body).Sleep()->LinearSum += (*body).Velocity()->Linear;
                (*body).Sleep()->AngularSum += (*body).Velocity()->Angular;

                float cluster = 1.0f + math.min(contacts, MaximumCountedContacts);
                float threshold = cluster * (*body).Sleep()->Threshold;
                float energy = NormalizedEnergy(body);
                if (energy >= threshold)
                {
                    float factor = threshold == 0.0f ? 2.0f : math.min(energy / threshold, 2.0f);
                    (*body).Sleep()->WakeCounter = factor * 0.5f * StoreMemory.WakeCounterReset + DeltaTime * (cluster - 1.0f);
                    (*body).Sleep()->LinearSum = float3.zero;
                    (*body).Sleep()->AngularSum = float3.zero;
                    return;
                }
            }

            (*body).Sleep()->WakeCounter = math.max(counter - DeltaTime, 0.0f);
        }

        private static float NormalizedEnergy(BodyData* body)
        {
            float3 angular = (*body).Sleep()->AngularSum;
            float inverseMass = (*body).Mass()->InverseMass != 0.0f ? (*body).Mass()->InverseMass : 1.0f;
            float rotational = math.dot(angular * angular, (*body).BodyMass()->Inertia) * inverseMass;
            return 0.5f * (rotational + math.lengthsq((*body).Sleep()->LinearSum));
        }
    }

    /// <summary>
    /// Processes contacts for <see cref="SleepJob"/>. It connects solved bodies that touch and counts each body's
    /// contact points. It also records whether each body touches a body that is not solved, including a static or
    /// kinematic body.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SleepContactsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        /// <summary>
        /// This list is read while <see cref="ReadImpulsesJob"/> runs. That job writes only the impulse fields, which
        /// this job does not read.
        /// </summary>
        [ReadOnly]
        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldRecord> Records;

        public NativeList<int> Parents;

        public NativeList<int> Contacts;

        public NativeList<bool> TouchesStill;

        public void Execute()
        {
            int count = (int)World->bodyCount;
            Parents.ResizeUninitialized(count);
            Contacts.ResizeUninitialized(count);
            TouchesStill.ResizeUninitialized(count);
            UnsafeUtility.MemClear(Contacts.GetUnsafePtr(), (long)count * sizeof(int));
            UnsafeUtility.MemClear(TouchesStill.GetUnsafePtr(), count);
            NativeArray<int> parents = Parents.AsArray();
            for (int i = 0; i < count; i++)
            {
                parents[i] = i;
            }

            for (int i = 0; i < Records.Length; i++)
            {
                ManifoldRecord record = Records[i];
                if (!record.Touching || record.Trigger)
                {
                    continue;
                }

                bool solvedA = SleepJob.Solved(World->bodies + record.RowA);
                bool solvedB = SleepJob.Solved(World->bodies + record.RowB);
                Contacts[record.RowA] += record.PointCount;
                Contacts[record.RowB] += record.PointCount;
                TouchesStill[record.RowA] |= !solvedB;
                TouchesStill[record.RowB] |= !solvedA;
                if (solvedA && solvedB)
                {
                    SleepJob.Union(parents, record.RowA, record.RowB);
                }
            }
        }
    }

    /// <summary>
    /// Puts the bodies listed by <see cref="SleepJob"/> to sleep and stops them at their current positions.
    /// </summary>
    [BurstCompile]
    internal unsafe struct FallAsleepJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        [ReadOnly]
        public NativeList<int> Falling;

        public void Execute()
        {
            for (int i = 0; i < Falling.Length; i++)
            {
                BodyData* body = World->bodies + Falling[i];
                body->flags |= BodyFlags.Sleeping;
                (*body).Sleep()->WakeCounter = 0.0f;
                (*body).Velocity()->Linear = float3.zero;
                (*body).Velocity()->Angular = float3.zero;
                (*body).Sleep()->LinearSum = float3.zero;
                (*body).Sleep()->AngularSum = float3.zero;
            }
        }
    }
}
