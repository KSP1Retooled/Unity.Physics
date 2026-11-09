using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Wakes the sleeping bodies a step's contacts reach, before the solver sees them, so that a sleeping body is pushed by
    /// the step that hits it rather than stopping what hit it like a wall. Waking spreads through touching bodies and joints,
    /// as it does through a PhysX island, and a sleeping body's contact with something at rest wakes nothing.
    ///
    /// It reads the manifolds from <see cref="ReadContactsJob"/>'s buckets rather than waiting for them to be merged, since
    /// the Jacobians wait for this job and nothing else on their way needs the merged list.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WakeTouchedBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        /// <summary>How many of the buckets this step filled.</summary>
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
            // The joints spread waking as the step began, so only a contact can start any here.
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

        /// <summary>Whether a contact between two bodies wakes either, as the contacts stood when they were read.</summary>
        internal static bool Wakes(NativeStore.BodyData* a, NativeStore.BodyData* b)
        {
            bool sleepingA = (a->flags & NativeStore.BodyFlags.Sleeping) != 0;
            bool sleepingB = (b->flags & NativeStore.BodyFlags.Sleeping) != 0;
            return (sleepingB && Moves(a)) || (sleepingA && Moves(b));
        }

        private static bool WakeFrom(NativeStore.BodyData* source, NativeStore.BodyData* target)
        {
            if ((target->flags & NativeStore.BodyFlags.Sleeping) == 0 || !Moves(source))
            {
                return false;
            }

            JointKernel.WakeRow(target);
            return true;
        }

        private static bool Moves(NativeStore.BodyData* body)
        {
            const NativeStore.BodyFlags resting = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Sleeping;
            if ((body->flags & resting) != 0)
            {
                return false;
            }

            return (body->flags & NativeStore.BodyFlags.Kinematic) == 0 || (body->flags & NativeStore.BodyFlags.HasKinematicTarget) != 0;
        }

        /// <summary>A body built asleep has infinite mass, and the solver reads a motion's mass only after this job.</summary>
        private void ReleaseWokenMotions()
        {
            for (int i = 0; i < DynamicIndices.Length; i++)
            {
                NativeStore.BodyData* body = World->bodies + DynamicIndices[i];
                const NativeStore.BodyFlags held = NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Sleeping;
                if ((body->flags & held) == 0)
                {
                    Motions[i].InverseMass = body->mass->InverseMass;
                    Motions[i].InverseInertia = body->mass->InverseInertia;
                }
            }
        }
    }

    /// <summary>
    /// PhysX's sleep rule, on what the step just solved. A body's wake counter runs down while its mass-normalized kinetic
    /// energy stays under its threshold and is topped up when it doesn't, and bodies sleep island by island: a body only
    /// sleeps once every body connected to it by contacts and joints has run out. Statics and kinematics don't connect
    /// bodies into an island, because they don't pass motion from one body to another.
    ///
    /// <see cref="SleepContactsJob"/> joins the bodies through their contacts as soon as the solve ends, because the contact
    /// report sorts the manifolds later. This joins them through their joints once the step has broken the ones it breaks,
    /// and only lists the bodies that fall asleep, for <see cref="FallAsleepJob"/>, because the contact report still has to
    /// see them awake.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SleepJob : IJob
    {
        private const int MaximumCountedContacts = 10;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        public NativeList<int> Parents;

        [ReadOnly]
        public NativeList<int> Contacts;

        [ReadOnly]
        public NativeList<bool> TouchesStill;

        public NativeList<int> Falling;

        public float DeltaTime;

        public void Execute()
        {
            Falling.Clear();
            int count = (int)World->bodyCount;
            NativeArray<int> parents = Parents.AsArray();
            for (uint i = 0; i < World->jointCount; i++)
            {
                NativeStore.JointData* joint = World->joints[i];
                if (joint->body0 != null && joint->body1 != null && (joint->state & JointConstraints.Inactive) == 0
                    && Solved(StoreOps.Hot(joint->body0)) && Solved(StoreOps.Hot(joint->body1)))
                {
                    Union(parents, (int)joint->body0->index, (int)joint->body1->index);
                }
            }

            var islandAwake = new NativeArray<bool>(count, Allocator.Temp);
            for (int i = 0; i < count; i++)
            {
                NativeStore.BodyData* body = World->bodies + i;
                if (Solved(body) && (body->flags & NativeStore.BodyFlags.Sleeping) == 0)
                {
                    UpdateWakeCounter(body, TouchesStill[i] ? Contacts[i] : 0);
                    islandAwake[Find(parents, i)] |= body->sleep->WakeCounter > 0.0f;
                }
            }

            for (int i = 0; i < count; i++)
            {
                NativeStore.BodyData* body = World->bodies + i;
                if (Solved(body) && (body->flags & NativeStore.BodyFlags.Sleeping) == 0 && !islandAwake[Find(parents, i)])
                {
                    Falling.Add(i);
                }
            }
        }

        internal static bool Solved(NativeStore.BodyData* body)
        {
            const NativeStore.BodyFlags immobile = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic;
            return (body->flags & immobile) == 0 && (body->flags & NativeStore.BodyFlags.Simulated) != 0;
        }

        /// <summary>
        /// The counter runs free for its first half, so a body caught at the top of a bounce isn't judged. After that it sums
        /// velocities rather than taking the step's own, which lets jitter around nothing cancel out while a slow steady drift
        /// adds up and keeps the body awake however small it is. PhysX raises the threshold with the contacts of a body touching
        /// something still, which lets a pile settle.
        /// </summary>
        private void UpdateWakeCounter(NativeStore.BodyData* body, int contacts)
        {
            float counter = body->sleep->WakeCounter;
            if (counter < StoreOps.WakeCounterReset * 0.5f || counter < DeltaTime)
            {
                body->sleep->LinearSum += body->velocity->Linear;
                body->sleep->AngularSum += body->velocity->Angular;

                float cluster = 1.0f + math.min(contacts, MaximumCountedContacts);
                float threshold = cluster * body->sleep->Threshold;
                float energy = NormalizedEnergy(body);
                if (energy >= threshold)
                {
                    float factor = threshold == 0.0f ? 2.0f : math.min(energy / threshold, 2.0f);
                    body->sleep->WakeCounter = factor * 0.5f * StoreOps.WakeCounterReset + DeltaTime * (cluster - 1.0f);
                    body->sleep->LinearSum = float3.zero;
                    body->sleep->AngularSum = float3.zero;
                    return;
                }
            }

            body->sleep->WakeCounter = math.max(counter - DeltaTime, 0.0f);
        }

        private static float NormalizedEnergy(NativeStore.BodyData* body)
        {
            float3 angular = body->sleep->AngularSum;
            float inverseMass = body->mass->InverseMass != 0.0f ? body->mass->InverseMass : 1.0f;
            float rotational = math.dot(angular * angular, body->bodyMass->Inertia) * inverseMass;
            return 0.5f * (rotational + math.lengthsq(body->sleep->LinearSum));
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
    /// The contacts' part of <see cref="SleepJob"/>: joins the solved bodies that touch, and counts each body's contact
    /// points and whether it touches something that doesn't move.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SleepContactsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        /// <summary>
        /// Read beside <see cref="ReadImpulsesJob"/>, which writes only the impulses, a field this job doesn't read.
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

    /// <summary>Puts to sleep the bodies <see cref="SleepJob"/> listed, stopping them where they are.</summary>
    [BurstCompile]
    internal unsafe struct FallAsleepJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeList<int> Falling;

        public void Execute()
        {
            for (int i = 0; i < Falling.Length; i++)
            {
                NativeStore.BodyData* body = World->bodies + Falling[i];
                body->flags |= NativeStore.BodyFlags.Sleeping;
                body->sleep->WakeCounter = 0.0f;
                body->velocity->Linear = float3.zero;
                body->velocity->Angular = float3.zero;
                body->sleep->LinearSum = float3.zero;
                body->sleep->AngularSum = float3.zero;
            }
        }
    }
}
