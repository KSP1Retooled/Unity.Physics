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
    /// </summary>
    [BurstCompile]
    internal unsafe struct WakeTouchedBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldRecord> Records;

        [ReadOnly]
        public NativeArray<int> DynamicIndices;

        [NativeDisableUnsafePtrRestriction]
        public MotionVelocity* Motions;

        public void Execute()
        {
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
            for (int i = 0; i < Records.Length; i++)
            {
                ManifoldRecord record = Records[i];
                if (record.Touching && !record.Trigger)
                {
                    woke |= WakeFrom(World->bodies + record.RowA, World->bodies + record.RowB);
                    woke |= WakeFrom(World->bodies + record.RowB, World->bodies + record.RowA);
                }
            }

            return woke;
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
                    Motions[i].InverseMass = body->inverseMass;
                    Motions[i].InverseInertia = body->inverseInertia;
                }
            }
        }
    }

    /// <summary>
    /// PhysX's sleep rule, on what the step just solved. A body's wake counter runs down while its mass-normalized kinetic
    /// energy stays under its threshold and is topped up when it doesn't, and bodies sleep island by island: a body only
    /// sleeps once every body connected to it by contacts and joints has run out. Statics and kinematics don't connect
    /// bodies into an island, because they don't pass motion from one body to another.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SleepJob : IJob
    {
        private const int MaximumCountedContacts = 10;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeList<ManifoldRecord> Records;

        public float DeltaTime;

        public void Execute()
        {
            int count = (int)World->bodyCount;
            var parents = new NativeArray<int>(count, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var contacts = new NativeArray<int>(count, Allocator.Temp);
            var touchesStill = new NativeArray<bool>(count, Allocator.Temp);
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

                bool solvedA = Solved(World->bodies + record.RowA);
                bool solvedB = Solved(World->bodies + record.RowB);
                contacts[record.RowA] += record.PointCount;
                contacts[record.RowB] += record.PointCount;
                touchesStill[record.RowA] |= !solvedB;
                touchesStill[record.RowB] |= !solvedA;
                if (solvedA && solvedB)
                {
                    Union(parents, record.RowA, record.RowB);
                }
            }

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
                    UpdateWakeCounter(body, touchesStill[i] ? contacts[i] : 0);
                    islandAwake[Find(parents, i)] |= body->wakeCounter > 0.0f;
                }
            }

            for (int i = 0; i < count; i++)
            {
                NativeStore.BodyData* body = World->bodies + i;
                if (Solved(body) && (body->flags & NativeStore.BodyFlags.Sleeping) == 0 && !islandAwake[Find(parents, i)])
                {
                    body->flags |= NativeStore.BodyFlags.Sleeping;
                    body->wakeCounter = 0.0f;
                    body->linearVelocity = float3.zero;
                    body->angularVelocity = float3.zero;
                    body->sleepLinearSum = float3.zero;
                    body->sleepAngularSum = float3.zero;
                }
            }
        }

        private static bool Solved(NativeStore.BodyData* body)
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
            float counter = body->wakeCounter;
            if (counter < StoreOps.WakeCounterReset * 0.5f || counter < DeltaTime)
            {
                quaternion worldFromInertia = math.mul(body->rotation, body->inertiaRotation);
                body->sleepLinearSum += body->linearVelocity;
                body->sleepAngularSum += math.rotate(math.conjugate(worldFromInertia), body->angularVelocity);

                float cluster = 1.0f + math.min(contacts, MaximumCountedContacts);
                float threshold = cluster * body->sleepThreshold;
                float energy = NormalizedEnergy(body);
                if (energy >= threshold)
                {
                    float factor = threshold == 0.0f ? 2.0f : math.min(energy / threshold, 2.0f);
                    body->wakeCounter = factor * 0.5f * StoreOps.WakeCounterReset + DeltaTime * (cluster - 1.0f);
                    body->sleepLinearSum = float3.zero;
                    body->sleepAngularSum = float3.zero;
                    return;
                }
            }

            body->wakeCounter = math.max(counter - DeltaTime, 0.0f);
        }

        private static float NormalizedEnergy(NativeStore.BodyData* body)
        {
            float3 angular = body->sleepAngularSum;
            float inverseMass = body->inverseMass != 0.0f ? body->inverseMass : 1.0f;
            float rotational = math.dot(angular * angular, body->inertia) * inverseMass;
            return 0.5f * (rotational + math.lengthsq(body->sleepLinearSum));
        }

        private static int Find(NativeArray<int> parents, int index)
        {
            while (parents[index] != index)
            {
                parents[index] = parents[parents[index]];
                index = parents[index];
            }

            return index;
        }

        private static void Union(NativeArray<int> parents, int a, int b)
        {
            parents[Find(parents, a)] = Find(parents, b);
        }
    }
}
