using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Wakes sleeping bodies reached through contacts during this step, before the solver runs. This lets a body
    /// respond to an impact during the same step instead of acting like a wall that stops the body that hit it. Bodies
    /// wake through chains of contacts and joints, as they do within a PhysX island. Contact with a body at rest does
    /// not wake a sleeping body.
    /// <para>
    /// This job reads manifolds directly from the buckets produced by <see cref="ReadContactsJob"/> instead of waiting
    /// for them to be merged. Jacobian construction waits for this job, and no other dependency before that
    /// construction needs the merged list.
    /// </para>
    /// </summary>
    [BurstCompile]
    internal unsafe struct WakeTouchedBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

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

        /// <summary>
        /// Restores mass and inertia for bodies that have woken. Sleeping bodies were built with infinite mass. The
        /// solver does not read their motion masses until after this job.
        /// </summary>
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
    /// Applies the PhysX sleep rule to the bodies after solving. A body's wake counter decreases while its kinetic
    /// energy, normalized by mass, stays below the threshold. The counter is increased when the energy reaches or
    /// exceeds the threshold. Bodies fall asleep as a whole island: every body connected through contacts and joints
    /// must have a counter of zero. Static and kinematic bodies do not connect dynamic bodies into an island, because
    /// they do not transfer motion between those bodies.
    /// <para>
    /// <see cref="SleepContactsJob"/> connects bodies through contacts as soon as solving ends, before the contact
    /// report sorts the manifolds. This job then connects them through joints, after joints that broke during the step
    /// have been processed. It lists the bodies that should fall asleep for <see cref="FallAsleepJob"/> to process
    /// later. They must remain awake until the contact report has read them. If every counter left by <see
    /// cref="UpdateWakeCountersJob"/> is above zero, every island stays awake, so there is no need to connect bodies
    /// into islands.
    /// </para>
    /// </summary>
    [BurstCompile]
    internal unsafe struct SleepJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

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
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeList<int> Contacts;

        [ReadOnly]
        public NativeList<bool> TouchesStill;

        [NativeDisableUnsafePtrRestriction]
        public int* RunOut;

        public float DeltaTime;

        public void Execute(int i)
        {
            NativeStore.BodyData* body = World->bodies + i;
            if (SleepJob.Solved(body) && (body->flags & NativeStore.BodyFlags.Sleeping) == 0)
            {
                UpdateWakeCounter(body, TouchesStill[i] ? Contacts[i] : 0);
                if (!(body->sleep->WakeCounter > 0.0f))
                {
                    *RunOut = 1;
                }
            }
        }

        /// <summary>
        /// The counter decreases without an energy check until it drops below half its reset value or below the
        /// duration of one step. This avoids testing energy while a body is briefly still at the top of a bounce. After
        /// that, energy is calculated from velocities summed across steps, rather than from the current step alone.
        /// Jitter around zero cancels out, while a steady drift accumulates and keeps the body awake, however slow the
        /// drift is. As in PhysX, the threshold increases with the contact count when the body touches a body that is
        /// not solved. This helps piles settle. Static and kinematic bodies, including moving kinematic bodies, are not
        /// solved.
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
        public NativeStore.WorldData* World;

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
