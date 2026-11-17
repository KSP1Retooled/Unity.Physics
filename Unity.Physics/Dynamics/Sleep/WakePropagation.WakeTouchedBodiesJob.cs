using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Dynamics.Contacts;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Dynamics.Sleep
{
    internal static partial class WakePropagation
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
            public NativeList<ContactBuckets.ContactBucket> Buckets;

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

                BodyKinematics.Wake(target);
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
    }
}
