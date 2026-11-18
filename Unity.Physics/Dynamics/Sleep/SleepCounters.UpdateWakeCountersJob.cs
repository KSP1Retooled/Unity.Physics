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

namespace Unity.Physics.Dynamics.Sleep
{
    internal static partial class SleepCounters
    {
        /// <summary>
        /// Updates the wake counter for each body as part of <see cref="FallAsleep.SleepJob"/>. It decreases or increases the counter
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
                if (FallAsleep.SleepJob.Solved(body) && (body->flags & BodyFlags.Sleeping) == 0)
                {
                    UpdateWakeCounter(body, TouchesStill[i] ? Contacts[i] : 0);
                    if (!(body->sleep->WakeCounter > 0.0f))
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
                float counter = body->sleep->WakeCounter;
                if (counter < StoreMemory.WakeCounterReset * 0.5f || counter < DeltaTime)
                {
                    body->sleep->LinearSum += body->velocity->Linear;
                    body->sleep->AngularSum += body->velocity->Angular;

                    float cluster = 1.0f + math.min(contacts, MaximumCountedContacts);
                    float threshold = cluster * body->sleep->Threshold;
                    float energy = NormalizedEnergy(body);
                    if (energy >= threshold)
                    {
                        float factor = threshold == 0.0f ? 2.0f : math.min(energy / threshold, 2.0f);
                        body->sleep->WakeCounter = factor * 0.5f * StoreMemory.WakeCounterReset + DeltaTime * (cluster - 1.0f);
                        body->sleep->LinearSum = float3.zero;
                        body->sleep->AngularSum = float3.zero;
                        return;
                    }
                }

                body->sleep->WakeCounter = math.max(counter - DeltaTime, 0.0f);
            }

            private static float NormalizedEnergy(BodyData* body)
            {
                float3 angular = body->sleep->AngularSum;
                float inverseMass = body->mass->InverseMass != 0.0f ? body->mass->InverseMass : 1.0f;
                float rotational = math.dot(angular * angular, body->bodyMass->Inertia) * inverseMass;
                return 0.5f * (rotational + math.lengthsq(body->sleep->LinearSum));
            }
        }
    }
}
