using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Reports;
using StoreContactPoint = Unity.Physics.Store.Reports.ContactPoint;
using TriggerEvent = Unity.Physics.Store.Reports.TriggerEvent;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Collectors;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Vessels.Detection;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Dynamics.Contacts
{
    internal static partial class ImpulseMemory
    {
        /// <summary>
        /// Records the solver impulse for each manifold. A step with no dynamic bodies has no solver execution or event
        /// stream, so this job does not run for such a step.
        /// </summary>
        [BurstCompile]
        internal struct ReadImpulsesJob : IJob
        {
            public NativeList<ManifoldRecord> Records;

            public CollisionEvents Events;

            // The job reads only the motions. The safety system would otherwise stop it while the dynamic tree is rebuilt,
            // which writes the built bodies.
            [ReadOnly]
            [NativeDisableContainerSafetyRestriction]
            public PhysicsWorld Built;

            public void Execute()
            {
                NativeHashMap<ManifoldKey, int> byKey = new NativeHashMap<ManifoldKey, int>(Records.Length, Allocator.Temp);
                for (int i = 0; i < Records.Length; i++)
                {
                    ref ManifoldRecord record = ref Records.ElementAt(i);
                    byKey.TryAdd(KeyOf(record.BuiltPair, record.BuiltKeys), i);
                    if (record.ArticulationImpulse != 0f)
                    {
                        record.Impulse = record.ArticulationImpulse;
                    }
                }

                foreach (CollisionEvent collision in Events)
                {
                    BodyIndexPair pair = new BodyIndexPair { BodyIndexA = collision.BodyIndexA, BodyIndexB = collision.BodyIndexB };
                    ColliderKeyPair keys = new ColliderKeyPair { ColliderKeyA = collision.ColliderKeyA, ColliderKeyB = collision.ColliderKeyB };
                    if (byKey.TryGetValue(KeyOf(pair, keys), out int index))
                    {
                        float impulse = collision.CalculateDetails(ref Built).EstimatedImpulse;
                        ref ManifoldRecord record = ref Records.ElementAt(index);
                        record.Impulse = math.isfinite(impulse) ? impulse + WarmStart.WarmStartJob.Fraction * record.WarmImpulse : 0.0f;
                        if (record.ArticulationImpulse != 0f)
                        {
                            record.Impulse += record.ArticulationImpulse;
                        }
                    }
                }
            }

            private static ManifoldKey KeyOf(BodyIndexPair pair, ColliderKeyPair keys)
            {
                return new ManifoldKey
                {
                    BodyA = pair.BodyIndexA,
                    BodyB = pair.BodyIndexB,
                    KeyA = keys.ColliderKeyA.Value,
                    KeyB = keys.ColliderKeyB.Value,
                };
            }
        }
    }
}
