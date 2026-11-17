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
    internal static partial class WarmStart
    {
        /// <summary>
        /// Index previous total impulses and shared contact counts by Jacobian key after this step's manifolds are
        /// known.
        /// </summary>
        [BurstCompile]
        internal struct IndexWarmStartsJob : IJob
        {
            [ReadOnly]
            [NativeDisableContainerSafetyRestriction]
            public NativeList<ManifoldRecord> Records;

            public NativeHashMap<JacobianKey, float2> WarmStarts;

            public void Execute()
            {
                WarmStarts.Clear();
                for (int i = 0; i < Records.Length; i++)
                {
                    ManifoldRecord record = Records[i];
                    if (record.WarmImpulse <= 0.0f)
                    {
                        continue;
                    }

                    JacobianKey key = new JacobianKey
                    {
                        BodyA = record.BuiltPair.BodyIndexA,
                        BodyB = record.BuiltPair.BodyIndexB,
                        Normal = record.BuiltNormal,
                    };

                    WarmStarts.TryGetValue(key, out float2 shared);
                    WarmStarts[key] = shared + new float2(record.WarmImpulse, record.PointCount);
                }
            }
        }
    }
}
