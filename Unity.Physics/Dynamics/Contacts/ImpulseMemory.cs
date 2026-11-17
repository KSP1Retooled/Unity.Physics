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
        /// Stores each manifold's impulse for the next step's warm start.
        /// </summary>
        [BurstCompile]
        internal struct RememberImpulsesJob : IJob
        {
            // A heavy body on a light one can oscillate for many steps. Following each oscillation in the warm start would
            // reinforce it.
            private const float Smoothing = 0.25f;

            [ReadOnly]
            public NativeList<ManifoldRecord> Records;

            public NativeHashMap<PersistentManifoldKey, float> LastImpulses;

            public void Execute()
            {
                LastImpulses.Clear();
                for (int i = 0; i < Records.Length; i++)
                {
                    ManifoldRecord record = Records[i];
                    if (!record.Trigger && record.Impulse > 0.0f)
                    {
                        float last = record.WarmImpulse > 0.0f ? record.WarmImpulse : record.Impulse;
                        LastImpulses[PersistentManifoldKey.Of(ref record)] = math.lerp(last, record.Impulse, Smoothing);
                    }
                }
            }
        }

        /// <summary>
        /// Discard cached impulses for detached shapes before their addresses can be reused.
        /// Leave the removal list for TouchingPairs.ForgetRemovedShapesJob.
        /// </summary>
        [BurstCompile]
        internal unsafe struct ForgetRemovedImpulsesJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            public NativeHashMap<PersistentManifoldKey, float> LastImpulses;

            public void Execute()
            {
                var removed = new NativeHashSet<IntPtr>((int)World->removedShapeCount, Allocator.Temp);
                for (uint i = 0; i < World->removedShapeCount; i++)
                {
                    removed.Add((IntPtr)World->removedShapes[i]);
                }

                NativeArray<PersistentManifoldKey> keys = LastImpulses.GetKeyArray(Allocator.Temp);
                foreach (PersistentManifoldKey key in keys)
                {
                    if (removed.Contains((IntPtr)key.ShapeA) || removed.Contains((IntPtr)key.ShapeB))
                    {
                        LastImpulses.Remove(key);
                    }
                }
            }
        }
    }
}
