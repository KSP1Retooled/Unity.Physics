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
    /// <summary>
    /// Identifies the same manifold across steps, even when body indices in the built world change.
    /// </summary>
    internal unsafe struct PersistentManifoldKey : IEquatable<PersistentManifoldKey>
    {
        public ShapeData* ShapeA;
        public ShapeData* ShapeB;
        public uint KeyA;
        public uint KeyB;

        internal static PersistentManifoldKey Of(ref ManifoldRecord record)
        {
            return new PersistentManifoldKey
            {
                ShapeA = record.ShapeA,
                ShapeB = record.ShapeB,
                KeyA = record.BuiltKeys.ColliderKeyA.Value,
                KeyB = record.BuiltKeys.ColliderKeyB.Value,
            };
        }

        public bool Equals(PersistentManifoldKey other)
        {
            return ShapeA == other.ShapeA && ShapeB == other.ShapeB && KeyA == other.KeyA && KeyB == other.KeyB;
        }

        public override int GetHashCode()
        {
            return ((IntPtr)ShapeA).GetHashCode() * 397 ^ ((IntPtr)ShapeB).GetHashCode() * 31 ^ (int)math.hash(new uint2(KeyA, KeyB));
        }
    }
}
