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

namespace Unity.Physics.Dynamics.Contacts
{
    /// <summary>
    /// Identifies a pair of touching shapes in pointer order, so that the key stays the same as bodies are added or
    /// removed.
    /// </summary>
    internal struct ShapePairKey : IEquatable<ShapePairKey>
    {
        public IntPtr First;
        public IntPtr Second;

        internal static unsafe ShapePairKey Of(ShapeData* a, ShapeData* b)
        {
            bool ordered = a < b;
            return new ShapePairKey { First = (IntPtr)(ordered ? a : b), Second = (IntPtr)(ordered ? b : a) };
        }

        public bool Equals(ShapePairKey other)
        {
            return First == other.First && Second == other.Second;
        }

        public override int GetHashCode()
        {
            return First.GetHashCode() * 397 ^ Second.GetHashCode();
        }
    }
}
