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
    /// Identifies a manifold's record from its solver event.
    /// </summary>
    internal struct ManifoldKey : IEquatable<ManifoldKey>
    {
        public int BodyA;
        public int BodyB;
        public uint KeyA;
        public uint KeyB;

        public bool Equals(ManifoldKey other)
        {
            return BodyA == other.BodyA && BodyB == other.BodyB && KeyA == other.KeyA && KeyB == other.KeyB;
        }

        public override int GetHashCode()
        {
            return (int)math.hash(new int4(BodyA, BodyB, (int)KeyA, (int)KeyB));
        }
    }
}
