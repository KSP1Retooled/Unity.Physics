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
    /// Identify a manifold by body pair and normal when the Jacobian has no collider keys.
    /// Coplanar manifolds with the same key share their impulse.
    /// </summary>
    internal struct JacobianKey : IEquatable<JacobianKey>
    {
        public int BodyA;
        public int BodyB;
        public float3 Normal;

        public bool Equals(JacobianKey other)
        {
            return BodyA == other.BodyA && BodyB == other.BodyB && Normal.Equals(other.Normal);
        }

        public override int GetHashCode()
        {
            return (int)math.hash(new int2(BodyA, BodyB)) ^ Normal.GetHashCode();
        }
    }
}
