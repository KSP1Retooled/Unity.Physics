using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Physics.Math;
using PhysicsJoint = Unity.Physics.Joint;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Vehicles;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Collectors;

namespace Unity.Physics.Dynamics.Vehicles
{
    /// <summary>
    /// Collect the nearest supporting wheel hit, excluding the wheel body, triggers, hidden shapes, ignored
    /// layers/pairs/groups and backfaces.
    /// </summary>
    internal unsafe struct WheelRayCollector : ICollector<RaycastHit>
    {
        internal RaycastHit Closest;
        internal ShapeData* Shape;

        private readonly NativeArray<RigidBody> bodies;
        private readonly BodyData* rows;
        private readonly StoreGlobals* globals;
        private readonly int ownRow;
        private readonly uint collidesWith;
        private readonly uint belongsTo;
        private readonly uint ignoreId;
        private readonly uint groupRecord;
        private readonly float3 direction;
        private uint closestSerial;

        internal WheelRayCollector(NativeArray<RigidBody> bodies, BodyData* rows, StoreGlobals* globals, int ownRow,
            uint collidesWith, uint belongsTo, uint ignoreId, uint groupRecord, float3 direction)
        {
            this.bodies = bodies;
            this.rows = rows;
            this.globals = globals;
            this.ownRow = ownRow;
            this.collidesWith = collidesWith;
            this.belongsTo = belongsTo;
            this.ignoreId = ignoreId;
            this.groupRecord = groupRecord;
            this.direction = direction;
            Closest = default;
            Shape = null;
            closestSerial = 0;
            MaxFraction = 1.0f;
            NumHits = 0;
        }

        public bool EarlyOutOnFirstHit => false;

        public float MaxFraction { get; private set; }

        public int NumHits { get; private set; }

        public bool AddHit(RaycastHit hit)
        {
            RigidBody body = bodies[hit.RigidBodyIndex];
            if (body.Entity.Version == 0 || body.Entity.Index == ownRow)
            {
                return false;
            }

            BodyData* row = rows + body.Entity.Index;
            ShapeData* shape = row->shapes[ColliderBuilder.ShapeIndexOf(body.Collider, hit.ColliderKey, ColliderBuilder.FirstSupported(row))];
            bool hidden = (shape->flags & StoreConstants.ShapeFlagSceneQuery) == 0 || (shape->flags & StoreConstants.ShapeFlagTrigger) != 0;
            if (hidden || (shape->belongsTo & collidesWith) == 0 || shape->collisionDisabled != 0)
            {
                return false;
            }

            // Apply the layer matrix from the shape filter; Unity gives wheel rays an all-layer filter.
            if (belongsTo != 0 && shape->collidesWith != 0 && (shape->collidesWith & belongsTo) == 0)
            {
                return false;
            }

            if (IgnoreTable.Contains(globals, ignoreId, shape->ignorePairId) || GroupTable.Ignores(globals, groupRecord, shape))
            {
                return false;
            }

            if (QueryCollectors.IsBackface((Collider*)body.Collider.GetUnsafePtr(), hit.ColliderKey, body.WorldFromBody, direction))
            {
                return false;
            }

            // Break equal-distance ties by body and collider key, independent of tree traversal order.
            uint serial = (uint)body.Entity.Version;
            if (Shape != null && !QueryCollectors.Precedes(hit.Fraction, serial, hit.ColliderKey, Closest.Fraction, closestSerial, Closest.ColliderKey))
            {
                return false;
            }

            Closest = hit;
            Shape = shape;
            closestSerial = serial;
            MaxFraction = QueryCollectors.JustAbove(hit.Fraction);
            NumHits = 1;
            return true;
        }
    }
}
