using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Keeps the closest ray hit that passes what a collision filter can't express: which kinds of shape the query
    /// wants, and whether the shape takes part in queries at all.
    /// </summary>
    internal unsafe struct QueryCollector : ICollector<RaycastHit>
    {
        // Physics.Raycast defaults to an infinite distance, which Unity Physics can't represent.
        internal const float MaxRayLength = 1e7f;

        private readonly NativeArray<RigidBody> bodies;
        private readonly QueryBody* queryBodies;
        private readonly QueryShape* queryShapes;
        private readonly uint filterOptions;
        private readonly float3 rayDirection;

        internal QueryCollector(QueryWorld* world, uint filterOptions, float3 rayDirection)
        {
            this.rayDirection = rayDirection;
            bodies = world->Collision.Bodies;
            queryBodies = world->Bodies;
            queryShapes = world->Shapes;
            this.filterOptions = filterOptions;
            MaxFraction = 1.0f;
            NumHits = 0;
            Hit = default;
            HitEngineObjectId = 0;
        }

        public bool EarlyOutOnFirstHit => false;

        public float MaxFraction { get; private set; }

        public int NumHits { get; private set; }

        internal RaycastHit Hit { get; private set; }

        internal ulong HitEngineObjectId { get; private set; }

        public bool AddHit(RaycastHit hit)
        {
            RigidBody rigidBody = bodies[hit.RigidBodyIndex];
            if (rigidBody.Entity.Version == 0)
            {
                return false;
            }

            QueryBody* body = queryBodies + hit.RigidBodyIndex;
            uint wanted = body->IsStatic != 0 ? NativeStore.QueryUseStaticBodies : NativeStore.QueryUseDynamicBodies;
            if ((filterOptions & wanted) == 0)
            {
                return false;
            }

            QueryShape* shape = queryShapes + body->FirstShape + ColliderBuilder.ShapeIndexOf(rigidBody.Collider, hit.ColliderKey, body->SingleShape);
            if ((shape->Flags & NativeStore.ShapeFlagSceneQuery) == 0)
            {
                return false;
            }

            bool trigger = (shape->Flags & NativeStore.ShapeFlagTrigger) != 0;
            if (trigger && (filterOptions & NativeStore.QueryUseTriggerShapes) == 0)
            {
                return false;
            }

            // A ray that starts inside a shape doesn't hit it under PhysX.
            if (hit.Fraction <= 0.0f)
            {
                return false;
            }

            if ((filterOptions & NativeStore.QueryAllowBackfaces) == 0 && IsBackface(rigidBody, hit.ColliderKey))
            {
                return false;
            }

            Hit = hit;
            HitEngineObjectId = shape->EngineObjectId;
            MaxFraction = hit.Fraction;
            NumHits = 1;
            return true;
        }

        /// <summary>
        /// Whether the ray met a triangle mesh from behind. Unity Physics hits polygons from both sides and turns the hit's
        /// normal to face the ray, so only the polygon's own winding can say, and a hit refused here lets the ray go on.
        /// </summary>
        private bool IsBackface(RigidBody rigidBody, ColliderKey key)
        {
            Collider* root = (Collider*)rigidBody.Collider.GetUnsafePtr();
            if (!Collider.GetLeafCollider(out ChildCollider leaf, root, key, rigidBody.WorldFromBody))
            {
                return false;
            }

            Collider* collider = leaf.Collider;
            if (collider->Type != ColliderType.Triangle && collider->Type != ColliderType.Quad)
            {
                return false;
            }

            BlobArray.Accessor<float3> vertices = ((PolygonCollider*)collider)->Vertices;
            float3 normal = math.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
            return math.dot(math.rotate(leaf.TransformFromChild.rot, normal), rayDirection) > 0.0f;
        }
    }
}
