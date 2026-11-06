using Unity.Collections;
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

        internal QueryCollector(QueryWorld* world, uint filterOptions)
        {
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

            QueryShape* shape = queryShapes + body->FirstShape + ShapeIndexOf(body, rigidBody, hit.ColliderKey);
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

            Hit = hit;
            HitEngineObjectId = shape->EngineObjectId;
            MaxFraction = hit.Fraction;
            NumHits = 1;
            return true;
        }

        private static int ShapeIndexOf(QueryBody* body, RigidBody rigidBody, ColliderKey key)
        {
            if (rigidBody.Collider.Value.Type != ColliderType.Compound)
            {
                return body->SingleShape;
            }

            rigidBody.Collider.Value.GetLeaf(key, out ChildCollider leaf);
            return leaf.Entity.Index;
        }
    }
}
