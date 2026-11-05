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
        private readonly NativeStore.BodyData* store;
        private readonly uint filterOptions;
        private readonly uint engineObjectIdOffset;

        internal QueryCollector(PhysicsWorld world, NativeStore.WorldData* store, uint filterOptions)
        {
            bodies = world.Bodies;
            this.store = store->bodies;
            this.filterOptions = filterOptions;
            engineObjectIdOffset = NativeStore.RetooledPhysics_Globals()->engineObjectIdOffset;
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

            NativeStore.BodyData* body = store + rigidBody.Entity.Index;
            bool isStatic = (body->flags & NativeStore.BodyFlags.Static) != 0;
            uint wanted = isStatic ? NativeStore.QueryUseStaticBodies : NativeStore.QueryUseDynamicBodies;
            if ((filterOptions & wanted) == 0)
            {
                return false;
            }

            NativeStore.ShapeData* shape = ShapeOf(body, rigidBody, hit.ColliderKey);
            if ((shape->flags & NativeStore.ShapeFlagSceneQuery) == 0)
            {
                return false;
            }

            bool trigger = (shape->flags & NativeStore.ShapeFlagTrigger) != 0;
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
            HitEngineObjectId = *(ulong*)((byte*)shape->userData + engineObjectIdOffset);
            MaxFraction = hit.Fraction;
            NumHits = 1;
            return true;
        }

        private static NativeStore.ShapeData* ShapeOf(NativeStore.BodyData* body, RigidBody rigidBody, ColliderKey key)
        {
            if (rigidBody.Collider.Value.Type != ColliderType.Compound)
            {
                for (int i = 0; i < body->shapeCount; i++)
                {
                    if (ColliderBuilder.IsSupported(body->shapes[i]))
                    {
                        return body->shapes[i];
                    }
                }
            }

            rigidBody.Collider.Value.GetLeaf(key, out ChildCollider leaf);
            return body->shapes[leaf.Entity.Index];
        }
    }
}
