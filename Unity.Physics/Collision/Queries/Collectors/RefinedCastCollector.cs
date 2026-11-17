using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Queries.Snapshot;

namespace Unity.Physics.Collision.Queries.Collectors
{
    /// <summary>
    /// Refine cast hits that Unity Physics reports before reaching contact, rejecting casts that pass the shape.
    /// Keep the latest hit after at most four additional passes, even if contact is still unconfirmed.
    /// </summary>
    internal unsafe struct RefinedCastCollector<C> : ICollector<ColliderCastHit>
        where C : struct, ICollector<ColliderCastHit>
    {
        // Each additional cast allows up to ten more advances.
        private const int MaxPasses = 4;

        internal C Inner;

        private readonly NativeArray<RigidBody> bodies;
        private readonly Collider* query;
        private readonly quaternion orientation;
        private readonly float3 start;
        private readonly float3 end;

        internal RefinedCastCollector(QueryWorld* world, C inner, ColliderCastInput input)
        {
            Inner = inner;
            bodies = world->Collision.Bodies;
            query = input.Collider;
            orientation = input.Orientation;
            start = input.Start;
            end = input.End;
        }

        public bool EarlyOutOnFirstHit => Inner.EarlyOutOnFirstHit;

        public float MaxFraction => Inner.MaxFraction;

        public int NumHits => Inner.NumHits;

        public bool AddHit(ColliderCastHit hit)
        {
            if (hit.Fraction > 0.0f && !Refine(ref hit))
            {
                return false;
            }

            return Inner.AddHit(hit);
        }

        private bool Refine(ref ColliderCastHit hit)
        {
            RigidBody body = bodies[hit.RigidBodyIndex];
            if (!Collider.GetLeafCollider(out ChildCollider leaf, (Collider*)body.Collider.GetUnsafePtr(), hit.ColliderKey, body.WorldFromBody))
            {
                return true;
            }

            RigidTransform worldFromLeaf = leaf.TransformFromChild;
            RigidTransform leafFromWorld = math.inverse(worldFromLeaf);
            float3 from = math.transform(leafFromWorld, start);
            float3 to = math.transform(leafFromWorld, end);
            quaternion rotation = math.mul(leafFromWorld.rot, orientation);
            float fraction = hit.Fraction;
            for (int pass = 0; pass < MaxPasses; pass++)
            {
                ColliderCastInput onward = new ColliderCastInput
                {
                    Collider = query,
                    Orientation = rotation,
                    Start = math.lerp(from, to, fraction),
                    End = to,
                };

                var closest = new ClosestHitCollector<ColliderCastHit>(1.0f);
                if (!leaf.Collider->CastCollider(onward, ref closest))
                {
                    return false;
                }

                ColliderCastHit found = closest.ClosestHit;
                hit.Position = math.transform(worldFromLeaf, found.Position);
                hit.SurfaceNormal = math.rotate(worldFromLeaf.rot, found.SurfaceNormal);
                if (found.Fraction <= 0.0f)
                {
                    break;
                }

                fraction += (1.0f - fraction) * found.Fraction;
                hit.Fraction = fraction;
            }

            return true;
        }
    }
}
