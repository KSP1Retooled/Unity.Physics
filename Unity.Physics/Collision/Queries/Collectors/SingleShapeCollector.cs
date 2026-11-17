using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;

namespace Unity.Physics.Collision.Queries.Collectors
{
    /// <summary>
    /// Collect ray or cast hits against one shape. Rays starting inside miss; casts report initial overlaps only
    /// when requested.
    /// </summary>
    internal unsafe struct SingleShapeCollector<T> : ICollector<T>
        where T : unmanaged, IQueryResult
    {
        internal T Hit;
        internal bool Backface;

        private readonly Collider* root;
        private readonly uint filterOptions;
        private readonly float3 direction;
        private readonly bool keepOverlaps;

        internal SingleShapeCollector(Collider* root, uint filterOptions, float3 direction, bool keepOverlaps = false)
        {
            this.root = root;
            this.filterOptions = filterOptions;
            this.direction = direction;
            this.keepOverlaps = keepOverlaps;
            Hit = default;
            Backface = false;
            MaxFraction = 1.0f;
            NumHits = 0;
        }

        public bool EarlyOutOnFirstHit => false;

        public float MaxFraction { get; private set; }

        public int NumHits { get; private set; }

        public bool AddHit(T hit)
        {
            if (hit.Fraction <= 0.0f && !keepOverlaps)
            {
                return false;
            }

            bool backface = QueryCollectors.IsBackface(root, hit.ColliderKey, RigidTransform.identity, direction);
            if (backface && (filterOptions & StoreConstants.QueryAllowBackfaces) == 0)
            {
                return false;
            }

            Hit = hit;
            Backface = backface;
            MaxFraction = hit.Fraction;
            NumHits = 1;
            return true;
        }
    }
}
