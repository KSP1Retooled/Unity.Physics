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
    /// Find the deepest convex overlap, excluding mesh triangles facing away from the shape's centre as PhysX does.
    /// </summary>
    internal unsafe struct PenetrationCollector : ICollector<DistanceHit>
    {
        internal DistanceHit Hit;

        private readonly Collider* target;
        private readonly float3 queryCentre;

        internal PenetrationCollector(Collider* target, float3 queryCentre, float maxDistance)
        {
            this.target = target;
            this.queryCentre = queryCentre;
            Hit = default;
            MaxFraction = maxDistance;
            NumHits = 0;
        }

        public bool EarlyOutOnFirstHit => false;

        public float MaxFraction { get; private set; }

        public int NumHits { get; private set; }

        public bool AddHit(DistanceHit hit)
        {
            if (QueryCollectors.IsBehind(target, hit.ColliderKey, queryCentre))
            {
                return false;
            }

            Hit = hit;
            MaxFraction = hit.Fraction;
            NumHits = 1;
            return true;
        }
    }
}
