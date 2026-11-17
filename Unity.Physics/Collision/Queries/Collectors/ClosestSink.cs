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
    /// Retain the closest hit, breaking distance ties with QueryCollectors.Precedes regardless of tree traversal
    /// order.
    /// </summary>
    internal unsafe struct ClosestSink<T> : IQuerySink<T>
        where T : unmanaged, IQueryResult
    {
        internal T Hit;
        internal QueryShape* Shape;
        internal uint Serial;
        internal bool Backface;

        public bool EarlyOut => false;

        /// <summary>
        /// Initial query length, then the next float above the nearest hit fraction to admit equal-distance ties.
        /// </summary>
        public float MaxFraction { get; set; }

        public int NumHits => Shape != null ? 1 : 0;

        public bool Add(in T hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            bool closer = Shape == null
                ? hit.Fraction < MaxFraction
                : QueryCollectors.Precedes(hit.Fraction, serial, hit.ColliderKey, Hit.Fraction, Serial, Hit.ColliderKey);

            if (closer)
            {
                Hit = hit;
                Shape = shape;
                Serial = serial;
                Backface = backface;
                MaxFraction = QueryCollectors.JustAbove(hit.Fraction);
            }

            return true;
        }
    }
}
