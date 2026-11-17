using Unity.Physics;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Collision.Queries.Collectors;

namespace KSP.Retooled.Physics.Queries
{
    /// <summary>Stores the first shape and stops the query.</summary>
    internal unsafe struct AnyShapeSink : IQuerySink<DistanceHit>
    {
        internal QueryShape* Shape;

        public bool EarlyOut => true;

        public float MaxFraction { get; set; }

        public int NumHits => Shape != null ? 1 : 0;

        public bool Add(in DistanceHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            Shape = shape;
            return false;
        }
    }
}
