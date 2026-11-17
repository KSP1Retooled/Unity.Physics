using Unity.Physics;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Collision.Queries.Collectors;
using KSP.Retooled.Physics.Interop;
using Unity.Burst;
using Unity.Physics.Store.Queries;

namespace KSP.Retooled.Physics.Queries
{
    /// <summary>Sends each ray hit to the plugin API's hit collector.</summary>
    internal unsafe struct RayListSink : IQuerySink<RaycastHit>
    {
        internal FunctionPointer<ManagedCallbacks.HitCollectorFunction> Collector;
        internal void* Context;
        internal float Distance;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in RaycastHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            RayCastHit record = QueryHitFormat.Record(hit.Position, hit.SurfaceNormal, hit.Fraction * Distance, shape, backface);
            NumHits++;
            return Collector.Invoke(Context, &record, 1) != 0;
        }
    }
}
