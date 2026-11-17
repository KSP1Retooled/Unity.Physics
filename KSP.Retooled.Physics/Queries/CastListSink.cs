using Unity.Physics;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Collision.Queries.Collectors;
using KSP.Retooled.Physics.Interop;
using Unity.Burst;
using Unity.Physics.Store.Queries;
using Unity.Mathematics;

namespace KSP.Retooled.Physics.Queries
{
    /// <summary>Sends each cast hit to the plugin API's hit collector.</summary>
    internal unsafe struct CastListSink : IQuerySink<ColliderCastHit>
    {
        internal FunctionPointer<ManagedCallbacks.HitCollectorFunction> Collector;
        internal void* Context;
        internal float Distance;
        internal float3 Direction;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in ColliderCastHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            RayCastHit record = overlappedAtStart
                ? QueryHitFormat.Record(float3.zero, -Direction, 0.0f, shape, false)
                : QueryHitFormat.Record(hit.Position, hit.SurfaceNormal, hit.Fraction * Distance, shape, backface);

            NumHits++;
            return Collector.Invoke(Context, &record, 1) != 0;
        }
    }
}
