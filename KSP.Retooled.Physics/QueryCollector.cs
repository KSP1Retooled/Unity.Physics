using KSP.Retooled.Physics.Interop;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Collision.Queries.Collectors;

namespace KSP.Retooled.Physics
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

    /// <summary>Sends each shape to the plugin API's overlap collector.</summary>
    internal unsafe struct ShapeListSink : IQuerySink<DistanceHit>
    {
        internal FunctionPointer<ManagedCallbacks.ShapeCollectorFunction> Collector;
        internal void* Context;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in DistanceHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            void* userData = shape->UserData;
            NumHits++;
            return Collector.Invoke(Context, &userData, 1) != 0;
        }
    }

    /// <summary>Sends each shape to the plugin API's broadphase collector.</summary>
    internal unsafe struct BroadphaseListSink : IQuerySink<DistanceHit>
    {
        internal FunctionPointer<ManagedCallbacks.BroadphaseCollectorFunction> Collector;
        internal void* Context;
        internal QueryBody* Bodies;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in DistanceHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            NumHits++;
            return Collector.Invoke(Context, shape->Handle, Bodies[hit.RigidBodyIndex].Owner, shape->UserData, (ushort)shape->GeometryType) != 0;
        }
    }

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
            RayCastHit record = QueryKernel.Record(hit.Position, hit.SurfaceNormal, hit.Fraction * Distance, shape, backface);
            NumHits++;
            return Collector.Invoke(Context, &record, 1) != 0;
        }
    }

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
                ? QueryKernel.Record(float3.zero, -Direction, 0.0f, shape, false)
                : QueryKernel.Record(hit.Position, hit.SurfaceNormal, hit.Fraction * Distance, shape, backface);

            NumHits++;
            return Collector.Invoke(Context, &record, 1) != 0;
        }
    }

}
