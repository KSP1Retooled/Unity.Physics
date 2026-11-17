using Unity.Physics;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Collision.Queries.Collectors;
using KSP.Retooled.Physics.Interop;
using Unity.Burst;

namespace KSP.Retooled.Physics.Queries
{
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
}
