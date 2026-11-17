using System.Runtime.InteropServices;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;

namespace Unity.Physics.Store.Queries
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct QueryInput
    {
        public quaternion rotation;
        public float3 position;
        public uint geometryType;

        public float3 extents;
        public int layerMask;

        public float3 direction;
        public float maxDistance;

        public uint filterOptions;
        public uint keepInitialOverlaps;

        public quaternion rotation2;
        public float3 position2;
        private readonly uint reserved0;

        public Shape* shape;
        public Shape* shape2;
        public Body* body;
        public void* collector;
        public void* collectorContext;
    }
}
