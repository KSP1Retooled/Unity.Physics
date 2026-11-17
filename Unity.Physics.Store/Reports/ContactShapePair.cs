using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.Reports
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct ContactShapePair
    {
        public ulong thisCollider;
        public ulong otherCollider;
        public ContactPoint* points;
        public uint pointCount;
        public ushort flags;
        public ushort events;
        public float3 impulseSum;
        private readonly uint reserved0;
    }
}
