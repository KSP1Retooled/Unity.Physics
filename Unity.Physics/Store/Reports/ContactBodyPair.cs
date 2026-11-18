using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.Reports
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct ContactBodyPair
    {
        public ulong thisBody;
        public ulong otherBody;
        public ContactShapePair* shapes;
        public uint shapeCount;
        public ushort flags;
        private readonly ushort reserved0;
        public float3 thisLinearVelocity;
        public float3 thisAngularVelocity;
        public float3 otherLinearVelocity;
        public float3 otherAngularVelocity;
    }
}
