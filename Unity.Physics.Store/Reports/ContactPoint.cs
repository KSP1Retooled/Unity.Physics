using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.Reports
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct ContactPoint
    {
        public float3 position;
        public float separation;
        public float3 normal;
        public uint faceIndex0;
        public float3 impulse;
        public uint faceIndex1;
    }
}
