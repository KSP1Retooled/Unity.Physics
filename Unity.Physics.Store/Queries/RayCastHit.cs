using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.Queries
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct RayCastHit
    {
        public float3 point;
        public float3 normal;
        public uint faceId;
        public float distance;
        public float2 uv;
        public ulong engineObjectId;
    }
}
