using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Shapes
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct CookInput
    {
        public void* vertices;
        public void* indices;
        public uint vertexCount;
        public uint indexCount;
        public uint vertexStride;
        public uint indexStride;
        public uint convex;
        public uint flippedNormals;
    }
}
