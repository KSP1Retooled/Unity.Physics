using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.Shapes
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct MeshData
    {
        public uint magic;
        public uint version;
        public uint convex;
        public uint vertexCount;
        public uint triangleCount;
        public float3 lower;
        public float3 upper;

        internal static float3* VerticesOf(MeshData* mesh)
        {
            return (float3*)(mesh + 1);
        }

        internal static int3* TrianglesOf(MeshData* mesh)
        {
            return (int3*)(VerticesOf(mesh) + mesh->vertexCount);
        }
    }
}
