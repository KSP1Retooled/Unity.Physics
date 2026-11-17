using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.Shapes
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct ShapeData
    {
        public quaternion localRotation;
        public float3 localPosition;
        public uint geometryType;

        public float3 extents;
        public uint flags;

        public uint belongsTo;
        public uint collidesWith;
        public uint ignorePairId;
        public uint collisionDisabled;

        public void* mesh;
        public MaterialData* material;
        public void* userData;
        public Shape* owner;
    }
}
