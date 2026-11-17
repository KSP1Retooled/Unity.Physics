using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Shapes
{
    /// <summary>
    /// Mirrored prefix of a mesh handle. The native reference count follows these fields.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct Mesh
    {
        public MeshData* data;
        public MeshMoments moments;
        public uint momentsKnown;

        /// <summary>Returns the cooked mesh of a shape, or null if the shape has no cooked mesh.</summary>
        internal static MeshData* MeshOf(ShapeData* shape)
        {
            return shape->mesh != null ? ((Mesh*)shape->mesh)->data : null;
        }
    }
}
