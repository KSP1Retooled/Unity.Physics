using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.Shapes
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct MeshMoments
    {
        public float volume;
        public float3 centroid;
        public float3 covarianceDiagonal;
        public float3 covarianceOffDiagonal;
    }
}
