using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Queries
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct QueryOutput
    {
        public RayCastHit hit;
        public void* shapeUserData;
    }
}
