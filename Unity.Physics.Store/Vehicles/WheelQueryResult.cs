using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.Vehicles
{
    /// <summary>Matches Physics::PhysXExt::WheelQueryResult. Vehicle.cpp checks this layout.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct WheelQueryResult
    {
        public float3 point;
        public float3 normal;
        public float3 forwardDir;
        public float3 sidewaysDir;
        public float force;
        public float forwardSlip;
        public float sidewaysSlip;
        public float suspensionCompression;
        public float steerAngle;
        private readonly uint reserved0;
        public void* userData;
    }
}
