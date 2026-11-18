using System.Runtime.InteropServices;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;

namespace Unity.Physics.Store.Vehicles
{
    /// <summary>
    /// Vehicle state followed by wheel slots in the same native allocation.
    /// Survives domain reloads and is freed with the world.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct VehicleData
    {
        public Body* body;
        public uint wheelCount;
        public uint implicitSprungMasses;
        public uint expansionLimit;
        public float substepSpeed;
        public uint substepsBelow;
        public uint substepsAbove;

        // The body's velocity in world space before the forces for this step are applied. This velocity is used to
        // calculate suspension damping.
        public float3 startLinear;
        public float3 startAngular;

        internal static WheelSlot* WheelsOf(VehicleData* vehicle)
        {
            return (WheelSlot*)(vehicle + 1);
        }
    }
}
