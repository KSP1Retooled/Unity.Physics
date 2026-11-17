using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Vehicles
{
    /// <summary>Matches Physics::PhysXExt::WheelSuspensionData.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct WheelSuspension
    {
        public float springStrength;
        public float springDamperRate;
        public float maxCompression;
        public float maxDroop;
        public float sprungMass;
    }
}
