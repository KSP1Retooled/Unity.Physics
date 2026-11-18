using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Vehicles
{
    /// <summary>Matches Physics::PhysXExt::WheelData.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct WheelBody
    {
        public float radius;
        public float mass;
        public float momentOfInertia;
        public float dampingRate;
    }
}
