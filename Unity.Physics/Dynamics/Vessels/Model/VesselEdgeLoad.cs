using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Vessels.Model
{
    /// <summary>
    /// Force on the child (or closure part B) and torque about the joint point, in world space.
    /// Ratio is the larger of force magnitude/breakForce and torque magnitude/breakTorque.
    /// </summary>
    internal struct VesselEdgeLoad
    {
        public double3 Force;
        public double3 Torque;
        public double Ratio;
    }
}
