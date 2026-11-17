using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics.Dynamics.Vessels.Structure
{
    /// <summary>A rigid section as the solve sees it: one body made of every part joined by locked edges.</summary>
    internal struct VesselSection
    {
        public double Mass;
        public double3 Center;

        /// <summary>World inertia about the centre, with every part's parallel-axis term, and its inverse.</summary>
        public double3x3 Inertia;

        public double3x3 InverseInertia;

        /// <summary>The momentum-conserving average of the parts' velocities before the solve.</summary>
        public double3 LinearVelocity;

        public double3 AngularVelocity;

        /// <summary>The parts' external impulses, summed, and their moments about the centre.</summary>
        public double3 LinearImpulse;

        public double3 AngularImpulse;
    }
}
