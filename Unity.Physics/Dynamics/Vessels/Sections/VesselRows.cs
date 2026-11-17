using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    // Cached link rows: three linear axes followed by three angular axes, reused after factorization.
    internal struct VesselRows
    {
        /// <summary>The world axes, as columns.</summary>
        public double3x3 Axes;

        /// <summary>The joint point on each side, from its section's centre.</summary>
        public double3 ChildArm;

        public double3 ParentArm;

        /// <summary>Each row's inverse compliance, h^2 k + h c.</summary>
        public double3 LinearWeight;

        public double3 AngularWeight;

        /// <summary>Each row's position term, inverse compliance times gamma / h times the error, which is h k times the error.</summary>
        public double3 LinearBias;

        public double3 AngularBias;

        /// <summary>The impulses the solve gave the rows, on the child side.</summary>
        public double3 LinearImpulse;

        public double3 AngularImpulse;
    }
}
