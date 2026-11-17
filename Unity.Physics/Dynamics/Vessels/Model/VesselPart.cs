using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Vessels.Model
{
    /// <summary>
    /// Per-step part state in double precision. Use world space except for principal inertia moments.
    /// </summary>
    internal struct VesselPart
    {
        public double Mass;

        /// <summary>The principal moments of inertia in the part's own frame.</summary>
        public double3 Inertia;

        /// <summary>The centre of mass in world space.</summary>
        public double3 Position;

        /// <summary>The rotation from the part's frame to world space, as a quaternion in x, y, z, w order.</summary>
        public double4 Orientation;

        public double3 LinearVelocity;
        public double3 AngularVelocity;

        /// <summary>
        /// The external impulse applied at the centre of mass over the step: force multiplied by the time step.
        /// </summary>
        public double3 LinearImpulse;

        public double3 AngularImpulse;
    }
}
