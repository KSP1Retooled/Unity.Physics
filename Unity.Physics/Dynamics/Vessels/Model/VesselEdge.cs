using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Vessels.Model
{
    /// <summary>
    /// Edge i joins child part i to Parent. Parents must precede children.
    /// </summary>
    internal struct VesselEdge
    {
        /// <summary>The index of the parent part, or -1 for the vessel's root.</summary>
        public int Parent;

        public VesselEdgeState State;

        /// <summary>
        /// Permit relocking below the load threshold. Disable for servos, docking ports and decouplers that need
        /// active rows.
        /// </summary>
        public bool MayLock;

        /// <summary>The joint point in the parent's frame, relative to its centre of mass.</summary>
        public double3 ParentAnchor;

        /// <summary>The joint point in the child's frame, relative to its centre of mass.</summary>
        public double3 ChildAnchor;

        /// <summary>
        /// The joint axes in the parent's frame, stored as columns: the twist axis followed by the two swing axes. The
        /// constraint rows act along these axes.
        /// </summary>
        public double3x3 Frame;

        /// <summary>The child's rest orientation relative to its parent, calculated as conj(parent) * child.</summary>
        public double4 Rest;

        /// <summary>Spring rate about each joint axis, in kN m/rad.</summary>
        public double3 AngularStiffness;

        /// <summary>Damping about each joint axis, in kN m s/rad.</summary>
        public double3 AngularDamping;

        public double TwistStiffness { get => AngularStiffness.x; set => AngularStiffness.x = value; }
        public double SwingStiffness { get => AngularStiffness.y; set => AngularStiffness.yz = new double2(value); }
        public double TwistDamping { get => AngularDamping.x; set => AngularDamping.x = value; }
        public double SwingDamping { get => AngularDamping.y; set => AngularDamping.yz = new double2(value); }

        /// <summary>The target twist angle in radians for the spring on a free axis.</summary>
        public double DriveTarget;

        /// <summary>The force in kN and torque in kN m at which the edge breaks.</summary>
        public double BreakForce;

        public double BreakTorque;

        /// <summary>The torque at which a compliant axis at a narrow angular stop has used all its travel.</summary>
        public double FlexYieldTorque;

        /// <summary>The squared sine of half the angle at which the edge should stay unlocked.</summary>
        public double UnlockRotationSquared;

        /// <summary>The fraction of the edge's break strength above which a locked edge starts to yield.</summary>
        public double YieldFraction;

        /// <summary>
        /// Relock threshold as a fraction of yield load. Keep below one to provide hysteresis.
        /// </summary>
        public double RelockFraction;
    }
}
