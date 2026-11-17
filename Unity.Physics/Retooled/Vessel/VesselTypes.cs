using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>The state of the edge joining a part to its parent in a vessel's part tree.</summary>
    public enum VesselEdgeState : byte
    {
        /// <summary>
        /// Part rigidly joined to its parent's section. Omit solver rows and recover edge loads by inverse
        /// dynamics.
        /// </summary>
        Locked,

        /// <summary>
        /// Uses three hard linear rows and three angular rows with stiff, damped springs: one twist and two swings.
        /// </summary>
        Yielding,

        /// <summary>
        /// A servo or hinge with five hard rows. The twist row uses a spring and damper with
        /// <see cref="VesselEdge.DriveTarget"/> as the target angle.
        /// </summary>
        FreeAxis,

        /// <summary>
        /// The edge is broken, so the part and its descendants can move independently of its former parent.
        /// </summary>
        Broken,
    }

    /// <summary>
    /// Per-step part state in double precision. Use world space except for principal inertia moments.
    /// </summary>
    public struct VesselPart
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

    /// <summary>
    /// Edge i joins child part i to Parent. Parents must precede children.
    /// </summary>
    public struct VesselEdge
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

        /// <summary>The angular spring rates in kN m/rad and damping coefficients in kN m s/rad.</summary>
        public double TwistStiffness;

        public double SwingStiffness;
        public double TwistDamping;
        public double SwingDamping;

        /// <summary>The target twist angle in radians for the spring on a free axis.</summary>
        public double DriveTarget;

        /// <summary>The force in kN and torque in kN m at which the edge breaks.</summary>
        public double BreakForce;

        public double BreakTorque;

        /// <summary>The fraction of the edge's break strength above which a locked edge starts to yield.</summary>
        public double YieldFraction;

        /// <summary>
        /// Relock threshold as a fraction of yield load. Keep below one to provide hysteresis.
        /// </summary>
        public double RelockFraction;
    }

    /// <summary>
    /// Loop-closing connection with six compliant rows. Omit closures internal to a rigid section.
    /// </summary>
    public struct VesselClosure
    {
        public int PartA;
        public int PartB;

        /// <summary>The joint point in each part's frame, relative to its centre of mass.</summary>
        public double3 AnchorA;

        public double3 AnchorB;

        /// <summary>The axes in part A's frame, as columns.</summary>
        public double3x3 Frame;

        /// <summary>The rest orientation of part B relative to part A, calculated as conj(A) * B.</summary>
        public double4 Rest;

        public double LinearStiffness;
        public double AngularStiffness;
        public double LinearDamping;
        public double AngularDamping;
    }

    /// <summary>
    /// Force on the child (or closure part B) and torque about the joint point, in world space.
    /// Ratio is the larger of force magnitude/breakForce and torque magnitude/breakTorque.
    /// </summary>
    public struct VesselEdgeLoad
    {
        public double3 Force;
        public double3 Torque;
        public double Ratio;
    }

    /// <summary>Settings for <see cref="VesselSolver"/>.</summary>
    public struct VesselSolverSettings
    {
        public double TimeStep;

        /// <summary>
        /// Hard-row spring period. Compute stiffness from reduced section mass or inertia so that light connectors
        /// do not soften heavy subtrees.
        /// </summary>
        public double HardPeriod;

        /// <summary>The damping for a hard row, as a fraction of critical damping for the same reduced mass.</summary>
        public double HardDampingRatio;

        /// <summary>Sections with at least this many active closures are treated as hubs and eliminated last.</summary>
        public int HubClosures;

        /// <summary>
        /// The approximate number of independent pieces of factorization work to distribute among job threads.
        /// </summary>
        public int ChunkTarget;

        /// <summary>
        /// Use minimum-degree ordering to reduce fill from struts. Otherwise order sections leaves-first with hubs
        /// last.
        /// </summary>
        public bool MinimumDegree;

        /// <summary>
        /// Include centripetal force in locked-edge loads. Disable to balance loads exactly against active-row
        /// impulses;
        /// the solver itself has no centripetal or gyroscopic terms.
        /// </summary>
        public bool CentripetalLoads;

        public static VesselSolverSettings Default => new VesselSolverSettings
        {
            TimeStep = 0.02,
            HardPeriod = 0.001,
            HardDampingRatio = 1.0,
            HubClosures = 16,
            ChunkTarget = 96,
            MinimumDegree = true,
            CentripetalLoads = true,
        };
    }
}
