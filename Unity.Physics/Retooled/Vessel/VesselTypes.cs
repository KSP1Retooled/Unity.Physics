using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>The state of the edge joining a part to its parent in a vessel's part tree.</summary>
    public enum VesselEdgeState : byte
    {
        /// <summary>
        /// The part has no motion relative to its parent and belongs to the same rigid section. The solver does not
        /// need rows for this edge. Its load is still calculated every step using inverse dynamics within the section.
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
    /// The state of one vessel part, supplied by the backend to <see cref="VesselSolver"/> each step and read back
    /// afterwards. All values use double precision. Values are in world space except for the principal moments of
    /// inertia.
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
    /// The edge that joins a part to its parent. Edges use the index of their child part: edge i joins part i to the
    /// part at <see cref="Parent"/>. Parents must appear before their children.
    /// </summary>
    public struct VesselEdge
    {
        /// <summary>The index of the parent part, or -1 for the vessel's root.</summary>
        public int Parent;

        public VesselEdgeState State;

        /// <summary>
        /// Whether a yielding edge may lock again when its load falls below the threshold. This is disabled for servos,
        /// docking ports and decouplers so that their rows remain active in the solver.
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
        /// The fraction of the yield load below which a yielding edge locks again. Keep this well below one to prevent
        /// an edge near its yield load from locking and unlocking on alternate steps.
        /// </summary>
        public double RelockFraction;
    }

    /// <summary>
    /// A strut, autostrut or other connection that closes a loop in the part tree. It has six compliant rows between
    /// two parts. A closure between parts in the same rigid section carries no load and is excluded from the solve.
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
    /// The force that the parent part applied to the child part and the torque about the joint point. For a closure,
    /// these are the force and torque that part A applied to part B. Both are in world space. The ratio is the larger
    /// of force divided by break force and torque divided by break torque.
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
        /// The spring period used for a hard row. Its stiffness is calculated from the reduced mass or inertia of the
        /// two sections that it joins. Use the sections rather than individual parts. Otherwise, a light part between
        /// heavy parts would produce the vessel's softest joint where it needs the greatest stiffness.
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
        /// Orders sections by minimum degree. Otherwise, sections are ordered from the leaves inward along the section
        /// tree, with hubs last. With tree ordering, each strut can introduce matrix fill along the entire tree path
        /// between its ends. On craft with struts, this creates a large block near the root that must be processed
        /// serially. Minimum degree ordering limits fill to roughly the entries introduced by the struts themselves.
        /// </summary>
        public bool MinimumDegree;

        /// <summary>
        /// Whether loads on locked edges include the centripetal force of a rotating section. The solver has no
        /// gyroscopic or centripetal terms. If centripetal force is excluded from these loads, they balance exactly
        /// against the impulses from active rows.
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
