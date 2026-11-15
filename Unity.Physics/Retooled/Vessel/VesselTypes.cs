using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>The state of the edge joining a part to its parent in a vessel's part tree.</summary>
    public enum VesselEdgeState : byte
    {
        /// <summary>
        /// No relative motion. The part folds into its parent's rigid section, so the edge costs nothing in the solve; its
        /// load still comes out every step, by inverse dynamics inside the section.
        /// </summary>
        Locked,

        /// <summary>Three hard linear rows and three angular rows that are stiff damped springs (twist and two swings).</summary>
        Yielding,

        /// <summary>
        /// A servo or hinge: five hard rows, and the twist row a spring and damper toward <see cref="VesselEdge.DriveTarget"/>.
        /// </summary>
        FreeAxis,

        /// <summary>No edge: the part and everything beyond it move on their own.</summary>
        Broken,
    }

    /// <summary>
    /// One part of a vessel, as the backend hands it to <see cref="VesselSolver"/> each step and reads it back after.
    /// Everything is in double precision and in world space, apart from the principal moments.
    /// </summary>
    public struct VesselPart
    {
        public double Mass;

        /// <summary>Principal moments of inertia, in the part's own frame.</summary>
        public double3 Inertia;

        /// <summary>Centre of mass.</summary>
        public double3 Position;

        /// <summary>The rotation from the part's frame to world, as a quaternion x, y, z, w.</summary>
        public double4 Orientation;

        public double3 LinearVelocity;
        public double3 AngularVelocity;

        /// <summary>External impulse over the step (force times the time step), applied at the centre of mass.</summary>
        public double3 LinearImpulse;

        public double3 AngularImpulse;
    }

    /// <summary>
    /// The edge from a part to its parent. Edges are indexed by their child part, so edge i joins part i to part
    /// <see cref="Parent"/>, and parents must come before their children.
    /// </summary>
    public struct VesselEdge
    {
        /// <summary>The parent part, or -1 for the vessel's root.</summary>
        public int Parent;

        public VesselEdgeState State;

        /// <summary>
        /// Whether a yielding edge whose load falls low enough may lock again. Servos, docking ports and decouplers stay
        /// live, so this is off for them.
        /// </summary>
        public bool MayLock;

        /// <summary>The joint point in the parent's frame, relative to its centre of mass.</summary>
        public double3 ParentAnchor;

        /// <summary>The joint point in the child's frame, relative to its centre of mass.</summary>
        public double3 ChildAnchor;

        /// <summary>
        /// The joint's axes in the parent's frame, as columns: twist, then the two swing axes. The rows act along these.
        /// </summary>
        public double3x3 Frame;

        /// <summary>The child's orientation relative to the parent's at rest, conj(parent) * child.</summary>
        public double4 Rest;

        /// <summary>Spring rates of the angular rows, kN m/rad, and their dampers, kN m s/rad.</summary>
        public double TwistStiffness;

        public double SwingStiffness;
        public double TwistDamping;
        public double SwingDamping;

        /// <summary>For a free axis, the twist angle its spring drives toward, in radians.</summary>
        public double DriveTarget;

        /// <summary>Force and torque at which the edge breaks, kN and kN m.</summary>
        public double BreakForce;

        public double BreakTorque;

        /// <summary>The fraction of its strength above which a locked edge starts to yield.</summary>
        public double YieldFraction;

        /// <summary>
        /// The fraction of the yield load below which a yielding edge locks again. Kept well under one, so that an edge
        /// near its yield load does not lock and unlock on alternate steps.
        /// </summary>
        public double RelockFraction;
    }

    /// <summary>
    /// A strut, autostrut or anything else that closes a loop in the part tree: six compliant rows between two parts. One
    /// between two parts of the same rigid section carries nothing and drops out of the solve.
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

        /// <summary>Part B's orientation relative to part A's at rest, conj(A) * B.</summary>
        public double4 Rest;

        public double LinearStiffness;
        public double AngularStiffness;
        public double LinearDamping;
        public double AngularDamping;
    }

    /// <summary>
    /// The load an edge carried over the step: the force and the torque about the joint point that the parent side applied
    /// to the child side (part B's side for a closure), with the larger of each against its strength.
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
        /// The period of a hard row, as a spring on the reduced mass (or inertia) of the two sections it joins. Taken from
        /// the sections, never from single parts: a light part between heavy ones would otherwise make the softest joint
        /// of the vessel exactly where it should be stiffest.
        /// </summary>
        public double HardPeriod;

        /// <summary>A hard row's damping as a fraction of critical damping on that same reduced mass.</summary>
        public double HardDampingRatio;

        /// <summary>Sections with at least this many live closures are eliminated last, as a hub.</summary>
        public int HubClosures;

        /// <summary>About how many independent pieces of work the factor is cut into for the job threads.</summary>
        public int ChunkTarget;

        /// <summary>
        /// Whether locked edges' loads include the centripetal force of a spinning section. The solve itself has no
        /// gyroscopic or centripetal terms, so loads without it close exactly against the live rows' impulses.
        /// </summary>
        public bool CentripetalLoads;

        public static VesselSolverSettings Default => new VesselSolverSettings
        {
            TimeStep = 0.02,
            HardPeriod = 0.001,
            HardDampingRatio = 1.0,
            HubClosures = 16,
            ChunkTarget = 96,
            CentripetalLoads = true,
        };
    }
}
