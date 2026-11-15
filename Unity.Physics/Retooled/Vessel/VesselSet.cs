using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// The persistent vessels of a step: what the caller compiled from its joints, handed to the simulation in
    /// <see cref="SimulationStepInput.VesselSet"/>. A vessel is a connected group of moving bodies held together by part
    /// joints, autostruts and struts. Its joints never become Unity Physics joints: the direct solver solves them with the
    /// vessel kernel's own rows (<see cref="VesselEdge"/>, <see cref="VesselClosure"/>), coupled to everything else in the
    /// island (contacts, the remaining joints) through the forest path's boundary block, and writes each edge's load back
    /// here.
    /// </summary>
    /// <remarks>
    /// The caller owns every array and keeps them alive until the step completes. Parts of a vessel are contiguous and in
    /// parent-first order; edge i belongs to part i. <see cref="VesselEdge.Parent"/>, <see cref="VesselClosure.PartA"/> and
    /// <see cref="VesselClosure.PartB"/> count from the vessel's first part. A body belongs to at most one vessel.
    /// Every vessel body is dynamic; a joint to the world, a static or a kinematic body is not a vessel edge and stays a
    /// Unity Physics joint.
    /// <para>
    /// A part's frame is its body's motion frame (<see cref="MotionData.WorldFromMotion"/>): its origin is the centre of
    /// mass, and its axes are the principal axes of inertia. Edge and closure anchors, frames and rest orientations are
    /// given in these frames. For a body whose inertia isn't rotated, the motion frame turns with the body's own frame.
    /// </para>
    /// <para>
    /// Hard rows: a <see cref="VesselEdge.TwistStiffness"/> or <see cref="VesselEdge.SwingStiffness"/> of
    /// <see cref="double.PositiveInfinity"/>, and a closure stiffness of <see cref="double.PositiveInfinity"/>, mean the axis
    /// is held as a hard row (<see cref="VesselSolverSettings.HardPeriod"/>), as a linear edge row always is.
    /// </para>
    /// </remarks>
    public unsafe struct VesselSet
    {
        public int VesselCount;

        [NativeDisableUnsafePtrRestriction]
        public VesselRecord* Vessels;

        /// <summary>Each part's body, as its index among the dynamics world's bodies this step.</summary>
        [NativeDisableUnsafePtrRestriction]
        public int* PartBodies;

        /// <summary>One edge per part; a vessel's first part is its root, whose edge has <see cref="VesselEdge.Parent"/> -1.</summary>
        [NativeDisableUnsafePtrRestriction]
        public VesselEdge* Edges;

        [NativeDisableUnsafePtrRestriction]
        public VesselClosure* Closures;

        /// <summary>
        /// Written by the solve, one per part: the force and torque each edge carried, averaged over the step (impulse over
        /// the step's time), with its ratio against its strength. A root's entry is zero.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        public VesselEdgeLoad* EdgeLoads;

        /// <summary>Written by the solve, one per closure, as <see cref="EdgeLoads"/>.</summary>
        [NativeDisableUnsafePtrRestriction]
        public VesselEdgeLoad* ClosureLoads;

        /// <summary>Written by the solve, one per part: whether the edge was locked (folded into its parent's section) this step.</summary>
        [NativeDisableUnsafePtrRestriction]
        public bool* EdgeLocked;

        public bool IsCreated => Vessels != null && VesselCount > 0;

        struct TestSetKey { }

        static readonly Unity.Burst.SharedStatic<VesselSet> TestSetShared = Unity.Burst.SharedStatic<VesselSet>.GetOrCreate<TestSetKey>();

        /// <summary>
        /// For tests, until the caller builds the set: a set that the step uses whenever its input has none. The test owns
        /// the arrays and keeps them alive while it steps. Set it to default to stop.
        /// </summary>
        public static VesselSet TestSet
        {
            get => TestSetShared.Data;
            set => TestSetShared.Data = value;
        }

        /// <summary>The step's set: <paramref name="input"/> if it has one, and otherwise <see cref="TestSet"/>.</summary>
        internal static VesselSet OrTestSet(in VesselSet input) => input.IsCreated ? input : TestSetShared.Data;
    }

    /// <summary>One vessel of a <see cref="VesselSet"/>.</summary>
    public struct VesselRecord
    {
        /// <summary>
        /// Unique for the life of the caller and never reused, so the solver can key what it keeps per vessel (its structure,
        /// its edges' lock states) on it. A vessel recompiled with different parts gets a new one.
        /// </summary>
        public ulong Id;

        /// <summary>Changes whenever the vessel's parts, edges' parents, edge states or closures change; not for parameters.</summary>
        public uint StructureVersion;

        public int FirstPart;
        public int PartCount;
        public int FirstClosure;
        public int ClosureCount;
    }
}
