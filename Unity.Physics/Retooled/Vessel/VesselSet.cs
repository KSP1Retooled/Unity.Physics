using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// The persistent vessels that the caller compiled from its joints and supplies through
    /// <see cref="SimulationStepInput.VesselSet"/>. A vessel is a connected group of moving bodies held together by
    /// part joints, autostruts and struts. Its joints are represented by the vessel kernel's rows
    /// (<see cref="VesselEdge"/>, <see cref="VesselClosure"/>), rather than Unity Physics joints. The direct solver
    /// couples these rows to the contacts and remaining joints in the island through the forest solver's boundary
    /// block. It writes the load on each edge back to this set.
    /// </summary>
    ///
    /// <remarks>
    /// The caller owns every array and must keep them alive until the step completes. Each vessel's parts occupy a
    /// contiguous range, with parents before their children. Edge i belongs to part i. The indices in
    /// <see cref="VesselEdge.Parent"/>, <see cref="VesselClosure.PartA"/> and <see cref="VesselClosure.PartB"/> are
    /// relative to the vessel's first part. A body belongs to at most one vessel. All vessel bodies are dynamic. Joints
    /// to the world, static bodies or kinematic bodies remain Unity Physics joints and are not vessel edges.
    /// <para>
    /// A part uses its body's motion frame (<see cref="MotionData.WorldFromMotion"/>). The origin is the centre of
    /// mass, and the axes are the principal axes of inertia. Anchors, frames and rest orientations for edges and
    /// closures are expressed in these motion frames. If the body's inertia is not rotated, its motion frame rotates
    /// with its body frame.
    /// </para>
    ///
    /// <para>
    /// An axis is constrained by a hard row when <see cref="VesselEdge.TwistStiffness"/>,
    /// <see cref="VesselEdge.SwingStiffness"/> or the corresponding closure stiffness is
    /// <see cref="double.PositiveInfinity"/>. See <see cref="VesselSolverSettings.HardPeriod"/>. Linear rows for edges
    /// are always hard rows.
    /// </para>
    ///
    /// </remarks>
    public unsafe struct VesselSet
    {
        public int VesselCount;

        [NativeDisableUnsafePtrRestriction]
        public VesselRecord* Vessels;

        /// <summary>The index of each part's body in the dynamics world for this step.</summary>
        [NativeDisableUnsafePtrRestriction]
        public int* PartBodies;

        /// <summary>
        /// One edge per part. The first part of a vessel is its root, with <see cref="VesselEdge.Parent"/> set to -1.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        public VesselEdge* Edges;

        [NativeDisableUnsafePtrRestriction]
        public VesselClosure* Closures;

        /// <summary>
        /// The solver writes one load per part: the force that the parent applied to the child through the edge, and
        /// the torque about the joint point. Both are in world space. Force and torque are calculated by dividing the
        /// impulses accumulated over the step by the step duration, giving the mean over all substeps. The ratio is the
        /// larger of force divided by break force and torque divided by break torque. The entry for a root is zero.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        public VesselEdgeLoad* EdgeLoads;

        /// <summary>
        /// The solver writes one load per closure, calculated as for <see cref="EdgeLoads"/>. It contains the force and
        /// torque that part A applied to part B, in world space, with torque measured about the joint point.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        public VesselEdgeLoad* ClosureLoads;

        /// <summary>
        /// The solver writes one value per part to indicate whether its edge was locked into the parent's section
        /// during this step.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        public bool* EdgeLocked;

        public bool IsCreated => Vessels != null && VesselCount > 0;

        struct TestSetKey { }

        static readonly Unity.Burst.SharedStatic<VesselSet> TestSetShared = Unity.Burst.SharedStatic<VesselSet>.GetOrCreate<TestSetKey>();

        /// <summary>
        /// A set for tests to use until the caller builds its own set. The simulation uses this set when no set is
        /// supplied in the step input. The test owns the arrays and must keep them alive while stepping. Set this to
        /// default to stop using it.
        /// </summary>
        public static VesselSet TestSet
        {
            get => TestSetShared.Data;
            set => TestSetShared.Data = value;
        }

        /// <summary>
        /// Returns <paramref name="input"/> if it contains vessels; otherwise returns <see cref="TestSet"/>.
        /// </summary>
        internal static VesselSet OrTestSet(in VesselSet input) => input.IsCreated ? input : TestSetShared.Data;
    }

    /// <summary>Describes one vessel in a <see cref="VesselSet"/>.</summary>
    public struct VesselRecord
    {
        /// <summary>
        /// An identifier that is unique for the caller's lifetime and is never reused. The solver uses it as the key
        /// for cached vessel structure and edge lock states. Assign a new identifier when recompiling a vessel with
        /// different parts.
        /// </summary>
        public ulong Id;

        /// <summary>
        /// Changes when the parts, edge parents, edge states or closures change. Parameter changes do not change this
        /// version.
        /// </summary>
        public uint StructureVersion;

        public int FirstPart;
        public int PartCount;
        public int FirstClosure;
        public int ClosureCount;
    }
}
