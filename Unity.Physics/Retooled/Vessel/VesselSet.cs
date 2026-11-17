using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Caller-owned persistent vessels supplied through SimulationStepInput.VesselSet.
    /// Vessel rows replace internal Unity joints and couple to contacts/other joints through the forest boundary.
    /// Keep all arrays alive until step completion. Parts are contiguous and parent-first; edge i belongs to part
    /// i.
    /// Parent and closure part indices are vessel-relative. A body belongs to at most one vessel and must be
    /// dynamic.
    /// Express anchors and joint frames in body motion frames. Infinite angular stiffness selects hard rows; edge
    /// linear rows are always hard.
    /// </summary>
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
        /// Mean edge force and torque about the joint point, in world space, averaged over substeps. Root entries
        /// are zero.
        /// Ratio is the larger of force magnitude/breakForce and torque magnitude/breakTorque.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        public VesselEdgeLoad* EdgeLoads;

        /// <summary>
        /// Mean closure force from A to B and torque about the joint point, in world space, as for EdgeLoads.
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
        /// Fallback vessel set for tests when no set is supplied in step input.
        /// The test owns array lifetimes. Assign default to stop using it.
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
        /// Unique ID never reused during the caller's lifetime, used for cached structure and locks.
        /// Assign a new ID when recompiling with different parts.
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
