namespace Unity.Physics.Dynamics.Vessels.Model
{
    /// <summary>Describes one vessel in a <see cref="VesselSet"/>.</summary>
    internal struct VesselRecord
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
