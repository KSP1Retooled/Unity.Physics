namespace Unity.Physics
{
    /// <summary>How the direct solve may treat a joint's rows, see <see cref="Joint.ForestRole"/>.</summary>
    public enum JointForestRole : byte
    {
        /// <summary>A row like any other: solved with the island's contacts and limits.</summary>
        None,

        /// <summary>An edge of the vessel forest's spanning tree.</summary>
        Tree,

        /// <summary>A forest joint that closes a loop, such as a strut.</summary>
        Closure,
    }
}
