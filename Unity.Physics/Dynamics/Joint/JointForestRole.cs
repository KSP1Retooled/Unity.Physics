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

        /// <summary>A servo that can join the vessel tree and keeps ordinary rows outside an active vessel.</summary>
        ServoTree,

        /// <summary>A sliding servo with the same topology rules as a rotary servo.</summary>
        PistonTree,
    }

    internal static class JointForestRoles
    {
        internal static bool IsServoTree(this JointForestRole role)
            => role == JointForestRole.ServoTree || role == JointForestRole.PistonTree;
    }

    internal enum ServoAxis : byte
    {
        None,
        AngularX,
        LinearX,
    }
}
