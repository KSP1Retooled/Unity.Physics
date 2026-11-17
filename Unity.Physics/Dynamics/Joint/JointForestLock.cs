using System;

namespace Unity.Physics
{
    /// <summary>
    /// Permitted locking operations for a forest body pair, based on owner load limits and forest deflection tests.
    /// </summary>
    [Flags]
    public enum JointForestLock : byte
    {
        None = 0,

        /// <summary>The joint's load is low enough for its pair to lock.</summary>
        Lockable = 1 << 0,

        /// <summary>The joint's load is low enough for a locked pair to stay locked.</summary>
        Holdable = 1 << 1,
    }
}
