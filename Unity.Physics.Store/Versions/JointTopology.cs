using System.Threading;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Store.Versions
{
    /// <summary>
    /// Shared version for changes to joint grouping or vessel forest roles.
    /// Unchanged worlds reuse groups; a change in any world invalidates all worlds.
    /// </summary>
    internal static class JointTopology
    {
        private readonly struct Key { }

        private static readonly SharedStatic<int> Shared = SharedStatic<int>.GetOrCreate<Key>();

        internal static uint Version => (uint)Shared.Data;

        /// <summary>Moves the version on and returns the new value.</summary>
        internal static unsafe uint Changed()
        {
            return (uint)System.Threading.Interlocked.Increment(ref UnsafeUtility.AsRef<int>(Shared.UnsafeDataPointer));
        }
    }
}
