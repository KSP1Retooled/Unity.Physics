using System.Threading;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Store.Versions
{
    /// <summary>
    /// Shared version for non-pose data copied into query snapshots, including body/shape membership and shape
    /// settings.
    /// Snapshot reuse also requires the same built body row order.
    /// </summary>
    internal static class QuerySnapshotVersion
    {
        private readonly struct Key { }

        private static readonly SharedStatic<int> Shared = SharedStatic<int>.GetOrCreate<Key>();

        internal static uint Version => (uint)Shared.Data;

        /// <summary>Moves the version on.</summary>
        internal static unsafe void Changed()
        {
            Interlocked.Increment(ref UnsafeUtility.AsRef<int>(Shared.UnsafeDataPointer));
        }
    }
}
