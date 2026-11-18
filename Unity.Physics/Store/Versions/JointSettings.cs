using System.Threading;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Store.Versions
{
    /// <summary>
    /// Shared version for joint settings that affect vessel constraints. Changed joints record the new version.
    /// Unchanged values and drive targets leave it unchanged, allowing cached edges and closures to be reused.
    /// </summary>
    internal static class JointSettings
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
