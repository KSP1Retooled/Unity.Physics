using System.Threading;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Store.Versions
{
    /// <summary>
    /// Shared version for trigger, collision filter and kinematic changes that require touching pairs to be
    /// reevaluated.
    /// </summary>
    internal static class PairFilterVersion
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
