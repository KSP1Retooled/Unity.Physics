using System.Threading;
using Unity.Burst;

namespace Unity.Physics
{
    /// <summary>Counts sorts that copied their output from the cache, for tests.</summary>
    internal readonly struct IslandPairSortCounters
    {
        private struct ReuseKey { }

        private static readonly SharedStatic<long> ReuseCount = SharedStatic<long>.GetOrCreate<ReuseKey>();

        internal static long Reuses => Interlocked.Read(ref ReuseCount.Data);

        internal static void RecordReuse() => Interlocked.Increment(ref ReuseCount.Data);
    }
}
