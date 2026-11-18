using System.Threading;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling.LowLevel.Unsafe;

namespace Unity.Physics.Dynamics.Solver.Sparse
{
    /// <summary>Accumulates sparse solve time across the steps that share a cache.</summary>
    internal unsafe struct SparseSolveCounters
    {
        [NativeDisableUnsafePtrRestriction]
        long* ticks;

        internal static SparseSolveCounters Create()
        {
            var counters = new SparseSolveCounters
            {
                ticks = (long*)UnsafeUtility.MallocTracked(sizeof(long), UnsafeUtility.AlignOf<long>(), Allocator.Persistent, 0),
            };
            *counters.ticks = 0;
            return counters;
        }

        internal void AddTime(long elapsed)
        {
            if (ticks != null)
            {
                Interlocked.Add(ref *ticks, elapsed);
            }
        }

        internal double Milliseconds => ticks != null ? *ticks * (ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
            / (double)ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6) : 0;

        internal void Dispose()
        {
            UnsafeUtility.FreeTracked(ticks, Allocator.Persistent);
            ticks = null;
        }
    }
}
