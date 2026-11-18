using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    /// <summary>Records forest solve counts and timings that the backend reads after stepping.</summary>
    internal unsafe struct ForestCounters
    {
        [NativeDisableUnsafePtrRestriction]
        long* forestCounters;

        internal const int ForestSolves = 0;
        internal const int ForestFallbacks = 1;
        internal const int ForestBuilds = 2;

        /// <summary>Timings use the profiler clock. See <see cref="GetSolveTimes"/>.</summary>
        internal const int ForestStartTime = 3;
        internal const int ForestSolveTime = 4;
        internal const int ForestFinishTime = 5;
        internal const int ForestFactorTime = 6;
        internal const int ForestBoundaryTime = 7;
        internal const int ForestDriftTime = 8;

        /// <summary>
        /// Counters for total sections and locked pairs across forest solves, structure rebuilds, lock changes, and
        /// time spent rebuilding.
        /// </summary>
        internal const int ForestSectionSum = 9;
        internal const int ForestLockedSum = 10;
        internal const int ForestStructureBuilds = 11;
        internal const int ForestLockChanges = 12;
        internal const int ForestStructureTime = 13;

        // Maximum inverse-dynamics momentum residual in units of 1e-9 N s, plus last-solve section/lock counts.
        internal const int ForestResidual = 14;
        internal const int ForestLastSections = 15;
        internal const int ForestLastLocked = 16;

        // Last recorded deflection in nanoradians. Replaced by NoteDeflection, without averaging.
        internal const int ForestLastDeflection = 17;

        /// <summary>
        /// Time spent after the initial forest solve computing body impulses and loads on locked pairs from the section
        /// results.
        /// </summary>
        internal const int ForestSettleTime = 18;

        /// <summary>
        /// Counters for lock changes on pairs with a contact on either body, and the maximum boundary row count among
        /// fallback solves.
        /// </summary>
        internal const int ForestContactLockChanges = 19;
        internal const int ForestFallbackBoundaryMax = 20;

        /// <summary>
        /// Time in timestamp ticks spent in serial portions of the forest solver path. See ForestRun.ClassifyTicks.
        /// </summary>
        internal const int ForestClassifyTime = 21;
        internal const int ForestDecideTime = 22;
        internal const int ForestSumsTime = 23;
        internal const int ForestBodyImpulseTime = 24;
        internal const int ForestSubtreeTime = 25;
        const int CounterCount = 26;

        internal static ForestCounters Create()
        {
            var counters = new ForestCounters();
            counters.forestCounters = (long*)UnsafeUtility.MallocTracked(CounterCount * sizeof(long), UnsafeUtility.AlignOf<long>(),
                Allocator.Persistent, 0);
            UnsafeUtility.MemClear(counters.forestCounters, CounterCount * sizeof(long));
            return counters;
        }

        /// <summary>
        /// Returns counts for forest solves, sparse fallbacks and forest structure builds since the cache was created.
        /// </summary>
        public void GetForestCounts(out long solves, out long fallbacks, out long builds)
        {
            solves = forestCounters != null ? forestCounters[ForestSolves] : 0;
            fallbacks = forestCounters != null ? forestCounters[ForestFallbacks] : 0;
            builds = forestCounters != null ? forestCounters[ForestBuilds] : 0;
        }

        /// <summary>
        /// Returns total forest preparation, solve and finish times across islands and substeps, in milliseconds.
        /// Factorization and boundary construction are parts of preparation.
        /// </summary>
        public void GetSolveTimes(out double forestStart, out double forestSolve, out double forestFinish,
            out double forestFactor, out double forestBoundary)
        {
            double scale = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6;
            forestStart = forestCounters != null ? forestCounters[ForestStartTime] * scale : 0;
            forestSolve = forestCounters != null ? forestCounters[ForestSolveTime] * scale : 0;
            forestFinish = forestCounters != null ? forestCounters[ForestFinishTime] * scale : 0;
            forestFactor = forestCounters != null ? forestCounters[ForestFactorTime] * scale : 0;
            forestBoundary = forestCounters != null ? forestCounters[ForestBoundaryTime] * scale : 0;
        }

        /// <summary>
        /// Returns milliseconds spent computing body impulses and loads after the initial forest solve. This time is
        /// also included in the forest finish time.
        /// </summary>
        public double ForestSettleMilliseconds => forestCounters != null ? forestCounters[ForestSettleTime]
            * (Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6) : 0;

        /// <summary>
        /// Returns milliseconds recorded for correcting predicted drift in the forest path. This time is also included
        /// in the forest finish time.
        /// </summary>
        public double ForestDriftMilliseconds => forestCounters != null ? forestCounters[ForestDriftTime]
            * (Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6) : 0;

        internal void AddTime(int counter, long ticks)
        {
            if (forestCounters != null)
            {
                System.Threading.Interlocked.Add(ref forestCounters[counter], ticks);
            }
        }

        internal void CountSections(int sections, int locked, bool rebuilt, int lockChanges, long structureTicks)
        {
            if (forestCounters == null)
            {
                return;
            }

            System.Threading.Interlocked.Add(ref forestCounters[ForestSectionSum], sections);
            System.Threading.Interlocked.Add(ref forestCounters[ForestLockedSum], locked);
            System.Threading.Interlocked.Add(ref forestCounters[ForestLockChanges], lockChanges);
            System.Threading.Interlocked.Exchange(ref forestCounters[ForestLastSections], sections);
            System.Threading.Interlocked.Exchange(ref forestCounters[ForestLastLocked], locked);
            if (rebuilt)
            {
                System.Threading.Interlocked.Increment(ref forestCounters[ForestStructureBuilds]);
                System.Threading.Interlocked.Add(ref forestCounters[ForestStructureTime], structureTicks);
            }
        }

        internal void AddCount(int counter, long value)
        {
            if (forestCounters != null)
            {
                System.Threading.Interlocked.Add(ref forestCounters[counter], value);
            }
        }

        internal void NoteMax(int counter, long value)
        {
            if (forestCounters == null)
            {
                return;
            }

            long seen = forestCounters[counter];
            while (value > seen)
            {
                long was = System.Threading.Interlocked.CompareExchange(ref forestCounters[counter], value, seen);
                if (was == seen)
                {
                    break;
                }

                seen = was;
            }
        }

        /// <summary>
        /// Returns a forest counter by index for tests and benchmarks.
        /// </summary>
        internal long ForestCounter(int counter) => forestCounters != null ? forestCounters[counter] : 0;

        /// <summary>
        /// Returns a forest time counter converted to milliseconds.
        /// </summary>
        internal double ForestMilliseconds(int counter) => ForestCounter(counter)
            * (Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6);

        internal void NoteDeflection(double radians)
        {
            if (forestCounters != null)
            {
                System.Threading.Interlocked.Exchange(ref forestCounters[ForestLastDeflection], (long)math.min(radians * 1e9, 9e18));
            }
        }

        /// <summary>
        /// Returns the most recently recorded deflection in radians, or zero if no deflection has been recorded.
        /// This property does not compute a maximum or an average over a step.
        /// </summary>
        public double LastDeflection => forestCounters != null ? forestCounters[ForestLastDeflection] * 1e-9 : 0;

        internal void NoteResidual(double residual)
        {
            if (forestCounters == null)
            {
                return;
            }

            long value = (long)math.min(residual * 1e9, 9e18);
            long seen = forestCounters[ForestResidual];
            while (value > seen)
            {
                long was = System.Threading.Interlocked.CompareExchange(ref forestCounters[ForestResidual], value, seen);
                if (was == seen)
                {
                    break;
                }

                seen = was;
            }
        }

        /// <summary>
        /// Return cumulative section/lock counts, rebuilds, lock changes, rebuild time and maximum inverse-dynamics
        /// residual.
        /// Also return counts from the last solve.
        /// </summary>
        public void GetSectionCounts(out long sectionSum, out long lockedSum, out long structureBuilds, out long lockChanges,
            out double structureMilliseconds, out double residual, out long lastSections, out long lastLocked)
        {
            bool any = forestCounters != null;
            sectionSum = any ? forestCounters[ForestSectionSum] : 0;
            lockedSum = any ? forestCounters[ForestLockedSum] : 0;
            structureBuilds = any ? forestCounters[ForestStructureBuilds] : 0;
            lockChanges = any ? forestCounters[ForestLockChanges] : 0;
            structureMilliseconds = any ? forestCounters[ForestStructureTime]
                * (Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                    / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6) : 0;
            residual = any ? forestCounters[ForestResidual] * 1e-9 : 0;
            lastSections = any ? forestCounters[ForestLastSections] : 0;
            lastLocked = any ? forestCounters[ForestLastLocked] : 0;
        }

        internal void CountForest(int counter)
        {
            if (forestCounters != null)
            {
                System.Threading.Interlocked.Increment(ref forestCounters[counter]);
            }
        }

        internal void Dispose()
        {
            UnsafeUtility.FreeTracked(forestCounters, Allocator.Persistent);
            forestCounters = null;
        }
    }
}
