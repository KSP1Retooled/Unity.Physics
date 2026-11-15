using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Sparse solvers kept from one solve to the next, so each island can reuse what its solver worked out last time. The
    /// caller of a step owns this and passes it in <see cref="SimulationStepInput"/>. Without one, nothing is reused.
    ///
    /// An island keeps its solver by its first body, the lowest-numbered, rather than by its place among the islands: a craft
    /// that drops a stage or lands beside another keeps its body numbers, while every island after a new one moves up a place.
    /// A solver handed to another island costs a new analysis and nothing else, because a solver checks everything it reuses
    /// against the problem it is given.
    /// </summary>
    public unsafe struct DirectSolverCache : IDisposable
    {
        // Beyond this many islands the rest solve from scratch, which the craft that matter never reach.
        const int Capacity = 64;

        struct Slot
        {
            public SparseConstraintSolver Solver;

            /// <summary>The island's vessel forest, or null until a forest solve first asks for one.</summary>
            public ForestIsland* Forest;

            /// <summary>The first body of the island the solver last solved, or -1 if it has never solved one.</summary>
            public int Key;

            public bool Created;

            /// <summary>The step this slot last went to an island.</summary>
            public long LastUsed;
        }

        [NativeDisableUnsafePtrRestriction]
        Slot* slots;

        /// <summary>
        /// Each island's slot this step, or -1 for an island that solves from scratch. Empty until a step assigns slots, in
        /// which case an island's slot is its index.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* slotOfIsland;

        [NativeDisableUnsafePtrRestriction]
        long* step;

        [NativeDisableUnsafePtrRestriction]
        ForestSettings* forestSettings;

        [NativeDisableUnsafePtrRestriction]
        long* forestCounters;

        internal const int ForestSolves = 0;
        internal const int ForestFallbacks = 1;
        internal const int ForestBuilds = 2;

        /// <summary>Time spent, in profiler timestamp units: see <see cref="GetSolveTimes"/>.</summary>
        internal const int SparseTime = 3;
        internal const int ForestStartTime = 4;
        internal const int ForestSolveTime = 5;
        internal const int ForestFinishTime = 6;
        internal const int ForestFactorTime = 7;
        internal const int ForestBoundaryTime = 8;
        internal const int ForestDriftTime = 9;

        /// <summary>Sections and locked pairs summed over forest solves, structure rebuilds, lock changes, rebuild time.</summary>
        internal const int ForestSectionSum = 10;
        internal const int ForestLockedSum = 11;
        internal const int ForestStructureBuilds = 12;
        internal const int ForestLockChanges = 13;
        internal const int ForestStructureTime = 14;

        /// <summary>The largest momentum left at a section's first body by inverse dynamics, in nano N s, and the last solve's sections.</summary>
        internal const int ForestResidual = 15;
        internal const int ForestLastSections = 16;
        internal const int ForestLastLocked = 17;

        /// <summary>The largest deflection the lock tests read after the last solve (a step's mean), in nanoradians.</summary>
        internal const int ForestLastDeflection = 18;

        /// <summary>Time the forest path spent turning sections' answers into bodies' and locked pairs' loads, once per solve.</summary>
        internal const int ForestSettleTime = 19;

        /// <summary>Lock changes of pairs with a contact on either body, and the most boundary rows a fallback had.</summary>
        internal const int ForestContactLockChanges = 20;
        internal const int ForestFallbackBoundaryMax = 21;

        /// <summary>Ticks in the forest path's one-lane parts, see ForestRun.ClassifyTicks.</summary>
        internal const int ForestClassifyTime = 22;
        internal const int ForestDecideTime = 23;
        internal const int ForestSumsTime = 24;
        internal const int ForestBodyImpulseTime = 25;
        internal const int ForestSubtreeTime = 26;
        const int CounterCount = 27;

        public static DirectSolverCache Create()
        {
            var cache = new DirectSolverCache
            {
                slots = (Slot*)UnsafeUtility.MallocTracked(Capacity * sizeof(Slot), UnsafeUtility.AlignOf<Slot>(), Allocator.Persistent, 0),
                slotOfIsland = UnsafeList<int>.Create(Capacity, Allocator.Persistent),
                step = (long*)UnsafeUtility.MallocTracked(sizeof(long), UnsafeUtility.AlignOf<long>(), Allocator.Persistent, 0),
                forestSettings = (ForestSettings*)UnsafeUtility.MallocTracked(sizeof(ForestSettings), UnsafeUtility.AlignOf<ForestSettings>(),
                    Allocator.Persistent, 0),
            };

            *cache.forestSettings = ForestSettings.Default;
            cache.forestCounters = (long*)UnsafeUtility.MallocTracked(CounterCount * sizeof(long), UnsafeUtility.AlignOf<long>(),
                Allocator.Persistent, 0);
            UnsafeUtility.MemClear(cache.forestCounters, CounterCount * sizeof(long));

            UnsafeUtility.MemClear(cache.slots, Capacity * sizeof(Slot));
            for (int slot = 0; slot < Capacity; slot++)
            {
                cache.slots[slot].Key = -1;
                cache.slots[slot].LastUsed = -1;
            }

            *cache.step = 0;
            return cache;
        }

        internal bool IsCreated => slots != null;

        /// <summary>
        /// Whether, and how, direct islands solve as a forest plus a boundary. A default cache has no forest. Set between
        /// steps; the setting reaches every copy of the cache.
        /// </summary>
        public ForestSettings Forest
        {
            get => forestSettings != null ? *forestSettings : default;
            set
            {
                if (forestSettings != null)
                {
                    *forestSettings = value;
                }
            }
        }

        /// <summary>
        /// The forest an island keeps between solves, or null if there is none to keep. Like <see cref="Solver"/>, an island
        /// asks only for its own slot's, so islands solving in parallel never share one.
        /// </summary>
        /// <summary>
        /// Since the cache was made: how many island solves took the forest path, how many forest-enabled solves fell back
        /// to the sparse solver, and how many times a forest structure was built.
        /// </summary>
        public void GetForestCounts(out long solves, out long fallbacks, out long builds)
        {
            solves = forestCounters != null ? forestCounters[ForestSolves] : 0;
            fallbacks = forestCounters != null ? forestCounters[ForestFallbacks] : 0;
            builds = forestCounters != null ? forestCounters[ForestBuilds] : 0;
        }

        /// <summary>
        /// Milliseconds spent since the cache was made, summed over islands and substeps: sparse solves from their start
        /// to their impulses applied, and the forest path's start (classify, factor, boundary block), its solves, and its
        /// finish (drift, application, push-out), with the factor and boundary block of the start apart.
        /// </summary>
        public void GetSolveTimes(out double sparse, out double forestStart, out double forestSolve, out double forestFinish,
            out double forestFactor, out double forestBoundary)
        {
            double scale = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6;
            sparse = forestCounters != null ? forestCounters[SparseTime] * scale : 0;
            forestStart = forestCounters != null ? forestCounters[ForestStartTime] * scale : 0;
            forestSolve = forestCounters != null ? forestCounters[ForestSolveTime] * scale : 0;
            forestFinish = forestCounters != null ? forestCounters[ForestFinishTime] * scale : 0;
            forestFactor = forestCounters != null ? forestCounters[ForestFactorTime] * scale : 0;
            forestBoundary = forestCounters != null ? forestCounters[ForestBoundaryTime] * scale : 0;
        }

        /// <summary>Milliseconds the forest path spent settling its sections after its first solve, inside its finish.</summary>
        public double ForestSettleMilliseconds => forestCounters != null ? forestCounters[ForestSettleTime]
            * (Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6) : 0;

        /// <summary>Milliseconds the forest path spent aiming at the predicted drift, inside its finish.</summary>
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

        /// <summary>A forest counter by its index, for tests and benches.</summary>
        internal long ForestCounter(int counter) => forestCounters != null ? forestCounters[counter] : 0;

        /// <summary>A forest time counter in milliseconds.</summary>
        internal double ForestMilliseconds(int counter) => ForestCounter(counter)
            * (Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6);

        /// <summary>
        /// For diagnostics, every kept forest's tree pairs by their deflection against the lock and yield angles: bins
        /// below half the lock angle, below it, below the yield angle, below twice it, and past it; each split by whether
        /// a contact touches the pair (index + 5) and counted again in <paramref name="locked"/> if the pair is locked.
        /// </summary>
        internal void DeflectionHistogram(float lockAngle, float yieldAngle, long* all, long* locked)
        {
            for (int i = 0; i < 10; i++)
            {
                all[i] = 0;
                locked[i] = 0;
            }

            for (int slot = 0; slots != null && slot < Capacity; slot++)
            {
                ForestIsland* forest = slots[slot].Forest;
                if (forest == null || slots[slot].LastUsed != *step)
                {
                    continue;
                }

                for (int p = 0; p < forest->PairCount; p++)
                {
                    if (forest->PairEdge[p] == 0 || forest->PairTree[p] == 0)
                    {
                        continue;
                    }

                    float d = forest->PairDeflection[p];
                    int bin = d < 0.5f * lockAngle ? 0 : d < lockAngle ? 1 : d < yieldAngle ? 2 : d < 2.0f * yieldAngle ? 3 : 4;
                    bin += forest->PairContact[p] != 0 ? 5 : 0;
                    all[bin]++;
                    locked[bin] += forest->PairLocked[p];
                }
            }
        }

        internal void NoteDeflection(double radians)
        {
            if (forestCounters != null)
            {
                System.Threading.Interlocked.Exchange(ref forestCounters[ForestLastDeflection], (long)math.min(radians * 1e9, 9e18));
            }
        }

        /// <summary>The largest deflection the lock tests read after the last forest solve (a step's mean), in radians.</summary>
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
        /// Since the cache was made: sections and locked pairs summed over forest solves, structure rebuilds, lock changes,
        /// milliseconds spent rebuilding, the largest inverse-dynamics residual in N s, and the last solve's sections and
        /// locked pairs.
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

        /// <summary>
        /// Forgets every island's forest: its pairs, lock states and structure. An island keeps them by its first body, so a
        /// new scene of the same shape would otherwise start with the last one's; tests that compare runs call this.
        /// </summary>
        public void ForgetForests()
        {
            for (int slot = 0; slots != null && slot < Capacity; slot++)
            {
                ForestIsland.Destroy(slots[slot].Forest);
                slots[slot].Forest = null;
            }
        }

        internal ForestIsland* ForestOf(int island)
        {
            if (slots == null)
            {
                return null;
            }

            int slot = slotOfIsland->Length > 0 ? (island < slotOfIsland->Length ? (*slotOfIsland)[island] : -1) : island;
            if (slot < 0 || slot >= Capacity)
            {
                return null;
            }

            if (slots[slot].Forest == null)
            {
                slots[slot].Forest = ForestIsland.Create();
            }

            return slots[slot].Forest;
        }

        /// <summary>
        /// Gives every island of a step its slot, from the first body of each. Runs on one thread before any island solves.
        /// An island takes back the slot its first body had, and the rest take the slots longest out of use.
        /// </summary>
        internal void Assign(NativeArray<int> firstBodies)
        {
            if (slots == null)
            {
                return;
            }

            long now = ++*step;
            int islandCount = firstBodies.Length;
            slotOfIsland->Resize(islandCount);

            for (int island = 0; island < islandCount; island++)
            {
                int found = -1;
                for (int slot = 0; slot < Capacity; slot++)
                {
                    if (slots[slot].Key == firstBodies[island])
                    {
                        found = slot;
                        break;
                    }
                }

                (*slotOfIsland)[island] = found;
                if (found >= 0)
                {
                    slots[found].LastUsed = now;
                }
            }

            for (int island = 0; island < islandCount; island++)
            {
                if ((*slotOfIsland)[island] >= 0)
                {
                    continue;
                }

                int oldest = -1;
                for (int slot = 0; slot < Capacity; slot++)
                {
                    if (slots[slot].LastUsed != now && (oldest < 0 || slots[slot].LastUsed < slots[oldest].LastUsed))
                    {
                        oldest = slot;
                    }
                }

                if (oldest < 0)
                {
                    continue;
                }

                slots[oldest].Key = firstBodies[island];
                slots[oldest].LastUsed = now;
                (*slotOfIsland)[island] = oldest;
            }
        }

        /// <summary>
        /// The solver for an island, or null if there is none to keep. Islands solve in parallel, so this creates a solver
        /// only for the island asking, and no two islands ever share a slot.
        /// </summary>
        internal SparseConstraintSolver* Solver(int island)
        {
            if (slots == null)
            {
                return null;
            }

            int slot = slotOfIsland->Length > 0 ? (island < slotOfIsland->Length ? (*slotOfIsland)[island] : -1) : island;
            if (slot < 0 || slot >= Capacity)
            {
                return null;
            }

            if (!slots[slot].Created)
            {
                slots[slot].Solver = new SparseConstraintSolver(Allocator.Persistent);
                slots[slot].Created = true;
            }

            return &slots[slot].Solver;
        }

        public void Dispose()
        {
            if (slots == null)
            {
                return;
            }

            for (int slot = 0; slot < Capacity; slot++)
            {
                if (slots[slot].Created)
                {
                    slots[slot].Solver.Dispose();
                }

                ForestIsland.Destroy(slots[slot].Forest);
            }

            UnsafeUtility.FreeTracked(forestSettings, Allocator.Persistent);
            UnsafeUtility.FreeTracked(forestCounters, Allocator.Persistent);
            forestSettings = null;
            forestCounters = null;

            UnsafeUtility.FreeTracked(slots, Allocator.Persistent);
            UnsafeUtility.FreeTracked(step, Allocator.Persistent);
            UnsafeList<int>.Destroy(slotOfIsland);
            slots = null;
            slotOfIsland = null;
            step = null;
        }
    }
}
