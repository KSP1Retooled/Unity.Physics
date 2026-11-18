using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Sparse;

namespace Unity.Physics.Dynamics.Solver.Islands
{
    /// <summary>
    /// Cache island solvers across steps, keyed by first-body identity rather than changing island indices.
    /// The caller owns the cache and passes it through SimulationStepInput.
    /// Solvers validate analysis before reuse; forests retain locks by body identity.
    /// </summary>
    internal unsafe struct IslandSolverCache : IDisposable
    {
        // Initial capacity of the substep island slot table. Grow when all slots were used recently.
        const int InitialCapacity = 64;

        // Maximum slot capacity. At capacity, reuse the least recently used unclaimed slot or solve without
        // caching.
        const int MaxSlots = 4096;

        // A slot that no step has used for this many steps can go to a new island. An island that went to the fused job
        // for a step or two, or slept for a moment, keeps its slot.
        internal const int SlotKeepSteps = 60;

        // The step in which a slot that was never assigned was last used. Such a slot can always go to a new island.
        const long NeverUsed = long.MinValue;

        /// <summary>
        /// Sentinel identity for islands that need no cache slot; no body uses this value.
        /// </summary>
        internal const ulong NoSlot = ulong.MaxValue;

        struct Slot
        {
            public SparseConstraintSolver Solver;

            /// <summary>
            /// The island's vessel forest, created when first needed and null until then.
            /// </summary>
            public ForestIsland* Forest;

            /// <summary>
            /// The first body of the island last assigned to this slot, or -1 if the slot has never been assigned.
            /// </summary>
            public int Key;

            /// <summary>
            /// Stable first-body identity assigned to the slot, or ulong.MaxValue if unused.
            /// </summary>
            public ulong Identity;

            public bool Created;

            /// <summary>
            /// The step in which this slot was last assigned to an island, or <see cref="NeverUsed"/>.
            /// </summary>
            public long LastUsed;
        }

        /// <summary>
        /// Substep solver slots. Assign grows the table serially before solving, keeping storage stable during
        /// parallel access.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<Slot>* slots;

        /// <summary>
        /// Map first-body identities to their previous slots. Only Assign modifies it.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeHashMap<ulong, int>* slotOfIdentity;

        /// <summary>A slot that holds nothing yet.</summary>
        static Slot EmptySlot => new Slot { Key = -1, Identity = NoSlot, LastUsed = NeverUsed };

        /// <summary>
        /// An unclaimed slot and its last-use step. Sort oldest first, breaking ties by slot index.
        /// </summary>
        struct FreeSlot : IComparable<FreeSlot>
        {
            public long LastUsed;
            public int Slot;

            public int CompareTo(FreeSlot other) => LastUsed != other.LastUsed ? LastUsed.CompareTo(other.LastUsed) : Slot.CompareTo(other.Slot);
        }

        // Initial number of fused vessel slots. Grow when all slots were used recently.
        const int FusedInitialCapacity = 256;

        // A slot whose vessel no step has solved for this many steps can go to a new vessel. A vessel that went to the
        // substeps' jobs for a step or two, or slept for a moment, keeps its slot.
        internal const int FusedKeepSteps = 60;

        /// <summary>
        /// Persistent fused island state, keyed by its first vessel ID rather than island or body index.
        /// </summary>
        struct FusedSlot
        {
            public ForestIsland* Forest;

            public ulong VesselId;

            /// <summary>The step in which the slot was last used.</summary>
            public long LastUsed;
        }

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<FusedSlot>* fusedSlots;

        /// <summary>
        /// Map vessel IDs to fused slots. Populate before fused jobs read it concurrently.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeHashMap<ulong, int>* fusedSlotOfVessel;

        /// <summary>
        /// Forest holding a vessel's newest lock states and the step when ownership was assigned.
        /// Copy these states when the vessel moves between fused or substep forests.
        /// </summary>
        internal struct VesselOwner
        {
            public ForestIsland* Forest;
            public long Step;
        }

        [NativeDisableUnsafePtrRestriction]
        UnsafeHashMap<ulong, VesselOwner>* vesselOwners;

        /// <summary>
        /// Shared world-body to island-body index, with -1 for entries not currently in use.
        /// Parallel islands write disjoint body entries and reset them to -1 after solving.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* compactOfWorld;

        /// <summary>
        /// When the record of owners holds more vessels than this, the vessels that no step has solved for
        /// <see cref="OwnerSteps"/> steps are dropped from it.
        /// </summary>
        const int OwnerCapacity = 1024;

        const int OwnerSteps = 600;

        /// <summary>
        /// Cache slot per island, or -1 if uncached. Before assignment, use the island index as a slot index.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* slotOfIsland;

        [NativeDisableUnsafePtrRestriction]
        long* step;

        /// <summary>
        /// Counts of unfused islands, first-substep rows, bodies and shareable solves, plus fused vessel part
        /// count.
        /// </summary>
        internal struct UnfusedWork
        {
            public int Islands;
            public int Rows;
            public int Bodies;
            public int TeamIslands;
            public int FusedParts;

            /// <summary>The step that the counts are from, or 0 if no step has recorded them.</summary>
            public long Step;
        }

        [NativeDisableUnsafePtrRestriction]
        UnfusedWork* unfusedWork;

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

        /// <summary>
        /// Counters for total sections and locked pairs across forest solves, structure rebuilds, lock changes, and
        /// time spent rebuilding.
        /// </summary>
        internal const int ForestSectionSum = 10;
        internal const int ForestLockedSum = 11;
        internal const int ForestStructureBuilds = 12;
        internal const int ForestLockChanges = 13;
        internal const int ForestStructureTime = 14;

        // Maximum inverse-dynamics momentum residual in units of 1e-9 N s, plus last-solve section/lock counts.
        internal const int ForestResidual = 15;
        internal const int ForestLastSections = 16;
        internal const int ForestLastLocked = 17;

        // Last recorded deflection in nanoradians. Replaced by NoteDeflection, without averaging.
        internal const int ForestLastDeflection = 18;

        /// <summary>
        /// Time spent after the initial forest solve computing body impulses and loads on locked pairs from the section
        /// results.
        /// </summary>
        internal const int ForestSettleTime = 19;

        /// <summary>
        /// Counters for lock changes on pairs with a contact on either body, and the maximum boundary row count among
        /// fallback solves.
        /// </summary>
        internal const int ForestContactLockChanges = 20;
        internal const int ForestFallbackBoundaryMax = 21;

        /// <summary>
        /// Time in timestamp ticks spent in serial portions of the forest solver path. See ForestRun.ClassifyTicks.
        /// </summary>
        internal const int ForestClassifyTime = 22;
        internal const int ForestDecideTime = 23;
        internal const int ForestSumsTime = 24;
        internal const int ForestBodyImpulseTime = 25;
        internal const int ForestSubtreeTime = 26;
        const int CounterCount = 27;

        public static IslandSolverCache Create()
        {
            var cache = new IslandSolverCache
            {
                slots = UnsafeList<Slot>.Create(InitialCapacity, Allocator.Persistent),
                slotOfIsland = UnsafeList<int>.Create(InitialCapacity, Allocator.Persistent),
                step = (long*)UnsafeUtility.MallocTracked(sizeof(long), UnsafeUtility.AlignOf<long>(), Allocator.Persistent, 0),
                forestSettings = (ForestSettings*)UnsafeUtility.MallocTracked(sizeof(ForestSettings), UnsafeUtility.AlignOf<ForestSettings>(),
                    Allocator.Persistent, 0),
            };

            *cache.forestSettings = ForestSettings.Default;
            cache.forestCounters = (long*)UnsafeUtility.MallocTracked(CounterCount * sizeof(long), UnsafeUtility.AlignOf<long>(),
                Allocator.Persistent, 0);
            UnsafeUtility.MemClear(cache.forestCounters, CounterCount * sizeof(long));

            // Make the first slots at once, so that a solve that runs before any assignment can use the island's index
            // as its slot, as Solver and ForestOf allow.
            for (int slot = 0; slot < InitialCapacity; slot++)
            {
                cache.slots->Add(EmptySlot);
            }

            cache.slotOfIdentity = (UnsafeHashMap<ulong, int>*)UnsafeUtility.MallocTracked(UnsafeUtility.SizeOf<UnsafeHashMap<ulong, int>>(),
                UnsafeUtility.AlignOf<UnsafeHashMap<ulong, int>>(), Allocator.Persistent, 0);
            *cache.slotOfIdentity = new UnsafeHashMap<ulong, int>(InitialCapacity, Allocator.Persistent);

            cache.fusedSlots = UnsafeList<FusedSlot>.Create(FusedInitialCapacity, Allocator.Persistent);
            cache.fusedSlotOfVessel = (UnsafeHashMap<ulong, int>*)UnsafeUtility.MallocTracked(UnsafeUtility.SizeOf<UnsafeHashMap<ulong, int>>(),
                UnsafeUtility.AlignOf<UnsafeHashMap<ulong, int>>(), Allocator.Persistent, 0);
            *cache.fusedSlotOfVessel = new UnsafeHashMap<ulong, int>(FusedInitialCapacity, Allocator.Persistent);
            cache.vesselOwners = (UnsafeHashMap<ulong, VesselOwner>*)UnsafeUtility.MallocTracked(UnsafeUtility.SizeOf<UnsafeHashMap<ulong, VesselOwner>>(),
                UnsafeUtility.AlignOf<UnsafeHashMap<ulong, VesselOwner>>(), Allocator.Persistent, 0);
            *cache.vesselOwners = new UnsafeHashMap<ulong, VesselOwner>(64, Allocator.Persistent);
            cache.compactOfWorld = UnsafeList<int>.Create(0, Allocator.Persistent);
            *cache.step = 0;
            cache.unfusedWork = (UnfusedWork*)UnsafeUtility.MallocTracked(sizeof(UnfusedWork), UnsafeUtility.AlignOf<UnfusedWork>(),
                Allocator.Persistent, 0);
            *cache.unfusedWork = default;
            return cache;
        }

        internal bool IsCreated => slots != null;

        /// <summary>
        /// Forest/boundary settings shared by all cache copies. Set between steps; default leaves forests disabled.
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
        /// Returns counts since cache creation: island solves that used the forest path, solves with forests enabled
        /// that fell back to the sparse solver, and forest structure builds.
        /// </summary>
        public void GetForestCounts(out long solves, out long fallbacks, out long builds)
        {
            solves = forestCounters != null ? forestCounters[ForestSolves] : 0;
            fallbacks = forestCounters != null ? forestCounters[ForestFallbacks] : 0;
            builds = forestCounters != null ? forestCounters[ForestBuilds] : 0;
        }

        /// <summary>
        /// Cumulative elapsed milliseconds across islands and substeps for sparse solving and forest preparation,
        /// solve and finish.
        /// Return factorization and boundary construction separately as subsets of preparation.
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

        /// <summary>
        /// Count edges in five deflection bins: below half-lock, lock, yield, twice-yield, and the rest.
        /// Use bins 5?9 for contact pairs. Count all pairs in all and locked pairs in locked.
        /// </summary>
        internal void DeflectionHistogram(float lockAngle, float yieldAngle, long* all, long* locked)
        {
            for (int i = 0; i < 10; i++)
            {
                all[i] = 0;
                locked[i] = 0;
            }

            for (int slot = 0; slots != null && slot < slots->Length; slot++)
            {
                ForestIsland* forest = slots->Ptr[slot].Forest;
                if (forest == null || slots->Ptr[slot].LastUsed != *step)
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

        /// <summary>
        /// Discard all cached forests, including lock states and structures. Used between independent test scenes.
        /// </summary>
        public void ForgetForests()
        {
            for (int slot = 0; slots != null && slot < slots->Length; slot++)
            {
                ForestIsland.Destroy(slots->Ptr[slot].Forest);
                slots->Ptr[slot].Forest = null;
            }

            for (int slot = 0; fusedSlots != null && slot < fusedSlots->Length; slot++)
            {
                ForestIsland.Destroy(fusedSlots->Ptr[slot].Forest);
            }

            if (fusedSlots != null)
            {
                fusedSlots->Clear();
            }

            if (fusedSlotOfVessel != null)
            {
                fusedSlotOfVessel->Clear();
            }

            if (vesselOwners != null)
            {
                vesselOwners->Clear();
            }
        }

        /// <summary>Returns the forest that holds the newest lock states of a vessel, or null if no forest holds them.</summary>
        internal ForestIsland* OwnerOf(ulong vesselId)
        {
            return vesselOwners != null && vesselOwners->TryGetValue(vesselId, out VesselOwner owner) ? owner.Forest : null;
        }

        /// <summary>
        /// Record the persistent forest holding a vessel's current locks, or forget temporary forests.
        /// </summary>
        internal void SetOwner(ulong vesselId, ForestIsland* forest)
        {
            if (vesselOwners == null)
            {
                return;
            }

            if (forest == null)
            {
                vesselOwners->Remove(vesselId);
                return;
            }

            (*vesselOwners)[vesselId] = new VesselOwner { Forest = forest, Step = *step };
        }

        /// <summary>
        /// Forget vessel ownership before its forest is destroyed. Without a forest argument, prune old entries
        /// only when the table is large.
        /// </summary>
        void ForgetOwners(ForestIsland* destroyed)
        {
            if (vesselOwners == null || (destroyed == null && vesselOwners->Count <= OwnerCapacity))
            {
                return;
            }

            long now = *step;
            var dropped = new NativeList<ulong>(Allocator.Temp);
            foreach (KVPair<ulong, VesselOwner> pair in *vesselOwners)
            {
                VesselOwner owner = pair.Value;
                if (destroyed != null ? owner.Forest == destroyed : owner.Step < now - OwnerSteps)
                {
                    dropped.Add(pair.Key);
                }
            }

            for (int i = 0; i < dropped.Length; i++)
            {
                vesselOwners->Remove(dropped[i]);
            }

            dropped.Dispose();
        }

        /// <summary>Forgets the vessels that no step has solved for a long time, once the record has grown large.</summary>
        internal void PruneOwners() => ForgetOwners(null);

        /// <summary>
        /// Find a cached fused forest by first vessel ID and mark its slot used this step, or return null.
        /// </summary>
        internal ForestIsland* FindFusedForest(ulong vesselId)
        {
            if (fusedSlots == null || !fusedSlotOfVessel->TryGetValue(vesselId, out int slot))
            {
                return null;
            }

            fusedSlots->Ptr[slot].LastUsed = *step;
            return fusedSlots->Ptr[slot].Forest;
        }

        /// <summary>
        /// Size the shared compact index for motionCount and reset all entries to -1 before island solving.
        /// </summary>
        internal void ClearCompactIndex(int motionCount)
        {
            if (compactOfWorld == null)
            {
                return;
            }

            if (compactOfWorld->Length < motionCount)
            {
                compactOfWorld->Resize(motionCount, NativeArrayOptions.UninitializedMemory);
            }

            UnsafeUtility.MemSet(compactOfWorld->Ptr, 0xFF, compactOfWorld->Length * (long)sizeof(int));
        }

        /// <summary>
        /// Return the shared compact index, or null if absent or too small. Null callers allocate their own index.
        /// </summary>
        internal int* CompactIndex(int motionCount)
        {
            return compactOfWorld != null && compactOfWorld->Length >= motionCount ? compactOfWorld->Ptr : null;
        }

        /// <summary>
        /// Find or create a fused forest by first vessel ID.
        /// Reuse slots unused for FusedKeepSteps, otherwise grow. Call serially before solving any islands.
        /// </summary>
        internal ForestIsland* FusedForestOf(ulong vesselId)
        {
            if (fusedSlots == null)
            {
                return null;
            }

            long now = *step;
            FusedSlot* table = fusedSlots->Ptr;
            if (!fusedSlotOfVessel->TryGetValue(vesselId, out int found))
            {
                // Only a vessel that has no slot yet looks through the slots, for the one that was used least recently.
                int oldest = -1;
                for (int slot = 0; slot < fusedSlots->Length; slot++)
                {
                    if (oldest < 0 || table[slot].LastUsed < table[oldest].LastUsed)
                    {
                        oldest = slot;
                    }
                }

                if (oldest >= 0 && table[oldest].LastUsed < now - FusedKeepSteps)
                {
                    // The slot's old forest belongs to another vessel, so start again from an empty one.
                    found = oldest;
                    fusedSlotOfVessel->Remove(table[found].VesselId);
                    ForgetOwners(table[found].Forest);
                    ForestIsland.Destroy(table[found].Forest);
                }
                else
                {
                    found = fusedSlots->Length;
                    fusedSlots->Add(default);
                    table = fusedSlots->Ptr;
                }

                table[found].Forest = ForestIsland.Create();
                table[found].VesselId = vesselId;
                fusedSlotOfVessel->Add(vesselId, found);
            }

            table[found].LastUsed = now;
            return table[found].Forest;
        }

        /// <summary>The number of slots that keep the state of fused vessel islands. Tests read it.</summary>
        internal int FusedSlotCount => fusedSlots != null ? fusedSlots->Length : 0;

        /// <summary>
        /// Return or create the forest in an island's assigned slot, or null if uncached.
        /// Parallel islands use distinct slots.
        /// </summary>
        internal ForestIsland* ForestOf(int island)
        {
            if (slots == null)
            {
                return null;
            }

            int slot = SlotOf(island);
            if (slot < 0)
            {
                return null;
            }

            Slot* entry = slots->Ptr + slot;
            if (entry->Forest == null)
            {
                entry->Forest = ForestIsland.Create();
            }

            return entry->Forest;
        }

        /// <summary>
        /// Assign slots by first-body identity before parallel solving. NoSlot islands use fused caches instead.
        /// Preserve existing matches, then reuse old slots or grow up to MaxSlots. Never assign a slot twice in one
        /// step.
        /// </summary>
        internal void Assign(NativeArray<int> firstBodies, NativeArray<ulong> firstIdentities)
        {
            if (slots == null)
            {
                return;
            }

            long now = ++*step;
            int islandCount = firstBodies.Length;
            slotOfIsland->Resize(islandCount);

            // An island that needs a slot and has none yet is marked with this value until the second pass below gives
            // it one.
            const int unmatched = -2;
            int unmatchedCount = 0;
            Slot* table = slots->Ptr;
            for (int island = 0; island < islandCount; island++)
            {
                ulong identity = firstIdentities[island];
                int found = -1;
                if (identity != NoSlot)
                {
                    if (slotOfIdentity->TryGetValue(identity, out int slot) && table[slot].LastUsed != now)
                    {
                        found = slot;
                        table[slot].LastUsed = now;
                    }
                    else
                    {
                        found = unmatched;
                        unmatchedCount++;
                    }
                }

                (*slotOfIsland)[island] = found;
            }

            if (unmatchedCount == 0)
            {
                return;
            }

            // The slots that no island of this step holds, the least recently used first.
            var free = new NativeList<FreeSlot>(slots->Length, Allocator.Temp);
            for (int slot = 0; slot < slots->Length; slot++)
            {
                if (table[slot].LastUsed != now)
                {
                    free.Add(new FreeSlot { LastUsed = table[slot].LastUsed, Slot = slot });
                }
            }

            free.Sort();
            int nextFree = 0;
            for (int island = 0; island < islandCount; island++)
            {
                if ((*slotOfIsland)[island] != unmatched)
                {
                    continue;
                }

                // The least recently used free slot goes to the island if no step has used it for a while, or if the
                // table can't grow. Otherwise the island gets a new slot.
                int chosen = -1;
                bool full = slots->Length >= MaxSlots;
                if (nextFree < free.Length && (free[nextFree].LastUsed < now - SlotKeepSteps || full))
                {
                    chosen = free[nextFree++].Slot;
                    ulong previous = table[chosen].Identity;
                    if (previous != NoSlot && slotOfIdentity->TryGetValue(previous, out int mapped) && mapped == chosen)
                    {
                        slotOfIdentity->Remove(previous);
                    }
                }
                else if (!full)
                {
                    chosen = slots->Length;
                    slots->Add(EmptySlot);
                    table = slots->Ptr;
                }

                (*slotOfIsland)[island] = chosen;
                if (chosen < 0)
                {
                    continue;
                }

                table[chosen].Key = firstBodies[island];
                table[chosen].Identity = firstIdentities[island];
                table[chosen].LastUsed = now;

                // When two islands of a step have the same identity, the map keeps the slot of the first one.
                slotOfIdentity->TryAdd(firstIdentities[island], chosen);
            }

            free.Dispose();
        }

        /// <summary>
        /// Record unfused substep workload after slot assignment.
        /// </summary>
        internal void NoteUnfusedWork(UnfusedWork work)
        {
            if (unfusedWork == null)
            {
                return;
            }

            work.Step = *step;
            *unfusedWork = work;
        }

        /// <summary>
        /// Return the last recorded unfused workload, or false if absent. Read after previous-step jobs complete.
        /// </summary>
        internal bool LastUnfusedWork(out UnfusedWork work)
        {
            work = unfusedWork != null ? *unfusedWork : default;
            return work.Step > 0;
        }

        /// <summary>The number of slots in the table of the substeps' islands. Tests read it.</summary>
        internal int SlotCount => slots != null ? slots->Length : 0;

        /// <summary>
        /// The slot of an island in this step, or -1 if the island has none. Before any assignment, the island's index
        /// is its slot.
        /// </summary>
        int SlotOf(int island)
        {
            int slot = slotOfIsland->Length > 0 ? (island < slotOfIsland->Length ? (*slotOfIsland)[island] : -1) : island;
            return slot >= 0 && slot < slots->Length ? slot : -1;
        }

        /// <summary>
        /// Return or create the solver in an island's assigned slot, or null if uncached.
        /// Parallel islands use distinct slots.
        /// </summary>
        internal SparseConstraintSolver* Solver(int island)
        {
            if (slots == null)
            {
                return null;
            }

            int slot = SlotOf(island);
            if (slot < 0)
            {
                return null;
            }

            Slot* entry = slots->Ptr + slot;
            if (!entry->Created)
            {
                entry->Solver = new SparseConstraintSolver(Allocator.Persistent);
                entry->Created = true;
            }

            return &entry->Solver;
        }

        public void Dispose()
        {
            if (slots == null)
            {
                return;
            }

            for (int slot = 0; slot < slots->Length; slot++)
            {
                if (slots->Ptr[slot].Created)
                {
                    slots->Ptr[slot].Solver.Dispose();
                }

                ForestIsland.Destroy(slots->Ptr[slot].Forest);
            }

            slotOfIdentity->Dispose();
            UnsafeUtility.FreeTracked(slotOfIdentity, Allocator.Persistent);
            slotOfIdentity = null;

            for (int slot = 0; slot < fusedSlots->Length; slot++)
            {
                ForestIsland.Destroy(fusedSlots->Ptr[slot].Forest);
            }

            UnsafeList<FusedSlot>.Destroy(fusedSlots);
            fusedSlots = null;
            fusedSlotOfVessel->Dispose();
            UnsafeUtility.FreeTracked(fusedSlotOfVessel, Allocator.Persistent);
            fusedSlotOfVessel = null;
            vesselOwners->Dispose();
            UnsafeUtility.FreeTracked(vesselOwners, Allocator.Persistent);
            vesselOwners = null;
            UnsafeList<int>.Destroy(compactOfWorld);
            compactOfWorld = null;
            UnsafeUtility.FreeTracked(forestSettings, Allocator.Persistent);
            UnsafeUtility.FreeTracked(unfusedWork, Allocator.Persistent);
            unfusedWork = null;
            UnsafeUtility.FreeTracked(forestCounters, Allocator.Persistent);
            forestSettings = null;
            forestCounters = null;

            UnsafeList<Slot>.Destroy(slots);
            UnsafeUtility.FreeTracked(step, Allocator.Persistent);
            UnsafeList<int>.Destroy(slotOfIsland);
            slots = null;
            slotOfIsland = null;
            step = null;
        }
    }
}
