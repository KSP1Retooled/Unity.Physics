using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Caches sparse solvers between solves so islands can reuse previous analysis. The caller owns this cache and
    /// passes it in <see cref="SimulationStepInput"/>. Without a cache, each solve starts without results from a
    /// previous solve.
    ///
    /// Slots are matched by the identity of each island's first body (see <see cref="BodyIdentity"/>), rather than by
    /// the island's index. Adding an island can change later island indices without changing their first bodies, for
    /// example when a craft drops a stage or lands near another craft. A solver can also be assigned to a different
    /// island: it validates cached analysis against the new problem and repeats analysis when necessary.
    ///
    /// The identity is not the body's index in the world. When a body is destroyed, the next body that is made can take
    /// its index, and a living body can move to another index when other bodies go. So a slot follows the body itself:
    /// a craft keeps its slot when its bodies move, and a new craft never takes over a destroyed craft's slot as its
    /// own. A forest's pairs keep their lock states by the identities of their bodies too.
    /// </summary>
    public unsafe struct DirectSolverCache : IDisposable
    {
        // The substeps' jobs solve the islands that the fused job doesn't take: craft whose joints can't join a
        // vessel, islands with loose bodies, and every island when the fused job is off. The table of their slots
        // starts with room for this many islands, and it grows when every slot holds an island that a recent step
        // solved. When the table had a fixed size of 64, every island past the 64th solved with a solver and a forest
        // for one step only, and built its structure again in every step.
        const int InitialCapacity = 64;

        // The table of slots grows to at most this many. Past that size, a new island takes the slot that was used
        // least recently among the slots that no island of the same step holds, and an island that finds no such slot
        // solves without cached data.
        const int MaxSlots = 4096;

        // A slot that no step has used for this many steps can go to a new island. An island that went to the fused job
        // for a step or two, or slept for a moment, keeps its slot.
        internal const int SlotKeepSteps = 60;

        // The step in which a slot that was never assigned was last used. Such a slot can always go to a new island.
        const long NeverUsed = long.MinValue;

        /// <summary>
        /// The identity that <see cref="Assign"/> takes as "this island needs no slot". No body has it, see
        /// <see cref="BodyIdentity"/>.
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
            /// The identity of the island's first body when the slot was assigned (see <see cref="BodyIdentity"/>), or
            /// <see cref="ulong.MaxValue"/>, which no body has, if the slot has never been assigned. Slots are matched
            /// by this.
            /// </summary>
            public ulong Identity;

            public bool Created;

            /// <summary>
            /// The step in which this slot was last assigned to an island, or <see cref="NeverUsed"/>.
            /// </summary>
            public long LastUsed;
        }

        /// <summary>
        /// The slots of the islands that the substeps' jobs solve. Only <see cref="Assign"/> adds slots, on one thread
        /// and before any island is solved, so the solvers don't move while islands use them.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<Slot>* slots;

        /// <summary>
        /// The slot that each identity of a first body was last assigned to (see <see cref="BodyIdentity"/>). Only
        /// <see cref="Assign"/> changes it.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeHashMap<ulong, int>* slotOfIdentity;

        /// <summary>A slot that holds nothing yet.</summary>
        static Slot EmptySlot => new Slot { Key = -1, Identity = NoSlot, LastUsed = NeverUsed };

        /// <summary>
        /// A slot that no island of the current step holds, and the step in which it was last used. Sorting these puts
        /// the slot that was used least recently first, and slots that were last used in the same step in index order.
        /// </summary>
        struct FreeSlot : IComparable<FreeSlot>
        {
            public long LastUsed;
            public int Slot;

            public int CompareTo(FreeSlot other) => LastUsed != other.LastUsed ? LastUsed.CompareTo(other.LastUsed) : Slot.CompareTo(other.Slot);
        }

        // The fused job takes vessel islands of every size, so a scene with many small craft or much debris has many
        // fused islands, and each of them keeps its state in a slot. The table of slots starts with room for this many
        // and grows when every slot holds a vessel that a recent step solved. When the table had a fixed size, every
        // fused island past that size built its structure again in every step: with 32 slots, 64 craft of 8 parts
        // took 1.0 ms a step instead of 0.4 ms.
        const int FusedInitialCapacity = 256;

        // A slot whose vessel no step has solved for this many steps can go to a new vessel. A vessel that went to the
        // substeps' jobs for a step or two, or slept for a moment, keeps its slot.
        internal const int FusedKeepSteps = 60;

        /// <summary>
        /// The persistent state of one fused vessel island. The key is the ID of the island's first vessel, which the
        /// backend keeps while the vessel's structure stays the same. The state does not depend on the island's index
        /// or on its first body.
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
        /// The slot that holds each vessel's state, by the vessel's ID, for the vessels that have a slot. Only the job
        /// that marks the fused islands changes it, before the fused job reads it from several threads.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeHashMap<ulong, int>* fusedSlotOfVessel;

        /// <summary>
        /// The forest that holds the newest lock states of a vessel, and the step in which that forest was handed the
        /// vessel. A vessel can move between forests: from the fused job's forest to an island's forest when it stops
        /// being fused for a step, from one island's forest to another's when its island's first body changes, and
        /// back. The forest that takes the vessel over copies the lock states from the forest named here, so that the
        /// lock states don't start again.
        /// </summary>
        internal struct VesselOwner
        {
            public ForestIsland* Forest;
            public long Step;
        }

        [NativeDisableUnsafePtrRestriction]
        UnsafeHashMap<ulong, VesselOwner>* vesselOwners;

        /// <summary>
        /// For each moving body in the world, its index among the bodies of the island that is solving it, or -1. All the
        /// islands of a step share this one index. A body belongs to one island in a step, so each island writes only
        /// the entries of its own bodies, and islands that solve at the same time never write the same entry. An island
        /// sets its entries to -1 again when its solve ends, so that every entry is -1 between solves. Before, each
        /// island filled an index as large as the world for each solve, and each fused island kept one of its own: with
        /// many small craft in a large world, that cost more than the solves themselves.
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
        /// The cache slot for each island in this step, or -1 if the island has no slot and must solve without cached
        /// data. Before slots have been assigned, this list is empty and lookups use the island index as the slot
        /// index.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* slotOfIsland;

        [NativeDisableUnsafePtrRestriction]
        long* step;

        /// <summary>
        /// How much work the islands that the fused job didn't take gave the substeps in one step: the islands, their
        /// rows at the first substep, their bodies, and the islands that are large enough to share their solve among
        /// lanes.
        /// </summary>
        internal struct UnfusedWork
        {
            public int Islands;
            public int Rows;
            public int Bodies;
            public int TeamIslands;

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

        /// <summary>
        /// Counters for the largest residual momentum at a section's first body after inverse dynamics, in units of
        /// 1e-9 N s, and the section and locked-pair counts from the last solve.
        /// </summary>
        internal const int ForestResidual = 15;
        internal const int ForestLastSections = 16;
        internal const int ForestLastLocked = 17;

        /// <summary>
        /// The most recently recorded deflection, stored in nanoradians. NoteDeflection replaces the previous value;
        /// it does not compute a maximum or an average.
        /// </summary>
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

        public static DirectSolverCache Create()
        {
            var cache = new DirectSolverCache
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
        /// Controls whether and how direct islands are solved using a forest and a boundary block. A default cache has
        /// no forest settings. Set this between steps; all copies of the cache refer to the same settings.
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
        /// Returns elapsed milliseconds since cache creation, summed across islands and substeps. Sparse time covers
        /// solving through impulse application. Forest times cover preparation (classification, factorization and
        /// boundary construction), solving, and finishing (drift correction, impulse application and push-out).
        /// Factorization and boundary construction times are also returned separately as portions of preparation.
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
        /// Builds a diagnostic histogram for tree edges in cached forests used in the current step. The five deflection
        /// bins are below half the lock angle, below the lock angle, below the yield angle, below twice the yield
        /// angle, and at or above twice the yield angle. Pairs with a contact on either body use the corresponding bin
        /// at index + 5. Counts each pair in <paramref name="all"/> and each locked pair in <paramref name="locked"/>.
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
        /// Returns statistics since cache creation: total sections and locked pairs across forest solves, structure
        /// rebuilds, lock changes, milliseconds spent rebuilding, and the largest residual from inverse dynamics in N
        /// s. Also returns the section and locked-pair counts from the last solve.
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
        /// Discards every cached forest, including its pairs, lock states and structure. Forests are matched by an
        /// island's first body, so a new scene with the same body indices could otherwise reuse the previous scene's
        /// forest. Tests call this when comparing independent runs.
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
        /// Records the forest that solves a vessel in this step. If the vessel solves with a forest that lasts for this
        /// step only, the cache forgets the vessel's forest instead.
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
        /// Forgets the vessels whose newest lock states are in a forest that is about to be destroyed. With no forest,
        /// and only when the record has grown large, forgets the vessels that no step has solved for a long time.
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
        /// Returns the persistent forest of the fused vessel island whose first vessel has this ID if a slot holds
        /// one, or null if no slot holds it. Marks the slot as used in this step, so that no other vessel takes the
        /// slot over in this step.
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
        /// Makes the shared index of the islands' bodies (see <see cref="compactOfWorld"/>) cover
        /// <paramref name="motionCount"/> moving bodies, and sets every entry to -1. One job calls this before any island
        /// of the step is solved. The islands keep every entry at -1 between their solves, so this only repairs an
        /// index that an earlier step left in another state.
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
        /// The shared index of the islands' bodies (see <see cref="compactOfWorld"/>), or null if the cache has none or if
        /// it covers fewer than <paramref name="motionCount"/> moving bodies. An island that gets null numbers its bodies
        /// in an index of its own.
        /// </summary>
        internal int* CompactIndex(int motionCount)
        {
            return compactOfWorld != null && compactOfWorld->Length >= motionCount ? compactOfWorld->Ptr : null;
        }

        /// <summary>
        /// Returns the persistent forest of the fused vessel island whose first vessel has this ID, and creates it if
        /// the vessel has none yet. A new vessel takes the least recently used slot if no step has used that slot for
        /// <see cref="FusedKeepSteps"/> steps. Otherwise the table grows by one slot, so that every fused island of a
        /// step keeps its state, however many there are. Only the job that marks the fused islands calls this, one
        /// island after another and before any island is solved, so no two threads change the slots at once. The fused
        /// job then finds the same forest with <see cref="FindFusedForest"/>.
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
        /// Returns the cached forest for an island, creating it if needed, or null if the island has no valid cache
        /// slot. Like <see cref="Solver"/>, this accesses only the island's assigned slot, so islands solved in
        /// parallel do not share a forest.
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
        /// Returns a number that tells this body apart from every other body that the world has had: the entity's
        /// version in the high half. The backend, which is the only creator of this cache, sets each body's entity
        /// version to the body's serial number, which it never gives to a second body. The entity's index is the body's
        /// row in the store, which changes when other bodies go, so the identity leaves it out. (In an ECS world the
        /// version alone would not tell entities apart; such a world doesn't use this cache.) A body without an entity
        /// is known by its index in the world, with zero in the high half, which no entity has as its version. When the
        /// caller has no bodies to give, every body is known by its index.
        /// </summary>
        internal static ulong BodyIdentity(RigidBody* bodies, int bodyCount, int worldBody)
        {
            if (bodies == null || (uint)worldBody >= (uint)bodyCount)
            {
                return (uint)worldBody;
            }

            int version = bodies[worldBody].Entity.Version;
            return version != 0 ? (ulong)(uint)version << 32 : (uint)worldBody;
        }

        /// <summary>
        /// Assigns cache slots for the step by the identity of each island's first body. Runs on one thread before any
        /// island is solved. An island whose identity is <see cref="NoSlot"/> gets no slot: the fused job keeps the
        /// state of its islands in slots of its own. Each other island first takes the slot that its identity had, which
        /// a hash map finds. The remaining islands take the slots that no step has used for <see cref="SlotKeepSteps"/>
        /// steps, the least recently used first, and otherwise new slots, until the table holds
        /// <see cref="MaxSlots"/> slots. No slot goes to two islands in one step, even if two first bodies had the same
        /// identity. A destroyed body's slot is never found again; it can still go to another island when no step has
        /// used it for a while. Its forest then gives the island no lock states, because a forest finds lock states by
        /// the identities of the bodies, see <see cref="ForestIsland.BuildPairs"/>. The caller gives each first body's
        /// index and identity, see <see cref="BodyIdentity"/>.
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
        /// Records how much work this step's islands that are not fused give the substeps. The job that marks the fused
        /// islands calls this, after <see cref="Assign"/>.
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
        /// The work that the islands that were not fused gave the substeps in the last step that recorded it. Returns
        /// false if no step has. The main thread reads this when it schedules a step, after the last step's jobs have
        /// finished.
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
        /// Returns the solver for an island, creating it if needed, or null if the island has no valid cache slot. Each
        /// island accesses only its assigned slot, so islands solved in parallel do not share a solver.
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
