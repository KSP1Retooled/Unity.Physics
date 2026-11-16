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
    /// Slots are matched by each island's first body, which has its lowest body index, rather than by
    /// the island's index. Adding an island can change later island indices without changing their body indices. The
    /// first body is a useful key while those body indices remain stable, including when a craft drops a stage or lands
    /// near another craft. A solver can also be assigned to a different island: it validates cached analysis against
    /// the new problem and repeats analysis when necessary.
    ///
    /// A body index alone does not tell two bodies apart: when a body is destroyed, the next body that is made can take
    /// its index. So the cache also keeps each slot's first body by its identity (see <see cref="BodyIdentity"/>), and a
    /// slot matches an island only when both the index and the identity match. A forest's pairs keep their lock states
    /// by the identities of their bodies too, so a new craft never takes over the lock states of a destroyed craft.
    /// </summary>
    public unsafe struct DirectSolverCache : IDisposable
    {
        // Cache at most this many islands per step. Any additional islands solve without cached data.
        const int Capacity = 64;

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

            /// <summary>The identity of the body at <see cref="Key"/> when the slot was assigned, see <see cref="BodyIdentity"/>.</summary>
            public ulong Identity;

            public bool Created;

            /// <summary>
            /// The step in which this slot was last assigned to an island.
            /// </summary>
            public long LastUsed;
        }

        [NativeDisableUnsafePtrRestriction]
        Slot* slots;

        // Keep the state of at most this many fused vessel islands. Any additional fused island solves with a
        // forest that it creates for the step only.
        const int FusedCapacity = 32;

        /// <summary>
        /// The persistent state of one fused vessel island. The key is the ID of the island's first vessel, which the
        /// backend keeps while the vessel's structure stays the same. The state does not depend on the island's index
        /// or on its first body.
        /// </summary>
        struct FusedSlot
        {
            public ForestIsland* Forest;

            public ulong VesselId;

            /// <summary>Whether the slot holds a vessel's state. A slot that was never used holds none.</summary>
            public bool Used;

            /// <summary>The step in which the slot was last used.</summary>
            public long LastUsed;
        }

        [NativeDisableUnsafePtrRestriction]
        FusedSlot* fusedSlots;

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

            cache.fusedSlots = (FusedSlot*)UnsafeUtility.MallocTracked(FusedCapacity * sizeof(FusedSlot), UnsafeUtility.AlignOf<FusedSlot>(),
                Allocator.Persistent, 0);
            UnsafeUtility.MemClear(cache.fusedSlots, FusedCapacity * sizeof(FusedSlot));
            cache.vesselOwners = (UnsafeHashMap<ulong, VesselOwner>*)UnsafeUtility.MallocTracked(UnsafeUtility.SizeOf<UnsafeHashMap<ulong, VesselOwner>>(),
                UnsafeUtility.AlignOf<UnsafeHashMap<ulong, VesselOwner>>(), Allocator.Persistent, 0);
            *cache.vesselOwners = new UnsafeHashMap<ulong, VesselOwner>(64, Allocator.Persistent);
            *cache.step = 0;
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
            for (int slot = 0; slots != null && slot < Capacity; slot++)
            {
                ForestIsland.Destroy(slots[slot].Forest);
                slots[slot].Forest = null;
            }

            for (int slot = 0; fusedSlots != null && slot < FusedCapacity; slot++)
            {
                ForestIsland.Destroy(fusedSlots[slot].Forest);
                fusedSlots[slot] = default;
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
            for (int slot = 0; fusedSlots != null && slot < FusedCapacity; slot++)
            {
                if (fusedSlots[slot].Used && fusedSlots[slot].VesselId == vesselId)
                {
                    fusedSlots[slot].LastUsed = *step;
                    return fusedSlots[slot].Forest;
                }
            }

            return null;
        }

        /// <summary>
        /// Returns the persistent forest of the fused vessel island whose first vessel has this ID, creating it if
        /// needed, or null if every slot was already used in this step. A slot that no vessel used in this step goes
        /// to a new vessel, the least recently used slot first. Only the job that marks the fused islands calls this,
        /// one island after another and before any island is solved, so no two threads change the slots at once. The
        /// fused job then finds the same forest with <see cref="FindFusedForest"/>.
        /// </summary>
        internal ForestIsland* FusedForestOf(ulong vesselId)
        {
            if (fusedSlots == null)
            {
                return null;
            }

            long now = *step;
            int found = -1, oldest = -1;
            for (int slot = 0; slot < FusedCapacity; slot++)
            {
                if (fusedSlots[slot].Used && fusedSlots[slot].VesselId == vesselId)
                {
                    found = slot;
                    break;
                }

                if ((!fusedSlots[slot].Used || fusedSlots[slot].LastUsed != now) && (oldest < 0 || !fusedSlots[slot].Used
                    || (fusedSlots[oldest].Used && fusedSlots[slot].LastUsed < fusedSlots[oldest].LastUsed)))
                {
                    oldest = slot;
                }
            }

            if (found < 0)
            {
                if (oldest < 0)
                {
                    return null;
                }

                // The slot's old forest belongs to another vessel, so start again from an empty one.
                found = oldest;
                ForgetOwners(fusedSlots[found].Forest);
                ForestIsland.Destroy(fusedSlots[found].Forest);
                fusedSlots[found].Forest = ForestIsland.Create();
                fusedSlots[found].VesselId = vesselId;
                fusedSlots[found].Used = true;
            }

            fusedSlots[found].LastUsed = now;
            return fusedSlots[found].Forest;
        }

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
        /// Returns a number that tells this body apart from every other body that the world has had, as long as the
        /// world's bodies have entities: the entity's version in the high half and its index in the low half. The
        /// backend sets each body's entity version to the body's serial number, which it never reuses. A body without
        /// an entity is known by its index in the world alone, with zero in the high half, which no entity has as its
        /// version. When the caller has no bodies to give, every body is known by its index.
        /// </summary>
        internal static ulong BodyIdentity(RigidBody* bodies, int bodyCount, int worldBody)
        {
            if (bodies == null || (uint)worldBody >= (uint)bodyCount)
            {
                return (uint)worldBody;
            }

            Unity.Entities.Entity entity = bodies[worldBody].Entity;
            return entity.Version != 0 ? ((ulong)(uint)entity.Version << 32) | (uint)entity.Index : (uint)worldBody;
        }

        /// <summary>
        /// Assigns cache slots for the step using each island's first body as its key. Runs on one thread before any
        /// island is solved. First reuses slots whose key and first body's identity both match, then assigns the
        /// remaining islands the least recently used slots that are still available. Islands beyond the cache capacity
        /// receive no slot. A slot whose key matches but whose identity doesn't belongs to a body that was destroyed,
        /// so no island takes it as its own. The slot can still go to another island as a least recently used slot. Its
        /// forest then gives the island no lock states, because a forest finds lock states by the identities of the
        /// bodies, see <see cref="ForestIsland.BuildPairs"/>. The caller gives each first body's identity, see
        /// <see cref="BodyIdentity"/>.
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

            for (int island = 0; island < islandCount; island++)
            {
                int found = -1;
                for (int slot = 0; slot < Capacity; slot++)
                {
                    if (slots[slot].Key == firstBodies[island] && slots[slot].Identity == firstIdentities[island])
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
                slots[oldest].Identity = firstIdentities[island];
                slots[oldest].LastUsed = now;
                (*slotOfIsland)[island] = oldest;
            }
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

            for (int slot = 0; slot < FusedCapacity; slot++)
            {
                ForestIsland.Destroy(fusedSlots[slot].Forest);
            }

            UnsafeUtility.FreeTracked(fusedSlots, Allocator.Persistent);
            fusedSlots = null;
            vesselOwners->Dispose();
            UnsafeUtility.FreeTracked(vesselOwners, Allocator.Persistent);
            vesselOwners = null;
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
