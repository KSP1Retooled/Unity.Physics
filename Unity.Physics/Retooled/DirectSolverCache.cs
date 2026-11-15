using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

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
            cache.forestCounters = (long*)UnsafeUtility.MallocTracked(3 * sizeof(long), UnsafeUtility.AlignOf<long>(), Allocator.Persistent, 0);
            UnsafeUtility.MemClear(cache.forestCounters, 3 * sizeof(long));

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

        internal void CountForest(int counter)
        {
            if (forestCounters != null)
            {
                System.Threading.Interlocked.Increment(ref forestCounters[counter]);
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
