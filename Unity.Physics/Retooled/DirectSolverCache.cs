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

        public static DirectSolverCache Create()
        {
            var cache = new DirectSolverCache
            {
                slots = (Slot*)UnsafeUtility.MallocTracked(Capacity * sizeof(Slot), UnsafeUtility.AlignOf<Slot>(), Allocator.Persistent, 0),
                slotOfIsland = UnsafeList<int>.Create(Capacity, Allocator.Persistent),
                step = (long*)UnsafeUtility.MallocTracked(sizeof(long), UnsafeUtility.AlignOf<long>(), Allocator.Persistent, 0),
            };

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
            }

            UnsafeUtility.FreeTracked(slots, Allocator.Persistent);
            UnsafeUtility.FreeTracked(step, Allocator.Persistent);
            UnsafeList<int>.Destroy(slotOfIsland);
            slots = null;
            slotOfIsland = null;
            step = null;
        }
    }
}
