using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Sparse solvers kept from one solve to the next, one for each island, so each can reuse what it worked out last time.
    /// The caller of a step owns this and passes it in <see cref="SimulationStepInput"/>. Without one, nothing is reused.
    ///
    /// An island's index can change between steps. That costs the solver a new analysis and nothing else, because a solver
    /// checks everything it reuses against the problem it is given.
    /// </summary>
    public unsafe struct DirectSolverCache : IDisposable
    {
        // Beyond this many islands the rest solve from scratch, which the craft that matter never reach.
        const int Capacity = 64;

        [NativeDisableUnsafePtrRestriction]
        SparseConstraintSolver* solvers;

        [NativeDisableUnsafePtrRestriction]
        bool* created;

        public static DirectSolverCache Create()
        {
            var cache = new DirectSolverCache
            {
                solvers = (SparseConstraintSolver*)UnsafeUtility.MallocTracked(Capacity * sizeof(SparseConstraintSolver),
                    UnsafeUtility.AlignOf<SparseConstraintSolver>(), Allocator.Persistent, 0),
                created = (bool*)UnsafeUtility.MallocTracked(Capacity, UnsafeUtility.AlignOf<bool>(), Allocator.Persistent, 0),
            };

            UnsafeUtility.MemClear(cache.created, Capacity);
            return cache;
        }

        /// <summary>
        /// The solver for an island, or null if there is none to keep. Islands solve in parallel, so this creates a solver
        /// only for the island asking, and no two threads ever ask for the same one.
        /// </summary>
        internal SparseConstraintSolver* Solver(int island)
        {
            if (solvers == null || island >= Capacity)
            {
                return null;
            }

            if (!created[island])
            {
                solvers[island] = new SparseConstraintSolver(Allocator.Persistent);
                created[island] = true;
            }

            return solvers + island;
        }

        public void Dispose()
        {
            if (solvers == null)
            {
                return;
            }

            for (int island = 0; island < Capacity; island++)
            {
                if (created[island])
                {
                    solvers[island].Dispose();
                }
            }

            UnsafeUtility.FreeTracked(solvers, Allocator.Persistent);
            UnsafeUtility.FreeTracked(created, Allocator.Persistent);
            solvers = null;
            created = null;
        }
    }
}
