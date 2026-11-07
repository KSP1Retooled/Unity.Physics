using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Everything managed keeps for one native world: a Unity Physics world built on demand from the store, and the
    /// simulation that steps it. None of it survives a detach, and the next step or query rebuilds it.
    /// </summary>
    internal sealed unsafe class WorldState : IDisposable
    {
        internal const int SolverIterations = 8;
        internal const int Substeps = 4;

        private readonly StepState* state;
        private readonly QueryWorld* query;

        internal WorldState(NativeStore.WorldData* store)
        {
            query = (QueryWorld*)UnsafeUtility.MallocTracked(sizeof(QueryWorld), UnsafeUtility.AlignOf<QueryWorld>(), Allocator.Persistent, 0);
            state = (StepState*)UnsafeUtility.MallocTracked(sizeof(StepState), UnsafeUtility.AlignOf<StepState>(), Allocator.Persistent, 0);
            *state = new StepState(store, NativeStore.RetooledPhysics_Globals(), query);
        }

        public void Dispose()
        {
            state->Dispose();
            UnsafeUtility.FreeTracked(state, Allocator.Persistent);
            UnsafeUtility.FreeTracked(query, Allocator.Persistent);
        }

        internal void Step(float deltaTime)
        {
            StepKernel.Step(state, deltaTime);
        }

        /// <summary>
        /// Brings the built world up to the store's poses without stepping and returns a QueryWorld constant. Main thread only.
        /// </summary>
        internal int BuildForQueries()
        {
            if (!state->CanWriteContainers())
            {
                return NativeStore.QueryWorldDeferred;
            }

            StepKernel.BuildForQueries(state);
            return NativeStore.QueryWorldBuilt;
        }
    }
}
