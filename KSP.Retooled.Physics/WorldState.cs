using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Systems;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Everything managed keeps for one native world: a Unity Physics world built on demand from the store, and what a step
    /// carries over. None of it survives a detach, and the next step or query rebuilds it.
    /// </summary>
    internal sealed unsafe class WorldState : IDisposable
    {
        internal const int SolverIterations = 12;
        internal const int DirectIslandSubsteps = 2;

        // The fastest, in m/s, that contacts push penetrating bodies apart: Rigidbody.maxDepenetrationVelocity's default.
        // PhysX applies it to each body, and stock never changes it.
        internal const float MaxDepenetrationVelocity = 10.0f;

        private readonly StepState* state;
        private readonly QueryWorld* query;

        internal WorldState(NativeStore.WorldData* store)
        {
            VesselForest.Initialize();
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

        /// <summary>
        /// Runs the physics system group on this world. Before Entities starts there is no group, and nothing steps: the
        /// first frame's simulation comes after it.
        /// </summary>
        internal void Step(PhysicsSystemGroup group, float deltaTime)
        {
            if (group == null || deltaTime <= 0.0f)
            {
                return;
            }

            StepKernel.Current = state;
            group.Step(deltaTime);
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
