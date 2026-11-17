using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Systems;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Stores the managed state for one native world. This includes a Unity Physics world built from the store on
    /// demand, plus data retained between steps. All of this state is discarded on detach. It is rebuilt for the next
    /// step or query.
    /// </summary>
    internal sealed unsafe class WorldState : IDisposable
    {
        internal const int SolverIterations = 12;
        internal const int DirectIslandSubsteps = 2;

        // The maximum speed, in m/s, at which contacts separate penetrating bodies. This is the default value of
        // Rigidbody.maxDepenetrationVelocity. PhysX applies it to each body, and the stock game never changes it.
        internal const float MaxDepenetrationVelocity = 10.0f;

        private readonly StepState* state;
        private readonly QueryWorld* query;

        internal WorldState(NativeStore.WorldData* store)
        {
            VesselForest.Initialize();
            VesselPipeline.Initialize();
            StepState.ReadBoundsCacheSwitches();
            query = (QueryWorld*)UnsafeUtility.MallocTracked(sizeof(QueryWorld), UnsafeUtility.AlignOf<QueryWorld>(), Allocator.Persistent, 0);
            state = (StepState*)UnsafeUtility.MallocTracked(sizeof(StepState), UnsafeUtility.AlignOf<StepState>(), Allocator.Persistent, 0);
            *state = new StepState(store, NativeStore.RetooledPhysics_Globals(), query);
        }

        /// <summary>Keeps the touching pairs for the next state of this world, before a detach discards this one.</summary>
        internal void SaveTouching()
        {
            state->SaveTouching();
        }

        public void Dispose()
        {
            state->Dispose();
            UnsafeUtility.FreeTracked(state, Allocator.Persistent);
            UnsafeUtility.FreeTracked(query, Allocator.Persistent);
        }

        /// <summary>
        /// Runs the physics system group for this world. The group does not exist before Entities starts, so no step
        /// runs before then. Simulation for the first frame occurs after Entities starts.
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
        /// Updates the built world to match the poses in the store without running a simulation step. Returns a
        /// QueryWorld constant. Call this method only on the main thread.
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
