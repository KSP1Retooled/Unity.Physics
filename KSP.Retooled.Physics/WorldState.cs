using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Systems;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Configuration;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Managed state and built Unity Physics world for one native world.
    /// Discard on detach and rebuild on the next step or query.
    /// </summary>
    internal sealed unsafe class WorldState : IDisposable
    {
        private readonly StepState* state;
        private readonly QueryWorld* query;
        private readonly WorldData* store;

        internal WorldState(WorldData* store)
        {
            this.store = store;
            ForestSwitches.Initialize();
            VesselSwitches.Initialize();
            StepState.ReadBoundsCacheSwitches();
            StepState.ReadTreeReuseSwitch();
            StepState.ReadStaticReuseSwitch();
            query = (QueryWorld*)UnsafeUtility.MallocTracked(sizeof(QueryWorld), UnsafeUtility.AlignOf<QueryWorld>(), Allocator.Persistent, 0);
            state = (StepState*)UnsafeUtility.MallocTracked(sizeof(StepState), UnsafeUtility.AlignOf<StepState>(), Allocator.Persistent, 0);
            *state = new StepState(store, NativeStore.RetooledPhysics_Globals(), query);
            TouchingHandOff.Restore(store, state->Touching);
        }

        /// <summary>Keeps the touching pairs for the next state of this world, before a detach discards this one.</summary>
        internal void SaveTouching()
        {
            TouchingHandOff.Save(store, state->Globals, state->Touching);
        }

        /// <summary>Waits for the refit of the dynamic tree that the last step left running, if there is one.</summary>
        internal void FinishPendingRefit()
        {
            state->FinishPendingRefit();
        }

        public void Dispose()
        {
            state->Dispose();
            UnsafeUtility.FreeTracked(state, Allocator.Persistent);
            UnsafeUtility.FreeTracked(query, Allocator.Persistent);
        }

        /// <summary>
        /// Run this world's physics systems when Entities has started. Before then, skip simulation.
        /// </summary>
        internal void Step(PhysicsSystemGroup group, float deltaTime)
        {
            // Refresh row pointers on every return path: Unity reads poses after this call, including skipped
            // or failed steps.
            try
            {
                if (group == null || deltaTime <= 0.0f)
                {
                    return;
                }

                // The last step's refit of the dynamic tree writes the built world, which this step builds again.
                state->FinishPendingRefit();
                StepKernel.Current = state;
                group.Step(deltaTime);
            }
            finally
            {
                BodyEntities.PrepareForJobs(store);
            }
        }

        /// <summary>
        /// Update built query data without simulation. Main thread only; return a QueryWorld result.
        /// </summary>
        internal int BuildForQueries()
        {
            // Inside a job, finish refit work directly and defer other world changes until a build outside the
            // job.
            if (Unity.Jobs.LowLevel.Unsafe.JobsUtility.IsExecutingJob)
            {
                state->HelpPendingRefit();
                return state->OnlyRefitPending ? StoreConstants.QueryWorldBuilt : StoreConstants.QueryWorldDeferred;
            }

            if (!state->CanWriteContainers())
            {
                state->HelpPendingRefit();
                return StoreConstants.QueryWorldDeferred;
            }

            StepKernel.BuildForQueries(state);
            return StoreConstants.QueryWorldBuilt;
        }

        /// <summary>
        /// Leaves the queries a world that they can read after a step threw: no world, and no refit to wait for.
        /// </summary>
        internal void AfterFailedStep()
        {
            state->ReleaseTreeAfterThrow();
        }
    }
}
