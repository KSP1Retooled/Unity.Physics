using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Systems;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Managed state and built Unity Physics world for one native world.
    /// Discard on detach and rebuild on the next step or query.
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
        private readonly NativeStore.WorldData* store;

        internal WorldState(NativeStore.WorldData* store)
        {
            this.store = store;
            VesselForest.Initialize();
            VesselPipeline.Initialize();
            StepState.ReadBoundsCacheSwitches();
            StepState.ReadTreeReuseSwitch();
            StepState.ReadStaticReuseSwitch();
            query = (QueryWorld*)UnsafeUtility.MallocTracked(sizeof(QueryWorld), UnsafeUtility.AlignOf<QueryWorld>(), Allocator.Persistent, 0);
            state = (StepState*)UnsafeUtility.MallocTracked(sizeof(StepState), UnsafeUtility.AlignOf<StepState>(), Allocator.Persistent, 0);
            *state = new StepState(store, NativeStore.RetooledPhysics_Globals(), query);
        }

        /// <summary>Keeps the touching pairs for the next state of this world, before a detach discards this one.</summary>
        internal void SaveTouching()
        {
            state->SaveTouching();
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
                return state->OnlyRefitPending ? NativeStore.QueryWorldBuilt : NativeStore.QueryWorldDeferred;
            }

            if (!state->CanWriteContainers())
            {
                state->HelpPendingRefit();
                return NativeStore.QueryWorldDeferred;
            }

            StepKernel.BuildForQueries(state);
            return NativeStore.QueryWorldBuilt;
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
