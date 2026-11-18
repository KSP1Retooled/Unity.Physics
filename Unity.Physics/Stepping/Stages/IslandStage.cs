using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Stepping.State;
using Unity.Physics.Store.World;
using Unity.Profiling;

namespace Unity.Physics.Stepping.Stages
{
    /// <summary>
    /// Filters body pairs, then schedules the island sort.
    /// </summary>
    internal static unsafe class IslandStage
    {
        private static readonly ProfilerMarker StepScheduleMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Schedule");

        private const int ParallelBatchSize = 64;

        /// <summary>
        /// Remove unsolved overlaps before island finding and supply this step's vessels.
        /// </summary>
        internal static JobHandle SortIslands(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref ContactState contactState, ref ColliderState colliders, ref JointBuildState jointState,
            ref VesselState vessels, ref StepSolverCaches solverCaches, SimulationSingleton singleton,
            ref SimulationStepInput input, JobHandle dependency)
        {
            using var span = StepScheduleMarker.Auto();
            input.VesselSet = vessels.vesselSet;

            Simulation* simulation = singleton.AsSimulationPtr();
            if (input.World.NumDynamicBodies == 0)
            {
                return dependency;
            }

            NativeList<DispatchPairSequencer.DispatchPair> pairs = simulation->StepContext.PairsToSortByIsland;
            JobHandle filtered = new PairFilter.DropIgnoredPairsJob
            {
                Pairs = pairs.AsDeferredJobArray(),
                Store = store->bodies,
                Globals = globals,
                Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
                Known = contactState.knownBodyPairs,
                WholeBuilds = colliders.wholeColliderBuilds,
                Learned = contactState.learnedBodyPairs.AsParallelWriter(),
                World = store,
                Units = jointState.jointUnits.AsDeferredJobArray(),
                BodyVessels = vessels.vesselBuild.BodyVessels.AsArray(),
                VesselActive = vessels.vesselBuild.Active.AsArray(),
                VesselPairUnits = vessels.vesselBuild.PairUnits,
            }.Schedule(pairs, ParallelBatchSize, dependency);

            // No job in this step reads the newly learned body pairs. Saving those pairs can therefore run in parallel
            // with finding islands.
            JobHandle remembered = new PairFilter.RememberBodyPairsJob
            {
                Known = contactState.knownBodyPairs,
                Learned = contactState.learnedBodyPairs,
                Bodies = (int)store->bodyCount,
            }.Schedule(filtered);

            SimulationStepInput keeping = input;
            keeping.SolverIslandCache = solverCaches.solverIslandCache;
            return JobHandle.CombineDependencies(simulation->ScheduleIslandJobs(keeping, filtered), remembered);
        }
    }
}
