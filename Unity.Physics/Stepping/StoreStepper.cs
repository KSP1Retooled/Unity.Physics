using Unity.Collections;
using Unity.Jobs;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Detection;
using Unity.Physics.Stepping.Stages;
using Unity.Physics.Stepping.State;
using Unity.Physics.Store.World;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Stepping
{
    /// <summary>
    /// Orders the stages for one store world and carries their state and job handles.
    /// </summary>
    internal unsafe partial struct StoreStepper
    {
        private readonly WorldData* store;
        private readonly StoreGlobals* globals;
        private readonly QueryWorld* query;

        private PhysicsWorld world;
        private StepSolverCaches SolverCaches;
        private ColliderState Colliders;
        internal QuerySnapshotState Queries;
        private SleepState Sleep;
        private ContactState Contacts;
        private VesselState VesselData;
        private JointBuildState Joints;
        internal TreeState Trees;
        internal StaticRows StaticBodies;
        private BodyPartition Partition;

        // Prepared simulation, or null since the last FinishStep.
        // Avoid reading the singleton here, which would wait for solver cleanup before scheduling write-back.
        internal Simulation* preparedSimulation;

        // Pre-solve jobs that FinishStepSystem completes before FinishStep reads their lists.
        internal JobHandle preparedInputs;

        // The job that ApplyForces leaves running. BuildWorld makes the build wait for it.
        private JobHandle forcesApplied;
        private JobHandle warmStartsIndexed;

        internal StoreStepper(WorldData* store, StoreGlobals* globals, QueryWorld* query)
        {
            this = default;
            VesselData.InitializeTombstones();
            this.store = store;
            this.globals = globals;
            this.query = query;
            world = new PhysicsWorld(0, 0, 0);
            Trees.InitializeStaticTree();
            Partition.InitializeRows();
            StaticBodies.Initialize();
            Partition.Initialize();
            Joints.Initialize(store);
            SolverCaches.InitializeDirectSolver();
            VesselData.Initialize();
            SolverCaches.Initialize();
            Queries.Initialize();
            Contacts.Initialize();
            Sleep.Initialize();
            Contacts.InitializeImpulseMemory();
            Colliders.Initialize();
            Trees.Initialize(store);
            Contacts.InitializePairCache();
        }

        internal void Dispose()
        {
            // The refit writes the world that is freed below.
            FinishPendingRefit();
            store->queryWorld = null;
            store->refit = null;
            Trees.DisposeRefit();

            forcesApplied.Complete();
            Colliders.collidersBuilt.Complete();

            // A step that threw before FinishStep leaves the fill jobs unfinished, and they write the vessel lists.
            VesselData.vesselsFilled.Complete();
            Colliders.ReleaseWorldColliders(store);

            store->queryDirty = 1;
            world.Dispose();
            Trees.DisposeStaticTree();
            Partition.DisposeRows();
            StaticBodies.Dispose();
            Partition.Dispose();
            Joints.Dispose(store);
            SolverCaches.DisposeDirectSolver();
            VesselData.Dispose();
            SolverCaches.Dispose();
            Queries.Dispose();
            Contacts.Dispose();
            Sleep.Dispose();
            Contacts.DisposeImpulseMemory();
            Colliders.Dispose();
            Trees.Dispose();
            Contacts.DisposePairCache();
        }

        /// <summary>The direct solver's cache, including the counts of solves that used the forest.</summary>
        internal readonly IslandSolverCache IslandSolverCache => SolverCaches.directSolverCache;

        /// <summary>The world built for the simulation systems to advance.</summary>
        internal readonly PhysicsWorld World => world;

        /// <summary>The index of each store body in the built world.</summary>
        internal readonly NativeList<int> BuiltIndices => Partition.builtIndices;

        internal readonly int StaticsChanged => StaticBodies.staticsChanged ? 1 : 0;

        /// <summary>The settings that the simulation systems use to build their input for a step.</summary>
        internal readonly PhysicsStep Settings => SolveStage.Settings;

        /// <summary>
        /// Complete outstanding work and invalidate queries before the caller prepares body pointers for jobs.
        /// </summary>
        internal void BeginForces()
        {
            ForcesStage.BeginForces(store, ref Trees, ref forcesApplied);
        }

        /// <summary>
        /// Wake bodies through joints, then schedule force integration.
        /// BuildWorld waits for the force jobs. The main thread may read flags while they run.
        /// </summary>
        internal void ApplyForces(float deltaTime)
        {
            ForcesStage.ApplyForces(store, ref Contacts, ref forcesApplied, deltaTime);
        }

        /// <summary>Builds the bodies, joints and broadphase for the simulation systems to advance.</summary>
        internal JobHandle BuildWorld(float deltaTime, JobHandle dependency)
        {
            return BuildStage.BuildWorld(store, globals, ref world, ref Partition, ref StaticBodies, ref Trees,
                ref Joints, ref VesselData, ref Colliders, ref forcesApplied, ref Queries.builtWhole,
                ref Contacts.stepCount, deltaTime, dependency);
        }

        /// <summary>The store updated by this simulation state. Used by tests.</summary>
        internal readonly WorldData* Store => store;

        /// <summary>The manifolds of the last step, with the impulses that they started with. Used by tests.</summary>
        internal readonly NativeList<ManifoldRecord> Manifolds => Contacts.manifolds;

        internal readonly StoreGlobals* Globals => globals;

        internal readonly NativeHashMap<ShapePairKey, ShapePairState> Touching => Contacts.touching;

        internal readonly NativeList<ManifoldPoint> ManifoldPoints => Contacts.manifoldPoints;

        internal readonly NativeList<ShapePairKey> RemovedTriggerPairs => Contacts.removedTriggerPairs;

        internal bool CanWriteContainers() => Partition.CanWriteContainers();

        internal uint NextContactStep() => ++Contacts.stepCount;

        /// <summary>The vessels compiled during the last build. Used by tests.</summary>
        internal readonly VesselBuild Vessels => VesselData.vesselBuild;

        /// <summary>The set of vessels supplied to the step during the last build. Used by tests.</summary>
        internal readonly VesselSet VesselSet => VesselData.vesselSet;

        /// <summary>
        /// Remove unsolved overlaps before island finding and supply this step's vessels.
        /// </summary>
        internal JobHandle SortIslands(SimulationSingleton singleton, ref SimulationStepInput input, JobHandle dependency)
        {
            return IslandStage.SortIslands(store, globals, ref world, ref Contacts, ref Colliders, ref Joints,
                ref VesselData, ref SolverCaches, singleton, ref input, dependency);
        }

        /// <summary>Reads the contacts produced by the narrow phase before the Jacobians are built from them.</summary>
        internal JobHandle ReadContacts(SimulationSingleton singleton, float deltaTime, JobHandle dependency)
        {
            return ContactStage.ReadContacts(store, globals, ref world, ref Partition, ref Contacts,
                ref warmStartsIndexed, singleton, deltaTime, dependency);
        }

        /// <summary>Warm starts the Jacobians.</summary>
        internal JobHandle PrepareSolve(SimulationSingleton singleton, ref SimulationStepInput input, JobHandle dependency)
        {
            return SolveStage.PrepareSolve(ref world, ref Partition, ref Contacts, ref VesselData, ref SolverCaches,
                warmStartsIndexed, out preparedSimulation, out preparedInputs, singleton, ref input, dependency);
        }

        /// <summary>
        /// Convert test-supplied vessel loads to joint loads and process breaking as at step completion.
        /// </summary>
        internal void ReadVesselLoads(float deltaTime)
        {
            FinishStage.ReadVesselLoads(store, globals, ref world, ref Partition, ref Joints, ref VesselData,
                deltaTime);
        }

        /// <summary>
        /// Writes back the solved world and prepares contact reports, query publication and sleep.
        /// </summary>
        internal void FinishBeforeContacts(Simulation simulation, float deltaTime,
            out JobHandle contactsJoined, out JobHandle snapshot, out JobHandle sleep)
        {
            FinishStage.SchedulePostSolve(store, globals, ref world, ref Partition, ref Joints, ref VesselData,
                ref Trees, ref Queries, warmStartsIndexed, ref simulation, deltaTime,
                out JobHandle writtenBack, out JobHandle vesselsSettled, out snapshot);
            FinishStage.CompleteBeforeContacts(store, globals, ref world, ref Joints, ref VesselData,
                ref Contacts, ref Sleep, ref simulation, deltaTime, writtenBack, vesselsSettled, out contactsJoined, out sleep);
        }

        /// <summary>Publish queries and finish sleep after the caller has run the contact report.</summary>
        internal void FinishAfterContacts(JobHandle snapshot, JobHandle sleep)
        {
            FinishStage.FinishAfterContacts(store, query, ref world, ref Queries, ref Trees, ref Partition,
                ref Sleep, snapshot, sleep);
        }

        /// <summary>Finish pending refit work and check whether queries need a new build.</summary>
        internal bool BeginQueryBuild()
        {
            return BuildStage.BeginQueryBuild(store, ref Trees, ref Queries);
        }

        /// <summary>Update the built world from prepared store rows without advancing time.</summary>
        internal void BuildForQueries()
        {
            BuildStage.BuildForQueries(store, globals, query, ref world, ref Partition, ref StaticBodies, ref Trees,
                ref Queries, ref Colliders, ref forcesApplied, ref VesselData.vesselsFilled);
        }

        /// <summary>
        /// Complete the pending dynamic tree refit on the main thread before reading or modifying the built world
        /// or bounds cache.
        /// Call this outside jobs. Query jobs use HelpPendingRefit.
        /// </summary>
        internal void FinishPendingRefit() => Trees.FinishPendingRefit(store);

        /// <summary>
        /// Finish pending refit work without completing a job handle, for callers already inside a job.
        /// FinishPendingRefit later completes the now-empty scheduled jobs.
        /// </summary>
        internal void HelpPendingRefit() => Trees.HelpPendingRefit(store);

        /// <summary>
        /// Whether the published query world waits only for the last step's refit, so that it is current once the refit
        /// has finished.
        /// </summary>
        internal readonly bool OnlyRefitPending => Queries.OnlyRefitPending(store);
    }
}
