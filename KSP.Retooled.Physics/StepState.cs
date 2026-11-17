using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Base.Threading;
using Unity.Physics.Collision.World.Trees;
using Unity.Physics.Dynamics.Simulation.Islands;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Reports;
using TriggerEvent = Unity.Physics.Store.Reports.TriggerEvent;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Configuration;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Dynamics.Vessels.Detection;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Unmanaged simulation state for one world, owned by WorldState and used by Burst step entry points.
    /// </summary>
    internal unsafe struct StepState
    {
        private const int ParallelBatchSize = 64;

        // Building a joint costs much more than building a body, so each batch contains fewer joints.
        private const int JointBatchSize = 8;

        // Copying the joints for one unit requires copying a few hundred bytes.
        private const int CopyBatchSize = 64;

        // WriteBackJob assigns these bodies a pose that differs from the result of integrating their solved motion.
        private const BodyFlags PosedByWriteBack = BodyFlags.Sleeping | BodyFlags.HasKinematicTarget;

        private readonly WorldData* store;
        private readonly StoreGlobals* globals;
        private readonly QueryWorld* query;

        private PhysicsWorld world;

        // Static tree jobs read these flags. keepStatics is constant after initialization;
        // ShiftStaticTreeJob sets rebuildStatics before tree jobs run.
        private NativeReference<int> rebuildStatics;
        private NativeReference<int> keepStatics;
        private ShiftStaticTreeJob.ShiftState treeShift;
        private bool staticsChanged;
        private bool treeMatchesHeld;

        // Store rows used for the retained static bodies. Reuse them and the anchor when KeepsStaticRows
        // permits.
        private NativeList<int> staticRows;
        private long keptStaticRowsCount;
        private NativeList<int> dynamicIndices;
        private NativeList<int> staticIndices;
        private NativeList<int> builtIndices;
        private NativeReference<int> partitionResult;
        private NativeList<JointGroup> jointGroups;
        private NativeList<JointGroupMember> jointGroupMembers;
        private NativeList<JointUnit> jointUnits;
        private NativeList<JointForestRole> jointRoles;
        private NativeList<int> jointsWritten;

        // The topology version and forest setting used to find the groups. The groups can be reused while these values
        // are unchanged. See JointTopology.
        private uint groupedTopology;
        private bool groupedForest;
        private bool grouped;
        private NativeList<JointLoad> jointLoads;
        private NativeList<JointLoad> groupLoads;

        // For each joint, 1 if its smoothed load exceeded its break limits in this step, for JointBreaking.BreakJointsJob. The jobs
        // that settle the loads write the flags. A byte for each joint is much less memory to write and to scan than
        // a load.
        private NativeList<byte> jointsOverloaded;

        // The load state of each joint, in the order of the store's list of joints. The joint commands keep it in that
        // order between steps. See JointLoadStates.
        private UnsafeList<JointLoadState>* jointStates;

        // The joints and groups outside the active vessels of this step. See VesselLoads.ListOtherJointsJob.
        private NativeList<int> otherJoints;
        private NativeList<int> otherGroups;
        private IslandSolverCache directSolverCache;

        // Prepared simulation, or null since the last FinishStep.
        // Avoid reading the singleton here, which would wait for solver cleanup before scheduling write-back.
        internal Simulation* preparedSimulation;

        // Pre-solve jobs that FinishStepSystem completes before FinishStep reads their lists.
        internal JobHandle preparedInputs;

        // The vessels compiled from the joints, and whether compilation has been completed for the current groups.
        private VesselBuild vesselBuild;
        private VesselSet vesselSet;
        private bool compiled;

        // Whether the vessel build holds cached edges and closures, and the joint settings version that they were
        // computed from. See VesselEdgeFill.FillVesselEdgesJob.RefillAll.
        private bool cacheFilled;
        private uint cachedSettings;

        // The jobs that fill the vessels' edges and closures for this step. Only the solve waits for them.
        private JobHandle vesselsFilled;

        // The step at which the oldest tombstone in the vessels expires, or -1. See VesselCompiler.CompileVesselsJob.
        private int tombstoneExpiry;
        private SolverIslandCache solverIslandCache;
        private NativeList<QueryBody> queryBodies;
        private NativeList<QueryShape> queryShapes;
        private NativeList<int> queryFirstShapes;

        // Reuse query snapshots while versions and body row order match.
        // Mark a scheduled snapshot incomplete until it finishes so that exceptions force a fresh copy.
        private bool snapshotTaken;
        private uint snapshotVersion;
        private NativeList<int> snapshotDynamic;
        private NativeList<int> snapshotStatic;
        private long keptSnapshots;
        private NativeList<ManifoldRecord> manifolds;
        private NativeList<ManifoldPoint> manifoldPoints;
        private NativeList<ContactBucket> contactBuckets;
        private NativeReference<int> filledContactBuckets;
        private NativeHashMap<ShapePairKey, ShapePairState> touching;

        // The events that end trigger pairs whose shapes were removed, which the next contact report sends to Unity.
        private NativeList<TriggerEvent> removedTriggerPairs;

        // The version of PairFilterVersion that the last step checked the touching pairs against.
        private uint seenPairFilterVersion;

        private NativeList<int> sleepParents;
        private NativeList<int> sleepContacts;
        private NativeList<bool> sleepTouchesStill;
        private NativeList<int> fallingAsleep;
        private NativeArray<int> sleepRunOut;
        private NativeHashMap<PersistentManifoldKey, float> lastImpulses;
        private NativeHashMap<JacobianKey, float2> warmStarts;
        private NativeList<ColliderBuilder.Work> colliderWork;
        private NativeList<long> colliderResults;
        private NativeReference<uint> wholeColliderBuilds;

        // The number of collider refreshes that gave any body a new collider, and the bounds of the dynamic bodies'
        // colliders that the refit at the end of a step leaves for the build at the start of the next.
        private NativeReference<uint> colliderChanges;
        private DynamicBoundsCache boundsCache;

        // Whether the last build of the dynamic tree used the bounds cache. Only then does the cache hold the current
        // colliders after the build.
        private bool boundsCacheBuilt;

        // Pending refit to solved poses. FinishPendingRefit protects the built world and bounds cache;
        // queries may help the refit through the native store pointer.
        private SharedDynamicTreeRefit* sharedRefit;
        private JobHandle refit;
        private bool refitPending;

        // The jobs that write the motions that the pending refit reads.
        private JobHandle refitInputs;

        // Dynamic store rows used for the current tree. Refit while row order, count and quality remain
        // suitable.
        private NativeList<int> treeRows;

        // What the jobs of the last refit left: whether a filter changed, and the tree's quality.
        private DynamicTreeRefitResults* treeRefitResults;

        // Whether the last build refitted the tree, the refits since the tree was last built, and the quality that the
        // first of those refits measured, or 0 before it has been read.
        private bool treeRefitted;
        private int treeRefits;
        private float treeBaseQuality;

        // Counts for tests.
        private long treeBuildCount;
        private long treeRefitCount;
        private long treeFilterRefitCount;

        /// <summary>
        /// Maximum consecutive refits before rebuilding the dynamic tree.
        /// </summary>
        internal const int MaxTreeRefits = 500;

        /// <summary>
        /// Maximum tree quality growth relative to the first refit after rebuilding, before another rebuild is
        /// required.
        /// </summary>
        internal const float MaxTreeQualityGrowth = 1.5f;

        /// <summary>
        /// Enable dynamic tree reuse. RETOOLED_TREE_REUSE=0 disables it.
        /// Stored in SharedStatic for Burst; the always-refit mode also bypasses count and quality limits.
        /// </summary>
        internal static bool TreeReuseEnabled
        {
            get => TreeReuseMode.Data != 1;
            set => TreeReuseMode.Data = value ? 0 : 1;
        }

        /// <summary>
        /// Whether every build refits the dynamic tree when the dynamic bodies are the same, however bad the tree has
        /// become. RETOOLED_TREE_REUSE=always turns it on, to check that a tree of any shape gives the same results.
        /// </summary>
        internal static bool TreeAlwaysRefitted
        {
            get => TreeReuseMode.Data == 2;
            set => TreeReuseMode.Data = value ? 2 : 0;
        }

        private sealed class TreeReuseModeKey
        {
        }

        private static readonly Unity.Burst.SharedStatic<int> TreeReuseMode = Unity.Burst.SharedStatic<int>.GetOrCreate<StepState, TreeReuseModeKey>();

        /// <summary>
        /// Reads RETOOLED_TREE_REUSE from the environment: 0 turns off the reuse of the dynamic tree, and "always" makes
        /// every build refit the tree when it can (<see cref="TreeAlwaysRefitted"/>).
        /// </summary>
        internal static void ReadTreeReuseSwitch()
        {
            string reuse = Environment.GetEnvironmentVariable("RETOOLED_TREE_REUSE");
            TreeReuseMode.Data = reuse == "0" ? 1 : reuse == "always" ? 2 : 0;
            if (reuse == "0" || reuse == "always")
            {
                UnityEngine.Debug.Log($"[StepState] RETOOLED_TREE_REUSE={reuse}: {(reuse == "0" ? "every build builds the dynamic tree from scratch" : "every build refits the dynamic tree when the dynamic bodies are the same")}");
            }
        }

        /// <summary>
        /// Reuse unchanged static body rows. RETOOLED_STATIC_REUSE=0 disables it.
        /// Stored in SharedStatic because Burst reads it.
        /// </summary>
        internal static bool StaticReuseEnabled
        {
            get => StaticReuseOff.Data == 0;
            set => StaticReuseOff.Data = value ? 0 : 1;
        }

        private sealed class StaticReuseOffKey
        {
        }

        private static readonly Unity.Burst.SharedStatic<int> StaticReuseOff = Unity.Burst.SharedStatic<int>.GetOrCreate<StepState, StaticReuseOffKey>();

        /// <summary>Reads RETOOLED_STATIC_REUSE from the environment: 0 turns off the reuse of the static rows.</summary>
        internal static void ReadStaticReuseSwitch()
        {
            string reuse = Environment.GetEnvironmentVariable("RETOOLED_STATIC_REUSE");
            StaticReuseEnabled = reuse != "0";
            if (reuse == "0")
            {
                UnityEngine.Debug.Log("[StepState] RETOOLED_STATIC_REUSE=0: every build writes the static bodies' rows from the store");
            }
        }

        /// <summary>The builds that kept the static bodies' rows of the last build. Used by tests.</summary>
        internal readonly long KeptStaticRowsCount => keptStaticRowsCount;

        /// <summary>The builds of the dynamic tree from scratch since this state was made. Used by tests.</summary>
        internal readonly long TreeBuildCount => treeBuildCount;

        /// <summary>The builds that refitted the dynamic tree instead of building it. Used by tests.</summary>
        internal readonly long TreeRefitCount => treeRefitCount;

        /// <summary>The refits that built the nodes' filters again because a body's filter had changed. Used by tests.</summary>
        internal readonly long TreeFilterRefitCount => treeFilterRefitCount;

        /// <summary>
        /// Reuse bounds cached by the previous refit. Tests can disable this for comparison.
        /// </summary>
        internal static bool BoundsCacheEnabled
        {
            get => BoundsCacheOff.Data == 0;
            set => BoundsCacheOff.Data = value ? 0 : 1;
        }

        private sealed class BoundsCacheOffKey
        {
        }

        private static readonly Unity.Burst.SharedStatic<int> BoundsCacheOff = Unity.Burst.SharedStatic<int>.GetOrCreate<StepState, BoundsCacheOffKey>();

        /// <summary>
        /// Read bounds cache switches: RETOOLED_BOUNDS_CACHE=0 disables caching;
        /// RETOOLED_BOUNDS_CHECK=1 validates cached bounds; RETOOLED_SHAPE_BOUNDS=0 disables cached collider
        /// geometry.
        /// </summary>
        internal static void ReadBoundsCacheSwitches()
        {
            string cache = Environment.GetEnvironmentVariable("RETOOLED_BOUNDS_CACHE");
            BoundsCacheEnabled = cache != "0";
            string check = Environment.GetEnvironmentVariable("RETOOLED_BOUNDS_CHECK");
            DynamicBoundsCache.CheckEnabled = check == "1";
            string shapes = Environment.GetEnvironmentVariable("RETOOLED_SHAPE_BOUNDS");
            DynamicBoundsCache.ShapesEnabled = shapes != "0";
            if (cache == "0" || check == "1" || shapes == "0")
            {
                UnityEngine.Debug.Log($"[StepState] RETOOLED_BOUNDS_CACHE={cache}, RETOOLED_BOUNDS_CHECK={check}, RETOOLED_SHAPE_BOUNDS={shapes}: bounds cache {(BoundsCacheEnabled ? "on" : "off")}, check {(DynamicBoundsCache.CheckEnabled ? "on" : "off")}, shape bounds {(DynamicBoundsCache.ShapesEnabled ? "on" : "off")}");
            }
        }
        private NativeHashMap<ulong, KnownBodyPair> knownBodyPairs;
        private NativeQueue<DropIgnoredPairsJob.KeyValue> learnedBodyPairs;
        private JobHandle collidersBuilt;

        // The job that ApplyForces leaves running. BuildWorld makes the build wait for it.
        private JobHandle forcesApplied;
        private JobHandle warmStartsIndexed;
        private uint stepCount;
        private uint builtStaticsVersion;
        private uint builtMaterialVersion;

        // False from the start of a build until the query world is published. If a build fails, the next build rebuilds
        // the whole world.
        private bool builtWhole;
        private bool motionsMatchStore;

        internal StepState(WorldData* store, StoreGlobals* globals, QueryWorld* query)
        {
            this = default;
            tombstoneExpiry = -1;
            this.store = store;
            this.globals = globals;
            this.query = query;
            world = new PhysicsWorld(0, 0, 0);
            rebuildStatics = new NativeReference<int>(1, Allocator.Persistent);
            keepStatics = new NativeReference<int>(0, Allocator.Persistent);
            treeShift = ShiftStaticTreeJob.ShiftState.Create();
            dynamicIndices = new NativeList<int>(Allocator.Persistent);
            staticIndices = new NativeList<int>(Allocator.Persistent);
            staticRows = new NativeList<int>(Allocator.Persistent);
            builtIndices = new NativeList<int>(Allocator.Persistent);
            partitionResult = new NativeReference<int>(1, Allocator.Persistent);
            jointGroups = new NativeList<JointGroup>(Allocator.Persistent);
            jointGroupMembers = new NativeList<JointGroupMember>(Allocator.Persistent);
            jointUnits = new NativeList<JointUnit>(Allocator.Persistent);
            jointRoles = new NativeList<JointForestRole>(Allocator.Persistent);
            jointsWritten = new NativeList<int>(Allocator.Persistent);
            jointLoads = new NativeList<JointLoad>(Allocator.Persistent);
            groupLoads = new NativeList<JointLoad>(Allocator.Persistent);
            jointsOverloaded = new NativeList<byte>(Allocator.Persistent);
            jointStates = UnsafeList<JointLoadState>.Create((int)store->jointCount, Allocator.Persistent);
            JointLoadStates.Register(store, jointStates);
            otherJoints = new NativeList<int>(Allocator.Persistent);
            otherGroups = new NativeList<int>(Allocator.Persistent);
            directSolverCache = IslandSolverCache.Create();
            vesselBuild = VesselBuild.Create();
            solverIslandCache = SolverIslandCache.Create();
            queryBodies = new NativeList<QueryBody>(Allocator.Persistent);
            queryShapes = new NativeList<QueryShape>(Allocator.Persistent);
            queryFirstShapes = new NativeList<int>(Allocator.Persistent);
            snapshotDynamic = new NativeList<int>(Allocator.Persistent);
            snapshotStatic = new NativeList<int>(Allocator.Persistent);
            manifolds =new NativeList<ManifoldRecord>(Allocator.Persistent);
            manifoldPoints = new NativeList<ManifoldPoint>(Allocator.Persistent);
            contactBuckets = new NativeList<ContactBucket>(Allocator.Persistent);
            filledContactBuckets = new NativeReference<int>(0, Allocator.Persistent);
            touching = new NativeHashMap<ShapePairKey, ShapePairState>(64, Allocator.Persistent);
            TouchingHandOff.Restore(store, touching);
            removedTriggerPairs = new NativeList<TriggerEvent>(Allocator.Persistent);
            sleepParents = new NativeList<int>(Allocator.Persistent);
            sleepContacts = new NativeList<int>(Allocator.Persistent);
            sleepTouchesStill = new NativeList<bool>(Allocator.Persistent);
            fallingAsleep = new NativeList<int>(Allocator.Persistent);
            sleepRunOut = new NativeArray<int>(1, Allocator.Persistent);
            lastImpulses = new NativeHashMap<PersistentManifoldKey, float>(64, Allocator.Persistent);
            warmStarts = new NativeHashMap<JacobianKey, float2>(64, Allocator.Persistent);
            colliderWork = new NativeList<ColliderBuilder.Work>(Allocator.Persistent);
            colliderResults = new NativeList<long>(Allocator.Persistent);
            wholeColliderBuilds = new NativeReference<uint>(0, Allocator.Persistent);
            colliderChanges = new NativeReference<uint>(0, Allocator.Persistent);
            boundsCache = new DynamicBoundsCache(colliderChanges.GetUnsafePtr(), Allocator.Persistent);
            sharedRefit = (SharedDynamicTreeRefit*)UnsafeUtility.MallocTracked(sizeof(SharedDynamicTreeRefit), UnsafeUtility.AlignOf<SharedDynamicTreeRefit>(), Allocator.Persistent, 0);
            UnsafeUtility.MemClear(sharedRefit, sizeof(SharedDynamicTreeRefit));
            store->refit = sharedRefit;
            store->treeReady = 1;
            treeRows = new NativeList<int>(Allocator.Persistent);
            treeRefitResults = (DynamicTreeRefitResults*)UnsafeUtility.MallocTracked(sizeof(DynamicTreeRefitResults), UnsafeUtility.AlignOf<DynamicTreeRefitResults>(), Allocator.Persistent, 0);
            UnsafeUtility.MemClear(treeRefitResults, sizeof(DynamicTreeRefitResults));
            knownBodyPairs = new NativeHashMap<ulong, KnownBodyPair>(1024, Allocator.Persistent);
            learnedBodyPairs = new NativeQueue<DropIgnoredPairsJob.KeyValue>(Allocator.Persistent);
        }

        internal void Dispose()
        {
            // The refit writes the world that is freed below.
            FinishPendingRefit();
            store->queryWorld = null;
            store->refit = null;
            sharedRefit->Dispose();
            UnsafeUtility.FreeTracked(sharedRefit, Allocator.Persistent);
            sharedRefit = null;

            forcesApplied.Complete();
            collidersBuilt.Complete();

            // A step that threw before FinishStep leaves the fill jobs unfinished, and they write the vessel lists.
            vesselsFilled.Complete();
            ReleaseDeadColliders();
            for (uint i = 0; i < store->bodyCount; i++)
            {
                BodyData* body = store->bodies + i;
                ColliderBuilder.Release((*body).Collider()->Value.GetUnsafePtr());
                ColliderBuilder.SetCollider(body, default);
                body->flags |= BodyFlags.ColliderDirty;
            }

            store->queryDirty = 1;
            world.Dispose();
            rebuildStatics.Dispose();
            keepStatics.Dispose();
            treeShift.Dispose();
            dynamicIndices.Dispose();
            staticIndices.Dispose();
            staticRows.Dispose();
            builtIndices.Dispose();
            partitionResult.Dispose();
            jointGroups.Dispose();
            jointGroupMembers.Dispose();
            jointUnits.Dispose();
            jointRoles.Dispose();
            jointsWritten.Dispose();
            jointLoads.Dispose();
            groupLoads.Dispose();
            jointsOverloaded.Dispose();
            JointLoadStates.Unregister(store);
            UnsafeList<JointLoadState>.Destroy(jointStates);
            otherJoints.Dispose();
            otherGroups.Dispose();
            directSolverCache.Dispose();
            vesselBuild.Dispose();
            solverIslandCache.Dispose();
            queryBodies.Dispose();
            queryShapes.Dispose();
            queryFirstShapes.Dispose();
            snapshotDynamic.Dispose();
            snapshotStatic.Dispose();
            manifolds.Dispose();
            manifoldPoints.Dispose();
            for (int i = 0; i < contactBuckets.Length; i++)
            {
                contactBuckets[i].Dispose();
            }

            contactBuckets.Dispose();
            filledContactBuckets.Dispose();
            touching.Dispose();
            removedTriggerPairs.Dispose();
            sleepParents.Dispose();
            sleepContacts.Dispose();
            sleepTouchesStill.Dispose();
            fallingAsleep.Dispose();
            sleepRunOut.Dispose();
            lastImpulses.Dispose();
            warmStarts.Dispose();
            colliderWork.Dispose();
            colliderResults.Dispose();
            wholeColliderBuilds.Dispose();
            boundsCache.Dispose();
            colliderChanges.Dispose();
            treeRows.Dispose();
            UnsafeUtility.FreeTracked(treeRefitResults, Allocator.Persistent);
            treeRefitResults = null;
            knownBodyPairs.Dispose();
            learnedBodyPairs.Dispose();
        }

        /// <summary>The direct solver's cache, including the counts of solves that used the forest.</summary>
        internal readonly IslandSolverCache IslandSolverCache => directSolverCache;

        /// <summary>The world built for the simulation systems to advance.</summary>
        internal readonly PhysicsWorld World => world;

        /// <summary>The index of each store body in the built world.</summary>
        internal readonly NativeList<int> BuiltIndices => builtIndices;

        internal readonly int StaticsChanged => staticsChanged ? 1 : 0;

        private readonly NativeReference<int>.ReadOnly BuildStaticTree => (staticsChanged ? rebuildStatics : keepStatics).AsReadOnly();

        /// <summary>The settings that the simulation systems use to build their input for a step.</summary>
        internal readonly PhysicsStep Settings
        {
            get
            {
                PhysicsStep settings = PhysicsStep.Default;
                settings.Gravity = float3.zero;
                settings.DirectIslandSubsteps = SolverConstants.DirectIslandSubsteps;
                settings.SolverIterationCount = SolverConstants.SolverIterations;
                settings.MaxDynamicDepenetrationVelocity = SolverConstants.MaxDepenetrationVelocity;
                settings.MaxStaticDepenetrationVelocity = SolverConstants.MaxDepenetrationVelocity;

                // FinishStep synchronizes the collision world while reading the results of the step.
                settings.SynchronizeCollisionWorld = 0;
                return settings;
            }
        }

        /// <summary>
        /// Wake bodies through joints, then schedule force integration.
        /// BuildWorld waits for the force jobs; the main thread may read flags while they run.
        /// </summary>
        internal void ApplyForces(float deltaTime)
        {
            // WorldState.Step has already finished the last step's refit. This is for a step that runs the systems
            // some other way.
            FinishPendingRefit();
            if (s_ThrowBeforeApplyForces.Data != 0)
            {
                ThrowBeforeApplyForces();
            }

            // A step that threw after this stage last time can have left the job running.
            forcesApplied.Complete();
            store->queryWorld = null;
            store->queryDirty = StoreMemory.QueryStale;
            BodyEntities.PrepareForJobs(store);
            VehicleKernel.RecordStartVelocities(store);

            using (BackendMarkers.StepApplyForces.Auto())
            {
                // Wake refiltered pairs before joint wake propagation and the narrow phase.
                uint filterVersion = PairFilterVersion.Version;
                if (filterVersion != seenPairFilterVersion)
                {
                    seenPairFilterVersion = filterVersion;
                    if (!touching.IsEmpty)
                    {
                        new WakeRefilteredPairsJob { World = store, Touching = touching }.Run();
                    }
                }

                new WakeJointedBodiesJob { World = store }.Run();

                forcesApplied = new ApplyForcesJob
                {
                    Bodies = store->bodies,
                    Gravity = store->gravity,
                    DeltaTime = deltaTime,
                }.Schedule((int)store->bodyCount, ParallelBatchSize);
                JobHandle.ScheduleBatchedJobs();
            }
        }

        /// <summary>Builds the bodies, joints and broadphase for the simulation systems to advance.</summary>
        internal JobHandle BuildWorld(float deltaTime, JobHandle dependency)
        {
            using var span = BackendMarkers.StepSchedule.Auto();

            // Every job below waits for the forces, including the collider jobs, which write the flags that the forces
            // job reads.
            dependency = JobHandle.CombineDependencies(dependency, forcesApplied);
            JobHandle colliders = RefreshColliders(dependency);
            JobHandle broadphase = Build(deltaTime, colliders, out JobHandle joints);

            using (BackendMarkers.BuildScheduleBroadphase.Auto())
            {
                // Build joints alongside the broadphase. Only dispatch pair creation needs to wait for them.
                world.DynamicsWorld.JointsBuilt = joints;
                return ScheduleWheels(deltaTime, broadphase, joints);
            }
        }

        /// <summary>
        /// Schedule wheel updates after the broadphase and before consumers of wheel-modified velocities and
        /// joints.
        /// Run only with wheels and a positive timestep.
        /// </summary>
        private JobHandle ScheduleWheels(float deltaTime, JobHandle broadphase, JobHandle joints)
        {
            if (store->vehicleCount == 0 || deltaTime <= 0.0f)
            {
                return broadphase;
            }

            JobHandle wheels = new WheelJob
            {
                Store = store,
                Globals = globals,
                Collision = world.CollisionWorld,
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                BuiltIndices = builtIndices.AsArray(),
                Joints = world.DynamicsWorld.InternalJointsList,
                AnchorIndex = dynamicIndices.Length + staticIndices.Length,
                DeltaTime = deltaTime,
            }.Schedule(JobHandle.CombineDependencies(broadphase, joints));
            world.DynamicsWorld.JointsBuilt = wheels;
            return wheels;
        }

        /// <summary>The store updated by this simulation state. Used by tests.</summary>
        internal readonly WorldData* Store => store;

        /// <summary>The manifolds of the last step, with the impulses that they started with. Used by tests.</summary>
        internal readonly NativeList<ManifoldRecord> Manifolds => manifolds;

        /// <summary>
        /// Save touching pairs for the next StepState before discarding this one on detach.
        /// </summary>
        internal readonly void SaveTouching()
        {
            TouchingHandOff.Save(store, globals, touching);
        }

        /// <summary>The vessels compiled during the last build. Used by tests.</summary>
        internal readonly VesselBuild Vessels => vesselBuild;

        /// <summary>The set of vessels supplied to the step during the last build. Used by tests.</summary>
        internal readonly VesselSet VesselSet => vesselSet;

        /// <summary>
        /// Remove unsolved overlaps before island finding and supply this step's vessels.
        /// </summary>
        internal JobHandle SortIslands(SimulationSingleton singleton, ref SimulationStepInput input, JobHandle dependency)
        {
            using var span = BackendMarkers.StepSchedule.Auto();
            input.VesselSet = vesselSet;

            Simulation* simulation = singleton.AsSimulationPtr();
            if (input.World.NumDynamicBodies == 0)
            {
                return dependency;
            }

            NativeList<DispatchPairSequencer.DispatchPair> pairs = simulation->StepContext.PairsToSortByIsland;
            JobHandle filtered = new DropIgnoredPairsJob
            {
                Pairs = pairs.AsDeferredJobArray(),
                Store = store->bodies,
                Globals = globals,
                Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
                Known = knownBodyPairs,
                WholeBuilds = wholeColliderBuilds,
                Learned = learnedBodyPairs.AsParallelWriter(),
                World = store,
                Units = jointUnits.AsDeferredJobArray(),
                BodyVessels = vesselBuild.BodyVessels.AsArray(),
                VesselActive = vesselBuild.Active.AsArray(),
                VesselPairUnits = vesselBuild.PairUnits,
            }.Schedule(pairs, ParallelBatchSize, dependency);

            // No job in this step reads the newly learned body pairs. Saving those pairs can therefore run in parallel
            // with finding islands.
            JobHandle remembered = new RememberBodyPairsJob
            {
                Known = knownBodyPairs,
                Learned = learnedBodyPairs,
                Bodies = (int)store->bodyCount,
            }.Schedule(filtered);

            SimulationStepInput keeping = input;
            keeping.SolverIslandCache = solverIslandCache;
            return JobHandle.CombineDependencies(simulation->ScheduleIslandJobs(keeping, filtered), remembered);
        }

        /// <summary>Reads the contacts produced by the narrow phase before the Jacobians are built from them.</summary>
        internal JobHandle ReadContacts(SimulationSingleton singleton, float deltaTime, JobHandle dependency)
        {
            using var span = BackendMarkers.StepSchedule.Auto();

            // Submit the batched jobs so that workers can run them while the main thread schedules the jobs below.
            JobHandle.ScheduleBatchedJobs();

            // Drop impulses for removed shapes before new contacts can reuse their addresses.
            if (store->removedShapeCount != 0 && lastImpulses.Count != 0)
            {
                new ForgetRemovedImpulsesJob { World = store, LastImpulses = lastImpulses }.Run();
            }

            manifolds.Clear();
            manifoldPoints.Clear();
            JobHandle read = dependency;
            JobHandle merged = dependency;
            NativeStream contacts = singleton.AsSimulationPtr()->StepContext.Contacts;
            if (contacts.IsCreated)
            {
                NativeStream.Reader reader = contacts.AsReader();
                read = new PrepareContactBucketsJob
                {
                    Contacts = reader,
                    Buckets = contactBuckets,
                    Filled = filledContactBuckets,
                }.Schedule(read);
                read = new ReadContactsJob
                {
                    Contacts = reader,
                    Buckets = contactBuckets,
                    Store = store->bodies,
                    Globals = globals,
                    Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
                    LastImpulses = lastImpulses,
                    DeltaTime = deltaTime,
                    BounceThreshold = store->bounceThreshold,
                }.ScheduleUnsafe(contacts, 1, read);

                merged = new MergeContactBucketsJob
                {
                    Filled = filledContactBuckets,
                    Buckets = contactBuckets,
                    Records = manifolds,
                    Points = manifoldPoints,
                }.Schedule(read);
            }

            // Only warm starting requires the merged manifolds before solving. Merging the manifolds can therefore run
            // in parallel with building the Jacobians.
            warmStartsIndexed = new IndexWarmStartsJob { Records = manifolds, WarmStarts = warmStarts }.Schedule(merged);

            // If there are no contacts, no more bodies need to be woken. Bodies connected by joints were already woken
            // at the start of the step.
            if (!contacts.IsCreated)
            {
                return read;
            }

            return new WakeTouchedBodiesJob
            {
                World = store,
                Filled = filledContactBuckets,
                Buckets = contactBuckets,
                DynamicIndices = dynamicIndices.AsArray(),
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
            }.Schedule(read);
        }

        /// <summary>Warm starts the Jacobians.</summary>
        internal JobHandle PrepareSolve(SimulationSingleton singleton, ref SimulationStepInput input, JobHandle dependency)
        {
            using var span = BackendMarkers.StepSchedule.Auto();

            directSolverCache.Forest = new ForestSettings
            {
                Enabled = ForestSwitches.Enabled || VesselSwitches.Enabled,
                BoundaryRowCap = ForestSwitches.BoundaryRowCap,
                Locking = ForestSwitches.Locking,
                LockAngle = ForestSwitches.LockAngle,
                YieldAngle = ForestSwitches.YieldAngle,
                MinimumHoldSteps = ForestSwitches.MinimumHoldSteps,
                DeflectionSmoothingSteps = ForestSwitches.DeflectionSmoothingSteps,
                Caps = ForestSwitches.Caps,
            };
            input.IslandSolverCache = directSolverCache;
            input.VesselSet = vesselSet;
            preparedSimulation = singleton.AsSimulationPtr();
            preparedInputs = JobHandle.CombineDependencies(dependency, warmStartsIndexed, vesselsFilled);
            return new WarmStartJob
            {
                WarmStarts = warmStarts,
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
                MotionCount = dynamicIndices.Length,
            }.Schedule(singleton, ref world, JobHandle.CombineDependencies(dependency, warmStartsIndexed, vesselsFilled));
        }

        /// <summary>
        /// Read impulse and vessel loads, then break overloaded joints.
        /// Wait for body pose write-back before converting impulse events to world space.
        /// </summary>
        private JobHandle ReadJointForces(NativeStream events, float deltaTime, JobHandle vesselsSettled)
        {
            const int groupsPerBatch = 16;

            // ScheduleVesselLoads has already sized the joint loads.
            int jointCount = jointLoads.Length;
            groupLoads.ResizeUninitialized(jointGroups.Length);
            UnsafeUtility.MemClear(groupLoads.GetUnsafePtr(), (long)groupLoads.Length * sizeof(JointLoad));

            bool listed = vesselSet.IsCreated;
            int seededJoints = listed ? otherJoints.Length : jointCount;
            int sharedGroups = listed ? otherGroups.Length : jointGroups.Length;
            JobHandle loaded = new JointLoads.SeedJointLoadsJob
            {
                World = store,
                States = jointStates->Ptr,
                Loads = jointLoads.AsArray(),
                Joints = otherJoints.AsArray(),
                Listed = listed,
            }.Schedule(seededJoints, ParallelBatchSize);
            if (events.IsCreated)
            {
                JobHandle rotated = new JointLoads.RotateJointImpulsesJob
                {
                    Events = events.AsReader(),
                    World = store,
                    Groups = jointGroups.AsArray(),
                    DeltaTime = deltaTime,
                }.ScheduleUnsafe(events, 1, default);

                loaded = new JointLoads.SumJointImpulsesJob
                {
                    Events = events.AsReader(),
                    JointLoads = jointLoads.AsArray(),
                    GroupLoads = groupLoads.AsArray(),
                }.Schedule(JobHandle.CombineDependencies(loaded, rotated));
            }

            JobHandle shared = new JointLoads.ShareGroupLoadsJob
            {
                World = store,
                Groups = jointGroups.AsArray(),
                Members = jointGroupMembers.AsArray(),
                GroupLoads = groupLoads.AsArray(),
                JointLoads = jointLoads.AsArray(),
                ListedGroups = otherGroups.AsArray(),
                Listed = listed,
            }.Schedule(sharedGroups, groupsPerBatch, loaded);

            JobHandle settled = new JointLoads.SettleJointLoadsJob
            {
                World = store,
                States = jointStates->Ptr,
                Loads = jointLoads.AsArray(),
                Overloaded = jointsOverloaded.AsArray(),
                Joints = otherJoints.AsArray(),
                Listed = listed,
                DeltaTime = deltaTime,
            }.Schedule(seededJoints, ParallelBatchSize, shared);

            settled = JobHandle.CombineDependencies(settled, vesselsSettled);
            return new JointBreaking.BreakJointsJob
            {
                World = store,
                Globals = globals,
                States = jointStates->Ptr,
                Overloaded = jointsOverloaded.AsArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(settled);
        }

        /// <summary>
        /// Convert test-supplied vessel loads to joint loads and process breaking as at step completion.
        /// </summary>
        internal void ReadVesselLoads(float deltaTime)
        {
            JobHandle vesselsSettled = ScheduleVesselLoads(deltaTime, default);
            ReadJointForces(default, deltaTime, vesselsSettled).Complete();
        }

        /// <summary>
        /// Schedule vessel joint load extraction after solving. Read solved motion poses without waiting for
        /// write-back.
        /// </summary>
        private JobHandle ScheduleVesselLoads(float deltaTime, JobHandle dependency)
        {
            const int vesselLoadsPerBatch = 32;

            // The jobs write the joint loads through arrays that the safety system doesn't track, so the list must not
            // change size after this point in the step.
            jointLoads.ResizeUninitialized((int)store->jointCount);
            jointsOverloaded.ResizeUninitialized((int)store->jointCount);
            if (jointStates->Length != (int)store->jointCount)
            {
                throw new InvalidOperationException("the joints' load states and the store's joints differ in number");
            }

            if (!vesselSet.IsCreated)
            {
                return default;
            }

            return new VesselLoads.VesselLoadsJob
            {
                World = store,
                States = jointStates->Ptr,
                Units = jointUnits.AsArray(),
                Groups = jointGroups.AsArray(),
                Members = jointGroupMembers.AsArray(),
                MotionDatas = world.MotionDatas,
                BuiltIndices = builtIndices.AsArray(),
                DeltaTime = deltaTime,
                PartUnits = vesselBuild.PartUnits.AsArray(),
                ClosureUnits = vesselBuild.ClosureUnits.AsArray(),
                PartIsBody0 = vesselBuild.PartIsBody0.AsArray(),
                UnitVessels = vesselBuild.UnitVessels.AsArray(),
                Active = vesselBuild.Active.AsArray(),
                EdgeLoads = vesselBuild.EdgeLoads.AsArray(),
                ClosureLoads = vesselBuild.ClosureLoads.AsArray(),
                JointLoads = jointLoads.AsArray(),
                Overloaded = jointsOverloaded.AsArray(),
            }.Schedule(vesselBuild.PartUnits.Length + vesselBuild.ClosureUnits.Length, vesselLoadsPerBatch, dependency);
        }

        /// <summary>Finishes the step by writing the solved world back to the store and to Unity.</summary>
        internal void FinishStep(Simulation simulation, float deltaTime)
        {
            // Schedule post-solve work now. Include manifold and warm-start dependencies even when nothing is
            // solved.
            JobHandle solved = JobHandle.CombineDependencies(simulation.FinalJobHandle, warmStartsIndexed, vesselsFilled);

            // The jobs below read only the solved motions, so they don't wait for the jobs that free the solve's
            // buffers and export the events of its direct islands. Those jobs only read the motions too.
            JobHandle motionsSolved = JobHandle.CombineDependencies(simulation.MotionsSolvedHandle, warmStartsIndexed, vesselsFilled);

            // Schedule write-back first because the main thread waits for it first.
            JobHandle writtenBack = new WriteBackJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
            }.Schedule(dynamicIndices.Length, ParallelBatchSize, motionsSolved);

            // Vessel loads read solved motions, so they can run alongside pose write-back.
            JobHandle vesselsSettled = ScheduleVesselLoads(deltaTime, motionsSolved);

            // Refit the dynamic tree to solved poses alongside write-back. The step returns without waiting for
            // the refit.
            // Clear treeReady until completion; later world access must finish the refit, and queries can help
            // it.
            // Snapshot copying reads only Static flags, which these concurrent jobs do not change.
            refit = SharedDynamicTreeRefit.Schedule(sharedRefit, ref world, deltaTime, float3.zero, motionsSolved,
                boundsCacheBuilt ? boundsCache : default, &store->treeReady, JobsUtility.JobWorkerCount + 1);
            refitPending = true;
            refitInputs = motionsSolved;

            // If the step throws, wait for solved motions before returning: queries helping the pending refit
            // read them.
            JobHandle snapshot;
            bool waitedForSolve = false;
            try
            {
                if (s_ThrowAfterRefitScheduled.Data != 0)
                {
                    ThrowAfterRefitScheduled();
                }

                snapshot = ScheduleQuerySnapshot(motionsSolved);
                JobHandle.ScheduleBatchedJobs();

                using (BackendMarkers.StepWaitForSolve.Auto())
                {
                    solved.Complete();
                }

                waitedForSolve = true;
            }
            finally
            {
                if (!waitedForSolve)
                {
                    motionsSolved.Complete();
                }
            }

            // The build of this step has completed, so every entry of the bounds cache that a body could take holds
            // the body's current collider: the build wrote it, and the refit writes it again now.
            if (boundsCacheBuilt)
            {
                boundsCache.MarkCurrent();
            }

            if (DynamicBoundsCache.CheckEnabled)
            {
                LogBoundsCheck();
            }

            // SleepContactsJob must read the manifolds before the contact report sorts them, so schedule it now.
            JobHandle contactsJoined = new SleepContactsJob
            {
                World = store,
                Records = manifolds,
                Parents = sleepParents,
                Contacts = sleepContacts,
                TouchesStill = sleepTouchesStill,
            }.Schedule();
            JobHandle.ScheduleBatchedJobs();

            using (BackendMarkers.StepWriteBack.Auto())
            {
                writtenBack.Complete();
            }

            JobHandle forces;
            using (BackendMarkers.StepReadJointForces.Auto())
            {
                forces = ReadJointForces(simulation.ImpulseEvents.EventDataStream, deltaTime, vesselsSettled);
            }

            // Wake bodies affected by shape removal after joint load reads but before sleep and contact
            // reporting.
            // Sleep grouping must also wait for joint breaking.
            if (store->removedShapeCount != 0)
            {
                forces.Complete();
                new ForgetRemovedShapesJob { World = store, Touching = touching, RemovedTriggerPairs = removedTriggerPairs }.Run();
            }

            JobHandle counted = new UpdateWakeCountersJob
            {
                World = store,
                Contacts = sleepContacts,
                TouchesStill = sleepTouchesStill,
                RunOut = (int*)sleepRunOut.GetUnsafePtr(),
                DeltaTime = deltaTime,
            }.Schedule((int)store->bodyCount, ParallelBatchSize, contactsJoined);
            JobHandle sleep = new SleepJob
            {
                World = store,
                Parents = sleepParents,
                Falling = fallingAsleep,
                RunOut = (int*)sleepRunOut.GetUnsafePtr(),
            }.Schedule(JobHandle.CombineDependencies(counted, forces));
            JobHandle.ScheduleBatchedJobs();

            // ReadImpulsesJob reads only the motions, which the rebuild of the tree doesn't change.
            using (BackendMarkers.StepReadImpulses.Auto())
            {
                if (world.NumDynamicBodies != 0)
                {
                    new ReadImpulsesJob { Records = manifolds, Events = simulation.CollisionEvents, Built = world }.Run();
                }

                new RememberImpulsesJob { Records = manifolds, LastImpulses = lastImpulses }.Run();
            }

            using (BackendMarkers.StepPublishContacts.Auto())
            {
                // The contact report sorts the manifolds that SleepContactsJob reads, so wait for that job first.
                contactsJoined.Complete();
                new PublishContactsJob
                {
                    World = store,
                    Globals = globals,
                    Records = manifolds,
                    Points = manifoldPoints,
                    Touching = touching,
                    RemovedTriggerPairs = removedTriggerPairs,
                    Step = ++stepCount,
                }.Run();
            }

            // Publish as current unless write-back changed poses. Queries finish the pending refit before
            // reading.
            using (BackendMarkers.StepPublishQueryWorld.Auto())
            {
                snapshot.Complete();
                FinishQuerySnapshot();
                // A refit that failed has set the ready word too, and leaves the world stale.
                bool treeReady = store->treeReady != 0;
                store->queryDirty = !motionsMatchStore || (treeReady && sharedRefit->Failed) ? StoreMemory.QueryStale
                    : !treeReady ? StoreMemory.QueryTreePending : 0u;
                PublishQueryWorld();
            }

            // FallAsleepJob sets the Sleeping flag. The snapshot reads only the Static flag, so it doesn't need to finish
            // first, although it has.
            using (BackendMarkers.StepSleep.Auto())
            {
                sleep.Complete();
                new FallAsleepJob { World = store, Falling = fallingAsleep }.Run();
            }
        }

        /// <summary>
        /// Updates the built world to match the store's poses without advancing time. Main thread only.
        /// </summary>
        internal void BuildForQueries()
        {
            FinishPendingRefit();

            // The world that the last step published is current now that its refit has finished.
            if (store->queryDirty == StoreMemory.QueryTreePending && builtWhole)
            {
                return;
            }

            store->queryWorld = null;
            BodyEntities.PrepareForJobs(store);

            // The last build is still valid except for the poses of static bodies. The query snapshot contains no data
            // that depends on a pose.
            if (store->queryDirty == StoreMemory.QueryStaticsMoved && builtWhole && globals->materialVersion == builtMaterialVersion)
            {
                PoseStatics();
                PublishQueryWorld();
                return;
            }

            // A step that threw between ApplyForces and BuildWorld can have left the forces job running, and the collider
            // jobs write the flags that it reads.
            forcesApplied.Complete();
            JobHandle colliders = RefreshColliders(default);
            JobHandle broadphase = Build(0.0f, colliders, out _);
            ScheduleQuerySnapshot(broadphase).Complete();
            broadphase.Complete();
            if (boundsCacheBuilt)
            {
                boundsCache.MarkCurrent();
            }
            FinishQuerySnapshot();
            PublishQueryWorld();
        }

        /// <summary>
        /// Complete the pending dynamic tree refit on the main thread before reading or modifying the built world
        /// or bounds cache.
        /// Call outside jobs; query jobs use HelpPendingRefit instead.
        /// </summary>
        internal void FinishPendingRefit()
        {
            if (!refitPending)
            {
                return;
            }

            using (BackendMarkers.StepWaitForTree.Auto())
            {
                // A job whose refit threw is complete too: the thread that threw marked the refit failed, and the
                // other threads stopped waiting for its piece.
                refit.Complete();

                // The jobs have done all of the work unless a test told them to leave it.
                WaitForRefit();
            }

            refit = default;
            refitPending = false;
            NoteFailedRefit();
        }

        /// <summary>
        /// Invalidate the published world after a refit fails. The next main-thread query rebuilds it;
        /// workers may continue querying pre-step poses until then.
        /// </summary>
        private void NoteFailedRefit()
        {
            if (!sharedRefit->Failed)
            {
                return;
            }

            // The refit may have stopped between moving a body and writing the body's bounds, so no entry of the
            // bounds cache can be trusted. The next build computes every body's bounds again.
            if (boundsCache.IsCreated)
            {
                boundsCache.Resize(0);
            }

            if (store->queryDirty == StoreMemory.QueryTreePending)
            {
                store->queryDirty = StoreMemory.QueryStale;
                UnityEngine.Debug.LogError("[StepState] the refit of the dynamic tree after the step failed; the next query on the main thread builds the world again");
            }
        }

        /// <summary>
        /// Finish pending refit work without completing a job handle, for callers already inside a job.
        /// FinishPendingRefit later completes the now-empty scheduled jobs.
        /// </summary>
        internal void HelpPendingRefit()
        {
            if (refitPending)
            {
                WaitForRefit();
            }
        }

        /// <summary>
        /// Does the refit's remaining work on this thread, and then waits for the threads that run its last pieces, if
        /// any.
        /// </summary>
        private void WaitForRefit()
        {
            sharedRefit->Help();
            int* ready = &store->treeReady;
            while (System.Threading.Interlocked.CompareExchange(ref *ready, 0, -1) == 0)
            {
                AddressWait.Wait(ready, 0);
                Unity.Burst.Intrinsics.Common.Pause();
            }
        }

        /// <summary>Whether a refit that the last step left running may still be running. Used by tests.</summary>
        internal readonly bool RefitPending => refitPending;

        /// <summary>
        /// Whether solved motions are ready for pending refit helpers. Used by tests.
        /// </summary>
        internal readonly bool RefitInputsComplete => !refitPending || refitInputs.IsCompleted;

        /// <summary>
        /// For tests: whether the next step throws right after it schedules the refit of the dynamic tree, while the
        /// solve's jobs still run. Only code that runs in Mono throws; Burst leaves the throw out.
        /// </summary>
        internal static bool ThrowAfterRefitScheduledForTests
        {
            get => s_ThrowAfterRefitScheduled.Data != 0;
            set => s_ThrowAfterRefitScheduled.Data = value ? 1 : 0;
        }

        /// <summary>For tests: the number of throws that <see cref="ThrowAfterRefitScheduledForTests"/> made.</summary>
        internal static int ThrowsAfterRefitScheduled => s_ThrowsAfterRefitScheduled;

        private static int s_ThrowsAfterRefitScheduled;

        [Unity.Burst.BurstDiscard]
        private static void ThrowAfterRefitScheduled()
        {
            s_ThrowAfterRefitScheduled.Data = 0;
            s_ThrowsAfterRefitScheduled++;
            throw new InvalidOperationException("a throw that a test injected into the step after the refit was scheduled");
        }

        private sealed class ThrowAfterRefitScheduledKey
        {
        }

        /// <summary>
        /// For tests: whether the next step throws before it applies the forces, which is also before it refreshes
        /// the pointers of the store's rows. Only code that runs in Mono throws; Burst leaves the throw out.
        /// </summary>
        internal static bool ThrowBeforeApplyForcesForTests
        {
            get => s_ThrowBeforeApplyForces.Data != 0;
            set => s_ThrowBeforeApplyForces.Data = value ? 1 : 0;
        }

        /// <summary>For tests: the number of throws that <see cref="ThrowBeforeApplyForcesForTests"/> made.</summary>
        internal static int ThrowsBeforeApplyForces => s_ThrowsBeforeApplyForces;

        private static int s_ThrowsBeforeApplyForces;

        [Unity.Burst.BurstDiscard]
        private static void ThrowBeforeApplyForces()
        {
            s_ThrowBeforeApplyForces.Data = 0;
            s_ThrowsBeforeApplyForces++;
            throw new InvalidOperationException("a throw that a test injected into the step before it applies the forces");
        }

        private sealed class ThrowBeforeApplyForcesKey
        {
        }

        private static readonly Unity.Burst.SharedStatic<int> s_ThrowBeforeApplyForces = Unity.Burst.SharedStatic<int>.GetOrCreate<StepState, ThrowBeforeApplyForcesKey>();

        private static readonly Unity.Burst.SharedStatic<int> s_ThrowAfterRefitScheduled = Unity.Burst.SharedStatic<int>.GetOrCreate<StepState, ThrowAfterRefitScheduledKey>();

        /// <summary>
        /// Whether the published query world waits only for the last step's refit, so that it is current once the refit
        /// has finished.
        /// </summary>
        internal readonly bool OnlyRefitPending => store->queryDirty == StoreMemory.QueryTreePending && builtWhole && store->queryWorld != null;

        /// <summary>
        /// Set treeReady after a failed step unless a pending refit owns completion, preventing queries from
        /// waiting forever.
        /// </summary>
        internal void ReleaseTreeAfterThrow()
        {
            if (!refitPending)
            {
                System.Threading.Interlocked.Exchange(ref store->treeReady, 1);
                AddressWait.WakeAll(&store->treeReady);
            }
        }

        /// <summary>
        /// Logs the check's mismatches of the bounds cache when there are new ones, and the counts every 1,000 steps.
        /// </summary>
        private static void LogBoundsCheck()
        {
            DynamicBoundsCache.CheckCounts* counts = DynamicBoundsCache.Counts;
            if (++counts->Steps % 1000 == 0)
            {
                // Burst compiles this method into the step, and it can't join strings, so each message is one string.
                UnityEngine.Debug.Log($"[BoundsCheck] {counts->Steps} steps: {counts->Mismatches} mismatches in {counts->Hits} bodies that took their bounds from the cache");
                UnityEngine.Debug.Log($"[BoundsCheck] {counts->ShapeMismatches} mismatches in {counts->ShapeHits} bodies that computed their bounds from the shape that the cache held");
            }

            if (counts->Mismatches != counts->LoggedMismatches)
            {
                counts->LoggedMismatches = counts->Mismatches;
                UnityEngine.Debug.LogError($"[BoundsCheck] {counts->Mismatches} mismatches, among the {counts->Hits} bodies that took their bounds from the cache and the bodies that computed them from its shapes; last at index {counts->LastIndex}, entity {counts->LastEntity}, computed min {counts->LastComputedMin}, cached min {counts->LastCachedMin}");
            }
        }

        /// <summary>
        /// Update built statics and their tree from store poses, recording the version to avoid redundant work.
        /// </summary>
        private void PoseStatics()
        {
            builtStaticsVersion = store->staticsVersion;
            staticsChanged = true;
            treeMatchesHeld = true;
            JobHandle posed = new PoseStaticsJob
            {
                Store = store->bodies,
                StaticIndices = staticIndices.AsArray(),
                StaticBodies = world.StaticBodies,
            }.Schedule(staticIndices.Length, ParallelBatchSize, default);

            world.CollisionWorld.ScheduleUpdateStaticTree(ref world, BuildStaticTree, ScheduleShiftStaticTree(posed), true).Complete();
        }

        /// <summary>
        /// Schedules the broadphase trees. Only building the static tree depends on the decision to rebuild it.
        /// Building the dynamic tree can run in parallel with that decision.
        /// </summary>
        private JobHandle ScheduleTrees(float deltaTime, JobHandle built)
        {
            int threads = JobsUtility.JobWorkerCount + 1;
            Broadphase broadphase = world.CollisionWorld.Broadphase;
            JobHandle statics = broadphase.ScheduleStaticTreeBuildJobs(ref world, threads, BuildStaticTree, ScheduleShiftStaticTree(built));
            boundsCacheBuilt = BoundsCacheEnabled;
            DynamicBoundsCache cache = boundsCacheBuilt ? boundsCache : default;
            JobHandle dynamics;
            if (RefitsDynamicTree())
            {
                dynamics = broadphase.ScheduleDynamicTreeRefitBuildJobs(ref world, deltaTime, float3.zero, built, cache, treeRefitResults);
                treeRefitted = true;
                treeRefits++;
                treeRefitCount++;
            }
            else
            {
                dynamics = broadphase.ScheduleDynamicTreeBuildJobs(ref world, deltaTime, float3.zero, threads, built, cache);
                treeRows.CopyFrom(dynamicIndices.AsArray());
                treeRefitted = false;
                treeRefits = 0;
                treeBaseQuality = 0.0f;
                treeBuildCount++;
            }

            return JobHandle.CombineDependencies(statics, dynamics);
        }

        /// <summary>
        /// Reuse the dynamic tree while the build completed, dynamic body rows and order match, and count and
        /// quality limits permit.
        /// Refitting preserves collision/query results, though traversal and multi-hit ordering can change.
        /// </summary>
        private bool RefitsDynamicTree()
        {
            if (treeRefitted && treeMatchesHeld)
            {
                // The prior refit has completed. Use the first refit after a build as the quality baseline.
                if (treeRefitResults->FiltersChanged != 0)
                {
                    treeFilterRefitCount++;
                }

                float quality = treeRefitResults->Quality;
                if (treeBaseQuality == 0.0f)
                {
                    treeBaseQuality = quality;
                }
                else if (quality > treeBaseQuality * MaxTreeQualityGrowth && !TreeAlwaysRefitted)
                {
                    return false;
                }
            }

            return TreeReuseEnabled && treeMatchesHeld && dynamicIndices.Length > 0
                && (treeRefits < MaxTreeRefits || TreeAlwaysRefitted) && Same(treeRows, dynamicIndices);
        }

        /// <summary>
        /// Reuse static body rows and the anchor when the previous build completed, staticsVersion is unchanged,
        /// and static row order matches. The caller must also verify that body storage was retained.
        /// </summary>
        private readonly bool KeepsStaticRows()
        {
            return StaticReuseEnabled && !staticsChanged && world.NumStaticBodies == staticIndices.Length + 1 && Same(staticRows, staticIndices);
        }

        /// <summary>
        /// Move retained static rows and the anchor after the new dynamic body range when its size changes.
        /// Call only while the built world is inaccessible to queries and jobs.
        /// </summary>
        private void MoveStaticRows(int dynamicBefore)
        {
            if (dynamicBefore == dynamicIndices.Length)
            {
                return;
            }

            RigidBody* bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies);
            UnsafeUtility.MemMove(bodies + dynamicIndices.Length, bodies + dynamicBefore, (long)(staticIndices.Length + 1) * sizeof(RigidBody));
        }

        /// <summary>
        /// After the static bodies are built, decides whether to rebuild or shift the static tree. If no static body
        /// changed, leaves the tree unchanged.
        /// </summary>
        private JobHandle ScheduleShiftStaticTree(JobHandle built)
        {
            if (!staticsChanged)
            {
                return built;
            }

            using var span = BackendMarkers.ShiftStaticTree.Auto();
            Broadphase.Tree tree = world.CollisionWorld.Broadphase.StaticTree;
            return new ShiftStaticTreeJob
            {
                StaticBodies = world.StaticBodies,
                Nodes = tree.Nodes.GetUnsafeList(),
                MeshSubAabbs = tree.MeshSubAabbs.GetUnsafeList(),
                State = treeShift,
                Store = store->bodies,
                Refreshed = colliderWork,
                TreeMatchesHeld = treeMatchesHeld,
                Margin = world.CollisionWorld.CollisionTolerance * 0.5f,
                Rebuild = rebuildStatics,
            }.Schedule(built);
        }

        /// <summary>
        /// Return whether safety handles permit container writes before starting a build.
        /// A main-thread query inside a job must keep using the published world if access is denied.
        /// After checking access outside jobs, pass the containers to build jobs again to restore ownership.
        /// </summary>
        internal bool CanWriteContainers()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            try
            {
                // These calls check write access without changing the containers.
                staticIndices.GetUnsafePtr();
                dynamicIndices.GetUnsafePtr();
            }
            catch (InvalidOperationException)
            {
                return false;
            }
#endif
            return true;
        }

        /// <summary>
        /// Copy query data from the store, or return dependency unchanged if the snapshot is still current.
        /// Do not retain pointers into movable store memory. Call FinishQuerySnapshot after completion.
        /// </summary>
        private JobHandle ScheduleQuerySnapshot(JobHandle dependency)
        {
            uint version = QuerySnapshotVersion.Version;
            if (snapshotTaken && version == snapshotVersion && Same(snapshotDynamic, dynamicIndices) && Same(snapshotStatic, staticIndices))
            {
                keptSnapshots++;
                return dependency;
            }

            snapshotTaken = false;
            snapshotVersion = version;
            snapshotDynamic.CopyFrom(dynamicIndices.AsArray());
            snapshotStatic.CopyFrom(staticIndices.AsArray());
            queryBodies.ResizeUninitialized(world.NumBodies);
            queryFirstShapes.ResizeUninitialized(world.NumBodies);
            JobHandle found = new QuerySnapshot.FindFirstShapesJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                StaticIndices = staticIndices.AsArray(),
                FirstShapes = queryFirstShapes.AsArray(),
                QueryShapes = queryShapes,
            }.Schedule(dependency);

            return new QuerySnapshot.SnapshotQueryDataJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                StaticIndices = staticIndices.AsArray(),
                FirstShapes = queryFirstShapes.AsArray(),
                QueryBodies = queryBodies.AsArray(),
                QueryShapes = queryShapes.AsDeferredJobArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(world.NumBodies, ParallelBatchSize, found);
        }

        /// <summary>Records that the snapshot scheduled by <see cref="ScheduleQuerySnapshot"/> has completed.</summary>
        private void FinishQuerySnapshot()
        {
            snapshotTaken = true;
        }

        private static bool Same(NativeList<int> kept, NativeList<int> built)
        {
            return kept.Length == built.Length
                && UnsafeUtility.MemCmp(kept.GetUnsafeReadOnlyPtr(), built.GetUnsafeReadOnlyPtr(), (long)kept.Length * sizeof(int)) == 0;
        }

        /// <summary>The number of steps and query builds that kept the last query snapshot. Used by tests.</summary>
        internal readonly long KeptSnapshots => keptSnapshots;

        /// <summary>
        /// Publish the built world and completed snapshot for queries.
        /// Colliders remain valid until the next build acquires the exclusive query lock.
        /// </summary>
        private void PublishQueryWorld()
        {
            query->Collision = world.CollisionWorld;
            query->Bodies = queryBodies.GetUnsafeReadOnlyPtr();
            query->Shapes = queryShapes.GetUnsafePtr();
            query->ShapeCount = queryShapes.Length;
            store->queryWorld = query;
            builtWhole = true;
        }

        /// <summary>
        /// Schedule requested collider updates. Consumers must wait for the returned handle before reading
        /// colliders.
        /// </summary>
        private JobHandle RefreshColliders(JobHandle dependency)
        {
            using var span = BackendMarkers.RefreshColliders.Auto();

            // Every step and query build completes the previous build, so this handle has already completed.
            collidersBuilt.Complete();
            ColliderBuilder.AskAgainForFailed(store, colliderWork, colliderResults);
            ReleaseDeadColliders();

            // Material changes can replace colliders at the same addresses. Invalidate static rows and the
            // static tree.
            uint materialVersion = globals->materialVersion;
            bool all = materialVersion != builtMaterialVersion;
            if (all)
            {
                store->staticsVersion++;
            }

            collidersBuilt = ColliderBuilder.ScheduleRefresh(store, all, colliderWork, colliderResults,
                wholeColliderBuilds, colliderChanges, dependency);
            builtMaterialVersion = materialVersion;
            return collidersBuilt;
        }

        private void ReleaseDeadColliders()
        {
            for (uint i = 0; i < store->deadColliderCount; i++)
            {
                ColliderBuilder.Release(store->deadColliders[i]);
            }

            store->deadColliderCount = 0;
        }

        /// <summary>
        /// Schedule body, broadphase and joint builds. Return tree completion and provide joint completion
        /// separately.
        /// </summary>
        private JobHandle Build(float deltaTime, JobHandle dependency, out JobHandle joints)
        {
            // An interrupted step may leave vessel fill jobs running. Finish them before resizing vessel lists.
            vesselsFilled.Complete();
            using (BackendMarkers.BuildPartition.Auto())
            {
                new PartitionBodiesJob
                {
                    World = store,
                    DynamicIndices = dynamicIndices,
                    StaticIndices = staticIndices,
                    BuiltIndices = builtIndices,
                    PosedByWriteBack = PosedByWriteBack,
                    MotionsMatchStore = partitionResult,
                }.Run();
                motionsMatchStore = partitionResult.Value != 0;

                // Indices in the static tree are relative to the static bodies. Queries and pair finding add the number
                // of dynamic bodies, so only changes to static bodies require rebuilding this tree.
                staticsChanged = !builtWhole || store->staticsVersion != builtStaticsVersion;
                builtStaticsVersion = store->staticsVersion;
                treeMatchesHeld = builtWhole;
                builtWhole = false;
            }

            // Compare cached bounds only when body storage and dynamic count are retained and the cache remains
            // valid.
            void* bodiesBefore = NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies);
            bool compareBodies = boundsCacheBuilt && BoundsCacheEnabled && world.NumDynamicBodies == dynamicIndices.Length;
            int dynamicBefore = world.NumDynamicBodies;
            bool keepStaticRows = KeepsStaticRows();

            // Reset the world with no joints; the joint build job will fill them. Unity Physics uses deferred arrays to
            // read the joints, so jobs can be scheduled before the joint count is known.
            using (BackendMarkers.BuildReset.Auto())
            {
                world.Reset(staticIndices.Length + 1, dynamicIndices.Length, 0);
                world.DynamicsWorld.EnableDirectSolver = true;
            }

            bool sameMemory = NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies) == bodiesBefore;
            compareBodies &= sameMemory;
            keepStaticRows &= sameMemory;
            if (keepStaticRows)
            {
                keptStaticRowsCount++;
                MoveStaticRows(dynamicBefore);
            }
            else if (StaticReuseEnabled)
            {
                // With the reuse off, every build writes the rows, so the rows that they come from don't matter.
                staticRows.CopyFrom(staticIndices.AsArray());
            }

            DynamicBoundsCache.View boundsView = BoundsCacheEnabled ? boundsCache.Resize(dynamicIndices.Length) : default;

            // Schedule bodies and trees first: pair creation and island finding depend on them.
            // Joint compilation and vessel work can proceed alongside them.
            JobHandle broadphase;
            using (BackendMarkers.BuildScheduleBodies.Auto())
            {
                var bodiesJob = new BuildBodiesJob
                {
                    Store = store->bodies,
                    DynamicIndices = dynamicIndices.AsArray(),
                    StaticIndices = staticIndices.AsArray(),
                    Bodies = world.Bodies,
                    MotionDatas = world.MotionDatas,
                    MotionVelocities = world.MotionVelocities,
                    DeltaTime = deltaTime,
                    Same = boundsView.Same,
                    CompareBodies = compareBodies,
                    KeepStatics = keepStaticRows,
                };

                JobHandle bodiesBuilt = bodiesJob.Schedule(bodiesJob.Count, ParallelBatchSize, dependency);
                JobHandle.ScheduleBatchedJobs();
                broadphase = ScheduleTrees(deltaTime, bodiesBuilt);
                JobHandle.ScheduleBatchedJobs();
            }

            // Queries need no solver joints. During simulation, joint builds wait for collider jobs that write
            // body flags.
            joints = dependency;
            if (deltaTime > 0.0f)
            {
                using var jointsSpan = BackendMarkers.BuildJoints.Auto();

                // Reuse the groups if the topology and forest setting have not changed since they were found. The joint
                // build refreshes group data that depends on other state.
                bool vessels = VesselSwitches.Enabled;
                bool forest = ForestSwitches.Enabled || vessels;
                uint topology = JointTopology.Version;
                JobHandle groupsFound = dependency;
                bool regrouped = !grouped || topology != groupedTopology || forest != groupedForest;
                if (regrouped)
                {
                    groupsFound = new JointGrouping.GroupJointsJob
                    {
                        World = store,
                        Groups = jointGroups,
                        Members = jointGroupMembers,
                        Units = jointUnits,
                        Roles = jointRoles,
                        Written = jointsWritten,
                        Forest = forest,
                    }.Schedule(dependency);
                    grouped = true;
                    groupedTopology = topology;
                    groupedForest = forest;
                }

                // Recompile vessels when topology or forest settings change, support is enabled, or a tombstone
                // expires.
                // Run synchronously because island scheduling needs the vessel count.
                bool tombstoneExpired = tombstoneExpiry >= 0 && (int)stepCount >= tombstoneExpiry;
                bool compiledThisStep = false;
                if (vessels && (!compiled || regrouped || tombstoneExpired))
                {
                    compiledThisStep = true;
                    groupsFound.Complete();
                    new VesselCompiler.CompileVesselsJob
                    {
                        World = store,
                        Units = jointUnits.AsArray(),
                        Roles = jointRoles.AsArray(),
                        Groups = jointGroups.AsArray(),
                        Members = jointGroupMembers.AsArray(),
                        Build = vesselBuild,
                        Step = (int)stepCount,
                        TombstoneSteps = VesselSwitches.TombstoneSteps,
                        MinimumParts = VesselSwitches.MinimumParts,
                    }.Run();
                    compiled = true;
                    tombstoneExpiry = vesselBuild.TombstoneExpiry(VesselSwitches.TombstoneSteps);
                }
                else if (!vessels && compiled)
                {
                    vesselBuild.Clear(0);
                    compiled = false;
                    tombstoneExpiry = -1;
                }

                vesselSet = default;
                if (vessels)
                {
                    vesselBuild.ChooseStepVessels(store, builtIndices.AsArray(), dynamicIndices.Length);
                    vesselSet = vesselBuild.Set();
                }

                // The number of joints produced by each unit is not known in advance. JointTranslation.BuildJointsJob therefore writes a
                // separate stream range for each unit.
                JobHandle allocated = NativeStream.ScheduleConstruct(out NativeStream stream, jointUnits, groupsFound, Allocator.TempJob);
                JobHandle emitted = new JointTranslation.BuildJointsJob
                {
                    World = store,
                    States = jointStates->Ptr,
                    BuiltIndices = builtIndices.AsArray(),
                    Units = jointUnits.AsDeferredJobArray(),
                    Output = stream.AsWriter(),
                    Written = jointsWritten.AsDeferredJobArray(),
                    Groups = jointGroups.AsDeferredJobArray(),
                    Members = jointGroupMembers.AsDeferredJobArray(),
                    DynamicCount = dynamicIndices.Length,
                    AnchorIndex = dynamicIndices.Length + staticIndices.Length,
                    DeltaTime = deltaTime,
                    Roles = jointRoles.AsDeferredJobArray(),
                    Locking = forest && ForestSwitches.Locking,
                    LockFraction = ForestSwitches.LockFraction,
                    YieldFraction = ForestSwitches.YieldFraction,
                    Vessels = vesselSet.IsCreated,
                    UnitVessels = vesselBuild.UnitVessels.AsArray(),
                    VesselActive = vesselBuild.Active.AsArray(),
                }.Schedule(jointUnits, JointBatchSize, allocated);

                // Fill vessel constraints alongside joint and broadphase work; only the solve needs their
                // results.
                vesselsFilled = default;
                if (vesselSet.IsCreated)
                {
                    // Reuse vessel constraints unless motion frames or joint settings changed.
                    // Check per-joint settings stamps only when the shared version changes.
                    uint settings = JointSettings.Version;
                    bool resized = vesselBuild.SizeCaches();
                    bool refillAll = resized || compiledThisStep || !cacheFilled;
                    bool settingsChanged = settings != cachedSettings;
                    cacheFilled = true;
                    cachedSettings = settings;
                    JobHandle edgesFilled = new VesselEdgeFill.FillVesselEdgesJob
                    {
                        World = store,
                        Units = jointUnits.AsDeferredJobArray(),
                        Groups = jointGroups.AsDeferredJobArray(),
                        Members = jointGroupMembers.AsDeferredJobArray(),
                        UnitVessels = vesselBuild.UnitVessels.AsArray(),
                        Active = vesselBuild.Active.AsArray(),
                        PartStoreBodies = vesselBuild.PartStoreBodies.AsArray(),
                        PartUnits = vesselBuild.PartUnits.AsArray(),
                        PartParents = vesselBuild.PartParents.AsArray(),
                        PartIsBody0 = vesselBuild.PartIsBody0.AsArray(),
                        PartVesselFirst = vesselBuild.PartVesselFirst.AsArray(),
                        Edges = vesselBuild.Edges.AsArray(),
                        RefillAll = refillAll,
                        SettingsChanged = settingsChanged,
                        EdgeCache = vesselBuild.EdgeCache.AsArray(),
                        EdgeCacheFrames = vesselBuild.EdgeCacheFrames.AsArray(),
                        EdgeCacheStamps = vesselBuild.EdgeCacheStamps.AsArray(),
                    }.Schedule(vesselBuild.PartUnits.Length, ParallelBatchSize, groupsFound);

                    JobHandle closuresFilled = new VesselClosureFill.FillVesselClosuresJob
                    {
                        World = store,
                        Units = jointUnits.AsDeferredJobArray(),
                        Groups = jointGroups.AsDeferredJobArray(),
                        Members = jointGroupMembers.AsDeferredJobArray(),
                        UnitVessels = vesselBuild.UnitVessels.AsArray(),
                        Active = vesselBuild.Active.AsArray(),
                        ClosureUnits = vesselBuild.ClosureUnits.AsArray(),
                        ClosurePartA = vesselBuild.ClosurePartA.AsArray(),
                        ClosurePartB = vesselBuild.ClosurePartB.AsArray(),
                        Vessels = vesselBuild.Vessels.AsArray(),
                        PartStoreBodies = vesselBuild.PartStoreBodies.AsArray(),
                        Closures = vesselBuild.Closures.AsArray(),
                        RefillAll = refillAll,
                        SettingsChanged = settingsChanged,
                        ClosureCache = vesselBuild.ClosureCache.AsArray(),
                        ClosureCacheFrames = vesselBuild.ClosureCacheFrames.AsArray(),
                        ClosureCacheStamps = vesselBuild.ClosureCacheStamps.AsArray(),
                    }.Schedule(vesselBuild.ClosureUnits.Length, ParallelBatchSize, groupsFound);
                    // The joint loads at the end of the step need this list. Waiting for JointTranslation.BuildJointsJob keeps the job
                    // from reading a group while JointTranslation.BuildJointsJob writes it.
                    JobHandle listed = new VesselLoads.ListOtherJointsJob
                    {
                        Units = jointUnits.AsDeferredJobArray(),
                        Groups = jointGroups.AsDeferredJobArray(),
                        Members = jointGroupMembers.AsDeferredJobArray(),
                        UnitVessels = vesselBuild.UnitVessels.AsArray(),
                        Active = vesselBuild.Active.AsArray(),
                        OtherJoints = otherJoints,
                        OtherGroups = otherGroups,
                    }.Schedule(emitted);
                    JobHandle loadsCleared = vesselBuild.ScheduleClearLoads(dependency);
                    vesselsFilled = JobHandle.CombineDependencies(JobHandle.CombineDependencies(edgesFilled, closuresFilled, listed), loadsCleared);
                }

                JobHandle laidOut = new JointGrouping.LayOutJointsJob
                {
                    Written = jointsWritten,
                    Output = world.DynamicsWorld.InternalJointsList,
                }.Schedule(emitted);

                JobHandle copied = new JointGrouping.CopyJointsJob
                {
                    Input = stream.AsReader(),
                    FirstJoints = jointsWritten.AsDeferredJobArray(),
                    Output = world.DynamicsWorld.InternalJointsList.AsDeferredJobArray(),
                }.Schedule(jointsWritten, CopyBatchSize, laidOut);

                joints = stream.Dispose(copied);
            }

            return broadphase;
        }
    }
}
