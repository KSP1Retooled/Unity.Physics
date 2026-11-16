using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The simulation state for a world, owned by <see cref="WorldState"/>. It is stored in unmanaged memory so that
    /// Burst can run an entire step: iterate over the bodies, schedule the stages and read their results.
    /// </summary>
    internal unsafe struct StepState
    {
        private const int ParallelBatchSize = 64;

        // Building a joint costs much more than building a body, so each batch contains fewer joints.
        private const int JointBatchSize = 8;

        // Copying the joints for one unit requires copying a few hundred bytes.
        private const int CopyBatchSize = 64;

        // WriteBackJob assigns these bodies a pose that differs from the result of integrating their solved motion.
        private const NativeStore.BodyFlags PosedByWriteBack = NativeStore.BodyFlags.Sleeping | NativeStore.BodyFlags.HasKinematicTarget;

        private readonly NativeStore.WorldData* store;
        private readonly NativeStore.StoreGlobals* globals;
        private readonly QueryWorld* query;

        private PhysicsWorld world;

        // The jobs for the static tree read one of these references to determine whether to rebuild the tree. After
        // initialization, keepStatics is never written. Only ShiftStaticTreeJob writes rebuildStatics, and each build
        // schedules that job before the jobs for the tree.
        private NativeReference<int> rebuildStatics;
        private NativeReference<int> keepStatics;
        private ShiftStaticTreeJob.ShiftState treeShift;
        private bool staticsChanged;
        private bool treeMatchesHeld;
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

        // For each joint, 1 if its smoothed load exceeded its break limits in this step, for BreakJointsJob. The jobs
        // that settle the loads write the flags. A byte for each joint is much less memory to write and to scan than
        // a load.
        private NativeList<byte> jointsOverloaded;

        // The joints and groups outside the active vessels of this step. See ListOtherJointsJob.
        private NativeList<int> otherJoints;
        private NativeList<int> otherGroups;
        private DirectSolverCache directSolverCache;

        // The simulation whose solve PrepareSolve prepared in this step, or null when no solve has been prepared since
        // the last FinishStep. FinishStepSystem takes it from here instead of reading the simulation singleton. Reading
        // the singleton on this thread would wait for every job of the solve, including the jobs that free its
        // buffers, before FinishStep could schedule the write-back.
        internal Simulation* preparedSimulation;

        // The jobs that the solve waits for: every job of the step before the solve. FinishStepSystem completes them
        // before FinishStep, because FinishStep reads lists that those jobs write, such as the joint groups, when it
        // schedules the jobs after the solve.
        internal JobHandle preparedInputs;

        // The vessels compiled from the joints, and whether compilation has been completed for the current groups.
        private VesselBuild vesselBuild;
        private VesselSet vesselSet;
        private bool compiled;

        // Whether the vessel build holds cached edges and closures, and the joint settings version that they were
        // computed from. See FillVesselEdgesJob.RefillAll.
        private bool cacheFilled;
        private uint cachedSettings;

        // The jobs that fill the vessels' edges and closures for this step. Only the solve waits for them.
        private JobHandle vesselsFilled;

        // The step at which the oldest tombstone in the vessels expires, or -1. See CompileVesselsJob.
        private int tombstoneExpiry;
        private SolverIslandCache solverIslandCache;
        private NativeList<QueryBody> queryBodies;
        private NativeList<QueryShape> queryShapes;
        private NativeList<int> queryFirstShapes;

        // The query snapshot holds no poses, so a step can keep the last one when nothing that it copied has changed.
        // The snapshot is kept while QuerySnapshotVersion has the value read when it was taken, and while the moving
        // and static bodies are built from the same store rows in the same order. snapshotTaken is false from the
        // moment a snapshot is scheduled until it completes, so a step that throws in between takes a new one.
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

        /// <summary>
        /// Whether the build of the dynamic tree takes the bounds that the refit of the last step left in the cache.
        /// Tests turn it off to compare the steps with and without the cache. Burst compiles the code that reads it,
        /// so it lives in a shared static, which holds whether the cache is off.
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

        internal StepState(NativeStore.WorldData* store, NativeStore.StoreGlobals* globals, QueryWorld* query)
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
            otherJoints = new NativeList<int>(Allocator.Persistent);
            otherGroups = new NativeList<int>(Allocator.Persistent);
            directSolverCache = DirectSolverCache.Create();
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
            knownBodyPairs = new NativeHashMap<ulong, KnownBodyPair>(1024, Allocator.Persistent);
            learnedBodyPairs = new NativeQueue<DropIgnoredPairsJob.KeyValue>(Allocator.Persistent);
        }

        internal void Dispose()
        {
            store->queryWorld = null;

            forcesApplied.Complete();
            collidersBuilt.Complete();

            // A step that threw before FinishStep leaves the fill jobs unfinished, and they write the vessel lists.
            vesselsFilled.Complete();
            ReleaseDeadColliders();
            for (uint i = 0; i < store->bodyCount; i++)
            {
                NativeStore.BodyData* body = store->bodies + i;
                ColliderBuilder.Release(body->collider->Value.GetUnsafePtr());
                body->collider->Value = default;
                body->flags |= NativeStore.BodyFlags.ColliderDirty;
            }

            store->queryDirty = 1;
            world.Dispose();
            rebuildStatics.Dispose();
            keepStatics.Dispose();
            treeShift.Dispose();
            dynamicIndices.Dispose();
            staticIndices.Dispose();
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
            knownBodyPairs.Dispose();
            learnedBodyPairs.Dispose();
        }

        /// <summary>The direct solver's cache, including the counts of solves that used the forest.</summary>
        internal readonly DirectSolverCache DirectSolverCache => directSolverCache;

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
                settings.DirectIslandSubsteps = WorldState.DirectIslandSubsteps;
                settings.SolverIterationCount = WorldState.SolverIterations;
                settings.MaxDynamicDepenetrationVelocity = WorldState.MaxDepenetrationVelocity;
                settings.MaxStaticDepenetrationVelocity = WorldState.MaxDepenetrationVelocity;

                // FinishStep synchronizes the collision world while reading the results of the step.
                settings.SynchronizeCollisionWorld = 0;
                return settings;
            }
        }

        /// <summary>
        /// The first stage of a step, which integrates forces into the store's velocities. The job that does this is
        /// left running: <see cref="BuildWorld"/> makes every job that it schedules wait for it, so the main thread can
        /// schedule the build while the workers apply the forces. Between the two stages, the main thread reads only
        /// the bodies' flags, which the job doesn't write. Waking bodies through joints writes the flags, so it runs here
        /// on the main thread before the job starts.
        /// </summary>
        internal void ApplyForces(float deltaTime)
        {
            // A step that threw after this stage last time can have left the job running.
            forcesApplied.Complete();
            store->queryWorld = null;
            store->queryDirty = StoreOps.QueryStale;
            BodyEntities.PrepareForJobs(store);
            VehicleKernel.RecordStartVelocities(store);

            using (BackendMarkers.StepApplyForces.Auto())
            {
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
                // The joints are first read when dispatch pairs are created, after overlaps have been found. Building
                // the joints can therefore run in parallel with building the trees and finding overlaps. Only the job
                // that creates dispatch pairs must wait for the joints.
                world.DynamicsWorld.JointsBuilt = joints;
                return ScheduleWheels(deltaTime, broadphase, joints);
            }
        }

        /// <summary>
        /// Updates the WheelColliders after the broadphase is ready for ray queries and before any job reads the
        /// velocities that the wheels change. All stages after the build wait for this job, so it runs only when there
        /// are wheels and deltaTime is positive. The job adds suspension limits to the joints. Jobs that depend on the
        /// joints must therefore also wait for the wheels.
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
        internal readonly NativeStore.WorldData* Store => store;

        /// <summary>The manifolds of the last step, with the impulses that they started with. Used by tests.</summary>
        internal readonly NativeList<ManifoldRecord> Manifolds => manifolds;

        /// <summary>The vessels compiled during the last build. Used by tests.</summary>
        internal readonly VesselBuild Vessels => vesselBuild;

        /// <summary>The set of vessels supplied to the step during the last build. Used by tests.</summary>
        internal readonly VesselSet VesselSet => vesselSet;

        /// <summary>
        /// Removes overlaps that will not be solved, then groups the remaining pairs into islands. The code that finds
        /// islands needs the vessels for this step, so they are added to the input here.
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

            // The store reuses the memory of a removed shape for the next shape that it makes, so a new shape can have
            // the address of a shape that was removed since the last step. The last impulses are kept by the shapes'
            // addresses, so a new contact would then start with the impulse of a contact that no longer exists. Drop
            // the impulses of removed shapes before the contacts read them. ForgetRemovedShapesJob clears the list of
            // removed shapes later in the step.
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
                Enabled = VesselForest.Enabled || VesselPipeline.Enabled,
                BoundaryRowCap = VesselForest.BoundaryRowCap,
                Locking = VesselForest.Locking,
                LockAngle = VesselForest.LockAngle,
                YieldAngle = VesselForest.YieldAngle,
                MinimumHoldSteps = VesselForest.MinimumHoldSteps,
                DeflectionSmoothingSteps = VesselForest.DeflectionSmoothingSteps,
                Caps = VesselForest.Caps,
            };
            input.DirectSolverCache = directSolverCache;
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
        /// Reads joint loads from the step's impulse events and vessel loads, then breaks overloaded joints. See
        /// <see cref="JointForceJobs"/>. The poses of the joint bodies must be written back first.
        /// </summary>
        ///
        /// <remarks>
        /// Schedules a chain of jobs in parallel with updating the tree and creating the query snapshot. Rotating the
        /// impulse events runs in parallel with initializing the joint loads. The impulses are added to the loads after
        /// both jobs finish. When the step has vessels, VesselLoadsJob handles the joints of active vessels by itself,
        /// beside that chain, and the chain handles only the other joints, which ListOtherJointsJob listed.
        /// </remarks>
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
            JobHandle loaded = new SeedJointLoadsJob
            {
                World = store,
                Loads = jointLoads.AsArray(),
                Joints = otherJoints.AsArray(),
                Listed = listed,
            }.Schedule(seededJoints, ParallelBatchSize);
            if (events.IsCreated)
            {
                JobHandle rotated = new RotateJointImpulsesJob
                {
                    Events = events.AsReader(),
                    World = store,
                    Groups = jointGroups.AsArray(),
                    DeltaTime = deltaTime,
                }.ScheduleUnsafe(events, 1, default);

                loaded = new SumJointImpulsesJob
                {
                    Events = events.AsReader(),
                    JointLoads = jointLoads.AsArray(),
                    GroupLoads = groupLoads.AsArray(),
                }.Schedule(JobHandle.CombineDependencies(loaded, rotated));
            }

            JobHandle shared = new ShareGroupLoadsJob
            {
                World = store,
                Groups = jointGroups.AsArray(),
                Members = jointGroupMembers.AsArray(),
                GroupLoads = groupLoads.AsArray(),
                JointLoads = jointLoads.AsArray(),
                ListedGroups = otherGroups.AsArray(),
                Listed = listed,
            }.Schedule(sharedGroups, groupsPerBatch, loaded);

            JobHandle settled = new SettleJointLoadsJob
            {
                World = store,
                Loads = jointLoads.AsArray(),
                Overloaded = jointsOverloaded.AsArray(),
                Joints = otherJoints.AsArray(),
                Listed = listed,
                DeltaTime = deltaTime,
            }.Schedule(seededJoints, ParallelBatchSize, shared);

            settled = JobHandle.CombineDependencies(settled, vesselsSettled);
            return new BreakJointsJob
            {
                World = store,
                Globals = globals,
                Overloaded = jointsOverloaded.AsArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(settled);
        }

        /// <summary>
        /// Converts vessel loads into joint loads, smooths them and breaks overloaded joints, as at the end of a step.
        /// Tests use this after writing vessel loads by hand following a step.
        /// </summary>
        internal void ReadVesselLoads(float deltaTime)
        {
            JobHandle vesselsSettled = ScheduleVesselLoads(deltaTime, default);
            ReadJointForces(default, deltaTime, vesselsSettled).Complete();
        }

        /// <summary>
        /// Sizes the joint loads for this step, and schedules <see cref="VesselLoadsJob"/> when the step has vessels.
        /// Vessel joints do not produce impulse events. The solver writes their loads into the vessel set instead. The
        /// dependency must include the solve. The job reads the poses of the vessel bodies from their solved motions,
        /// so it doesn't need to wait for the poses to be written back.
        /// </summary>
        private JobHandle ScheduleVesselLoads(float deltaTime, JobHandle dependency)
        {
            const int vesselLoadsPerBatch = 32;

            // The jobs write the joint loads through arrays that the safety system doesn't track, so the list must not
            // change size after this point in the step.
            jointLoads.ResizeUninitialized((int)store->jointCount);
            jointsOverloaded.ResizeUninitialized((int)store->jointCount);
            if (!vesselSet.IsCreated)
            {
                return default;
            }

            return new VesselLoadsJob
            {
                World = store,
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
            // This thread usually gets here long before the solve ends. Schedule the work that needs only the solved
            // motions now, so that the workers start it as soon as the solve ends, rather than when this thread wakes
            // up and schedules it. Solving depends on warm starting, which waits for the merged manifolds. A step with
            // nothing to solve does not have that dependency, so add it explicitly here.
            JobHandle solved = JobHandle.CombineDependencies(simulation.FinalJobHandle, warmStartsIndexed, vesselsFilled);

            // The jobs below read only the solved motions, so they don't wait for the jobs that free the solve's
            // buffers and export the events of its direct islands. Those jobs only read the motions too.
            JobHandle motionsSolved = JobHandle.CombineDependencies(simulation.MotionsSolvedHandle, warmStartsIndexed, vesselsFilled);

            // The workers start the jobs that become ready together in the order in which this thread scheduled them.
            // Schedule the write-back first, because the step waits for it first. When the tree's jobs came first, they
            // took the workers, and the write-back started up to 0.1 ms after the solve ended.
            JobHandle writtenBack = new WriteBackJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
            }.Schedule(dynamicIndices.Length, ParallelBatchSize, motionsSolved);

            // The loads of the vessel joints come from the solve, not from impulse events. A group shares its load in
            // the frame of its leader's body, but the job takes that body's pose from its solved motion, so it runs
            // beside the write-back. Neither job writes what the other reads.
            JobHandle vesselsSettled = ScheduleVesselLoads(deltaTime, motionsSolved);

            // Only queries read the dynamic tree at this stage. Move it to the solved poses while the motions are
            // written back, and while this thread schedules the jobs that read joint loads and publishes the contacts.
            // The build at the start of this step made the tree for the same bodies in the same order, so fitting its
            // bounds to the new poses is enough. Queries then find the same hits as in a rebuilt tree, but they can
            // report the hits of a query that returns many in a different order. Neither Unity nor PhysX promises an
            // order for those queries, and the order already changed from step to step as the bodies moved.
            // The query snapshot reads only the store, so it doesn't wait for the tree either. The snapshot also
            // runs while WriteBackJob and ForgetRemovedShapesJob write the bodies' flags. Of those flags, the snapshot
            // reads only Static, and neither job changes it.
            JobHandle tree = world.CollisionWorld.ScheduleRefitDynamicTree(ref world, deltaTime, float3.zero, motionsSolved,
                boundsCacheBuilt ? boundsCache : default);
            JobHandle snapshot = ScheduleQuerySnapshot(motionsSolved);
            JobHandle.ScheduleBatchedJobs();

            using (BackendMarkers.StepWaitForSolve.Auto())
            {
                solved.Complete();
            }

            // The build of this step has completed, so every entry of the bounds cache that a body could take holds
            // the body's current collider: the build wrote it, and the refit writes it again now.
            if (boundsCacheBuilt)
            {
                boundsCache.MarkCurrent();
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

            // Removing a shape wakes bodies that it supported. Do this after the joint jobs read which bodies are
            // asleep, but before the sleep jobs and contact report read that state. Waiting here is needed only in the
            // rare case of removed shapes. The remaining sleep jobs and query snapshot do not need the impulses or
            // contact report, so workers can run them while this thread waits for the tree and publishes contacts.
            // SleepJob groups bodies through unbroken joints, so it must wait for BreakJointsJob.
            if (store->removedShapeCount != 0)
            {
                forces.Complete();
                new ForgetRemovedShapesJob { World = store, Touching = touching }.Run();
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
                    Step = ++stepCount,
                }.Run();
            }

            // The built world matches the motions written back during this step unless a body received a different pose
            // during write-back. In that case, queries need a rebuild. The store cannot otherwise change during a step.
            using (BackendMarkers.StepWaitForTree.Auto())
            {
                tree.Complete();
            }

            using (BackendMarkers.StepPublishQueryWorld.Auto())
            {
                snapshot.Complete();
                FinishQuerySnapshot();
                store->queryDirty = motionsMatchStore ? 0u : StoreOps.QueryStale;
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
            store->queryWorld = null;
            BodyEntities.PrepareForJobs(store);

            // The last build is still valid except for the poses of static bodies. The query snapshot contains no data
            // that depends on a pose.
            if (store->queryDirty == StoreOps.QueryStaticsMoved && builtWhole && globals->materialVersion == builtMaterialVersion)
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
        /// Updates the built static bodies and static tree to match the poses in the store. Recording the version
        /// avoids repeating this work next step unless more static bodies move first.
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
            JobHandle dynamics = broadphase.ScheduleDynamicTreeBuildJobs(ref world, deltaTime, float3.zero, threads, built, boundsCacheBuilt ? boundsCache : default);
            return JobHandle.CombineDependencies(statics, dynamics);
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
        /// Returns false when safety checks prevent writes to the containers. This happens when the main thread runs a
        /// job while waiting for other work: the safety system releases containers from completed jobs only when the
        /// main thread accesses them outside a job. Checking before starting a build leaves the published world
        /// readable for the rest of the current job. With safety checks disabled, the build can proceed and the current
        /// job sees the completed build.
        /// <para>
        /// Checking access outside a job releases the containers being checked. Every build must therefore pass those
        /// containers to a job again. Both a full build and <see cref="PoseStatics"/> pass the static indices to jobs.
        /// Checking the dynamic indices also covers worlds with no static bodies, which always require a full build.
        /// </para>
        ///
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
        /// Copies the store data needed by queries for <see cref="PublishQueryWorld"/>. Workers read this snapshot
        /// until the next publication, so it must not contain pointers to memory that the store can move or free.
        /// Returns <paramref name="dependency"/> without scheduling anything when the last snapshot still matches the
        /// store. The caller must call <see cref="FinishQuerySnapshot"/> after the returned handle completes.
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
            JobHandle found = new FindFirstShapesJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                StaticIndices = staticIndices.AsArray(),
                FirstShapes = queryFirstShapes.AsArray(),
                QueryShapes = queryShapes,
            }.Schedule(dependency);

            return new SnapshotQueryDataJob
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
        /// Makes the built world and completed snapshot available to query functions. Sharing the colliders is safe
        /// because they are released only during the next build, while the query lock is held exclusively.
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
        /// Schedules the collider updates requested by the store. The bodies have their updated colliders when the
        /// returned handle completes. No collider is read before the jobs that depend on that handle run.
        /// </summary>
        private JobHandle RefreshColliders(JobHandle dependency)
        {
            using var span = BackendMarkers.RefreshColliders.Auto();

            // Every step and query build completes the previous build, so this handle has already completed.
            collidersBuilt.Complete();
            ColliderBuilder.AskAgainForFailed(store, colliderWork, colliderResults);
            ReleaseDeadColliders();

            uint materialVersion = globals->materialVersion;
            collidersBuilt = ColliderBuilder.ScheduleRefresh(store, materialVersion != builtMaterialVersion, colliderWork, colliderResults,
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
        /// Schedules the body build, the broadphase trees and the joint build. Returns the handle for the trees and
        /// supplies the joint handle separately, so the broadphase does not need to wait for the joints.
        /// </summary>
        private JobHandle Build(float deltaTime, JobHandle dependency, out JobHandle joints)
        {
            // FinishStep waits for the jobs that fill the vessels' edges and closures. A step that threw before
            // FinishStep leaves them running, and this build compiles the vessels again and resizes their lists. In the
            // usual case, the jobs have already finished.
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

            // The bounds cache holds the bounds of the bodies that the world holds now. The body build compares each
            // dynamic body with the body that it overwrites, but only if the world keeps its bodies' memory and the
            // same number of dynamic bodies, and if the last build and refit kept the cache.
            void* bodiesBefore = NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies);
            bool compareBodies = boundsCacheBuilt && BoundsCacheEnabled && world.NumDynamicBodies == dynamicIndices.Length;

            // Reset the world with no joints; the joint build job will fill them. Unity Physics uses deferred arrays to
            // read the joints, so jobs can be scheduled before the joint count is known.
            using (BackendMarkers.BuildReset.Auto())
            {
                world.Reset(staticIndices.Length + 1, dynamicIndices.Length, 0);
                world.DynamicsWorld.EnableDirectSolver = true;
            }

            compareBodies &= NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies) == bodiesBefore;
            DynamicBoundsCache.View boundsView = BoundsCacheEnabled ? boundsCache.Resize(dynamicIndices.Length) : default;

            // The bodies and the broadphase trees lead the longest chain before the solve: the pairs and islands wait
            // for the trees, and the trees wait for the bodies. Schedule them before this thread groups and compiles
            // the joints and chooses the step's vessels, and before the joint and vessel fill jobs, so that the workers
            // take them first. The joint jobs read none of what these jobs write, and the trees don't need the joints.
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
                };

                JobHandle bodiesBuilt = bodiesJob.Schedule(bodiesJob.Count, ParallelBatchSize, dependency);
                JobHandle.ScheduleBatchedJobs();
                broadphase = ScheduleTrees(deltaTime, bodiesBuilt);
                JobHandle.ScheduleBatchedJobs();
            }

            // Queries do not run the solver, so build joints only for a simulation step, which also supplies the time
            // step for motors. Like the body build jobs, the joint build jobs wait for collider jobs because those jobs
            // write body flags that the joint build reads.
            joints = dependency;
            if (deltaTime > 0.0f)
            {
                using var jointsSpan = BackendMarkers.BuildJoints.Auto();

                // Reuse the groups if the topology and forest setting have not changed since they were found. The joint
                // build refreshes group data that depends on other state.
                bool vessels = VesselPipeline.Enabled;
                bool forest = VesselForest.Enabled || vessels;
                uint topology = JointTopology.Version;
                JobHandle groupsFound = dependency;
                bool regrouped = !grouped || topology != groupedTopology || forest != groupedForest;
                if (regrouped)
                {
                    groupsFound = new GroupJointsJob
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

                // Vessels are compiled from joint groups and forest roles. Compile them when vessel support is first
                // enabled or when the groups are rebuilt because the topology or forest setting changed. Compilation
                // runs synchronously because the vessel count must be known before scheduling the jobs that find
                // islands.
                // A tombstone also needs a compilation when it expires.
                bool tombstoneExpired = tombstoneExpiry >= 0 && (int)stepCount >= tombstoneExpiry;
                bool compiledThisStep = false;
                if (vessels && (!compiled || regrouped || tombstoneExpired))
                {
                    compiledThisStep = true;
                    groupsFound.Complete();
                    new CompileVesselsJob
                    {
                        World = store,
                        Units = jointUnits.AsArray(),
                        Roles = jointRoles.AsArray(),
                        Groups = jointGroups.AsArray(),
                        Members = jointGroupMembers.AsArray(),
                        Build = vesselBuild,
                        Step = (int)stepCount,
                        TombstoneSteps = VesselPipeline.TombstoneSteps,
                        MinimumParts = VesselPipeline.MinimumParts,
                    }.Run();
                    compiled = true;
                    tombstoneExpiry = vesselBuild.TombstoneExpiry(VesselPipeline.TombstoneSteps);
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

                // The number of joints produced by each unit is not known in advance. BuildJointsJob therefore writes a
                // separate stream range for each unit.
                JobHandle allocated = NativeStream.ScheduleConstruct(out NativeStream stream, jointUnits, groupsFound, Allocator.TempJob);
                JobHandle emitted = new BuildJointsJob
                {
                    World = store,
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
                    Locking = forest && VesselForest.Locking,
                    LockFraction = VesselForest.LockFraction,
                    YieldFraction = VesselForest.YieldFraction,
                    Vessels = vesselSet.IsCreated,
                    UnitVessels = vesselBuild.UnitVessels.AsArray(),
                    VesselActive = vesselBuild.Active.AsArray(),
                }.Schedule(jointUnits, JointBatchSize, allocated);

                // Only the solve reads the edges and closures, so these jobs run beside the joint build and the
                // broadphase, and only the solve waits for them (see PrepareSolve). They prepare the groups of active
                // vessels, which BuildJointsJob skips.
                vesselsFilled = default;
                if (vesselSet.IsCreated)
                {
                    // Reuse the edges and closures from earlier steps unless the vessels were compiled again. The fill
                    // jobs compute an entry again when a body's motion frame changed, or when the settings of one of its
                    // joints changed. They look at the joints' settings stamps only when the version moved.
                    uint settings = JointSettings.Version;
                    bool resized = vesselBuild.SizeCaches();
                    bool refillAll = resized || compiledThisStep || !cacheFilled;
                    bool settingsChanged = settings != cachedSettings;
                    cacheFilled = true;
                    cachedSettings = settings;
                    JobHandle edgesFilled = new FillVesselEdgesJob
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

                    JobHandle closuresFilled = new FillVesselClosuresJob
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
                    // The joint loads at the end of the step need this list. Waiting for BuildJointsJob keeps the job
                    // from reading a group while BuildJointsJob writes it.
                    JobHandle listed = new ListOtherJointsJob
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

                JobHandle laidOut = new LayOutJointsJob
                {
                    Written = jointsWritten,
                    Output = world.DynamicsWorld.InternalJointsList,
                }.Schedule(emitted);

                JobHandle copied = new CopyJointsJob
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
