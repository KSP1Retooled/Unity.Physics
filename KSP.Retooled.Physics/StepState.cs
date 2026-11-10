using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// A world's simulation state, in unmanaged memory so that Burst can run a whole step: walking the bodies, scheduling
    /// the stages and reading their results. <see cref="WorldState"/> owns it.
    /// </summary>
    internal unsafe struct StepState
    {
        private const int ParallelBatchSize = 64;

        // Building a joint costs much more than building a body, so a batch takes fewer of them.
        private const int JointBatchSize = 8;

        // The write-back gives these bodies a pose other than their solved motion.
        private const NativeStore.BodyFlags PosedByWriteBack = NativeStore.BodyFlags.Sleeping | NativeStore.BodyFlags.HasKinematicTarget;

        private readonly NativeStore.WorldData* store;
        private readonly NativeStore.StoreGlobals* globals;
        private readonly QueryWorld* query;

        private PhysicsWorld world;

        // The static tree's jobs read whether to rebuild from one of these, and neither is ever written again, so a build never
        // writes to a container that those jobs may still hold.
        private NativeReference<int> rebuildStatics;
        private NativeReference<int> keepStatics;
        private bool staticsChanged;
        private NativeList<int> dynamicIndices;
        private NativeList<int> staticIndices;
        private NativeList<int> builtIndices;
        private NativeList<JointGroup> jointGroups;
        private NativeList<JointGroupMember> jointGroupMembers;
        private NativeList<JointUnit> jointUnits;
        private NativeList<JointLoad> jointLoads;
        private NativeList<JointLoad> groupLoads;
        private DirectSolverCache directSolverCache;
        private NativeList<QueryBody> queryBodies;
        private NativeList<QueryShape> queryShapes;
        private NativeList<int> queryFirstShapes;
        private NativeList<ManifoldRecord> manifolds;
        private NativeList<ManifoldPoint> manifoldPoints;
        private NativeList<ContactBucket> contactBuckets;
        private NativeReference<int> filledContactBuckets;
        private NativeHashMap<ShapePairKey, ShapePairState> touching;
        private NativeList<int> sleepParents;
        private NativeList<int> sleepContacts;
        private NativeList<bool> sleepTouchesStill;
        private NativeList<int> fallingAsleep;
        private NativeHashMap<PersistentManifoldKey, float> lastImpulses;
        private NativeHashMap<JacobianKey, float2> warmStarts;
        private NativeList<ColliderBuilder.Work> colliderWork;
        private NativeList<long> colliderResults;
        private NativeReference<uint> wholeColliderBuilds;
        private NativeHashMap<ulong, KnownBodyPair> knownBodyPairs;
        private NativeQueue<DropIgnoredPairsJob.KeyValue> learnedBodyPairs;
        private JobHandle collidersBuilt;
        private JobHandle warmStartsIndexed;
        private uint stepCount;
        private uint builtStaticsVersion;
        private uint builtMaterialVersion;

        // False from the start of a build until it finished, so a build that fails is followed by a whole one.
        private bool builtWhole;
        private bool motionsMatchStore;

        internal StepState(NativeStore.WorldData* store, NativeStore.StoreGlobals* globals, QueryWorld* query)
        {
            this = default;
            this.store = store;
            this.globals = globals;
            this.query = query;
            world = new PhysicsWorld(0, 0, 0);
            rebuildStatics = new NativeReference<int>(1, Allocator.Persistent);
            keepStatics = new NativeReference<int>(0, Allocator.Persistent);
            dynamicIndices = new NativeList<int>(Allocator.Persistent);
            staticIndices = new NativeList<int>(Allocator.Persistent);
            builtIndices = new NativeList<int>(Allocator.Persistent);
            jointGroups = new NativeList<JointGroup>(Allocator.Persistent);
            jointGroupMembers = new NativeList<JointGroupMember>(Allocator.Persistent);
            jointUnits = new NativeList<JointUnit>(Allocator.Persistent);
            jointLoads = new NativeList<JointLoad>(Allocator.Persistent);
            groupLoads = new NativeList<JointLoad>(Allocator.Persistent);
            directSolverCache = DirectSolverCache.Create();
            queryBodies = new NativeList<QueryBody>(Allocator.Persistent);
            queryShapes = new NativeList<QueryShape>(Allocator.Persistent);
            queryFirstShapes = new NativeList<int>(Allocator.Persistent);
            manifolds =new NativeList<ManifoldRecord>(Allocator.Persistent);
            manifoldPoints = new NativeList<ManifoldPoint>(Allocator.Persistent);
            contactBuckets = new NativeList<ContactBucket>(Allocator.Persistent);
            filledContactBuckets = new NativeReference<int>(0, Allocator.Persistent);
            touching = new NativeHashMap<ShapePairKey, ShapePairState>(64, Allocator.Persistent);
            sleepParents = new NativeList<int>(Allocator.Persistent);
            sleepContacts = new NativeList<int>(Allocator.Persistent);
            sleepTouchesStill = new NativeList<bool>(Allocator.Persistent);
            fallingAsleep = new NativeList<int>(Allocator.Persistent);
            lastImpulses = new NativeHashMap<PersistentManifoldKey, float>(64, Allocator.Persistent);
            warmStarts = new NativeHashMap<JacobianKey, float2>(64, Allocator.Persistent);
            colliderWork = new NativeList<ColliderBuilder.Work>(Allocator.Persistent);
            colliderResults = new NativeList<long>(Allocator.Persistent);
            wholeColliderBuilds = new NativeReference<uint>(0, Allocator.Persistent);
            knownBodyPairs = new NativeHashMap<ulong, KnownBodyPair>(1024, Allocator.Persistent);
            learnedBodyPairs = new NativeQueue<DropIgnoredPairsJob.KeyValue>(Allocator.Persistent);
        }

        internal void Dispose()
        {
            store->queryWorld = null;

            collidersBuilt.Complete();
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
            dynamicIndices.Dispose();
            staticIndices.Dispose();
            builtIndices.Dispose();
            jointGroups.Dispose();
            jointGroupMembers.Dispose();
            jointUnits.Dispose();
            jointLoads.Dispose();
            groupLoads.Dispose();
            directSolverCache.Dispose();
            queryBodies.Dispose();
            queryShapes.Dispose();
            queryFirstShapes.Dispose();
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
            lastImpulses.Dispose();
            warmStarts.Dispose();
            colliderWork.Dispose();
            colliderResults.Dispose();
            wholeColliderBuilds.Dispose();
            knownBodyPairs.Dispose();
            learnedBodyPairs.Dispose();
        }

        /// <summary>The built world, which the simulation's systems step.</summary>
        internal readonly PhysicsWorld World => world;

        internal readonly int StaticsChanged => staticsChanged ? 1 : 0;

        private readonly NativeReference<int>.ReadOnly BuildStaticTree => (staticsChanged ? rebuildStatics : keepStatics).AsReadOnly();

        /// <summary>The settings the simulation's systems build their step input from.</summary>
        internal readonly PhysicsStep Settings
        {
            get
            {
                PhysicsStep settings = PhysicsStep.Default;
                settings.Gravity = float3.zero;
                settings.DirectIslandSubsteps = WorldState.DirectIslandSubsteps;
                settings.SolverIterationCount = WorldState.SolverIterations;

                // FinishStep synchronizes the collision world itself, while it reads the step's results.
                settings.SynchronizeCollisionWorld = 0;
                return settings;
            }
        }

        /// <summary>The first stage of a step, which integrates forces into the store's velocities.</summary>
        internal void ApplyForces(float deltaTime)
        {
            store->queryWorld = null;
            store->queryDirty = StoreOps.QueryStale;
            BodyEntities.PrepareForJobs(store);

            using (BackendMarkers.StepApplyForces.Auto())
            {
                new WakeJointedBodiesJob { World = store }.Run();

                new ApplyForcesJob
                {
                    Bodies = store->bodies,
                    Gravity = store->gravity,
                    DeltaTime = deltaTime,
                }.Schedule((int)store->bodyCount, ParallelBatchSize).Complete();
            }
        }

        /// <summary>Builds the bodies, joints and broadphase that the simulation's systems go on to step.</summary>
        internal JobHandle BuildWorld(float deltaTime, JobHandle dependency)
        {
            using var span = BackendMarkers.StepSchedule.Auto();

            JobHandle colliders = RefreshColliders(dependency);
            JobHandle built = Build(deltaTime, colliders, out JobHandle joints);

            using (BackendMarkers.BuildScheduleBroadphase.Auto())
            {
                // Nothing reads the joints until the dispatch pairs are made after the overlaps, so they build alongside
                // the trees and the overlaps, and only the dispatch pairs wait for them.
                world.DynamicsWorld.JointsBuilt = joints;
                JobHandle broadphase = world.CollisionWorld.ScheduleBuildBroadphaseJobs(ref world, deltaTime, float3.zero, BuildStaticTree, built, true);
                return ScheduleWheels(deltaTime, broadphase);
            }
        }

        /// <summary>
        /// Drives the WheelColliders once their rays have a broadphase to hit, and before anything reads the velocities they
        /// change. Everything after the build waits for it, so it only runs when there are wheels.
        /// </summary>
        private JobHandle ScheduleWheels(float deltaTime, JobHandle dependency)
        {
            if (store->vehicleCount == 0)
            {
                return dependency;
            }

            return new WheelJob
            {
                Store = store,
                Globals = globals,
                Collision = world.CollisionWorld,
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                BuiltIndices = builtIndices.AsArray(),
                DeltaTime = deltaTime,
            }.Schedule(dependency);
        }

        /// <summary>Drops the overlaps nothing will solve, then sorts what's left into islands.</summary>
        internal JobHandle SortIslands(SimulationSingleton singleton, in SimulationStepInput input, JobHandle dependency)
        {
            using var span = BackendMarkers.StepSchedule.Auto();

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
            }.Schedule(pairs, ParallelBatchSize, dependency);

            // Nothing in this step reads what it learned, so keeping it runs beside the islands.
            JobHandle remembered = new RememberBodyPairsJob
            {
                Known = knownBodyPairs,
                Learned = learnedBodyPairs,
                Bodies = (int)store->bodyCount,
            }.Schedule(filtered);

            return JobHandle.CombineDependencies(simulation->ScheduleIslandJobs(input, filtered), remembered);
        }

        /// <summary>What the narrow phase decided, read before the Jacobians are built from it.</summary>
        internal JobHandle ReadContacts(SimulationSingleton singleton, float deltaTime, JobHandle dependency)
        {
            using var span = BackendMarkers.StepSchedule.Auto();

            // Jobs wait in a batch until something flushes it, so without this the workers sit idle through the scheduling below.
            JobHandle.ScheduleBatchedJobs();

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

            // Only the warm starts need the merged manifolds before the solve, so the merge runs beside the Jacobian build.
            warmStartsIndexed = new IndexWarmStartsJob { Records = manifolds, WarmStarts = warmStarts }.Schedule(merged);

            // Without contacts nothing is woken: the joints already spread waking as the step began.
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

            input.DirectSolverCache = directSolverCache;
            return new WarmStartJob
            {
                WarmStarts = warmStarts,
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
                MotionCount = dynamicIndices.Length,
            }.Schedule(singleton, ref world, JobHandle.CombineDependencies(dependency, warmStartsIndexed));
        }

        /// <summary>
        /// Reads what the joints carried from the step's impulse events and breaks the overloaded ones, see
        /// <see cref="JointForceJobs"/>. Their bodies' poses have to be written back first.
        /// </summary>
        private void ReadJointForces(Simulation simulation, float deltaTime)
        {
            const int groupsPerBatch = 16;

            int jointCount = (int)store->jointCount;
            jointLoads.ResizeUninitialized(jointCount);
            groupLoads.ResizeUninitialized(jointGroups.Length);
            UnsafeUtility.MemClear(groupLoads.GetUnsafePtr(), (long)groupLoads.Length * sizeof(JointLoad));

            JobHandle seeded = new SeedJointLoadsJob { World = store, Loads = jointLoads.AsArray() }.Schedule(jointCount, ParallelBatchSize);
            NativeStream events = simulation.ImpulseEvents.EventDataStream;
            if (events.IsCreated)
            {
                JobHandle.ScheduleBatchedJobs();
                new RotateJointImpulsesJob
                {
                    Events = events.AsReader(),
                    World = store,
                    Groups = jointGroups.AsArray(),
                    DeltaTime = deltaTime,
                }.ScheduleUnsafe(events, 1, default).Complete();

                seeded.Complete();
                new SumJointImpulsesJob
                {
                    Events = events.AsReader(),
                    JointLoads = jointLoads.AsArray(),
                    GroupLoads = groupLoads.AsArray(),
                }.Run();
            }

            JobHandle shared = new ShareGroupLoadsJob
            {
                World = store,
                Groups = jointGroups.AsArray(),
                Members = jointGroupMembers.AsArray(),
                GroupLoads = groupLoads.AsArray(),
                JointLoads = jointLoads.AsArray(),
            }.Schedule(jointGroups.Length, groupsPerBatch, seeded);

            new SettleJointLoadsJob { World = store, Loads = jointLoads.AsArray(), DeltaTime = deltaTime }
                .Schedule(jointCount, ParallelBatchSize, shared).Complete();

            new BreakJointsJob
            {
                World = store,
                Globals = globals,
                Loads = jointLoads.AsArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Run();
        }

        /// <summary>The last stage of a step, which hands the solved world back to the store and to Unity.</summary>
        internal void FinishStep(Simulation simulation, float deltaTime)
        {
            using (BackendMarkers.StepWaitForSolve.Auto())
            {
                simulation.FinalJobHandle.Complete();

                // The solve waits for the merged manifolds through the warm starts, but a step with nothing to solve doesn't.
                warmStartsIndexed.Complete();
            }

            // Only the query world reads the dynamic tree, so it is rebuilt at the solved poses while this thread writes the
            // motions back and reads the joints. Reading the impulses has to wait for it: it holds the whole world, whose
            // bodies the rebuild writes.
            JobHandle tree = world.CollisionWorld.ScheduleUpdateDynamicTree(ref world, deltaTime, float3.zero, default);

            // The sleep reads the manifolds before the contact report sorts them, so it starts on them now.
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
                new WriteBackJob
                {
                    Store = store->bodies,
                    DynamicIndices = dynamicIndices.AsArray(),
                    MotionDatas = world.MotionDatas,
                    MotionVelocities = world.MotionVelocities,
                }.Schedule(dynamicIndices.Length, ParallelBatchSize).Complete();
            }

            using (BackendMarkers.StepReadJointForces.Auto())
            {
                ReadJointForces(simulation, deltaTime);
            }

            // What a removed shape held wakes after the joints read who sleeps, and before the sleep and the contact report
            // do. The rest of the sleep and the query snapshot don't need the impulses or the report, so they run on workers
            // while this thread waits for the tree and publishes the contacts.
            new ForgetRemovedShapesJob { World = store, Touching = touching }.Run();
            JobHandle sleep = new SleepJob
            {
                World = store,
                Parents = sleepParents,
                Contacts = sleepContacts,
                TouchesStill = sleepTouchesStill,
                Falling = fallingAsleep,
                DeltaTime = deltaTime,
            }.Schedule(contactsJoined);
            JobHandle snapshot = ScheduleQuerySnapshot(tree);
            JobHandle.ScheduleBatchedJobs();

            using (BackendMarkers.StepWaitForTree.Auto())
            {
                tree.Complete();
            }

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
                // The report sorts the manifolds the sleep reads.
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

            // The step synchronized the built world to the motions it wrote back, which spares the queries after it a
            // rebuild. The store can't change during a step, so these bodies are the only way it can differ.
            using (BackendMarkers.StepPublishQueryWorld.Auto())
            {
                snapshot.Complete();
                store->queryDirty = motionsMatchStore ? 0u : StoreOps.QueryStale;
                PublishQueryWorld();
            }

            // After the snapshot, which reads the flags this sets.
            using (BackendMarkers.StepSleep.Auto())
            {
                sleep.Complete();
                new FallAsleepJob { World = store, Falling = fallingAsleep }.Run();
            }
        }

        /// <summary>Brings the built world up to the store's poses without stepping. Main thread only.</summary>
        internal void BuildForQueries()
        {
            store->queryWorld = null;
            BodyEntities.PrepareForJobs(store);

            // The last build stands but for the statics' poses. Nothing in the snapshot depends on a pose.
            if (store->queryDirty == StoreOps.QueryStaticsMoved && builtWhole && globals->materialVersion == builtMaterialVersion)
            {
                PoseStatics();
                PublishQueryWorld();
                return;
            }

            JobHandle colliders = RefreshColliders(default);
            JobHandle built = Build(0.0f, colliders, out _);
            JobHandle broadphase = world.CollisionWorld.ScheduleBuildBroadphaseJobs(ref world, 0.0f, float3.zero, BuildStaticTree, built, true);
            ScheduleQuerySnapshot(broadphase).Complete();
            PublishQueryWorld();
        }

        /// <summary>
        /// Moves the built statics to where the store has them and builds the static tree again. Recording the version spares
        /// the next step the same rebuild, unless more statics move first.
        /// </summary>
        private void PoseStatics()
        {
            builtStaticsVersion = store->staticsVersion;
            staticsChanged = true;
            JobHandle posed = new PoseStaticsJob
            {
                Store = store->bodies,
                StaticIndices = staticIndices.AsArray(),
                StaticBodies = world.StaticBodies,
            }.Schedule(staticIndices.Length, ParallelBatchSize, default);

            world.CollisionWorld.ScheduleUpdateStaticTree(ref world, BuildStaticTree, posed, true).Complete();
        }

        /// <summary>
        /// False while the main thread is running a query job it picked up inside a Complete, where the safety system refuses
        /// writes. Asking before the build starts leaves the published world readable for the rest of that job. Without
        /// safety checks nothing refuses the build, and the job that asked simply sees it finished. Asking clears the
        /// dynamic indices, so nothing between here and the next whole build may read them.
        /// </summary>
        internal bool CanWriteContainers()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            try
            {
                dynamicIndices.Clear();
            }
            catch (InvalidOperationException)
            {
                return false;
            }
#endif
            return true;
        }

        /// <summary>
        /// Copies everything a query needs from the store, for <see cref="PublishQueryWorld"/>. Workers keep reading the copy
        /// until the next publish, so nothing in it may point at memory the store can move or free.
        /// </summary>
        private JobHandle ScheduleQuerySnapshot(JobHandle dependency)
        {
            queryBodies.ResizeUninitialized(world.NumBodies);
            queryFirstShapes.ResizeUninitialized(world.NumBodies);
            JobHandle found = new FindFirstShapesJob
            {
                Store = store->bodies,
                Bodies = world.Bodies,
                FirstShapes = queryFirstShapes.AsArray(),
                QueryShapes = queryShapes,
            }.Schedule(dependency);

            return new SnapshotQueryDataJob
            {
                Store = store->bodies,
                Bodies = world.Bodies,
                FirstShapes = queryFirstShapes.AsArray(),
                QueryBodies = queryBodies.AsArray(),
                QueryShapes = queryShapes.AsDeferredJobArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(world.NumBodies, ParallelBatchSize, found);
        }

        /// <summary>
        /// Hands the built world and the finished snapshot to the query functions. The colliders are safe to share: they are
        /// only released by the next build, which holds the query lock exclusively.
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
        /// Schedules the colliders the store asks for. The bodies get them when the returned handle completes, and nothing
        /// between here and the jobs that depend on it reads a collider.
        /// </summary>
        private JobHandle RefreshColliders(JobHandle dependency)
        {
            using var span = BackendMarkers.RefreshColliders.Auto();

            // Every step and query build completes the one before it, so this is only ever a wait on a finished handle.
            collidersBuilt.Complete();
            ColliderBuilder.AskAgainForFailed(store, colliderWork, colliderResults);
            ReleaseDeadColliders();

            uint materialVersion = globals->materialVersion;
            collidersBuilt = ColliderBuilder.ScheduleRefresh(store, materialVersion != builtMaterialVersion, colliderWork, colliderResults,
                wholeColliderBuilds, dependency);
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
        /// Schedules the bodies, returned, and the joints, handed out separately so the broadphase needn't wait for them.
        /// </summary>
        private JobHandle Build(float deltaTime, JobHandle dependency, out JobHandle joints)
        {
            using (BackendMarkers.BuildPartition.Auto())
            {
                dynamicIndices.Clear();
                staticIndices.Clear();
                builtIndices.ResizeUninitialized((int)store->bodyCount);
                motionsMatchStore = true;
                for (int i = 0; i < store->bodyCount; i++)
                {
                    NativeStore.BodyFlags flags = store->bodies[i].flags;
                    builtIndices[i] = -1;
                    if ((flags & NativeStore.BodyFlags.Simulated) == 0)
                    {
                        continue;
                    }

                    if ((flags & NativeStore.BodyFlags.Static) != 0)
                    {
                        builtIndices[i] = staticIndices.Length;
                        staticIndices.Add(i);
                    }
                    else
                    {
                        builtIndices[i] = dynamicIndices.Length;
                        dynamicIndices.Add(i);
                        if ((flags & PosedByWriteBack) != 0)
                        {
                            motionsMatchStore = false;
                        }
                    }
                }

                // The static tree numbers the statics among themselves, and queries and pair finding add the dynamic count,
                // so only the statics decide whether it is built again.
                staticsChanged = !builtWhole || store->staticsVersion != builtStaticsVersion;
                builtStaticsVersion = store->staticsVersion;
                builtWhole = false;

                for (int i = 0; i < builtIndices.Length; i++)
                {
                    if (builtIndices[i] >= 0 && (store->bodies[i].flags & NativeStore.BodyFlags.Static) != 0)
                    {
                        builtIndices[i] += dynamicIndices.Length;
                    }
                }
            }

            // The joints start empty and their job fills them. Unity Physics reads them through deferred arrays, so nothing
            // scheduled before that job finishes needs to know how many there are.
            using (BackendMarkers.BuildReset.Auto())
            {
                world.Reset(staticIndices.Length + 1, dynamicIndices.Length, 0);
                world.DynamicsWorld.EnableDirectSolver = true;
            }

            // A query world never solves, so only a step needs the joints, and only a step has a time to give their motors.
            // Like the bodies, they wait for the collider jobs, which write the body flags the joints read.
            joints = dependency;
            if (deltaTime > 0.0f)
            {
                using var jointsSpan = BackendMarkers.BuildJoints.Auto();

                JobHandle grouped = new GroupJointsJob
                {
                    World = store,
                    Groups = jointGroups,
                    Members = jointGroupMembers,
                    Units = jointUnits,
                }.Schedule(dependency);

                // Each unit builds a number of joints only it knows, so each writes its own run of a stream.
                JobHandle allocated = NativeStream.ScheduleConstruct(out NativeStream stream, jointUnits, grouped, Allocator.TempJob);
                JobHandle emitted = new BuildJointsJob
                {
                    World = store,
                    BuiltIndices = builtIndices.AsArray(),
                    Units = jointUnits.AsDeferredJobArray(),
                    Output = stream.AsWriter(),
                    Groups = jointGroups.AsDeferredJobArray(),
                    Members = jointGroupMembers.AsDeferredJobArray(),
                    DynamicCount = dynamicIndices.Length,
                    AnchorIndex = dynamicIndices.Length + staticIndices.Length,
                    DeltaTime = deltaTime,
                }.Schedule(jointUnits, JointBatchSize, allocated);

                JobHandle gathered = new GatherJointsJob
                {
                    Input = stream.AsReader(),
                    Output = world.DynamicsWorld.InternalJointsList,
                }.Schedule(emitted);

                joints = stream.Dispose(gathered);
            }

            using var scheduleSpan = BackendMarkers.BuildScheduleBodies.Auto();

            var bodiesJob = new BuildBodiesJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                StaticIndices = staticIndices.AsArray(),
                Bodies = world.Bodies,
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                DeltaTime = deltaTime,
            };

            return bodiesJob.Schedule(bodiesJob.Count, ParallelBatchSize, dependency);
        }
    }
}
