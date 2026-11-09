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
        private NativeReference<int> staticsChanged;
        private NativeList<int> dynamicIndices;
        private NativeList<int> staticIndices;
        private NativeList<int> builtIndices;
        private NativeList<JointGroup> jointGroups;
        private NativeList<JointGroupMember> jointGroupMembers;
        private NativeList<JointUnit> jointUnits;
        private DirectSolverCache directSolverCache;
        private NativeList<QueryBody> queryBodies;
        private NativeList<QueryShape> queryShapes;
        private NativeList<int> queryFirstShapes;
        private NativeList<ManifoldRecord> manifolds;
        private NativeList<ManifoldPoint> manifoldPoints;
        private NativeList<ContactBucket> contactBuckets;
        private NativeHashMap<ShapePairKey, ShapePairState> touching;
        private NativeHashMap<PersistentManifoldKey, float> lastImpulses;
        private NativeHashMap<JacobianKey, float2> warmStarts;
        private NativeList<ColliderBuilder.Work> colliderWork;
        private NativeList<long> colliderResults;
        private NativeReference<uint> wholeColliderBuilds;
        private NativeHashMap<ulong, KnownBodyPair> knownBodyPairs;
        private NativeQueue<DropIgnoredPairsJob.KeyValue> learnedBodyPairs;
        private JobHandle collidersBuilt;
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
            staticsChanged = new NativeReference<int>(1, Allocator.Persistent);
            dynamicIndices = new NativeList<int>(Allocator.Persistent);
            staticIndices = new NativeList<int>(Allocator.Persistent);
            builtIndices = new NativeList<int>(Allocator.Persistent);
            jointGroups = new NativeList<JointGroup>(Allocator.Persistent);
            jointGroupMembers = new NativeList<JointGroupMember>(Allocator.Persistent);
            jointUnits = new NativeList<JointUnit>(Allocator.Persistent);
            directSolverCache = DirectSolverCache.Create();
            queryBodies = new NativeList<QueryBody>(Allocator.Persistent);
            queryShapes = new NativeList<QueryShape>(Allocator.Persistent);
            queryFirstShapes = new NativeList<int>(Allocator.Persistent);
            manifolds =new NativeList<ManifoldRecord>(Allocator.Persistent);
            manifoldPoints = new NativeList<ManifoldPoint>(Allocator.Persistent);
            contactBuckets = new NativeList<ContactBucket>(Allocator.Persistent);
            touching = new NativeHashMap<ShapePairKey, ShapePairState>(64, Allocator.Persistent);
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
            staticsChanged.Dispose();
            dynamicIndices.Dispose();
            staticIndices.Dispose();
            builtIndices.Dispose();
            jointGroups.Dispose();
            jointGroupMembers.Dispose();
            jointUnits.Dispose();
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
            touching.Dispose();
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

        internal readonly int StaticsChanged => staticsChanged.Value;

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
            store->queryDirty = 1;
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
                return world.CollisionWorld.ScheduleBuildBroadphaseJobs(ref world, deltaTime, float3.zero, staticsChanged.AsReadOnly(), built, true);
            }
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
            JobHandle built = dependency;
            NativeStream contacts = singleton.AsSimulationPtr()->StepContext.Contacts;
            if (contacts.IsCreated)
            {
                NativeStream.Reader reader = contacts.AsReader();
                built = new PrepareContactBucketsJob { Contacts = reader, Buckets = contactBuckets }.Schedule(built);
                built = new ReadContactsJob
                {
                    Contacts = reader,
                    Buckets = contactBuckets,
                    Store = store->bodies,
                    Globals = globals,
                    Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
                    LastImpulses = lastImpulses,
                    DeltaTime = deltaTime,
                    BounceThreshold = store->bounceThreshold,
                }.ScheduleUnsafe(contacts, 1, built);

                built = new MergeContactBucketsJob
                {
                    Contacts = reader,
                    Buckets = contactBuckets,
                    Records = manifolds,
                    Points = manifoldPoints,
                }.Schedule(built);
            }

            built = new WakeTouchedBodiesJob
            {
                World = store,
                Records = manifolds,
                DynamicIndices = dynamicIndices.AsArray(),
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
            }.Schedule(built);

            return new IndexWarmStartsJob { Records = manifolds, WarmStarts = warmStarts }.Schedule(built);
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
            }.Schedule(singleton, ref world, dependency);
        }

        /// <summary>The last stage of a step, which hands the solved world back to the store and to Unity.</summary>
        internal void FinishStep(Simulation simulation, float deltaTime)
        {
            using (BackendMarkers.StepWaitForSolve.Auto())
            {
                simulation.FinalJobHandle.Complete();
            }

            // Only the query world reads the dynamic tree, so it is rebuilt at the solved poses while this thread writes the
            // motions back and reads the joints. Reading the impulses has to wait for it: it holds the whole world, whose
            // bodies the rebuild writes.
            JobHandle tree = world.CollisionWorld.ScheduleUpdateDynamicTree(ref world, deltaTime, float3.zero, default);
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
                new ReadJointForcesJob
                {
                    World = store,
                    Globals = globals,
                    Events = simulation.ImpulseEvents,
                    Groups = jointGroups.AsArray(),
                    Members = jointGroupMembers.AsArray(),
                    DeltaTime = deltaTime,
                    EngineObjectIdOffset = globals->engineObjectIdOffset,
                }.Run();
            }

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

            using (BackendMarkers.StepSleep.Auto())
            {
                new SleepJob { World = store, Records = manifolds, DeltaTime = deltaTime }.Run();
            }

            // The step synchronized the built world to the motions it wrote back, which spares the queries after it a
            // rebuild. The store can't change during a step, so these bodies are the only way it can differ.
            using (BackendMarkers.StepPublishQueryWorld.Auto())
            {
                store->queryDirty = motionsMatchStore ? 0u : 1u;
                PublishQueryWorld();
            }
        }

        /// <summary>Brings the built world up to the store's poses without stepping. Main thread only.</summary>
        internal void BuildForQueries()
        {
            store->queryWorld = null;
            BodyEntities.PrepareForJobs(store);
            JobHandle colliders = RefreshColliders(default);
            JobHandle built = Build(0.0f, colliders, out _);
            world.CollisionWorld.ScheduleBuildBroadphaseJobs(ref world, 0.0f, float3.zero, staticsChanged.AsReadOnly(), built, true).Complete();
            PublishQueryWorld();
        }

        /// <summary>
        /// False while the main thread is running a query job it picked up inside a Complete, where the safety system refuses
        /// writes. Asking before the build starts leaves the published world readable for the rest of that job. Without
        /// safety checks nothing refuses the build, and the job that asked simply sees it finished.
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
        /// Hands the built world to the query functions, with a copy of everything a query needs from the store. Workers keep
        /// reading this until the next publish, so nothing here may point at memory the store can move or free. The colliders
        /// are safe: they are only released by the next build, which holds the query lock exclusively.
        /// </summary>
        private void PublishQueryWorld()
        {
            queryBodies.ResizeUninitialized(world.NumBodies);
            queryShapes.ResizeUninitialized(FindFirstShapes());
            new SnapshotQueryDataJob
            {
                Store = store->bodies,
                Bodies = world.Bodies,
                FirstShapes = queryFirstShapes.AsArray(),
                QueryBodies = queryBodies.AsArray(),
                QueryShapes = queryShapes.AsArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(world.NumBodies, ParallelBatchSize).Complete();

            query->Collision = world.CollisionWorld;
            query->Bodies = queryBodies.GetUnsafeReadOnlyPtr();
            query->Shapes = queryShapes.GetUnsafePtr();
            query->ShapeCount = queryShapes.Length;
            store->queryWorld = query;
            builtWhole = true;
        }

        /// <summary>Sets where each built body's shapes start among the query's shapes, and returns how many there are.</summary>
        private int FindFirstShapes()
        {
            queryFirstShapes.ResizeUninitialized(world.NumBodies);
            int count = 0;
            for (int i = 0; i < world.NumBodies; i++)
            {
                queryFirstShapes[i] = count;
                Unity.Entities.Entity entity = world.Bodies[i].Entity;
                if (entity.Version != 0)
                {
                    count += (int)store->bodies[entity.Index].shapeCount;
                }
            }

            return count;
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
                bool changed = !builtWhole || store->staticsVersion != builtStaticsVersion;
                staticsChanged.Value = changed ? 1 : 0;
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
