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

        // Below this many dynamic bodies a stage's parallel jobs cost more to run than they save. Measured on two craft: a
        // step of 73 bodies is 12% faster in one job per stage, and one of 713 is 51% slower, mostly in the narrow phase
        // and Jacobians. The value is interpolated between those two.
        private const int MinParallelBodies = 128;

        // The write-back gives these bodies a pose other than their solved motion.
        private const NativeStore.BodyFlags PosedByWriteBack = NativeStore.BodyFlags.Sleeping | NativeStore.BodyFlags.HasKinematicTarget;

        private readonly NativeStore.WorldData* store;
        private readonly NativeStore.StoreGlobals* globals;
        private readonly QueryWorld* query;

        private PhysicsWorld world;
        private Simulation simulation;
        private NativeReference<int> staticsChanged;
        private NativeList<int> dynamicIndices;
        private NativeList<int> staticIndices;
        private NativeList<int> builtIndices;
        private NativeList<Joint> builtJoints;
        private NativeList<bool> directBodies;
        private NativeReference<bool> directBodyTouched;
        private DirectSolverCache directSolverCache;
        private NativeList<QueryBody> queryBodies;
        private NativeList<QueryShape> queryShapes;
        private NativeList<ManifoldRecord> manifolds;
        private NativeList<ManifoldPoint> manifoldPoints;
        private NativeHashMap<ShapePairKey, ShapePairState> touching;
        private NativeHashMap<PersistentManifoldKey, float> lastImpulses;
        private NativeHashMap<JacobianKey, float2> warmStarts;
        private uint stepCount;
        private uint builtStaticsVersion;
        private uint builtMaterialVersion;
        private int builtDynamicCount;

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
            simulation = Simulation.Create();
            staticsChanged = new NativeReference<int>(1, Allocator.Persistent);
            dynamicIndices = new NativeList<int>(Allocator.Persistent);
            staticIndices = new NativeList<int>(Allocator.Persistent);
            builtIndices = new NativeList<int>(Allocator.Persistent);
            builtJoints = new NativeList<Joint>(Allocator.Persistent);
            directBodies = new NativeList<bool>(Allocator.Persistent);
            directBodyTouched = new NativeReference<bool>(Allocator.Persistent);
            directSolverCache = DirectSolverCache.Create();
            queryBodies = new NativeList<QueryBody>(Allocator.Persistent);
            queryShapes = new NativeList<QueryShape>(Allocator.Persistent);
            manifolds = new NativeList<ManifoldRecord>(Allocator.Persistent);
            manifoldPoints = new NativeList<ManifoldPoint>(Allocator.Persistent);
            touching = new NativeHashMap<ShapePairKey, ShapePairState>(64, Allocator.Persistent);
            lastImpulses = new NativeHashMap<PersistentManifoldKey, float>(64, Allocator.Persistent);
            warmStarts = new NativeHashMap<JacobianKey, float2>(64, Allocator.Persistent);
        }

        internal void Dispose()
        {
            store->queryWorld = null;

            ReleaseDeadColliders();
            for (uint i = 0; i < store->bodyCount; i++)
            {
                NativeStore.BodyData* body = store->bodies + i;
                ColliderBuilder.Release(body->collider);
                body->collider = null;
                body->flags |= NativeStore.BodyFlags.ColliderDirty;
            }

            store->queryDirty = 1;
            simulation.Dispose();
            world.Dispose();
            staticsChanged.Dispose();
            dynamicIndices.Dispose();
            staticIndices.Dispose();
            builtIndices.Dispose();
            builtJoints.Dispose();
            directBodies.Dispose();
            directBodyTouched.Dispose();
            directSolverCache.Dispose();
            queryBodies.Dispose();
            queryShapes.Dispose();
            manifolds.Dispose();
            manifoldPoints.Dispose();
            touching.Dispose();
            lastImpulses.Dispose();
            warmStarts.Dispose();
        }

        /// <summary>One simulation step: integrate forces, build, solve, write back.</summary>
        internal void Step(float deltaTime)
        {
            if (deltaTime <= 0.0f)
            {
                return;
            }

            store->queryWorld = null;
            store->queryDirty = 1;
            BodyEntities.PrepareForJobs(store);
            RefreshColliders();

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

            BackendMarkers.StepSchedule.Begin();
            float3 gravity = float3.zero;
            JobHandle built = Build(deltaTime).Schedule();
            bool multiThreaded = dynamicIndices.Length >= MinParallelBodies;
            built = world.CollisionWorld.ScheduleBuildBroadphaseJobs(ref world, deltaTime, gravity, staticsChanged.AsReadOnly(), built, multiThreaded);

            directBodies.ResizeUninitialized(dynamicIndices.Length);
            new MarkDirectBodiesJob { Joints = builtJoints.AsArray(), Bodies = directBodies.AsArray() }.Run();
            directBodyTouched.Value = false;

            SimulationStepInput input = new SimulationStepInput
            {
                World = world,
                TimeStep = deltaTime,
                Gravity = gravity,
                NumSubsteps = WorldState.Substeps,
                NumSolverIterations = WorldState.SolverIterations,
                DirectSolverSettings = Solver.DirectSolverSettings.Default,
                DirectSolverCache = directSolverCache,
                SolverStabilizationHeuristicSettings = Solver.StabilizationHeuristicSettings.Default,
                MaxDynamicDepenetrationVelocity = SimulationStepInput.DefaultMaxDynamicDepenetrationVelocity,
                MaxStaticDepenetrationVelocity = SimulationStepInput.DefaultMaxStaticDepenetrationVelocity,
                HaveStaticBodiesChanged = staticsChanged.AsReadOnly(),
                SynchronizeCollisionWorld = true,
            };

            // These are ScheduleStepJobs' stages, with a gap after the narrow phase for what the contacts decide.
            built = simulation.ScheduleBroadphaseJobs(input, built, multiThreaded).FinalExecutionHandle;
            built = simulation.ScheduleNarrowphaseJobs(input, built, multiThreaded).FinalExecutionHandle;

            // Jobs wait in a batch until something flushes it, so without this the workers sit idle through the scheduling below.
            JobHandle.ScheduleBatchedJobs();

            SimulationSingleton singleton = default;
            singleton.InitializeFromSimulation(ref simulation);
            manifolds.Clear();
            manifoldPoints.Clear();
            built = new ReadContactsJob
            {
                Store = store->bodies,
                Globals = globals,
                Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
                Records = manifolds,
                Points = manifoldPoints,
                LastImpulses = lastImpulses,
                DirectBodies = directBodies.AsArray(),
                DirectBodyTouched = directBodyTouched,
                DeltaTime = deltaTime,
                BounceThreshold = store->bounceThreshold,
            }.Schedule(singleton, ref world, built);

            built = new WakeTouchedBodiesJob
            {
                World = store,
                Records = manifolds,
                DynamicIndices = dynamicIndices.AsArray(),
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
            }.Schedule(built);

            built = new IndexWarmStartsJob { Records = manifolds, WarmStarts = warmStarts }.Schedule(built);
            built = simulation.ScheduleCreateJacobiansJobs(input, built, multiThreaded).FinalExecutionHandle;
            built = new WarmStartJob
            {
                WarmStarts = warmStarts,
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
                MotionCount = dynamicIndices.Length,
            }.Schedule(singleton, ref world, built);
            BackendMarkers.StepSchedule.End();

            using (BackendMarkers.StepWaitForContacts.Auto())
            {
                built.Complete();
            }

            using (BackendMarkers.StepScheduleSolve.Auto())
            {
                input.SkipDirectCoupling = !directBodyTouched.Value;
                simulation.ScheduleSolveAndIntegrateJobs(input, built, multiThreaded);
            }

            using (BackendMarkers.StepWaitForSolve.Auto())
            {
                simulation.FinalJobHandle.Complete();
            }

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
                    DeltaTime = deltaTime,
                    EngineObjectIdOffset = globals->engineObjectIdOffset,
                }.Run();
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
            RefreshColliders();
            JobHandle built = Build(0.0f).Schedule();
            bool multiThreaded = dynamicIndices.Length >= MinParallelBodies;
            world.CollisionWorld.ScheduleBuildBroadphaseJobs(ref world, 0.0f, float3.zero, staticsChanged.AsReadOnly(), built, multiThreaded).Complete();
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
            queryShapes.ResizeUninitialized(ShapeCount());
            new SnapshotQueryDataJob
            {
                Store = store->bodies,
                Bodies = world.Bodies,
                QueryBodies = queryBodies.AsArray(),
                QueryShapes = queryShapes.AsArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule().Complete();

            query->Collision = world.CollisionWorld;
            query->Bodies = queryBodies.GetUnsafeReadOnlyPtr();
            query->Shapes = queryShapes.GetUnsafePtr();
            query->ShapeCount = queryShapes.Length;
            store->queryWorld = query;
            builtWhole = true;
        }

        private int ShapeCount()
        {
            int count = 0;
            for (int i = 0; i < world.NumBodies; i++)
            {
                Unity.Entities.Entity entity = world.Bodies[i].Entity;
                if (entity.Version != 0)
                {
                    count += (int)store->bodies[entity.Index].shapeCount;
                }
            }

            return count;
        }

        private void RefreshColliders()
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.RefreshColliders.Auto();

            ReleaseDeadColliders();

            uint materialVersion = globals->materialVersion;
            ColliderBuilder.Refresh(store, materialVersion != builtMaterialVersion);
            builtMaterialVersion = materialVersion;
        }

        private void ReleaseDeadColliders()
        {
            for (uint i = 0; i < store->deadColliderCount; i++)
            {
                ColliderBuilder.Release(store->deadColliders[i]);
            }

            store->deadColliderCount = 0;
        }

        private BuildBodiesJob Build(float deltaTime)
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

            // The static tree addresses bodies relative to the dynamic ones before them.
            bool changed = !builtWhole || store->staticsVersion != builtStaticsVersion || dynamicIndices.Length != builtDynamicCount;
            staticsChanged.Value = changed ? 1 : 0;
            builtStaticsVersion = store->staticsVersion;
            builtDynamicCount = dynamicIndices.Length;
            builtWhole = false;

            for (int i = 0; i < builtIndices.Length; i++)
            {
                if (builtIndices[i] >= 0 && (store->bodies[i].flags & NativeStore.BodyFlags.Static) != 0)
                {
                    builtIndices[i] += dynamicIndices.Length;
                }
            }

            // A query world never solves, so only a step needs the joints, and only a step has a time to give their motors.
            builtJoints.Clear();
            if (deltaTime > 0.0f)
            {
                new BuildJointsJob
                {
                    World = store,
                    BuiltIndices = builtIndices.AsArray(),
                    Output = builtJoints,
                    DynamicCount = dynamicIndices.Length,
                    AnchorIndex = dynamicIndices.Length + staticIndices.Length,
                    DeltaTime = deltaTime,
                }.Run();
            }

            world.Reset(staticIndices.Length + 1, dynamicIndices.Length, builtJoints.Length);
            world.DynamicsWorld.Joints.CopyFrom(builtJoints.AsArray());
            world.DynamicsWorld.EnableDirectSolver = true;

            return new BuildBodiesJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                StaticIndices = staticIndices.AsArray(),
                Bodies = world.Bodies,
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                DeltaTime = deltaTime,
            };
        }
    }
}
