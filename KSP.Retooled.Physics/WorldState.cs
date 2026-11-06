using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Everything managed keeps for one native world: a Unity Physics world built on demand from the store, and the
    /// simulation that steps it. None of it survives a detach, and the next step or query rebuilds it.
    /// </summary>
    internal sealed unsafe class WorldState : IDisposable
    {
        private const int SolverIterations = 8;
        private const int Substeps = 4;

        private const int ParallelBatchSize = 64;

        // The write-back gives these bodies a pose other than their solved motion.
        private const NativeStore.BodyFlags PosedByWriteBack = NativeStore.BodyFlags.Sleeping | NativeStore.BodyFlags.HasKinematicTarget;

        private readonly NativeStore.WorldData* store;
        private readonly QueryWorld* query;

        private PhysicsWorld world = new PhysicsWorld(0, 0, 0);
        private Simulation simulation = Simulation.Create();
        private NativeReference<int> staticsChanged = new NativeReference<int>(1, Allocator.Persistent);
        private NativeList<int> dynamicIndices = new NativeList<int>(Allocator.Persistent);
        private NativeList<int> staticIndices = new NativeList<int>(Allocator.Persistent);
        private NativeList<int> builtIndices = new NativeList<int>(Allocator.Persistent);
        private NativeList<Joint> builtJoints = new NativeList<Joint>(Allocator.Persistent);
        private NativeList<QueryBody> queryBodies = new NativeList<QueryBody>(Allocator.Persistent);
        private NativeList<QueryShape> queryShapes = new NativeList<QueryShape>(Allocator.Persistent);
        private uint builtStaticsVersion;
        private uint builtMaterialVersion;
        private int builtDynamicCount;

        // False from the start of a build until it finished, so a build that fails is followed by a whole one.
        private bool builtWhole;
        private bool motionsMatchStore;

        internal WorldState(NativeStore.WorldData* store)
        {
            this.store = store;
            query = (QueryWorld*)UnsafeUtility.MallocTracked(sizeof(QueryWorld), UnsafeUtility.AlignOf<QueryWorld>(), Allocator.Persistent, 0);
        }

        public void Dispose()
        {
            store->queryWorld = null;
            UnsafeUtility.FreeTracked(query, Allocator.Persistent);

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
            queryBodies.Dispose();
            queryShapes.Dispose();
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
            RefreshColliders();

            new WakeJointedBodiesJob { World = store }.Run();

            new ApplyForcesJob
            {
                Bodies = store->bodies,
                Gravity = store->gravity,
                DeltaTime = deltaTime,
            }.Schedule((int)store->bodyCount, ParallelBatchSize).Complete();

            float3 gravity = float3.zero;
            JobHandle built = Build(deltaTime).Schedule();
            built = world.CollisionWorld.ScheduleBuildBroadphaseJobs(ref world, deltaTime, gravity, staticsChanged.AsReadOnly(), built);

            SimulationStepInput input = new SimulationStepInput
            {
                World = world,
                TimeStep = deltaTime,
                Gravity = gravity,
                NumSubsteps = Substeps,
                NumSolverIterations = SolverIterations,
                SolverStabilizationHeuristicSettings = Solver.StabilizationHeuristicSettings.Default,
                MaxDynamicDepenetrationVelocity = SimulationStepInput.DefaultMaxDynamicDepenetrationVelocity,
                MaxStaticDepenetrationVelocity = SimulationStepInput.DefaultMaxStaticDepenetrationVelocity,
                HaveStaticBodiesChanged = staticsChanged.AsReadOnly(),
                SynchronizeCollisionWorld = true,
            };

            // These are ScheduleStepJobs' stages, with a gap after the narrow phase for what the contacts decide.
            built = simulation.ScheduleBroadphaseJobs(input, built).FinalExecutionHandle;
            built = simulation.ScheduleNarrowphaseJobs(input, built).FinalExecutionHandle;

            SimulationSingleton singleton = default;
            singleton.InitializeFromSimulation(ref simulation);
            built = new ReadContactsJob
            {
                Store = store->bodies,
                Globals = NativeStore.RetooledPhysics_Globals(),
                Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
            }.Schedule(singleton, ref world, built);

            built = simulation.ScheduleCreateJacobiansJobs(input, built).FinalExecutionHandle;
            simulation.ScheduleSolveAndIntegrateJobs(input, built);
            simulation.FinalJobHandle.Complete();

            new WriteBackJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
            }.Schedule(dynamicIndices.Length, ParallelBatchSize).Complete();

            new ReadJointForcesJob
            {
                World = store,
                Globals = NativeStore.RetooledPhysics_Globals(),
                Events = simulation.ImpulseEvents,
                DeltaTime = deltaTime,
                EngineObjectIdOffset = NativeStore.RetooledPhysics_Globals()->engineObjectIdOffset,
            }.Run();

            // The step synchronized the built world to the motions it wrote back, which spares the queries after it a
            // rebuild. The store can't change during a step, so these bodies are the only way it can differ.
            store->queryDirty = motionsMatchStore ? 0u : 1u;
            PublishQueryWorld();
        }

        /// <summary>
        /// Brings the built world up to the store's poses without stepping and returns a QueryWorld constant. Main thread only.
        /// </summary>
        internal int BuildForQueries()
        {
            if (!CanWriteContainers())
            {
                return NativeStore.QueryWorldDeferred;
            }

            store->queryWorld = null;
            RefreshColliders();
            JobHandle built = Build(0.0f).Schedule();
            world.CollisionWorld.ScheduleBuildBroadphaseJobs(ref world, 0.0f, float3.zero, staticsChanged.AsReadOnly(), built).Complete();
            PublishQueryWorld();
            return NativeStore.QueryWorldBuilt;
        }

        /// <summary>
        /// False while the main thread is running a query job it picked up inside a Complete, where the safety system refuses
        /// writes. Asking before the build starts leaves the published world readable for the rest of that job. Without
        /// safety checks nothing refuses the build, and the job that asked simply sees it finished.
        /// </summary>
        private bool CanWriteContainers()
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
                EngineObjectIdOffset = NativeStore.RetooledPhysics_Globals()->engineObjectIdOffset,
            }.Schedule().Complete();

            query->Collision = world.CollisionWorld;
            query->Bodies = queryBodies.GetUnsafeReadOnlyPtr();
            query->Shapes = queryShapes.GetUnsafeReadOnlyPtr();
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
            ReleaseDeadColliders();

            uint materialVersion = NativeStore.RetooledPhysics_Globals()->materialVersion;
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
