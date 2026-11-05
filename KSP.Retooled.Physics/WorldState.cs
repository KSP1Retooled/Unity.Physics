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

        private readonly NativeStore.WorldData* store;

        private PhysicsWorld world = new PhysicsWorld(0, 0, 0);
        private Simulation simulation = Simulation.Create();
        private NativeReference<int> staticsChanged = new NativeReference<int>(1, Allocator.Persistent);
        private NativeList<int> dynamicIndices = new NativeList<int>(Allocator.Persistent);
        private NativeList<int> staticIndices = new NativeList<int>(Allocator.Persistent);
        private uint builtStaticsVersion;
        private uint builtMaterialVersion;
        private int builtDynamicCount;
        private bool everBuilt;

        internal WorldState(NativeStore.WorldData* store)
        {
            this.store = store;
        }

        public void Dispose()
        {
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
        }

        /// <summary>One simulation step: integrate forces, build, solve, write back.</summary>
        internal void Step(float deltaTime)
        {
            if (deltaTime <= 0.0f)
            {
                return;
            }

            RefreshColliders();

            new ApplyForcesJob
            {
                Bodies = store->bodies,
                Gravity = store->gravity,
                DeltaTime = deltaTime,
            }.Schedule((int)store->bodyCount, ParallelBatchSize).Complete();

            float3 gravity = float3.zero;
            JobHandle built = Build(deltaTime);
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
                SynchronizeCollisionWorld = false,
            };

            simulation.ScheduleStepJobs(input, built);
            simulation.FinalJobHandle.Complete();

            new WriteBackJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
            }.Schedule(dynamicIndices.Length, ParallelBatchSize).Complete();
        }

        /// <summary>Brings the built world up to the store's poses without stepping, on the calling thread.</summary>
        internal void BuildForQueries()
        {
            RefreshColliders();
            Build(0.0f).Complete();
            world.CollisionWorld.BuildBroadphase(ref world, 0.0f, float3.zero, staticsChanged.Value != 0);
        }

        internal bool RayCast(NativeStore.RayCastInput* input, NativeStore.RayCastHit* hit)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = math.min(input->maxDistance, QueryCollector.MaxRayLength);
            RaycastInput ray = new RaycastInput
            {
                Start = input->origin,
                End = input->origin + direction * distance,
                Filter = new CollisionFilter
                {
                    BelongsTo = uint.MaxValue,
                    CollidesWith = (uint)input->layerMask,
                },
            };

            QueryCollector collector = new QueryCollector(world, store, input->filterOptions);
            if (!world.CollisionWorld.CastRay(ray, ref collector))
            {
                return false;
            }

            hit->point = collector.Hit.Position;
            hit->normal = collector.Hit.SurfaceNormal;
            hit->faceId = 0;
            hit->distance = collector.Hit.Fraction * distance;
            hit->uv = float2.zero;
            hit->engineObjectId = collector.HitEngineObjectId;
            return true;
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

        private JobHandle Build(float deltaTime)
        {
            dynamicIndices.Clear();
            staticIndices.Clear();
            for (int i = 0; i < store->bodyCount; i++)
            {
                NativeStore.BodyFlags flags = store->bodies[i].flags;
                if ((flags & NativeStore.BodyFlags.Simulated) == 0)
                {
                    continue;
                }

                if ((flags & NativeStore.BodyFlags.Static) != 0)
                {
                    staticIndices.Add(i);
                }
                else
                {
                    dynamicIndices.Add(i);
                }
            }

            // The static tree addresses bodies relative to the dynamic ones before them.
            bool changed = !everBuilt || store->staticsVersion != builtStaticsVersion || dynamicIndices.Length != builtDynamicCount;
            staticsChanged.Value = changed ? 1 : 0;
            builtStaticsVersion = store->staticsVersion;
            builtDynamicCount = dynamicIndices.Length;
            everBuilt = true;

            world.Reset(staticIndices.Length + 1, dynamicIndices.Length, 0);

            return new BuildBodiesJob
            {
                Store = store->bodies,
                DynamicIndices = dynamicIndices.AsArray(),
                StaticIndices = staticIndices.AsArray(),
                Bodies = world.Bodies,
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                DeltaTime = deltaTime,
            }.Schedule();
        }
    }
}
