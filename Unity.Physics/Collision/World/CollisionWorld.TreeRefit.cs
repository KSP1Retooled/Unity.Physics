using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.Physics
{
    public partial struct CollisionWorld
    {
        /// <summary>
        /// Move dynamic bodies to solved motion poses and refit their existing tree, optionally caching new
        /// collider bounds.
        /// Require the same bodies in the same order. Query hit sets match rebuilding, but hit order may differ.
        /// </summary>
        internal JobHandle ScheduleRefitDynamicTree(ref PhysicsWorld world, float timeStep, float3 gravity, JobHandle inputDeps,
            DynamicBoundsCache boundsCache = default)
        {
            int count = world.NumDynamicBodies;
            if (count == 0)
            {
                return inputDeps;
            }

            DynamicBoundsCache.View cache = boundsCache.IsCreated && boundsCache.CurrentView.Length == count ? boundsCache.CurrentView : default;

            var aabbs = new NativeArray<Aabb>(count, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            JobHandle handle = new UpdateDynamicBoundsJob
            {
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                RigidBodies = m_Bodies.GetSubArray(0, count),
                Aabbs = aabbs,
                AabbMargin = world.CollisionWorld.CollisionTolerance * 0.5f, // each body contributes half, as in the build
                TimeStep = timeStep,
                Gravity = gravity,
                Cache = cache,
            }.Schedule(count, 32, inputDeps);

            // Refit subtrees in parallel, then fit the nodes above the cut in one job.
            Broadphase.Tree tree = Broadphase.DynamicTree;
            handle = new RefitDynamicSubtreesJob
            {
                Tree = tree,
                Aabbs = aabbs,
            }.Schedule(BoundingVolumeHierarchy.RefitSubtreeCount, 1, handle);

            return new RefitDynamicTreeTopJob
            {
                Tree = tree,
                Aabbs = aabbs,
            }.Schedule(handle);
        }

        [BurstCompile]
        private struct UpdateRigidBodyTransformsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<MotionData> MotionDatas;
            public NativeArray<RigidBody> RigidBodies;

            public void Execute(int i)
            {
                ExecuteImpl(i, MotionDatas, RigidBodies);
            }

            internal static void ExecuteImpl(int i, NativeArray<MotionData> motionDatas, NativeArray<RigidBody> rigidBodies)
            {
                RigidBody rb = rigidBodies[i];
                rb.WorldFromBody = math.mul(motionDatas[i].WorldFromMotion, math.inverse(motionDatas[i].BodyFromMotion));
                rigidBodies[i] = rb;
            }
        }

        /// <summary>
        /// Moves each dynamic body to its motion's pose, as <see cref="UpdateRigidBodyTransformsJob"/> does, and
        /// computes the body's bounds as the broadphase build does.
        /// </summary>
        [BurstCompile]
        internal struct UpdateDynamicBoundsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<MotionData> MotionDatas;
            [ReadOnly] public NativeArray<MotionVelocity> MotionVelocities;
            public NativeArray<RigidBody> RigidBodies;
            [WriteOnly] public NativeArray<Aabb> Aabbs;
            public float AabbMargin;
            public float TimeStep;
            public float3 Gravity;

            /// <summary>The cache of the bodies' bounds for the next build, or a view that is not created.</summary>
            public DynamicBoundsCache.View Cache;

            public unsafe void Execute(int i)
            {
                UpdateBody(i, (MotionData*)MotionDatas.GetUnsafeReadOnlyPtr(), (MotionVelocity*)MotionVelocities.GetUnsafeReadOnlyPtr(),
                    (RigidBody*)RigidBodies.GetUnsafePtr(), (Aabb*)Aabbs.GetUnsafePtr(), AabbMargin, Gravity, TimeStep, Cache);
            }

            /// <summary>
            /// Moves dynamic body i to its motion's pose, and writes its bounds for the tree and for the cache. Also used
            /// by <see cref="SharedDynamicTreeRefit"/>.
            /// </summary>
            internal static unsafe void UpdateBody(int i, MotionData* motionDatas, MotionVelocity* motionVelocities, RigidBody* bodies,
                Aabb* aabbs, float aabbMargin, float3 gravity, float timeStep, in DynamicBoundsCache.View cache)
            {
                RigidBody body = bodies[i];
                body.WorldFromBody = math.mul(motionDatas[i].WorldFromMotion, math.inverse(motionDatas[i].BodyFromMotion));
                bodies[i] = body;
                Aabb bounds = cache.IsCreated ? cache.ColliderBounds(i, &body) : body.CalculateAabb();
                aabbs[i] = Broadphase.PrepareDynamicBodyDataJob.ExpandAabb(bounds, aabbMargin, gravity, timeStep, motionVelocities[i]);

                // Cache bounds at the current pose. Write-back normalization may change rotation bits,
                // in which case the next build recomputes bounds.
                if (cache.IsCreated && body.Collider.IsCreated)
                {
                    cache.Bounds[i] = bounds;
                }
            }
        }

        /// <summary>
        /// Fits one subtree below the cut of the dynamic tree to the bounds from <see cref="UpdateDynamicBoundsJob"/>
        /// in each iteration. No two iterations fit the same node.
        /// </summary>
        [BurstCompile]
        struct RefitDynamicSubtreesJob : IJobParallelFor
        {
            [NativeDisableContainerSafetyRestriction]
            public Broadphase.Tree Tree;

            [ReadOnly]
            public NativeArray<Aabb> Aabbs;

            public void Execute(int slot)
            {
                BoundingVolumeHierarchy bvh = Tree.BoundingVolumeHierarchy;
                int root = bvh.FindRefitSubtree(slot);
                if (root != 0)
                {
                    bvh.RefitSubtree(root, Aabbs);
                }
            }
        }

        /// <summary>Fits the nodes above the cut of the dynamic tree, after <see cref="RefitDynamicSubtreesJob"/>.</summary>
        [BurstCompile]
        struct RefitDynamicTreeTopJob : IJob
        {
            [NativeDisableContainerSafetyRestriction]
            public Broadphase.Tree Tree;

            [ReadOnly, DeallocateOnJobCompletion]
            public NativeArray<Aabb> Aabbs;

            public void Execute()
            {
                Tree.BoundingVolumeHierarchy.RefitAboveCut();
            }
        }
    }
}
