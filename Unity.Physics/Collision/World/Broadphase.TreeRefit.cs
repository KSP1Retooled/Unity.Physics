using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Assertions;
using static Unity.Physics.BoundingVolumeHierarchy;

namespace Unity.Physics
{
    partial struct Broadphase
    {
        /// <summary>
        /// Refit the existing dynamic tree using the same bounds, filters and responses as a fresh build.
        /// Require the same dynamic body count and preferably the same body order to retain useful spatial
        /// grouping.
        /// Keep results at a fixed address and read them only after completion. Report filter changes and tree
        /// quality.
        /// </summary>
        internal unsafe JobHandle ScheduleDynamicTreeRefitBuildJobs(ref PhysicsWorld world, float timeStep, float3 gravity,
            JobHandle inputDeps, DynamicBoundsCache boundsCache, DynamicTreeRefitResults* results)
        {
            using var scope = ScheduleDynamicBuildJobsMarker.Auto();

            int count = world.NumDynamicBodies;
            Assert.AreEqual(count, DynamicTree.NumBodies);
            Assert.IsTrue(count > 0);

            results->FiltersChanged = 0;
            var aabbs = new NativeArray<Aabb>(count, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            JobHandle handle = new PrepareDynamicBodyDataJob
            {
                RigidBodies = world.DynamicBodies,
                MotionVelocities = world.MotionVelocities,
                Aabbs = aabbs,
                FiltersOut = DynamicTree.BodyFilters.AsArray(),
                RespondsToCollisionOut = DynamicTree.RespondsToCollision.AsArray(),
                BodySolverTypesOut = DynamicTree.BodySolverTypes.AsArray(),
                AabbMargin = world.CollisionWorld.CollisionTolerance * 0.5f, // each body contributes half
                TimeStep = timeStep,
                Gravity = gravity,
                Incremental = DynamicTree.IncrementalRef,
                Cache = boundsCache.IsCreated ? boundsCache.Resize(count) : default,
                FiltersChanged = &results->FiltersChanged,
            }.Schedule(count, 32, inputDeps);

            // As in CollisionWorld.ScheduleRefitDynamicTree, the subtrees below the cut are fitted in parallel, and then
            // one job fits the few nodes above the cut.
            handle = new RefitBuiltSubtreesJob
            {
                Tree = DynamicTree,
                Aabbs = aabbs,
                Results = results,
            }.Schedule(BoundingVolumeHierarchy.RefitSubtreeCount, 1, handle);

            return new RefitBuiltTreeTopJob
            {
                Tree = DynamicTree,
                Aabbs = aabbs,
                Results = results,
            }.Schedule(handle);
        }

        /// <summary>
        /// Fits one subtree below the cut of the dynamic tree in each iteration, for
        /// <see cref="ScheduleDynamicTreeRefitBuildJobs"/>, and records the subtree's areas.
        /// </summary>
        [BurstCompile]
        unsafe struct RefitBuiltSubtreesJob : IJobParallelFor
        {
            [NativeDisableContainerSafetyRestriction]
            public Tree Tree;

            [ReadOnly]
            public NativeArray<Aabb> Aabbs;

            [NativeDisableUnsafePtrRestriction]
            public DynamicTreeRefitResults* Results;

            public void Execute(int slot)
            {
                BoundingVolumeHierarchy bvh = Tree.BoundingVolumeHierarchy;
                BoundingVolumeHierarchy.MeasuringFitter fitter = Results->Fitter(Tree, (Aabb*)Aabbs.GetUnsafeReadOnlyPtr());
                int root = bvh.FindRefitSubtree(slot);
                if (root != 0)
                {
                    bvh.RefitSubtree(root, ref fitter);
                }

                Results->SlotNodeAreas[slot] = fitter.NodeArea;
                Results->SlotLeafAreas[slot] = fitter.LeafArea;
            }
        }

        /// <summary>
        /// Fits the nodes above the cut of the dynamic tree after <see cref="RefitBuiltSubtreesJob"/>, and adds up the
        /// areas of the whole tree.
        /// </summary>
        [BurstCompile]
        unsafe struct RefitBuiltTreeTopJob : IJob
        {
            [NativeDisableContainerSafetyRestriction]
            public Tree Tree;

            [ReadOnly, DeallocateOnJobCompletion]
            public NativeArray<Aabb> Aabbs;

            [NativeDisableUnsafePtrRestriction]
            public DynamicTreeRefitResults* Results;

            public void Execute()
            {
                BoundingVolumeHierarchy.MeasuringFitter fitter = Results->Fitter(Tree, (Aabb*)Aabbs.GetUnsafeReadOnlyPtr());
                Tree.BoundingVolumeHierarchy.RefitAboveCut(ref fitter);
                float nodeArea = fitter.NodeArea;
                float leafArea = 0.0f;
                for (int slot = 0; slot < BoundingVolumeHierarchy.RefitSubtreeCount; slot++)
                {
                    nodeArea += Results->SlotNodeAreas[slot];
                    leafArea += Results->SlotLeafAreas[slot];
                }

                Results->NodeArea = nodeArea;
                Results->LeafArea = leafArea;
            }
        }
    }
}
