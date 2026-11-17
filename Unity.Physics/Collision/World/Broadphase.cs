//#define BVH_CHECK_INTEGRITY

using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine.Assertions;
using static Unity.Physics.BoundingVolumeHierarchy;

namespace Unity.Physics
{
    // A bounding volume around a collection of rigid bodies
    [NoAlias]
    struct Broadphase : IDisposable
    {
        internal struct OverlapResult
        {
            public BodyIndexPair BodyPair;
            public SolverType SolverType;

            public OverlapResult(int bodyIndexA, int bodyIndexB, SolverType solverType = Solver.kDefaultSolverType)
            {
                BodyPair = new() { BodyIndexA = bodyIndexA, BodyIndexB = bodyIndexB };
                SolverType = solverType;
            }
        }

        internal struct InsertionData
        {
            public Aabb Aabb;
            public PointAndIndex PointAndIndex;
            public CollisionFilter Filter;
        }

        [NoAlias]
        public Tree StaticTree;  // The tree of static rigid bodies
        [NoAlias]
        public Tree DynamicTree; // The tree of dynamic rigid bodies

        public Aabb Domain =>
            Aabb.Union(StaticTree.BoundingVolumeHierarchy.Domain, DynamicTree.BoundingVolumeHierarchy.Domain);

        public int NumStaticBodies => StaticTree.NumBodies;

        public int NumDynamicBodies => DynamicTree.NumBodies;

        public Broadphase(int numStaticBodies, int numDynamicBodies)
        {
            StaticTree = new Tree(numStaticBodies);
            DynamicTree = new Tree(numDynamicBodies);
        }

        internal Broadphase(Tree staticTree, Tree dynamicTree)
        {
            StaticTree = staticTree;
            DynamicTree = dynamicTree;
        }

        public void Reset(int numStaticBodies, int numDynamicBodies)
        {
            StaticTree.Reset(numStaticBodies);
            DynamicTree.Reset(numDynamicBodies);
        }

        public void Dispose()
        {
            StaticTree.Dispose();
            DynamicTree.Dispose();
        }

        public Broadphase Clone()
        {
            return new Broadphase
            {
                StaticTree = StaticTree.Clone(),
                DynamicTree = DynamicTree.Clone()
            };
        }

        #region Build

        /// <summary>
        /// Build the broadphase based on the given world.
        /// </summary>
        public void Build(NativeArray<RigidBody> staticBodies, NativeArray<RigidBody> dynamicBodies,
            NativeArray<MotionVelocity> motionVelocities, float collisionTolerance, float timeStep, float3 gravity, bool buildStaticTree = true, bool reset = true, bool incrementalDynamicTree = false, bool incrementalStaticTree = false)
        {
            float aabbMargin = collisionTolerance * 0.5f; // each body contributes half

            if (buildStaticTree)
            {
                if (reset)
                {
                    StaticTree.Reset(staticBodies.Length);
                }
                BuildStaticTree(staticBodies, aabbMargin, incrementalStaticTree);
            }

            if (reset)
            {
                DynamicTree.Reset(dynamicBodies.Length);
            }
            BuildDynamicTree(dynamicBodies, motionVelocities, gravity, timeStep, aabbMargin, incrementalDynamicTree);
        }

        /// <summary>
        /// Build the static tree of the broadphase based on the given array of rigid bodies.
        /// </summary>
        public void BuildStaticTree(NativeArray<RigidBody> staticBodies, float aabbMargin, bool incremental = false)
        {
            Assert.AreEqual(staticBodies.Length, StaticTree.NumBodies);

            if (staticBodies.Length == 0)
            {
                StaticTree.BoundingVolumeHierarchy.Clear();
                return;
            }
            // else:

            if (incremental)
            {
                if (StaticTree.InsertBodyDataStream.IsCreated && (StaticTree.InsertBodyDataStream.Count() == staticBodies.Length-1))
                {
                    // Trigger a full rebuild when everything has changed. This might be a scene reload
                    StaticTree.Incremental = false;
                }

                StaticTree.BuiltMultiThreaded = false;
                StaticTree.BuildIncremental();
#if BVH_CHECK_INTEGRITY
                unsafe
                {
                    // Note: Remove one body since the static tree does not contain the default static body. It does not have a collider and is not included in the tree creation.
                    var expectedNumBodies = staticBodies.Length - 1;
                    StaticTree.BoundingVolumeHierarchy.CheckIntegrity(expectedNumBodies, StaticTree.BodyFilters.GetUnsafePtr());
                }
#endif
                return;
            }
            // else:

            StaticTree.BuiltMultiThreaded = false;

            // Read bodies
            var aabbs = new NativeArray<Aabb>(staticBodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var points = new NativeArray<PointAndIndex>(staticBodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int i = 0; i < staticBodies.Length; i++)
            {
                PrepareStaticBodyDataJob.ExecuteImpl(i, aabbMargin, staticBodies, aabbs, points,
                    StaticTree.BodyFilters.AsArray(), StaticTree.RespondsToCollision.AsArray(), StaticTree.BodySolverTypes.AsArray(),
                    StaticTree.MeshSubAabbs.AsArray(), StaticTree.IncrementalRef);
            }

            // Build tree
            var bvh = StaticTree.BoundingVolumeHierarchy;
            bvh.Build(points, aabbs, out int nodeCount);

            // Build node filters
            bvh.BuildCombinedCollisionFilter(StaticTree.BodyFilters.AsArray(), 1, nodeCount - 1);

            points.Dispose();
            aabbs.Dispose();
        }

#if ENABLE_PROFILER
        static readonly ProfilerMarker s_RemovalsMarker = new(ProfilerCategory.Physics, "BVH.RemovalsAndUpdates");
        static readonly ProfilerMarker s_InsertionsMarker = new(ProfilerCategory.Physics, "BVH.Insertions");
#endif

        /// <summary>
        /// Build the dynamic tree of the broadphase based on the given array of rigid bodies and motions.
        /// </summary>
        public void BuildDynamicTree(NativeArray<RigidBody> dynamicBodies,
            NativeArray<MotionVelocity> motionVelocities, float3 gravity, float timeStep, float aabbMargin, bool incremental = false)
        {
            Assert.AreEqual(dynamicBodies.Length, DynamicTree.NumBodies);

            if (dynamicBodies.Length == 0)
            {
                DynamicTree.BoundingVolumeHierarchy.Clear();
                return;
            }
            // else:

            if (incremental)
            {
                if (DynamicTree.InsertBodyDataStream.IsCreated && (DynamicTree.InsertBodyDataStream.Count() == dynamicBodies.Length))
                {
                    // Trigger a full rebuild when everything has changed. This might be a scene reload
                    DynamicTree.Incremental = false;
                }

                DynamicTree.BuildIncremental();
#if BVH_CHECK_INTEGRITY
                unsafe
                {
                    DynamicTree.BoundingVolumeHierarchy.CheckIntegrity(dynamicBodies.Length, DynamicTree.BodyFilters.GetUnsafePtr());
                }
#endif
                return;
            }
            // else:

            var aabbs = new NativeArray<Aabb>(dynamicBodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var points = new NativeArray<PointAndIndex>(dynamicBodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            for (int i = 0; i < dynamicBodies.Length; i++)
            {
                PrepareDynamicBodyDataJob.ExecuteImpl(i, aabbMargin, gravity, timeStep, dynamicBodies, motionVelocities, aabbs, points,
                    DynamicTree.BodyFilters.AsArray(), DynamicTree.RespondsToCollision.AsArray(), DynamicTree.BodySolverTypes.AsArray(), DynamicTree.IncrementalRef);
            }

            var bvh = DynamicTree.BoundingVolumeHierarchy;

            bvh.Build(points, aabbs, out var nodeCount);
            bvh.BuildCombinedCollisionFilter(DynamicTree.BodyFilters.AsArray(), 1, nodeCount - 1);

            points.Dispose();
            aabbs.Dispose();
        }

        /// <summary>
        /// Schedule a set of jobs to build the broadphase based on the given world.
        /// </summary>
        public JobHandle ScheduleBuildJobs(ref PhysicsWorld world, float timeStep, float3 gravity, NativeReference<int>.ReadOnly buildStaticTree, JobHandle inputDeps,
            bool multiThreaded = true, bool reset = true, bool incrementalDynamicTree = false, bool incrementalStaticTree = false)
        {
            if (!multiThreaded)
            {
                return new BuildBroadphaseJob
                {
                    StaticBodies = world.StaticBodies,
                    DynamicBodies = world.DynamicBodies,
                    MotionVelocities = world.MotionVelocities,
                    CollisionTolerance = world.CollisionWorld.CollisionTolerance,
                    TimeStep = timeStep,
                    Gravity = gravity,
                    BuildStaticTree = buildStaticTree,
                    Reset = reset,
                    Broadphase = this,
                    IncrementalDynamicTree = incrementalDynamicTree,
                    IncrementalStaticTree = incrementalStaticTree
                }.Schedule(inputDeps);
            }
            // else:

            // +1 for main thread
            int threadCount = JobsUtility.JobWorkerCount + 1;
            return JobHandle.CombineDependencies(
                ScheduleStaticTreeBuildJobs(ref world, threadCount, buildStaticTree, inputDeps, incrementalStaticTree),
                ScheduleDynamicTreeBuildJobs(ref world, timeStep, gravity, threadCount, inputDeps, incrementalDynamicTree));
        }

        static readonly ProfilerMarker ScheduleStaticTreeBuildJobsMarker = new("ScheduleStaticTreeBuildJobs");

        /// <summary>
        /// Schedule a set of jobs to build the static tree of the broadphase based on the given world.
        /// </summary>
        public JobHandle ScheduleStaticTreeBuildJobs(
            ref PhysicsWorld world, int numThreadsHint, NativeReference<int>.ReadOnly shouldDoWork, JobHandle inputDeps, bool incremental = false)
        {
            using var scope = ScheduleStaticTreeBuildJobsMarker.Auto();

            Assert.AreEqual(world.NumStaticBodies, StaticTree.NumBodies);

            if (world.NumStaticBodies == 0)
            {
                StaticTree.BoundingVolumeHierarchy.Clear();
                return inputDeps;
            }
            // else:

            if (incremental)
            {
                return new BuildIncrementalTreeJob
                {
                    Tree = StaticTree,
                    IsStaticTree = true
                }.Schedule(inputDeps);
            }
            // else:

            var aabbs = new NativeArray<Aabb>(world.NumStaticBodies, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var points = new NativeArray<PointAndIndex>(world.NumStaticBodies, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            var numStaticBodiesArray = new NativeReference<int>(Allocator.TempJob);
            JobHandle handle = new PrepareNumStaticBodiesJob
            {
                NumStaticBodies = world.NumStaticBodies,
                BuildStaticTree = shouldDoWork,
                NumStaticBodiesArray = numStaticBodiesArray,
                BuiltMultiThreaded = StaticTree.BuiltMultiThreadedRef
            }.Schedule(inputDeps);

            handle = new PrepareStaticBodyDataJob
            {
                RigidBodies = world.StaticBodies,
                Aabbs = aabbs,
                Points = points,
                FiltersOut = StaticTree.BodyFilters.AsArray(),
                RespondsToCollisionOut = StaticTree.RespondsToCollision.AsArray(),
                BodySolverTypesOut = StaticTree.BodySolverTypes.AsArray(),
                MeshSubAabbsOut = StaticTree.MeshSubAabbs.AsArray(),
                AabbMargin = world.CollisionWorld.CollisionTolerance * 0.5f, // each body contributes half
                Incremental = StaticTree.IncrementalRef
            }.ScheduleUnsafe(numStaticBodiesArray, 32, handle);

            var buildHandle = StaticTree.BoundingVolumeHierarchy.ScheduleBuildJobs(
                points, aabbs, StaticTree.BodyFilters.AsArray(), shouldDoWork, numThreadsHint, handle, StaticTree.Ranges, StaticTree.BranchCount);

            return JobHandle.CombineDependencies(buildHandle, numStaticBodiesArray.Dispose(handle));
        }

        static readonly ProfilerMarker ScheduleDynamicBuildJobsMarker = new("ScheduleDynamicTreeBuildJobs");

        /// <summary>
        /// Schedule a set of jobs to build the dynamic tree of the broadphase based on the given world.
        /// </summary>
        public JobHandle ScheduleDynamicTreeBuildJobs(
            ref PhysicsWorld world, float timeStep, float3 gravity, int numThreadsHint, JobHandle inputDeps, bool incremental = false)
        {
            return ScheduleDynamicTreeBuildJobs(ref world, timeStep, gravity, numThreadsHint, inputDeps, default, incremental);
        }

        /// <summary>
        /// Schedules the jobs that build the dynamic tree, as the overload without a cache does. A body that is the
        /// same as before the build (<see cref="DynamicBoundsCache.View.Same"/>, which the code that built the bodies
        /// must have set) takes its collider's bounds from <paramref name="boundsCache"/> instead of computing them
        /// from its collider. The job writes the bounds of every other body into the cache. See
        /// <see cref="DynamicBoundsCache"/>.
        /// </summary>
        internal unsafe JobHandle ScheduleDynamicTreeBuildJobs(ref PhysicsWorld world, float timeStep, float3 gravity, int numThreadsHint,
            JobHandle inputDeps, DynamicBoundsCache boundsCache, bool incremental = false)
        {
            using var scope = ScheduleDynamicBuildJobsMarker.Auto();

            Assert.AreEqual(world.NumDynamicBodies, DynamicTree.NumBodies);

            // A build that doesn't run the job below leaves no entry of the cache for the current colliders, so it
            // empties the cache.
            if (boundsCache.IsCreated && (world.NumDynamicBodies == 0 || incremental))
            {
                boundsCache.Resize(0);
            }

            if (world.NumDynamicBodies == 0)
            {
                DynamicTree.BoundingVolumeHierarchy.Clear();
                return inputDeps;
            }
            // else:

            if (incremental)
            {
                return new BuildIncrementalTreeJob
                {
                    Tree = DynamicTree,
                    IsStaticTree =  false
                }.Schedule(inputDeps);
            }
            // else:

            var aabbs = new NativeArray<Aabb>(world.NumDynamicBodies, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var points = new NativeArray<PointAndIndex>(world.NumDynamicBodies, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            JobHandle handle = new PrepareDynamicBodyDataJob
            {
                RigidBodies = world.DynamicBodies,
                MotionVelocities = world.MotionVelocities,
                Aabbs = aabbs,
                Points = points,
                FiltersOut = DynamicTree.BodyFilters.AsArray(),
                RespondsToCollisionOut = DynamicTree.RespondsToCollision.AsArray(),
                BodySolverTypesOut = DynamicTree.BodySolverTypes.AsArray(),
                AabbMargin = world.CollisionWorld.CollisionTolerance * 0.5f, // each body contributes half
                TimeStep = timeStep,
                Gravity = gravity,
                Incremental = DynamicTree.IncrementalRef,
                Cache = boundsCache.IsCreated ? boundsCache.Resize(world.NumDynamicBodies) : default,
            }.Schedule(world.NumDynamicBodies, 32, inputDeps);

            var shouldDoWork = new NativeReference<int>(1, Allocator.TempJob);

            // Note: points and aabbs deallocated via ScheduleBuildJobs
            handle = DynamicTree.BoundingVolumeHierarchy.ScheduleBuildJobs(
                points, aabbs, DynamicTree.BodyFilters.AsArray(), shouldDoWork, numThreadsHint, handle,
                DynamicTree.Ranges, DynamicTree.BranchCount);

            return shouldDoWork.Dispose(handle);
        }

        /// <summary>
        /// Schedules jobs that do the work of <see cref="ScheduleDynamicTreeBuildJobs(ref PhysicsWorld, float, float3, int, JobHandle, DynamicBoundsCache, bool)"/>
        /// without building the tree again: they compute every body's bounds, filter and response as the build does, and
        /// then fit the bounds of the tree that the last build left to them. The tree keeps its shape, so each leaf
        /// holds the same body indices as before. Use it only when the last build of the tree, or the last refit of
        /// this kind, was for the same number of dynamic bodies; the caller should also make sure that the bodies are
        /// mostly the bodies that the tree was built for, because the tree is only as good as the grouping of the
        /// bodies that it was built for.
        /// <para>
        /// The leaves get exactly the bounds that a build would give them, and every node's bounds and filter hold
        /// all of its children's, as in a built tree. So the tree finds exactly the pairs of bodies that a built tree
        /// finds, though in another order, and queries find exactly the same hits. When a body's filter differs from
        /// the filter that the tree held for its index, the refit builds the nodes' filters again from the bodies'.
        /// </para>
        /// <para>
        /// The jobs leave in <paramref name="results"/> whether a filter changed, and two surface areas that tell how
        /// good the tree still is (see <see cref="DynamicTreeRefitResults.Quality"/>). The caller keeps the results
        /// at a fixed address and reads them only after the jobs have completed.
        /// </para>
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

        #endregion

        #region Find overlaps

        // Write all overlapping body pairs to the given streams,
        // where at least one of the bodies is dynamic. The results are unsorted.
        public void FindOverlaps(ref NativeStream.Writer dynamicVsDynamicPairsWriter, ref NativeStream.Writer staticVsDynamicPairsWriter)
        {
            // Dynamic-dynamic
            {
                dynamicVsDynamicPairsWriter.BeginForEachIndex(0);
                DynamicVsDynamicFindOverlappingPairsJob.ExecuteImpl(
                    new int2(1, 1), DynamicTree, ref dynamicVsDynamicPairsWriter);
                dynamicVsDynamicPairsWriter.EndForEachIndex();
            }

            // Static-dynamic
            {
                staticVsDynamicPairsWriter.BeginForEachIndex(0);
                StaticVsDynamicFindOverlappingPairsJob.ExecuteImpl(
                    new int2(1, 1), StaticTree, DynamicTree, ref staticVsDynamicPairsWriter);
                staticVsDynamicPairsWriter.EndForEachIndex();
            }
        }

        // Schedule a set of jobs which will write all overlapping body pairs to the given stream,
        // where at least one of the bodies is dynamic. The results are unsorted.
        public SimulationJobHandles ScheduleFindOverlapsJobs(out NativeStream dynamicVsDynamicPairsStream, out NativeStream staticVsDynamicPairsStream,
            JobHandle inputDeps, bool multiThreaded = true, bool incrementalDynamicTree = false, bool incrementalStaticTree = false)
        {
            SimulationJobHandles returnHandles = default;

            if (!multiThreaded)
            {
                dynamicVsDynamicPairsStream = new NativeStream(1, Allocator.TempJob);
                staticVsDynamicPairsStream = new NativeStream(1, Allocator.TempJob);
                returnHandles.FinalExecutionHandle = new FindOverlapsJob
                {
                    Broadphase = this,
                    DynamicVsDynamicPairsWriter = dynamicVsDynamicPairsStream.AsWriter(),
                    StaticVsDynamicPairsWriter = staticVsDynamicPairsStream.AsWriter()
                }.Schedule(inputDeps);

                return returnHandles;
            }

            var dynamicVsDynamicNodePairIndices = new NativeList<int2>(Allocator.TempJob);
            var staticVsDynamicNodePairIndices = new NativeList<int2>(Allocator.TempJob);

            if (incrementalDynamicTree || incrementalStaticTree)
            {
                JobHandle dynamicTreeSplitJobHandle = default;
                JobHandle staticTreeSplitJobHandle = default;

                var threadCount = JobsUtility.JobWorkerCount + 1;
                if (incrementalDynamicTree)
                {
                    dynamicTreeSplitJobHandle = new CalculateTreeSplitDataJob
                    {
                        Tree = DynamicTree,
                        ThreadCount = threadCount
                    }.Schedule(inputDeps);
                }

                if (incrementalStaticTree)
                {
                    staticTreeSplitJobHandle = new CalculateTreeSplitDataJob
                    {
                        Tree = StaticTree,
                        ThreadCount = threadCount
                    }.Schedule(inputDeps);
                }

                inputDeps = JobHandle.CombineDependencies(dynamicTreeSplitJobHandle, staticTreeSplitJobHandle);
            }

            JobHandle allocateDeps = new AllocateDynamicVsStaticNodePairs
            {
                dynamicVsDynamicNodePairIndices = dynamicVsDynamicNodePairIndices,
                staticVsDynamicNodePairIndices = staticVsDynamicNodePairIndices,
                dynamicBranchCount = DynamicTree.BranchCount,
                staticBranchCount = StaticTree.BranchCount
            }.Schedule(inputDeps);

            // Build pairs of branch node indices
            JobHandle dynamicVsDynamicPairs = new DynamicVsDynamicBuildBranchNodePairsJob
            {
                Ranges = DynamicTree.Ranges,
                NumBranches = DynamicTree.BranchCount,
                NodePairIndices = dynamicVsDynamicNodePairIndices.AsDeferredJobArray()
            }.Schedule(allocateDeps);

            JobHandle staticVsDynamicPairs = new StaticVsDynamicBuildBranchNodePairsJob
            {
                DynamicRanges = DynamicTree.Ranges,
                StaticRanges = StaticTree.Ranges,
                NumStaticBranches = StaticTree.BranchCount,
                NumDynamicBranches = DynamicTree.BranchCount,
                NodePairIndices = staticVsDynamicNodePairIndices.AsDeferredJobArray()
            }.Schedule(allocateDeps);

            JobHandle dynamicConstruct = NativeStream.ScheduleConstruct(out dynamicVsDynamicPairsStream, dynamicVsDynamicNodePairIndices, allocateDeps, Allocator.TempJob);
            JobHandle staticConstruct = NativeStream.ScheduleConstruct(out staticVsDynamicPairsStream, staticVsDynamicNodePairIndices, allocateDeps, Allocator.TempJob);

            // Write all overlaps to the stream (also deallocates nodePairIndices)
            JobHandle dynamicVsDynamicHandle = new DynamicVsDynamicFindOverlappingPairsJob
            {
                DynamicTree = DynamicTree,
                PairWriter = dynamicVsDynamicPairsStream.AsWriter(),
                NodePairIndices = dynamicVsDynamicNodePairIndices.AsDeferredJobArray()
            }.Schedule(dynamicVsDynamicNodePairIndices, 1, JobHandle.CombineDependencies(dynamicVsDynamicPairs, dynamicConstruct));

            // Write all overlaps to the stream (also deallocates nodePairIndices)
            JobHandle staticVsDynamicHandle = new StaticVsDynamicFindOverlappingPairsJob
            {
                StaticTree = StaticTree,
                DynamicTree = DynamicTree,
                PairWriter = staticVsDynamicPairsStream.AsWriter(),
                NodePairIndices = staticVsDynamicNodePairIndices.AsDeferredJobArray()
            }.Schedule(staticVsDynamicNodePairIndices, 1, JobHandle.CombineDependencies(staticVsDynamicPairs, staticConstruct));

            // Dispose node pair lists
            var disposeOverlapPairs0 = dynamicVsDynamicNodePairIndices.Dispose(dynamicVsDynamicHandle);
            var disposeOverlapPairs1 = staticVsDynamicNodePairIndices.Dispose(staticVsDynamicHandle);

            returnHandles.FinalDisposeHandle = JobHandle.CombineDependencies(disposeOverlapPairs0, disposeOverlapPairs1);
            returnHandles.FinalExecutionHandle = JobHandle.CombineDependencies(dynamicVsDynamicHandle, staticVsDynamicHandle);

            return returnHandles;
        }

        #endregion

        #region Tree

        // A tree of rigid bodies
        [NoAlias]
        public struct Tree : IDisposable
        {
            [NoAlias] public NativeList<Node> Nodes;    // The nodes of the bounding volume; used in bvh
            [NoAlias] public NativeList<CollisionFilter> NodeFilters;   // The collision filter for each node (a union of all its children); used in bvh
            [NativeDisableContainerSafetyRestriction]
            [NoAlias] public NativeList<CollisionFilter> BodyFilters;  // A copy of the collision filter of each body; used when finding overlap pairs.
            [NativeDisableContainerSafetyRestriction]
            [NoAlias] public NativeList<bool> RespondsToCollision; // A copy of the RespondsToCollision flag of each body
            [NoAlias] public NativeList<SolverType> BodySolverTypes; // A copy of the SolverType of each body
            [NativeDisableContainerSafetyRestriction]
            [NoAlias] public NativeList<FourTransposedAabbs> MeshSubAabbs; // Cached world-space sub-AABBs of mesh BVH root children, for raycast pre-filtering
            [NoAlias] internal NativeArray<Builder.Range> Ranges;   // Element ranges used during building; Root node index of the ranges is used as input
                                                                    // to parallel implementation of finding the overlap pairs. See Broadphase.ScheduleFindOverlapsJobs.
            [NoAlias] internal NativeArray<int> BranchCount; // Number of branches built processed in parallel in multi-threaded implementation of bvh building.
                                                             // Used as input to parallel implementation of finding the overlap pairs. See Broadphase.ScheduleFindOverlapsJobs.

            [NoAlias]
            [NativeDisableContainerSafetyRestriction]
            NativeStream m_RemoveBodyDataStream;
            [NoAlias]
            [NativeDisableContainerSafetyRestriction]
            NativeStream m_UpdateBodyDataStream;
            [NoAlias]
            [NativeDisableContainerSafetyRestriction]
            NativeStream m_InsertBodyDataStream;
            [NoAlias]
            [NativeDisableContainerSafetyRestriction]
            NativeList<ElementLocationData> m_UpdatedElementLocationDataList;
            [NoAlias]
            IncrementalInsertionContext m_IncrementalInsertionContext;
            [NoAlias]
            NativeReference<bool> m_Incremental;
            [NoAlias]
            NativeReference<bool> m_BuiltMultiThreaded;

            // Data stream representing rigid bodies that need to be removed from the tree, e.g., due to their deletion.
            // Used as part of the incremental broadphase.
            public NativeStream RemoveBodyDataStream
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => m_RemoveBodyDataStream;
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                set => m_RemoveBodyDataStream = value;
            }

            // Data stream representing rigid bodies for which data needs to be updated in the tree, i.e., their
            // Aabb or collision filter.
            // Used as part of the incremental broadphase.
            public NativeStream UpdateBodyDataStream
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => m_UpdateBodyDataStream;
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                set => m_UpdateBodyDataStream = value;
            }

            // Data stream representing rigid bodies that need to be inserted into the tree, e.g., due to their creation.
            // Used as part of the incremental broadphase.
            public NativeStream InsertBodyDataStream
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => m_InsertBodyDataStream;
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                set => m_InsertBodyDataStream = value;
            }

            // Latest incremental location data for rigid bodies which were inserted into the tree or moved within the tree.
            // Used for updating temporal coherence data for rigid bodies as part of the incremental broadphase.
            public NativeList<ElementLocationData> UpdatedElementLocationDataList
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => m_UpdatedElementLocationDataList;
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                set => m_UpdatedElementLocationDataList = value;
            }

            internal Allocator Allocator;

            internal bool Incremental
            {
                get => m_Incremental.Value;
                set => m_Incremental.Value = value;
            }

            internal bool BuiltMultiThreaded
            {
                get => m_BuiltMultiThreaded.Value;
                set => m_BuiltMultiThreaded.Value = value;
            }

            internal NativeReference<bool> BuiltMultiThreadedRef => m_BuiltMultiThreaded;

            internal NativeReference<bool> IncrementalRef => m_Incremental;

            /// <summary>
            /// Whether the next <see cref="BuildIncremental"/> discards the tree's content, because the tree is empty (the root node never holds bodies itself) or was last built from scratch.
            /// </summary>
            internal bool RequiresClearOnIncrementalBuild => BoundingVolumeHierarchy.NodeCount <= 2 || !Incremental;

            public BoundingVolumeHierarchy BoundingVolumeHierarchy
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get
                {
                    unsafe
                    {
                        return new BoundingVolumeHierarchy(Nodes.GetUnsafeList(), NodeFilters.GetUnsafeList(), clear: false);
                    }
                }
            }

            public int NumBodies => BodyFilters.IsCreated ? BodyFilters.Length : 0;

            public Tree(int numBodies, Allocator allocator = Allocator.Persistent)
            {
                this = default;
                Allocator = allocator;
                SetCapacity(numBodies);
                Ranges = new NativeArray<BoundingVolumeHierarchy.Builder.Range>(
                    BoundingVolumeHierarchy.Constants.MaxNumTreeBranches, allocator, NativeArrayOptions.UninitializedMemory);
                BranchCount = new NativeArray<int>(1, allocator, NativeArrayOptions.ClearMemory);

                m_RemoveBodyDataStream = default;
                m_UpdateBodyDataStream = default;
                m_InsertBodyDataStream = default;
                m_UpdatedElementLocationDataList = default;
                m_IncrementalInsertionContext = new IncrementalInsertionContext(128, allocator);

                m_Incremental = new NativeReference<bool>(allocator);
                m_Incremental.Value = false;
                m_BuiltMultiThreaded = new NativeReference<bool>(allocator);
                m_BuiltMultiThreaded.Value = false;
            }

            public void Reset(int numBodies)
            {
                if (numBodies != BodyFilters.Length)
                {
                    SetCapacity(numBodies);
                }

                m_RemoveBodyDataStream = default;
                m_UpdateBodyDataStream = default;
                m_InsertBodyDataStream = default;
                m_UpdatedElementLocationDataList = default;

                m_IncrementalInsertionContext.Clear();
            }

            private void SetCapacity(int numBodies)
            {
                int numNodes = numBodies + BoundingVolumeHierarchy.Constants.MaxNumTreeBranches;

                if (!Nodes.IsCreated)
                {
                    Nodes = new NativeList<BoundingVolumeHierarchy.Node>(numNodes, Allocator);
                    // Always initialize first 2 nodes as empty, to gracefully return from queries on an empty tree
                    Nodes.ResizeUninitialized(2);
                    Nodes[0] = Nodes[1] = BoundingVolumeHierarchy.Node.Empty;
                }
                else if (numNodes > Nodes.Capacity)
                {
                    Nodes.SetCapacity(numNodes);
                }

                if (!NodeFilters.IsCreated)
                {
                    NodeFilters = new NativeList<CollisionFilter>(numNodes, Allocator);
                    NodeFilters.ResizeUninitialized(2);
                    NodeFilters[0] = NodeFilters[1] = CollisionFilter.Zero;
                }
                else if (numNodes > NodeFilters.Capacity)
                {
                    NodeFilters.SetCapacity(numNodes);
                }

                if (!BodyFilters.IsCreated)
                {
                    BodyFilters = new NativeList<CollisionFilter>(numBodies, Allocator);
                }
                BodyFilters.ResizeUninitialized(numBodies);

                if (!RespondsToCollision.IsCreated)
                {
                    RespondsToCollision = new NativeList<bool>(numBodies, Allocator);
                }
                RespondsToCollision.ResizeUninitialized(numBodies);

                if (!BodySolverTypes.IsCreated)
                {
                    BodySolverTypes = new NativeList<SolverType>(numBodies, Allocator);
                }
                BodySolverTypes.ResizeUninitialized(numBodies);

                if (!MeshSubAabbs.IsCreated)
                {
                    MeshSubAabbs = new NativeList<FourTransposedAabbs>(numBodies, Allocator);
                }
                MeshSubAabbs.ResizeUninitialized(numBodies);
            }

            public Tree Clone()
            {
                var clone = new Tree
                {
                    Allocator = Allocator,
                    Nodes = new NativeList<Node>(Nodes.Capacity, Allocator), // use capacity for multithreaded use
                    NodeFilters = new NativeList<CollisionFilter>(NodeFilters.Capacity, Allocator), // use capacity for multithreaded use
                    BodyFilters = new NativeList<CollisionFilter>(BodyFilters.Length, Allocator),
                    RespondsToCollision = new NativeList<bool>(RespondsToCollision.Length, Allocator),
                    BodySolverTypes = new NativeList<SolverType>(BodySolverTypes.Length, Allocator),
                    MeshSubAabbs = new NativeList<FourTransposedAabbs>(MeshSubAabbs.Length, Allocator),
                    Ranges = new NativeArray<BoundingVolumeHierarchy.Builder.Range>(Ranges, Allocator),
                    BranchCount = new NativeArray<int>(BranchCount, Allocator),
                    m_RemoveBodyDataStream = default,
                    m_UpdateBodyDataStream = default,
                    m_InsertBodyDataStream = default,
                    m_UpdatedElementLocationDataList = default,
                    m_IncrementalInsertionContext = new IncrementalInsertionContext(128, Allocator.Persistent),
                    m_Incremental = new NativeReference<bool>(Allocator.Persistent),
                    m_BuiltMultiThreaded = new NativeReference<bool>(Allocator.Persistent)
                };
                clone.Nodes.CopyFrom(Nodes);
                clone.NodeFilters.CopyFrom(NodeFilters);
                clone.BodyFilters.CopyFrom(BodyFilters);
                clone.RespondsToCollision.CopyFrom(RespondsToCollision);
                clone.BodySolverTypes.CopyFrom(BodySolverTypes);
                clone.MeshSubAabbs.CopyFrom(MeshSubAabbs);
                clone.Incremental = Incremental;
                clone.BuiltMultiThreaded = BuiltMultiThreaded;

                return clone;
            }

            public void Dispose()
            {
                if (Nodes.IsCreated)
                    Nodes.Dispose();

                if (NodeFilters.IsCreated)
                    NodeFilters.Dispose();

                if (BodyFilters.IsCreated)
                    BodyFilters.Dispose();

                if (RespondsToCollision.IsCreated)
                    RespondsToCollision.Dispose();

                if (BodySolverTypes.IsCreated)
                    BodySolverTypes.Dispose();

                if (MeshSubAabbs.IsCreated)
                    MeshSubAabbs.Dispose();

                if (Ranges.IsCreated)
                    Ranges.Dispose();

                if (BranchCount.IsCreated)
                    BranchCount.Dispose();

                if (m_IncrementalInsertionContext.IsCreated)
                    m_IncrementalInsertionContext.Dispose();

                if (m_Incremental.IsCreated)
                    m_Incremental.Dispose();

                if (m_BuiltMultiThreaded.IsCreated)
                    m_BuiltMultiThreaded.Dispose();
            }

            [GenerateTestsForBurstCompatibility]
            public void BuildIncremental()
            {
                var bvh = BoundingVolumeHierarchy;

                // NodeCount=0 for when Tree has not yet been initialized. First two nodes in an empty tree are
                //    0: invalid node, 1: root node.
                // Incremental=false for if Tree was previously not built incrementally. We need to clear out any
                //   previous content to make sure we don't retain any leftover, untracked bodies.
                bool requiresTreeClear = RequiresClearOnIncrementalBuild;
                if (requiresTreeClear) bvh.Clear();

                Incremental = true;

                var removalsRequested = RemoveBodyDataStream.IsCreated && !RemoveBodyDataStream.IsEmpty();
                var updatesRequested = UpdateBodyDataStream.IsCreated && !UpdateBodyDataStream.IsEmpty();
                if (!requiresTreeClear && // don't want to try to remove or update if the tree was cleared
                    (removalsRequested || updatesRequested))
                {
                    var removals = removalsRequested ?
                        RemoveBodyDataStream.ToNativeArray<RemovalData>(Allocator.Temp) : new NativeArray<RemovalData>();
                    var updates = updatesRequested ?
                        UpdateBodyDataStream.ToNativeArray<UpdateData>(Allocator.Temp) : new NativeArray<UpdateData>();

                    // remove and update bodies
                    if (removals.Length + updates.Length != 0)
                    {
#if ENABLE_PROFILER
                        s_RemovalsMarker.Begin();
#endif
                        bvh.RemoveOrUpdate(removals, updates, BodyFilters.AsArray());
#if ENABLE_PROFILER
                        s_RemovalsMarker.End();
#endif
                    }

                    if (removals.IsCreated) removals.Dispose();
                    if (updates.IsCreated) updates.Dispose();
                }

                var insertionsRequested = InsertBodyDataStream.IsCreated && !InsertBodyDataStream.IsEmpty();
                if (insertionsRequested)
                {
                    // insert bodies
#if ENABLE_PROFILER
                    s_InsertionsMarker.Begin();
#endif
                    m_IncrementalInsertionContext.Clear();

                    var insertionReader = InsertBodyDataStream.AsReader();
                    for (int i = 0; i < insertionReader.ForEachCount; ++i)
                    {
                        insertionReader.BeginForEachIndex(i);

                        while (insertionReader.RemainingItemCount > 0)
                        {
                            var data = insertionReader.Read<InsertionData>();
                            m_IncrementalInsertionContext.Insert(ref bvh, data.Aabb, data.PointAndIndex, data.Filter);
                        }

                        insertionReader.EndForEachIndex();
                    }

                    // copy updated element location data from insertion context to output list
                    SafetyChecks.CheckAreEqualAndThrow(true, UpdatedElementLocationDataList.IsCreated);
                    UpdatedElementLocationDataList.Clear();
                    m_IncrementalInsertionContext.CopyTo(UpdatedElementLocationDataList);

#if ENABLE_PROFILER
                    s_InsertionsMarker.End();
#endif
                }

                // todo: Going forward, rebuild everything from scratch if first time (no previous tree available) or if too many changes have been detected.
                // This should  be done in parallel via different jobs scheduled when scheduling the "build broadphase" jobs.
                // The check for previously existing tree can easily be done at that point.
                // Note though that if we are in incremental mode, we must also set the node count in the underlying BVH UnsafeList to the correct value.
                // Otherwise we can not incrementally operate on the "from scratch" created BVH. This is also required for the BVH optimization phase where
                // we would create a completely new, from scratch built BVH and then overwrite the existing one with it.
                // The node count of the UnsafeList is currently not yet set when building from scratch. See BoundingVolumeHierarchy.Build() (easy, build function knows node count)
                // and BuildBranch().
                // For BuildBranch() we will need to do obtain the final node count that is used in a post process, such as FinalizeTreeJob.
                // FinalizeTreeJob performs the remaining refit for the first portion of the tree (see
                // BoundingVolumeHierarchy.Refit(startNodeIndex, endNodeIndex <=== here it sets the minimum branch offset (starting node of the branch) from all BuildBranchJobs).
                // And at this point in time, we should be able to get the max node count as well, if the BuildBranchJobs communicate that information.
                // This could be done by providing a NativeArray<int> to the BuildBranchJobs, which is filled with the max node count per job, which we know within the
                // BuildBranch() function. In the FinalizeTreeJob we can then get the total node count by finding the max in the array.
            }
        }

        #endregion

        #region Queries

        internal struct RigidBodyOverlapsCollector : IOverlapCollector
        {
            public NativeList<int> RigidBodyIndices;

            public unsafe void AddRigidBodyIndices(int* indices, int count)
            {
                RigidBodyIndices.AddRange(indices, count);
            }

            public unsafe void AddColliderKeys(ColliderKey* keys, int count) => SafetyChecks.ThrowNotSupportedException();

            public void PushCompositeCollider(ColliderKeyPath compositeKey) => SafetyChecks.ThrowNotSupportedException();

            public void PopCompositeCollider(uint numCompositeKeyBits) => SafetyChecks.ThrowNotSupportedException();
        }

        // Test broadphase nodes against the aabb in input. For any overlapping
        // tree leaves, put the body indices into the output leafIndices.
        public void OverlapAabb(OverlapAabbInput input, NativeArray<RigidBody> rigidBodies, ref NativeList<int> rigidBodyIndices)
        {
            if (input.Filter.IsEmpty)
                return;
            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());
            var leafCollector = new RigidBodyOverlapsCollector { RigidBodyIndices = rigidBodyIndices };

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;
            StaticTree.BoundingVolumeHierarchy.AabbOverlap(input, ref leafProcessor, ref leafCollector);

            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            DynamicTree.BoundingVolumeHierarchy.AabbOverlap(input, ref leafProcessor, ref leafCollector);
        }

        public bool CastRay<T>(RaycastInput input, NativeArray<RigidBody> rigidBodies, ref T collector)
            where T : struct, ICollector<RaycastHit>
        {
            if (input.Filter.IsEmpty)
                return false;

            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;
            bool hasHit = StaticTree.BoundingVolumeHierarchy.Raycast(input, ref leafProcessor, ref collector);

            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            hasHit |= DynamicTree.BoundingVolumeHierarchy.Raycast(input, ref leafProcessor, ref collector);

            return hasHit;
        }

#if BVH_COLLECT_METRICS
        // An overload for gathering performance BVH Metrics of the Static BVH Tree
        internal bool CastRay<T>(RaycastInput input, NativeArray<RigidBody> rigidBodies, ref T collector,
            out BVHTraversalMetrics metrics)
            where T : struct, ICollector<RaycastHit>
        {
            metrics = default;
            if (input.Filter.IsEmpty)
                return false;

            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;

            bool hasHit = StaticTree.BoundingVolumeHierarchy.RaycastWithMetrics(input, ref leafProcessor, ref collector,
                out var staticMetrics);

            metrics.NodesVisited = staticMetrics.NodesVisited;
            metrics.NodesPruned = staticMetrics.NodesPruned;
            metrics.LeafNodesProcessed = staticMetrics.LeafNodesProcessed;
            metrics.SortOperations = staticMetrics.SortOperations;
            metrics.MaxStackDepth = staticMetrics.MaxStackDepth;
            metrics.PrimitivesTestedTotal = staticMetrics.PrimitivesTestedTotal;

            // Dynamic leafProcessor is not using pre-fitlered MeshSubAabbs. This is because dynamic bodies require
            // updated AABBs each frame and this data would need to be recomputed. Cost/benefit would need analysis
            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            hasHit |= DynamicTree.BoundingVolumeHierarchy.Raycast(input, ref leafProcessor, ref collector);

            return hasHit;
        }
#endif

        public unsafe bool CastCollider<T>(ColliderCastInput input, NativeArray<RigidBody> rigidBodies, ref T collector)
            where T : struct, ICollector<ColliderCastHit>
        {
            Assert.IsTrue(input.Collider != null);
            if (input.Collider->GetCollisionFilter().IsEmpty)
                return false;

            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;
            bool hasHit = StaticTree.BoundingVolumeHierarchy.ColliderCast(input, ref leafProcessor, ref collector);

            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            hasHit |= DynamicTree.BoundingVolumeHierarchy.ColliderCast(input, ref leafProcessor, ref collector);

            return hasHit;
        }

#if BVH_COLLECT_METRICS
        // An overload for gathering performance BVH Metrics of the Static BVH Tree
        public unsafe bool CastCollider<T>(ColliderCastInput input, NativeArray<RigidBody> rigidBodies, ref T collector,
            out BVHTraversalMetrics metrics)
            where T : struct, ICollector<ColliderCastHit>
        {
            metrics = default;
            Assert.IsTrue(input.Collider != null);
            if (input.Collider->GetCollisionFilter().IsEmpty)
                return false;

            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;

            bool hasHit = StaticTree.BoundingVolumeHierarchy.ColliderCastWithMetrics(input, ref leafProcessor, ref collector,
                out var staticMetrics);

            metrics.NodesVisited = staticMetrics.NodesVisited;
            metrics.NodesPruned = staticMetrics.NodesPruned;
            metrics.LeafNodesProcessed = staticMetrics.LeafNodesProcessed;
            metrics.SortOperations = staticMetrics.SortOperations;
            metrics.MaxStackDepth = staticMetrics.MaxStackDepth;
            metrics.PrimitivesTestedTotal = staticMetrics.PrimitivesTestedTotal;

            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            hasHit |= DynamicTree.BoundingVolumeHierarchy.ColliderCast(input, ref leafProcessor, ref collector);

            return hasHit;
        }
#endif

        public bool CalculateDistance<T>(PointDistanceInput input, NativeArray<RigidBody> rigidBodies, ref T collector)
            where T : struct, ICollector<DistanceHit>
        {
            if (input.Filter.IsEmpty)
                return false;
            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;
            bool hasHit = StaticTree.BoundingVolumeHierarchy.Distance(input, ref leafProcessor, ref collector);

            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            hasHit |= DynamicTree.BoundingVolumeHierarchy.Distance(input, ref leafProcessor, ref collector);

            return hasHit;
        }

#if BVH_COLLECT_METRICS
        // An overload for gathering performance BVH Metrics of the Static BVH Tree
        public bool CalculateDistance<T>(PointDistanceInput input, NativeArray<RigidBody> rigidBodies, ref T collector,
            out BVHTraversalMetrics metrics)
            where T : struct, ICollector<DistanceHit>
        {
            metrics = default;
            if (input.Filter.IsEmpty)
                return false;
            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;

            bool hasHit = StaticTree.BoundingVolumeHierarchy.DistanceWithMetrics(input, ref leafProcessor, ref collector, out var staticMetrics);

            metrics.NodesVisited = staticMetrics.NodesVisited;
            metrics.NodesPruned = staticMetrics.NodesPruned;
            metrics.LeafNodesProcessed = staticMetrics.LeafNodesProcessed;
            metrics.SortOperations = staticMetrics.SortOperations;
            metrics.MaxStackDepth = staticMetrics.MaxStackDepth;
            metrics.PrimitivesTestedTotal = staticMetrics.PrimitivesTestedTotal;

            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            hasHit |= DynamicTree.BoundingVolumeHierarchy.Distance(input, ref leafProcessor, ref collector);

            return hasHit;
        }
#endif

        public unsafe bool CalculateDistance<T>(ColliderDistanceInput input, NativeArray<RigidBody> rigidBodies, ref T collector)
            where T : struct, ICollector<DistanceHit>
        {
            Assert.IsTrue(input.Collider != null);
            if (input.Collider->GetCollisionFilter().IsEmpty)
                return false;
            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;
            bool hasHit = StaticTree.BoundingVolumeHierarchy.Distance(input, ref leafProcessor, ref collector);

            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            hasHit |= DynamicTree.BoundingVolumeHierarchy.Distance(input, ref leafProcessor, ref collector);

            return hasHit;
        }

#if BVH_COLLECT_METRICS
        // An overload for gathering performance BVH Metrics of the Static BVH Tree
        public unsafe bool CalculateDistance<T>(ColliderDistanceInput input, NativeArray<RigidBody> rigidBodies, ref T collector,
            out BVHTraversalMetrics metrics)
            where T : struct, ICollector<DistanceHit>
        {
            metrics = default;
            Assert.IsTrue(input.Collider != null);
            if (input.Collider->GetCollisionFilter().IsEmpty)
                return false;
            var leafProcessor = new BvhLeafProcessor(rigidBodies, StaticTree.MeshSubAabbs.AsArray());

            leafProcessor.BaseRigidBodyIndex = DynamicTree.NumBodies;

            bool hasHit = StaticTree.BoundingVolumeHierarchy.DistanceWithMetrics(input, ref leafProcessor, ref collector,
                out var staticMetrics);

            metrics.NodesVisited = staticMetrics.NodesVisited;
            metrics.NodesPruned = staticMetrics.NodesPruned;
            metrics.LeafNodesProcessed = staticMetrics.LeafNodesProcessed;
            metrics.SortOperations = staticMetrics.SortOperations;
            metrics.MaxStackDepth = staticMetrics.MaxStackDepth;
            metrics.PrimitivesTestedTotal = staticMetrics.PrimitivesTestedTotal;

            leafProcessor = new BvhLeafProcessor(rigidBodies);
            leafProcessor.BaseRigidBodyIndex = 0;
            hasHit |= DynamicTree.BoundingVolumeHierarchy.Distance(input, ref leafProcessor, ref collector);

            return hasHit;
        }
    #endif

        internal struct BvhLeafProcessor :
            BoundingVolumeHierarchy.IRaycastLeafProcessor,
                BoundingVolumeHierarchy.IColliderCastLeafProcessor,
                BoundingVolumeHierarchy.IPointDistanceLeafProcessor,
                BoundingVolumeHierarchy.IColliderDistanceLeafProcessor,
                BoundingVolumeHierarchy.IAabbOverlapLeafProcessor
        {
            private readonly NativeArray<RigidBody> m_Bodies;
            private readonly NativeArray<FourTransposedAabbs> m_MeshSubAabbs;
            public int BaseRigidBodyIndex;

            public BvhLeafProcessor(NativeArray<RigidBody> bodies, NativeArray<FourTransposedAabbs> meshSubAabbs = default)
            {
                m_Bodies = bodies;
                m_MeshSubAabbs = meshSubAabbs;
                BaseRigidBodyIndex = 0;
            }

            public bool AabbOverlap(int rigidBodyIndex, ref NativeList<int> allHits)
            {
                allHits.Add(BaseRigidBodyIndex + rigidBodyIndex);
                return true;
            }

            public bool RayLeaf<T>(RaycastInput input, int rigidBodyIndex, ref T collector) where T : struct, ICollector<RaycastHit>
            {
                rigidBodyIndex += BaseRigidBodyIndex;
                RigidBody body = m_Bodies[rigidBodyIndex];

                if (m_MeshSubAabbs.IsCreated &&
                    body.Collider.IsCreated && body.Collider.Value.Type == ColliderType.Mesh)
                {
                    int localIndex = rigidBodyIndex - BaseRigidBodyIndex;

                    bool4 hits = m_MeshSubAabbs[localIndex].Raycast(input.Ray, collector.MaxFraction, out _);
                    if (!math.any(hits))
                        return false;

                    input.QueryContext.MeshBvhChildIndex = QueryContext.SingleHitChildIndex(hits);
                }
                else
                {
                    input.QueryContext.MeshBvhChildIndex = -1;
                }

                input.QueryContext.IsInitialized = true;
                input.QueryContext.RigidBodyIndex = rigidBodyIndex;

                return body.CastRay(input, ref collector);
            }

            public unsafe bool ColliderCastLeaf<T>(ColliderCastInput input, int rigidBodyIndex, ref T collector)
                where T : struct, ICollector<ColliderCastHit>
            {
                rigidBodyIndex += BaseRigidBodyIndex;

                RigidBody body = m_Bodies[rigidBodyIndex];

                // Pre-filter: for mesh colliders, test swept collider AABB against cached mesh sub-AABBs
                if (m_MeshSubAabbs.IsCreated &&
                    body.Collider.IsCreated && body.Collider.Value.Type == ColliderType.Mesh)
                {
                    int localIndex = rigidBodyIndex - BaseRigidBodyIndex;
                    Aabb castAabb = input.Collider->CalculateAabb(
                        new RigidTransform { pos = input.Start, rot = input.Orientation },
                        input.QueryContext.InvTargetScale * input.QueryColliderScale);
                    FourTransposedAabbs expanded = m_MeshSubAabbs[localIndex];
                    expanded.Lx -= castAabb.Extents.x;
                    expanded.Ly -= castAabb.Extents.y;
                    expanded.Lz -= castAabb.Extents.z;
                    Ray aabbRay = input.Ray;
                    aabbRay.Origin = castAabb.Min;

                    bool4 hits = expanded.Raycast(aabbRay, collector.MaxFraction, out _);
                    if (!math.any(hits))
                        return false;

                    input.QueryContext.MeshBvhChildIndex = QueryContext.SingleHitChildIndex(hits);
                }
                else
                {
                    input.QueryContext.MeshBvhChildIndex = -1;
                }

                input.QueryContext.IsInitialized = true;
                input.QueryContext.RigidBodyIndex = rigidBodyIndex;

                return body.CastCollider(input, ref collector);
            }

            public bool DistanceLeaf<T>(PointDistanceInput input, int rigidBodyIndex, ref T collector)
                where T : struct, ICollector<DistanceHit>
            {
                rigidBodyIndex += BaseRigidBodyIndex;

                RigidBody body = m_Bodies[rigidBodyIndex];

                // Pre-filter: for mesh colliders, test point distance against cached mesh sub-AABBs
                if (m_MeshSubAabbs.IsCreated &&
                    body.Collider.IsCreated && body.Collider.Value.Type == ColliderType.Mesh)
                {
                    int localIndex = rigidBodyIndex - BaseRigidBodyIndex;
                    var pointT = new Math.FourTransposedPoints(input.Position);
                    float4 distSq = m_MeshSubAabbs[localIndex].DistanceFromPointSquared(ref pointT);
                    float maxDistSq = collector.MaxFraction * collector.MaxFraction;

                    bool4 hits = distSq <= new float4(maxDistSq);
                    if (!math.any(hits))
                        return false;

                    input.QueryContext.MeshBvhChildIndex = QueryContext.SingleHitChildIndex(hits);
                }
                else
                {
                    input.QueryContext.MeshBvhChildIndex = -1;
                }

                input.QueryContext.IsInitialized = true;
                input.QueryContext.RigidBodyIndex = rigidBodyIndex;

                return body.CalculateDistance(input, ref collector);
            }

            public unsafe bool DistanceLeaf<T>(ColliderDistanceInput input, int rigidBodyIndex, ref T collector)
                where T : struct, ICollector<DistanceHit>
            {
                rigidBodyIndex += BaseRigidBodyIndex;

                RigidBody body = m_Bodies[rigidBodyIndex];

                // Pre-filter: for mesh colliders, test collider AABB distance against cached mesh sub-AABBs
                if (m_MeshSubAabbs.IsCreated &&
                    body.Collider.IsCreated && body.Collider.Value.Type == ColliderType.Mesh)
                {
                    int localIndex = rigidBodyIndex - BaseRigidBodyIndex;
                    Aabb queryAabb = input.Collider->CalculateAabb(input.Transform,
                        input.QueryContext.InvTargetScale * input.Scale);
                    FourTransposedAabbs queryAabbT;
                    (&queryAabbT)->SetAllAabbs(queryAabb);
                    float4 distSq = m_MeshSubAabbs[localIndex].DistanceFromAabbSquared(ref queryAabbT);
                    float maxDistSq = collector.MaxFraction * collector.MaxFraction;

                    bool4 hits = distSq <= new float4(maxDistSq);
                    if (!math.any(hits))
                        return false;

                    input.QueryContext.MeshBvhChildIndex = QueryContext.SingleHitChildIndex(hits);
                }
                else
                {
                    input.QueryContext.MeshBvhChildIndex = -1;
                }

                input.QueryContext.IsInitialized = true;
                input.QueryContext.RigidBodyIndex = rigidBodyIndex;

                return body.CalculateDistance(input, ref collector);
            }

            public unsafe void AabbLeaf<T>(OverlapAabbInput input, int rigidBodyIndex, ref T collector)
                where T : struct, IOverlapCollector
            {
                rigidBodyIndex += BaseRigidBodyIndex;
                RigidBody body = m_Bodies[rigidBodyIndex];
                if (body.Collider.IsCreated && CollisionFilter.IsCollisionEnabled(input.Filter, body.Collider.Value.GetCollisionFilter()))
                {
                    collector.AddRigidBodyIndices(&rigidBodyIndex, 1);
                }
            }
        }

        #endregion

        #region Jobs

        // Builds the broadphase in a single job.
        [BurstCompile]
        struct BuildBroadphaseJob : IJob
        {
            [ReadOnly] public NativeArray<RigidBody> StaticBodies;
            [ReadOnly] public NativeArray<RigidBody> DynamicBodies;
            [ReadOnly] public NativeArray<MotionVelocity> MotionVelocities;
            [ReadOnly] public float CollisionTolerance;
            [ReadOnly] public float TimeStep;
            [ReadOnly] public float3 Gravity;
            public NativeReference<int>.ReadOnly BuildStaticTree;

            public Broadphase Broadphase;

            [ReadOnly] public bool Reset;
            [ReadOnly] public bool IncrementalDynamicTree;
            [ReadOnly] public bool IncrementalStaticTree;

            public void Execute()
            {
                Broadphase.Build(StaticBodies, DynamicBodies, MotionVelocities, CollisionTolerance, TimeStep, Gravity, BuildStaticTree.Value == 1, Reset, IncrementalDynamicTree, IncrementalStaticTree);
            }
        }

        [BurstCompile]
        struct BuildIncrementalTreeJob : IJob
        {
            public Tree Tree;
            public bool IsStaticTree;

            public void Execute()
            {
                int numBodies = IsStaticTree ? Tree.NumBodies - 1 : Tree.NumBodies; // manage default static body
                if (Tree.InsertBodyDataStream.IsCreated && (Tree.InsertBodyDataStream.Count() == numBodies))
                {
                    // Trigger a full rebuild when everything has changed. This might be a scene reload
                    Tree.Incremental = false;
                }

                Tree.BuildIncremental();
                Tree.BuiltMultiThreaded = true;
            }
        }

        // Writes pairs of overlapping broadphase AABBs to a stream, in a single job.
        [BurstCompile]
        struct FindOverlapsJob : IJob
        {
            [ReadOnly] public Broadphase Broadphase;
            public NativeStream.Writer DynamicVsDynamicPairsWriter;
            public NativeStream.Writer StaticVsDynamicPairsWriter;

            public void Execute()
            {
                Broadphase.FindOverlaps(ref DynamicVsDynamicPairsWriter, ref StaticVsDynamicPairsWriter);
            }
        }

        [BurstCompile]
        internal struct CalculateTreeSplitDataJob : IJob
        {
            public Tree Tree;
            [ReadOnly] public int ThreadCount;

            public void Execute()
            {
                Tree.BoundingVolumeHierarchy.CalculateTreeSplitData(Tree.Ranges, ThreadCount, out var branchCount);
                Tree.BranchCount[0] = branchCount;
            }
        }

        // Allocate memory for pair indices
        [BurstCompile]
        struct AllocateDynamicVsStaticNodePairs : IJob
        {
            [ReadOnly] public NativeArray<int> dynamicBranchCount;
            [ReadOnly] public NativeArray<int> staticBranchCount;

            [WriteOnly] public NativeList<int2> dynamicVsDynamicNodePairIndices;
            [WriteOnly] public NativeList<int2> staticVsDynamicNodePairIndices;

            public void Execute()
            {
                // number of dynamic/dynamic branch overlap pairs; identical to number of elements in a symmetric table of size (dynamicBranchCount, dynamicBranchCount)
                // without the diagonal, that is, the number of elements in the lower triangular part of the table.
                int numDynamicVsDynamicBranchOverlapPairs = dynamicBranchCount[0] * (dynamicBranchCount[0] + 1) / 2;
                dynamicVsDynamicNodePairIndices.ResizeUninitialized(numDynamicVsDynamicBranchOverlapPairs);

                // number of static/dynamic branch overlap pairs; identical to number of elements in a table of size (staticBranchCount, dynamicBranchCount)
                int numStaticVsDynamicBranchOverlapPairs = staticBranchCount[0] * dynamicBranchCount[0];
                staticVsDynamicNodePairIndices.ResizeUninitialized(numStaticVsDynamicBranchOverlapPairs);
            }
        }

        // Reads broadphase data from dynamic rigid bodies
        [BurstCompile]
        internal unsafe struct PrepareDynamicBodyDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<RigidBody> RigidBodies;
            [ReadOnly] public NativeArray<MotionVelocity> MotionVelocities;
            [ReadOnly] public float TimeStep;
            [ReadOnly] public float3 Gravity;
            [ReadOnly] public float AabbMargin;

            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<PointAndIndex> Points;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<Aabb> Aabbs;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<CollisionFilter> FiltersOut;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<bool> RespondsToCollisionOut;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<SolverType> BodySolverTypesOut;
            [WriteOnly, NativeDisableParallelForRestriction]
            public NativeReference<bool> Incremental;

            /// <summary>The cache of the bodies' bounds, or a view that is not created.</summary>
            public DynamicBoundsCache.View Cache;

            /// <summary>
            /// Null for a build of the tree. For a refit of the tree that the last build left (see
            /// <see cref="ScheduleDynamicTreeRefitBuildJobs"/>), the job sets this word to 1 when any body's filter
            /// differs from the filter that the arrays held before, because the nodes' filters must then be built
            /// again. A refit doesn't read <see cref="Points"/>, so the job writes them only when they are created.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public int* FiltersChanged;

            public unsafe void Execute(int index)
            {
                if (!Cache.IsCreated && FiltersChanged == null)
                {
                    ExecuteImpl(index, AabbMargin, Gravity, TimeStep, RigidBodies, MotionVelocities, Aabbs, Points, FiltersOut, RespondsToCollisionOut, BodySolverTypesOut, Incremental);
                    return;
                }

                if (index == 0)
                    Incremental.Value = false;

                // A body that the cache holds at this pose with this collider takes its collider's bounds and filter
                // from the cache, and the job doesn't read its collider at all. Every other body computes them as
                // ExecuteImpl does and leaves them in the cache.
                RigidBody body = RigidBodies[index];
                Aabb bounds;
                CollisionFilter filter;
                bool responds;
                if (Cache.IsCreated && Cache.Holds(index, body))
                {
                    if (Cache.Check != null)
                    {
                        Cache.CheckEntry(index, body);
                    }

                    bounds = Cache.Bounds[index];
                    filter = Cache.Colliders[index].Filter;
                    responds = Cache.Colliders[index].RespondsToCollision != 0;
                }
                else if (body.Collider.IsCreated)
                {
                    bounds = body.CalculateAabb();
                    filter = body.Collider.Value.GetCollisionFilter();
                    responds = body.Collider.Value.RespondsToCollision;
                    if (Cache.IsCreated)
                    {
                        Cache.Write(index, body, bounds);
                    }
                }
                else
                {
                    bounds = body.CalculateAabb();
                    filter = CollisionFilter.Zero;
                    responds = false;
                }

                // The filters that the last build left belong to the bodies at the same indices, and the nodes'
                // filters were built from them.
                if (FiltersChanged != null)
                {
                    var before = (CollisionFilter*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(FiltersOut);
                    if (!before[index].Equals(filter))
                    {
                        *FiltersChanged = 1;
                    }
                }

                var aabb = ExpandAabb(bounds, AabbMargin, Gravity, TimeStep, MotionVelocities[index]);
                FiltersOut[index] = filter;
                RespondsToCollisionOut[index] = responds;
                BodySolverTypesOut[index] = body.Collider.IsCreated ? body.SolverType : Solver.kDefaultSolverType;
                Aabbs[index] = aabb;
                if (Points.IsCreated)
                {
                    Points[index] = new PointAndIndex
                    {
                        Position = aabb.Center,
                        Index = index
                    };
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal static Aabb CalculateAabb(ref RigidBody body, float aabbMargin, float3 gravity, float timeStep, in MotionVelocity motionVelocity)
            {
                return ExpandAabb(body.CalculateAabb(), aabbMargin, gravity, timeStep, motionVelocity);
            }

            /// <summary>
            /// Expands the bounds of a body's collider by how far the body can move in the time step, and by the margin.
            /// <see cref="CalculateAabb"/> does the same with the bounds that it computes from the collider.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal static Aabb ExpandAabb(Aabb colliderAabb, float aabbMargin, float3 gravity, float timeStep, in MotionVelocity motionVelocity)
            {
                // Apply gravity only on a copy to get proper expansion for the AABB,
                // actual applying of gravity will be done later in the physics step
                var mv = motionVelocity;
                mv.LinearVelocity += gravity * timeStep * mv.GravityFactor;

                MotionExpansion expansion = mv.CalculateExpansion(timeStep);
                var aabb = expansion.ExpandAabb(colliderAabb);
                aabb.Expand(aabbMargin);

                return aabb;
            }

            internal static void ExecuteImpl(int index, float aabbMargin, float3 gravity, float timeStep,
                NativeArray<RigidBody> rigidBodies, NativeArray<MotionVelocity> motionVelocities,
                NativeArray<Aabb> aabbs, NativeArray<PointAndIndex> points,
                NativeArray<CollisionFilter> filtersOut, NativeArray<bool> respondsToCollisionOut,
                NativeArray<SolverType> solverTypesOut, NativeReference<bool> incremental)
            {
                // A from-scratch build resets this flag, so we need to update the NativeReference in this job, so we use index 0 to only write it once.
                // This is viable because nothing in this job reads this NativeReference.
                if (index == 0)
                    incremental.Value = false;

                RigidBody body = rigidBodies[index];

                var aabb = CalculateAabb(ref body, aabbMargin, gravity, timeStep, motionVelocities[index]);

                if (body.Collider.IsCreated)
                {
                    filtersOut[index] = body.Collider.Value.GetCollisionFilter();
                    respondsToCollisionOut[index] = body.Collider.Value.RespondsToCollision;
                    solverTypesOut[index] = body.SolverType;
                }
                else
                {
                    filtersOut[index] = CollisionFilter.Zero;
                    respondsToCollisionOut[index] = false;
                    solverTypesOut[index] = Solver.kDefaultSolverType;
                }

                aabbs[index] = aabb;
                points[index] = new PointAndIndex
                {
                    Position = aabb.Center,
                    Index = index
                };
            }
        }

        // Prepares the NumStaticBodies value for PrepareStaticBodyDataJob
        [BurstCompile]
        struct PrepareNumStaticBodiesJob : IJob
        {
            public int NumStaticBodies;
            public NativeReference<int>.ReadOnly BuildStaticTree;
            public NativeReference<int> NumStaticBodiesArray;
            public NativeReference<bool> BuiltMultiThreaded;

            public void Execute()
            {
                if (BuildStaticTree.Value == 1)
                {
                    NumStaticBodiesArray.Value = NumStaticBodies;
                    BuiltMultiThreaded.Value = true;
                }
                else
                {
                    NumStaticBodiesArray.Value = 0;
                }
            }
        }

        // Reads broadphase data from static rigid bodies
        [BurstCompile]
        internal struct PrepareStaticBodyDataJob : IJobParallelForDefer
        {
            [ReadOnly] public NativeArray<RigidBody> RigidBodies;
            [ReadOnly] public float AabbMargin;

            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<Aabb> Aabbs;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<PointAndIndex> Points;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<CollisionFilter> FiltersOut;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<bool> RespondsToCollisionOut;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<SolverType> BodySolverTypesOut;
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<FourTransposedAabbs> MeshSubAabbsOut;
            [WriteOnly, NativeDisableParallelForRestriction]
            public NativeReference<bool> Incremental;

            public void Execute(int index)
            {
                ExecuteImpl(index, AabbMargin, RigidBodies, Aabbs, Points, FiltersOut, RespondsToCollisionOut,
                    BodySolverTypesOut, MeshSubAabbsOut, Incremental);
            }

            internal static Aabb CalculateAabb(ref RigidBody body, float aabbMargin)
            {
                var aabb = body.CalculateAabb();
                aabb.Expand(aabbMargin);

                return aabb;
            }

            internal static unsafe FourTransposedAabbs ComputeMeshSubAabbs(ref RigidBody body)
            {
                if (!body.Collider.IsCreated || body.Collider.Value.Type != ColliderType.Mesh)
                    return FourTransposedAabbs.Empty;

                var meshCollider = (MeshCollider*)body.Collider.GetUnsafePtr();
                ref var rootNode = ref meshCollider->Mesh.BoundingVolumeHierarchy.GetNode(1);

                var worldSubAabbs = FourTransposedAabbs.Empty;
                for (int i = 0; i < 4; i++)
                {
                    if (rootNode.IsChildValid(i))
                    {
                        Aabb localAabb = rootNode.Bounds.GetAabb(i);
                        Aabb worldAabb = Math.TransformAabb(localAabb, body.WorldFromBody, body.Scale);
                        worldSubAabbs.SetAabb(i, worldAabb);
                    }
                }

                return worldSubAabbs;
            }

            internal static void ExecuteImpl(int index, float aabbMargin,
                NativeArray<RigidBody> rigidBodies, NativeArray<Aabb> aabbs, NativeArray<PointAndIndex> points,
                NativeArray<CollisionFilter> filtersOut, NativeArray<bool> respondsToCollisionOut, NativeArray<SolverType> solverTypesOut,
                NativeArray<FourTransposedAabbs> meshSubAabbsOut, NativeReference<bool> incremental)
            {
                // A from-scratch build resets this flag, so we need to update the NativeReference in this job, so we use index 0 to only write it once.
                // This is viable because nothing in this job reads this NativeReference.
                if (index == 0)
                    incremental.Value = false;

                RigidBody body = rigidBodies[index];

                Aabb aabb = CalculateAabb(ref body, aabbMargin);

                if (body.Collider.IsCreated)
                {
                    filtersOut[index] = body.Collider.Value.GetCollisionFilter();
                    respondsToCollisionOut[index] = body.Collider.Value.RespondsToCollision;
                    solverTypesOut[index] = body.SolverType;
                }
                else
                {
                    filtersOut[index] = CollisionFilter.Zero;
                    respondsToCollisionOut[index] = false;
                    solverTypesOut[index] = Solver.kDefaultSolverType;
                }

                aabbs[index] = aabb;
                points[index] = new PointAndIndex
                {
                    Position = aabb.Center,
                    Index = index
                };
                meshSubAabbsOut[index] = ComputeMeshSubAabbs(ref body);
            }
        }

        // Builds a list of branch node index pairs (an input to FindOverlappingPairsJob)
        [BurstCompile]
        internal struct DynamicVsDynamicBuildBranchNodePairsJob : IJob
        {
            [ReadOnly] public NativeArray<Builder.Range> Ranges;
            [ReadOnly] public NativeArray<int> NumBranches;
            public NativeArray<int2> NodePairIndices;

            public void Execute()
            {
                int numBranches = NumBranches[0];

                int arrayIndex = 0;

                // First add all branch self overlaps.
                // Start with largest branch
                for (int i = 0; i < numBranches; i++)
                {
                    NodePairIndices[arrayIndex++] = new int2(Ranges[i].Root, Ranges[i].Root);
                }

                for (int i = 0; i < numBranches; i++)
                {
                    for (int j = i + 1; j < numBranches; j++)
                    {
                        var pair = new int2 { x = Ranges[i].Root, y = Ranges[j].Root };
                        NodePairIndices[arrayIndex++] = pair;
                    }
                }
            }
        }

        // Builds a list of branch node index pairs (an input to FindOverlappingPairsJob)
        [BurstCompile]
        struct StaticVsDynamicBuildBranchNodePairsJob : IJob
        {
            [ReadOnly] public NativeArray<Builder.Range> StaticRanges;
            [ReadOnly] public NativeArray<Builder.Range> DynamicRanges;
            [ReadOnly] public NativeArray<int> NumStaticBranches;
            [ReadOnly] public NativeArray<int> NumDynamicBranches;
            public NativeArray<int2> NodePairIndices;

            public void Execute()
            {
                int numStaticBranches = NumStaticBranches[0];
                int numDynamicBranches = NumDynamicBranches[0];

                int arrayIndex = 0;
                for (int i = 0; i < numStaticBranches; i++)
                {
                    for (int j = 0; j < numDynamicBranches; j++)
                    {
                        var pair = new int2 { x = StaticRanges[i].Root, y = DynamicRanges[j].Root };
                        NodePairIndices[arrayIndex++] = pair;
                    }
                }
            }
        }

        internal static SolverType GetCombinedSolverType(SolverType solverTypeA, SolverType solverTypeB)
        {
            return (SolverType)math.min((int)solverTypeA, (int)solverTypeB);
        }

        // An implementation of IOverlapCollector which filters and writes body pairs to a native stream
        internal unsafe struct BodyPairWriter : ITreeOverlapCollector
        {
            const int k_Capacity = 256;
            const int k_Margin = 64;
            const int k_Threshold = k_Capacity - k_Margin;

            fixed int m_PairsLeft[k_Capacity];
            fixed int m_PairsRight[k_Capacity];

            readonly NativeStream.Writer* m_CollidingPairs;
            readonly CollisionFilter* m_BodyFiltersLeft;
            readonly CollisionFilter* m_BodyFiltersRight;
            readonly bool* m_BodyRespondsToCollisionLeft;
            readonly bool* m_BodyRespondsToCollisionRight;
            readonly SolverType* m_BodySolverTypesLeft;
            readonly SolverType* m_BodySolverTypesRight;
            readonly int m_BodyIndexABase;
            readonly int m_BodyIndexBBase;
            int m_Count;

            public BodyPairWriter(NativeStream.Writer* collidingPairs, CollisionFilter* bodyFiltersLeft, CollisionFilter* bodyFiltersRight,
                                  bool* bodyRespondsToCollisionLeft, bool* bodyRespondsToCollisionRight,
                                  SolverType* bodySolverTypesLeft, SolverType* bodySolverTypesRight,
                                  int bodyIndexABase, int bodyIndexBBase)
            {
                m_CollidingPairs = collidingPairs;
                m_BodyFiltersLeft = bodyFiltersLeft;
                m_BodyFiltersRight = bodyFiltersRight;
                m_BodyRespondsToCollisionLeft = bodyRespondsToCollisionLeft;
                m_BodyRespondsToCollisionRight = bodyRespondsToCollisionRight;
                m_BodySolverTypesLeft = bodySolverTypesLeft;
                m_BodySolverTypesRight = bodySolverTypesRight;
                m_BodyIndexABase = bodyIndexABase;
                m_BodyIndexBBase = bodyIndexBBase;
                m_Count = 0;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void AddPairs(int4 pairsLeft, int4 pairsRight, int count, bool swapped = false)
            {
                if (swapped)
                {
                    fixed(int* l = m_PairsRight)
                    {
                        *((int4*)(l + m_Count)) = pairsLeft;
                    }

                    fixed(int* r = m_PairsLeft)
                    {
                        *((int4*)(r + m_Count)) = pairsRight;
                    }
                }
                else
                {
                    fixed(int* l = m_PairsLeft)
                    {
                        *((int4*)(l + m_Count)) = pairsLeft;
                    }

                    fixed(int* r = m_PairsRight)
                    {
                        *((int4*)(r + m_Count)) = pairsRight;
                    }
                }

                m_Count += count;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void AddPairs(int pairLeft, int4 pairsRight, int countR)
            {
                fixed(int* l = m_PairsLeft)
                {
                    *((int4*)(l + m_Count)) = new int4(pairLeft);
                }

                fixed(int* r = m_PairsRight)
                {
                    *((int4*)(r + m_Count)) = pairsRight;
                }

                m_Count += countR;
            }

            public void Close()
            {
                Flush();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void FlushIfNeeded()
            {
                if (m_Count >= k_Threshold)
                {
                    Flush();
                }
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            private void Flush()
            {
                if (m_Count != 0)
                {
                    fixed(int* l = m_PairsLeft)
                    {
                        fixed(int* r = m_PairsRight)
                        {
                            for (int i = 0; i < m_Count; i++)
                            {
                                int bodyALocalIndex = l[i];
                                int bodyBLocalIndex = r[i];
                                if (m_BodyRespondsToCollisionLeft[bodyALocalIndex] && m_BodyRespondsToCollisionRight[bodyBLocalIndex])
                                {
                                    if (CollisionFilter.IsCollisionEnabled(m_BodyFiltersLeft[bodyALocalIndex], m_BodyFiltersRight[bodyBLocalIndex]))
                                    {
                                        m_CollidingPairs->Write(new OverlapResult
                                        {
                                            BodyPair = new BodyIndexPair
                                            {
                                                BodyIndexA = bodyALocalIndex + m_BodyIndexABase,
                                                BodyIndexB = bodyBLocalIndex + m_BodyIndexBBase
                                            },
                                            SolverType = GetCombinedSolverType(m_BodySolverTypesLeft[bodyALocalIndex], m_BodySolverTypesRight[bodyBLocalIndex]),
                                        });
                                    }
                                }
                            }
                        }
                    }

                    m_Count = 0;
                }
            }
        }

        // Writes pairs of overlapping broadphase AABBs to a stream.
        [BurstCompile]
        internal struct DynamicVsDynamicFindOverlappingPairsJob : IJobParallelForDefer
        {
            [ReadOnly] public Tree DynamicTree;
            [ReadOnly] public NativeArray<int2> NodePairIndices;
            public NativeStream.Writer PairWriter;

            public void Execute(int index)
            {
                PairWriter.BeginForEachIndex(index);

                int2 pair = NodePairIndices[index];
                ExecuteImpl(pair, DynamicTree, ref PairWriter);

                PairWriter.EndForEachIndex();
            }

            internal static unsafe void ExecuteImpl(int2 pair, Tree dynamicTree, ref NativeStream.Writer pairWriter)
            {
                var bodyFilters = dynamicTree.BodyFilters.GetUnsafeReadOnlyPtr();
                var bodyRespondsToCollision = dynamicTree.RespondsToCollision.GetUnsafeReadOnlyPtr();
                var bodySolverTypes = dynamicTree.BodySolverTypes.GetUnsafeReadOnlyPtr();
                var bufferedPairs = new BodyPairWriter((NativeStream.Writer*)UnsafeUtility.AddressOf(ref pairWriter),
                    bodyFilters, bodyFilters, bodyRespondsToCollision, bodyRespondsToCollision, bodySolverTypes, bodySolverTypes, 0, 0);
                new BoundingVolumeHierarchy(dynamicTree.Nodes.AsArray(), dynamicTree.NodeFilters.AsArray()).SelfBvhOverlap(ref bufferedPairs, pair.x, pair.y);
                bufferedPairs.Close();
            }
        }

        // Writes pairs of overlapping broadphase AABBs to a stream.
        [BurstCompile]
        struct StaticVsDynamicFindOverlappingPairsJob : IJobParallelForDefer
        {
            [ReadOnly] public Tree StaticTree;
            [ReadOnly] public Tree DynamicTree;
            [ReadOnly] public NativeArray<int2> NodePairIndices;
            public NativeStream.Writer PairWriter;

            public void Execute(int index)
            {
                PairWriter.BeginForEachIndex(index);

                int2 pair = NodePairIndices[index];
                ExecuteImpl(pair, StaticTree, DynamicTree, ref PairWriter);

                PairWriter.EndForEachIndex();
            }

            internal static unsafe void ExecuteImpl(int2 pair, Tree staticTree, Tree dynamicTree, ref NativeStream.Writer pairWriter)
            {
                var staticBvh = new BoundingVolumeHierarchy(staticTree.Nodes.AsArray(), staticTree.NodeFilters.AsArray());
                var dynamicBvh = new BoundingVolumeHierarchy(dynamicTree.Nodes.AsArray(), dynamicTree.NodeFilters.AsArray());

                var bodyPairWriter = new BodyPairWriter((NativeStream.Writer*)UnsafeUtility.AddressOf(ref pairWriter),
                    staticTree.BodyFilters.GetUnsafeReadOnlyPtr(), dynamicTree.BodyFilters.GetUnsafeReadOnlyPtr(),
                    staticTree.RespondsToCollision.GetUnsafeReadOnlyPtr(), dynamicTree.RespondsToCollision.GetUnsafeReadOnlyPtr(),
                    staticTree.BodySolverTypes.GetUnsafeReadOnlyPtr(), dynamicTree.BodySolverTypes.GetUnsafeReadOnlyPtr(),
                    dynamicTree.NumBodies, 0);

                staticBvh.BvhOverlap(ref bodyPairWriter, dynamicBvh, pair.x, pair.y);

                bodyPairWriter.Close();
            }
        }

        #endregion
    }

    /// <summary>
    /// The bounds of the dynamic bodies' colliders, kept from the end of one step to the build of the dynamic tree at
    /// the start of the next. Without it, every dynamic body's collider bounds were computed twice for each step: once
    /// when the step refits the tree to the solved poses (<see cref="CollisionWorld.ScheduleRefitDynamicTree"/>), and
    /// again when the next step builds the tree, although most bodies are where the refit put them.
    /// <para>
    /// Each index of the world's dynamic bodies has an entry: the bounds of the collider of the world's body at that
    /// index, at that body's pose, and the collider with its filter. The refit and the build each leave the bounds of
    /// every body with a collider in the cache, so that after either, entry i holds the bounds of the body that the
    /// world holds at index i. The code that builds the next world's bodies sets <see cref="View.Same"/> for each
    /// body that has exactly the collider and pose, to the last bit, that the world's body at its index had before.
    /// Only such a body takes its bounds from the cache. Anything that moves a body between the steps, such as the
    /// game, a shift of the floating origin or a kinematic target, makes the body compute its bounds again.
    /// </para>
    /// <para>
    /// A collider's memory can be freed and given to a new collider, so the address of a collider is not enough to
    /// know it. The owner of the cache counts every change of the bodies' colliders in the number that the cache's
    /// colliderVersion points to, and <see cref="MarkCurrent"/> records that number when the entries hold the current
    /// colliders. While the two numbers differ, no body takes its bounds from the cache.
    /// </para>
    /// </summary>
    internal unsafe struct DynamicBoundsCache : IDisposable
    {
        /// <summary>
        /// The collider of an entry and the body that it belonged to, with the collider's filter, which the build would
        /// read from it.
        /// </summary>
        internal struct ColliderInfo
        {
            public long Collider;
            public Unity.Entities.Entity Body;
            public CollisionFilter Filter;
            public int RespondsToCollision;
        }

        /// <summary>The cache's entries and versions, for a job to read and write.</summary>
        internal struct View
        {
            [NativeDisableUnsafePtrRestriction]
            public Aabb* Bounds;
            [NativeDisableUnsafePtrRestriction]
            public ColliderInfo* Colliders;

            /// <summary>
            /// For each body of the world being built, 1 if the body has exactly the collider and pose that the
            /// world's body at the same index had before the build, and 0 otherwise.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public byte* Same;
            public int Length;
            [NativeDisableUnsafePtrRestriction]
            public uint* HeldVersion;
            [NativeDisableUnsafePtrRestriction]
            public uint* ColliderVersion;

            /// <summary>The counts of the check, while the check is on (<see cref="CheckEnabled"/>), or null.</summary>
            [NativeDisableUnsafePtrRestriction]
            public CheckCounts* Check;

            public bool IsCreated => Bounds != null;

            /// <summary>
            /// Computes, from the collider, the bounds, filter and response of a body that took them from entry i,
            /// and counts the body, and counts it again as a mismatch if any of them differs from the entry's.
            /// </summary>
            public void CheckEntry(int i, in RigidBody body)
            {
                Aabb computed = body.CalculateAabb();
                Aabb cached = Bounds[i];
                bool same = math.all(math.asuint(computed.Min) == math.asuint(cached.Min)) && math.all(math.asuint(computed.Max) == math.asuint(cached.Max))
                    && body.Collider.Value.GetCollisionFilter().Equals(Colliders[i].Filter)
                    && body.Collider.Value.RespondsToCollision == (Colliders[i].RespondsToCollision != 0);
                System.Threading.Interlocked.Increment(ref Check->Hits);
                if (!same)
                {
                    System.Threading.Interlocked.Increment(ref Check->Mismatches);
                    Check->LastIndex = i;
                    Check->LastEntity = body.Entity.Index;
                    Check->LastComputedMin = computed.Min;
                    Check->LastCachedMin = cached.Min;
                }
            }

            /// <summary>
            /// Whether entry i holds the bounds and filter of the body's collider at the body's pose: the body is the
            /// same as before the build, and the entries hold the current colliders. A body with a scale other than
            /// 1 never matches.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool Holds(int i, in RigidBody body)
            {
                long collider = (long)body.Collider.GetUnsafePtr();
                return Same[i] != 0 && Colliders[i].Collider == collider && collider != 0 && body.Scale == 1.0f
                    && *HeldVersion == *ColliderVersion;
            }

            /// <summary>
            /// Writes entry i for a body whose collider is created, with the collider's bounds at the body's pose. The
            /// filter is read from the collider only when the entry held another collider or another body's collider.
            /// </summary>
            /// <remarks>
            /// The heap can give a new body's collider the address of a collider that was freed, so the address alone
            /// doesn't tell that the entry holds this collider. When the entry kept the filter of a destroyed body's
            /// collider at the same address, the new body took that filter from the cache in the next step: a trigger
            /// pair of the new body ended one step after it entered, because the old body didn't detect collisions.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Write(int i, in RigidBody body, Aabb bounds)
            {
                Bounds[i] = bounds;
                long collider = body.Scale == 1.0f ? (long)body.Collider.GetUnsafePtr() : 0;
                if (Colliders[i].Collider != collider || Colliders[i].Body != body.Entity)
                {
                    Colliders[i] = new ColliderInfo
                    {
                        Collider = collider,
                        Body = body.Entity,
                        Filter = body.Collider.Value.GetCollisionFilter(),
                        RespondsToCollision = body.Collider.Value.RespondsToCollision ? 1 : 0,
                    };
                }
            }

            /// <summary>
            /// Returns whether a body that is built at index i is the body that the world held at that index before,
            /// with exactly the same collider and pose. The code that builds the bodies calls it before it overwrites
            /// the body, and stores the answer in <see cref="Same"/>.
            /// </summary>
            /// <remarks>
            /// The entity must match too, because the heap can give a new body's collider the address of a collider
            /// that was freed. That happens when a body is destroyed and a new body takes its index. Without the check,
            /// the new body took the old body's bounds, filter and response from the cache: a trigger pair of a new
            /// body ended one step after it entered, because the cache gave the new body the filter of an old body on a
            /// layer that didn't collide with the trigger. The same body keeps its entity, and its collider can't take
            /// the address of the collider that the world held, because the backend builds the new collider before it
            /// frees the old one.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Matches(RigidBody before, RigidBody after)
            {
                // The collider of the body that the world held before can have been freed since. So read the
                // colliders' addresses as they are stored, at the start of the body. BlobAssetReference.GetUnsafePtr
                // checks the collider's header first, and that check fails and throws for a freed collider.
                return *(void**)&before == *(void**)&after && before.Entity == after.Entity && before.Scale == after.Scale
                    && math.all(math.asuint(before.WorldFromBody.pos) == math.asuint(after.WorldFromBody.pos))
                    && math.all(math.asuint(before.WorldFromBody.rot.value) == math.asuint(after.WorldFromBody.rot.value));
            }
        }

        /// <summary>The counts of the check of the cache. Read them on the main thread between steps.</summary>
        internal struct CheckCounts
        {
            public long Steps;
            public long Hits;
            public long Mismatches;
            public long LoggedMismatches;
            public int LastIndex;
            public int LastEntity;
            public float3 LastComputedMin;
            public float3 LastCachedMin;
        }

        sealed class CheckCountsKey
        {
        }

        sealed class CheckOnKey
        {
        }

        static readonly Unity.Burst.SharedStatic<CheckCounts> SharedCheckCounts = Unity.Burst.SharedStatic<CheckCounts>.GetOrCreate<DynamicBoundsCache, CheckCountsKey>();
        static readonly Unity.Burst.SharedStatic<int> SharedCheckOn = Unity.Burst.SharedStatic<int>.GetOrCreate<DynamicBoundsCache, CheckOnKey>();

        /// <summary>
        /// Whether the build computes the bounds, filter and response of every body that takes them from the cache
        /// too, and counts the bodies whose cached values differ (<see cref="Counts"/>). For tests and for finding a
        /// stale entry; the build still uses the cached values.
        /// </summary>
        public static bool CheckEnabled
        {
            get => SharedCheckOn.Data != 0;
            set => SharedCheckOn.Data = value ? 1 : 0;
        }

        /// <summary>The counts of the check since the program started.</summary>
        internal static CheckCounts* Counts => (CheckCounts*)UnsafeUtility.AddressOf(ref SharedCheckCounts.Data);

        NativeList<Aabb> bounds;
        NativeList<ColliderInfo> colliders;
        NativeList<byte> same;
        NativeReference<uint> heldVersion;
        uint* colliderVersion;

        /// <summary>
        /// Makes an empty cache. The owner adds one to the number that <paramref name="colliderVersion"/> points to
        /// whenever a body's collider changes, and keeps that number alive as long as the cache.
        /// </summary>
        public DynamicBoundsCache(uint* colliderVersion, Allocator allocator)
        {
            bounds = new NativeList<Aabb>(allocator);
            colliders = new NativeList<ColliderInfo>(allocator);
            same = new NativeList<byte>(allocator);
            heldVersion = new NativeReference<uint>(*colliderVersion - 1, allocator);
            this.colliderVersion = colliderVersion;
        }

        public bool IsCreated => bounds.IsCreated;

        /// <summary>
        /// Gives the cache one entry for each of the given number of bodies, and returns the view for the jobs. New
        /// entries hold no collider and no body is the same as before. Call it only on the main thread while no job
        /// uses the cache.
        /// </summary>
        internal View Resize(int count)
        {
            if (bounds.Length != count)
            {
                bounds.Resize(count, NativeArrayOptions.ClearMemory);
                colliders.Resize(count, NativeArrayOptions.ClearMemory);
                same.Resize(count, NativeArrayOptions.ClearMemory);
            }

            return CurrentView;
        }

        /// <summary>The view of the entries as they are, for a job to read and write.</summary>
        internal View CurrentView => new View
        {
            Bounds = bounds.GetUnsafePtr(),
            Colliders = colliders.GetUnsafePtr(),
            Same = same.GetUnsafePtr(),
            Length = bounds.Length,
            HeldVersion = heldVersion.GetUnsafePtr(),
            ColliderVersion = colliderVersion,
            Check = CheckEnabled ? Counts : null,
        };

        /// <summary>
        /// Records that the entries hold the current colliders. Call it on the main thread when no job changes the
        /// colliders, and only after every entry that a body could take holds that body's current collider: after a
        /// build of the tree since the colliders last changed.
        /// </summary>
        public void MarkCurrent()
        {
            heldVersion.Value = *colliderVersion;
        }

        public void Dispose()
        {
            if (bounds.IsCreated)
            {
                bounds.Dispose();
                colliders.Dispose();
                same.Dispose();
                heldVersion.Dispose();
            }
        }
    }

    /// <summary>
    /// What the jobs of <see cref="Broadphase.ScheduleDynamicTreeRefitBuildJobs"/> leave for the code that decides
    /// whether the next build of the dynamic tree may refit the tree again. The owner keeps it at a fixed address, and
    /// reads it on the main thread only after the jobs have completed.
    /// </summary>
    internal unsafe struct DynamicTreeRefitResults
    {
        /// <summary>1 when a body's filter differed from the filter that the tree held for its index, and 0 otherwise.</summary>
        public int FiltersChanged;

        /// <summary>The sum of the surface areas of the bounds of every node below the root.</summary>
        public float NodeArea;

        /// <summary>The sum of the surface areas of the bounds of every body in the tree.</summary>
        public float LeafArea;

        /// <summary>The two sums for each subtree below the cut, which the subtree's iteration writes.</summary>
        public fixed float SlotNodeAreas[BoundingVolumeHierarchy.RefitSubtreeCount];
        public fixed float SlotLeafAreas[BoundingVolumeHierarchy.RefitSubtreeCount];

        /// <summary>
        /// The node area for each unit of leaf area, or 0 when the tree has no area. A tree that groups bodies that
        /// are close together has small nodes, so this number grows as the bodies of a node move apart. The bounds of
        /// the bodies themselves grow or shrink with their speed, and dividing by the leaf area keeps such a change
        /// from looking like a worse tree.
        /// </summary>
        public float Quality => LeafArea > 0.0f ? NodeArea / LeafArea : 0.0f;

        /// <summary>The fitter for the refit's jobs, which builds the nodes' filters again when a body's filter changed.</summary>
        internal BoundingVolumeHierarchy.MeasuringFitter Fitter(Broadphase.Tree tree, Aabb* bounds)
        {
            return new BoundingVolumeHierarchy.MeasuringFitter
            {
                Bounds = bounds,
                LeafFilters = FiltersChanged != 0 ? tree.BodyFilters.GetUnsafeList()->Ptr : null,
                NodeFilters = tree.NodeFilters.GetUnsafeList()->Ptr,
            };
        }
    }
}
