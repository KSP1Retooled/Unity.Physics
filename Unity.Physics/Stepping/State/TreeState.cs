using Unity.Physics.Configuration;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Base.Threading;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.World.Trees;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using Unity.Profiling;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the broadphase reuse data, the bounds cache, and the pending refit.
    /// Reuses the dynamic tree after a completed build while the row order, refit count, and tree quality allow it.
    /// </summary>
    internal unsafe struct TreeState
    {
        // The static tree jobs read these flags. keepStatics stays constant after initialization.
        // StaticTreeShift.ShiftStaticTreeJob sets rebuildStatics before the tree jobs run.
        private NativeReference<int> rebuildStatics;
        private NativeReference<int> keepStatics;
        private StaticTreeShift.ShiftState treeShift;
        internal bool treeMatchesHeld;
        // The number of collider refreshes that gave any body a new collider, and the bounds of the dynamic bodies'
        // colliders that the refit at the end of a step leaves for the build at the start of the next.
        internal NativeReference<uint> colliderChanges;
        private DynamicBoundsCache boundsCache;

        // Whether the last build of the dynamic tree used the bounds cache. Only then does the cache hold the current
        // colliders after the build.
        private bool boundsCacheBuilt;

        // FinishPendingRefit waits before callers read the built world or the bounds cache.
        // Queries may help the pending refit through the native store pointer.
        internal SharedDynamicTreeRefit* sharedRefit;
        private JobHandle refit;
        private bool refitPending;

        // The jobs that write the motions that the pending refit reads.
        private JobHandle refitInputs;

        // Dynamic store rows used for the current tree. Refit while row order, count and quality remain
        // suitable.
        private NativeList<int> treeRows;

        // The last refit records whether a filter changed and how much overlap the tree contains.
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
        private static readonly ProfilerMarker ShiftStaticTreeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.ShiftStaticTree");
        private static readonly ProfilerMarker StepWaitForTreeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForTree");

        /// <summary>
        /// Maximum consecutive refits before rebuilding the dynamic tree.
        /// </summary>
        internal const int MaxTreeRefits = 500;

        /// <summary>
        /// Maximum tree quality growth relative to the first refit after rebuilding, before another rebuild is
        /// required.
        /// </summary>
        internal const float MaxTreeQualityGrowth = 1.5f;

        /// <summary>The builds of the dynamic tree from scratch since this state was made. Used by tests.</summary>
        internal readonly long TreeBuildCount => treeBuildCount;

        /// <summary>The builds that refitted the dynamic tree instead of building it. Used by tests.</summary>
        internal readonly long TreeRefitCount => treeRefitCount;

        /// <summary>The refits that built the nodes' filters again because a body's filter had changed. Used by tests.</summary>
        internal readonly long TreeFilterRefitCount => treeFilterRefitCount;

        /// <summary>Whether a refit that the last step left running may still be running. Used by tests.</summary>
        internal readonly bool RefitPending => refitPending;

        /// <summary>
        /// Whether solved motions are ready for pending refit helpers. Used by tests.
        /// </summary>
        internal readonly bool RefitInputsComplete => !refitPending || refitInputs.IsCompleted;

        internal void InitializeStaticTree()
        {
            rebuildStatics = new NativeReference<int>(1, Allocator.Persistent);
            keepStatics = new NativeReference<int>(0, Allocator.Persistent);
            treeShift = StaticTreeShift.ShiftState.Create();
        }

        internal void Initialize(WorldData* store)
        {
            colliderChanges = new NativeReference<uint>(0, Allocator.Persistent);
            boundsCache = new DynamicBoundsCache(colliderChanges.GetUnsafePtr(), Allocator.Persistent);
            sharedRefit = (SharedDynamicTreeRefit*)UnsafeUtility.MallocTracked(sizeof(SharedDynamicTreeRefit), UnsafeUtility.AlignOf<SharedDynamicTreeRefit>(), Allocator.Persistent, 0);
            UnsafeUtility.MemClear(sharedRefit, sizeof(SharedDynamicTreeRefit));
            store->refit = sharedRefit;
            store->treeReady = 1;
            treeRows = new NativeList<int>(Allocator.Persistent);
            treeRefitResults = (DynamicTreeRefitResults*)UnsafeUtility.MallocTracked(sizeof(DynamicTreeRefitResults), UnsafeUtility.AlignOf<DynamicTreeRefitResults>(), Allocator.Persistent, 0);
            UnsafeUtility.MemClear(treeRefitResults, sizeof(DynamicTreeRefitResults));
        }

        internal void DisposeRefit()
        {
            sharedRefit->Dispose();
            UnsafeUtility.FreeTracked(sharedRefit, Allocator.Persistent);
            sharedRefit = null;
        }

        internal void DisposeStaticTree()
        {
            rebuildStatics.Dispose();
            keepStatics.Dispose();
            treeShift.Dispose();
        }

        internal void Dispose()
        {
            boundsCache.Dispose();
            colliderChanges.Dispose();
            treeRows.Dispose();
            UnsafeUtility.FreeTracked(treeRefitResults, Allocator.Persistent);
            treeRefitResults = null;
        }

        /// <summary>
        /// Complete the pending dynamic tree refit on the main thread before reading or modifying the built world
        /// or bounds cache.
        /// Call this outside jobs. Query jobs use HelpPendingRefit.
        /// </summary>
        internal void FinishPendingRefit(WorldData* store)
        {
            if (!refitPending)
            {
                return;
            }

            using (StepWaitForTreeMarker.Auto())
            {
                // If a thread threw, it marked the refit as failed. The other threads stopped waiting for its work.
                refit.Complete();

                // The jobs have done all of the work unless a test told them to leave it.
                WaitForRefit(store);
            }

            refit = default;
            refitPending = false;
            NoteFailedRefit(store);
        }

        /// <summary>
        /// Invalidate the published world after a refit fails. The next main-thread query rebuilds it.
        /// Workers may continue querying the poses from before the step until then.
        /// </summary>
        private void NoteFailedRefit(WorldData* store)
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
                UnityEngine.Debug.LogError("[StoreStepper] the refit of the dynamic tree after the step failed; the next query on the main thread builds the world again");
            }
        }

        /// <summary>
        /// Finish pending refit work without completing a job handle, for callers already inside a job.
        /// FinishPendingRefit later completes the now-empty scheduled jobs.
        /// </summary>
        internal void HelpPendingRefit(WorldData* store)
        {
            if (refitPending)
            {
                WaitForRefit(store);
            }
        }

        /// <summary>
        /// Does the refit's remaining work on this thread, and then waits for the threads that run its last pieces, if
        /// any.
        /// </summary>
        private void WaitForRefit(WorldData* store)
        {
            sharedRefit->Help();
            int* ready = &store->treeReady;
            while (System.Threading.Interlocked.CompareExchange(ref *ready, 0, -1) == 0)
            {
                AddressWait.Wait(ready, 0);
                Unity.Burst.Intrinsics.Common.Pause();
            }
        }

        /// <summary>
        /// Logs new bounds cache mismatches and reports the counts every 1,000 steps.
        /// </summary>
        internal static void LogBoundsCheck()
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
        /// Reuse the dynamic tree while the build completed, dynamic body rows and order match, and count and
        /// quality limits permit.
        /// Refitting preserves collision/query results, though traversal and multi-hit ordering can change.
        /// </summary>
        private bool RefitsDynamicTree(ref BodyPartition partition)
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
                else if (quality > treeBaseQuality * MaxTreeQualityGrowth && !BroadphaseSwitches.TreeAlwaysRefitted)
                {
                    return false;
                }
            }

            return BroadphaseSwitches.TreeReuseEnabled && treeMatchesHeld && partition.dynamicIndices.Length > 0
                && (treeRefits < MaxTreeRefits || BroadphaseSwitches.TreeAlwaysRefitted) && BodyPartition.SameRows(treeRows, partition.dynamicIndices);
        }

        /// <summary>
        /// After the static bodies are built, decides whether to rebuild or shift the static tree. If no static body
        /// changed, leaves the tree unchanged.
        /// </summary>
        internal JobHandle ScheduleShiftStaticTree(WorldData* store, ref PhysicsWorld world, bool staticsChanged, NativeList<ColliderBuilder.Work> colliderWork, JobHandle built)
        {
            if (!staticsChanged)
            {
                return built;
            }

            using var span = ShiftStaticTreeMarker.Auto();
            Broadphase.Tree tree = world.CollisionWorld.Broadphase.StaticTree;
            return new StaticTreeShift.ShiftStaticTreeJob
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
        /// Schedules the broadphase trees. Only building the static tree depends on the decision to rebuild it.
        /// Building the dynamic tree can run in parallel with that decision.
        /// </summary>
        internal JobHandle ScheduleTrees(WorldData* store, ref PhysicsWorld world, ref StaticRows staticRows,
            ref BodyPartition partition, NativeList<ColliderBuilder.Work> colliderWork, float deltaTime, JobHandle built)
        {
            int threads = JobsUtility.JobWorkerCount + 1;
            Broadphase broadphase = world.CollisionWorld.Broadphase;
            JobHandle statics = broadphase.ScheduleStaticTreeBuildJobs(ref world, threads, BuildStaticTree(staticRows.staticsChanged), ScheduleShiftStaticTree(store, ref world, staticRows.staticsChanged, colliderWork, built));
            boundsCacheBuilt = BroadphaseSwitches.BoundsCacheEnabled;
            DynamicBoundsCache cache = boundsCacheBuilt ? boundsCache : default;
            JobHandle dynamics;
            if (RefitsDynamicTree(ref partition))
            {
                dynamics = broadphase.ScheduleDynamicTreeRefitBuildJobs(ref world, deltaTime, float3.zero, built, cache, treeRefitResults);
                treeRefitted = true;
                treeRefits++;
                treeRefitCount++;
            }
            else
            {
                dynamics = broadphase.ScheduleDynamicTreeBuildJobs(ref world, deltaTime, float3.zero, threads, built, cache);
                treeRows.CopyFrom(partition.dynamicIndices.AsArray());
                treeRefitted = false;
                treeRefits = 0;
                treeBaseQuality = 0.0f;
                treeBuildCount++;
            }

            return JobHandle.CombineDependencies(statics, dynamics);
        }

        internal DynamicBoundsCache.View PrepareBounds(ref BodyPartition partition)
        {
            return BroadphaseSwitches.BoundsCacheEnabled ? boundsCache.Resize(partition.dynamicIndices.Length) : default;
        }

        internal readonly bool CanCompareBounds(ref PhysicsWorld world, ref BodyPartition partition)
        {
            return boundsCacheBuilt && BroadphaseSwitches.BoundsCacheEnabled && world.NumDynamicBodies == partition.dynamicIndices.Length;
        }

        internal readonly NativeReference<int>.ReadOnly BuildStaticTree(bool staticsChanged) => (staticsChanged ? rebuildStatics : keepStatics).AsReadOnly();
        internal void SchedulePendingRefit(WorldData* store, ref PhysicsWorld world, float deltaTime, JobHandle motionsSolved)
        {
            refit = SharedDynamicTreeRefit.Schedule(sharedRefit, ref world, deltaTime, float3.zero, motionsSolved,
                boundsCacheBuilt ? boundsCache : default, &store->treeReady, JobsUtility.JobWorkerCount + 1);
            refitPending = true;
            refitInputs = motionsSolved;
        }

        internal void MarkBoundsCurrent()
        {
            if (boundsCacheBuilt)
            {
                boundsCache.MarkCurrent();
            }
        }

    }
}