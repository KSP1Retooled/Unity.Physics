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
    /// The trees can be reused while the build and their body rows permit it.
    /// </summary>
    internal unsafe struct TreeState
    {
        // Static tree jobs read these flags. keepStatics is constant after initialization;
        // StaticTreeShift.ShiftStaticTreeJob sets rebuildStatics before tree jobs run.
        internal NativeReference<int> rebuildStatics;
        internal NativeReference<int> keepStatics;
        internal StaticTreeShift.ShiftState treeShift;
        internal bool treeMatchesHeld;
        // The number of collider refreshes that gave any body a new collider, and the bounds of the dynamic bodies'
        // colliders that the refit at the end of a step leaves for the build at the start of the next.
        internal NativeReference<uint> colliderChanges;
        internal DynamicBoundsCache boundsCache;

        // Whether the last build of the dynamic tree used the bounds cache. Only then does the cache hold the current
        // colliders after the build.
        internal bool boundsCacheBuilt;

        // Pending refit to solved poses. FinishPendingRefit protects the built world and bounds cache;
        // queries may help the refit through the native store pointer.
        internal SharedDynamicTreeRefit* sharedRefit;
        internal JobHandle refit;
        internal bool refitPending;

        // The jobs that write the motions that the pending refit reads.
        internal JobHandle refitInputs;

        // Dynamic store rows used for the current tree. Refit while row order, count and quality remain
        // suitable.
        internal NativeList<int> treeRows;

        // What the jobs of the last refit left: whether a filter changed, and the tree's quality.
        internal DynamicTreeRefitResults* treeRefitResults;

        // Whether the last build refitted the tree, the refits since the tree was last built, and the quality that the
        // first of those refits measured, or 0 before it has been read.
        internal bool treeRefitted;
        internal int treeRefits;
        internal float treeBaseQuality;

        // Counts for tests.
        internal long treeBuildCount;
        internal long treeRefitCount;
        internal long treeFilterRefitCount;
        internal static readonly ProfilerMarker ShiftStaticTreeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.ShiftStaticTree");
        internal static readonly ProfilerMarker StepWaitForTreeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForTree");

        /// <summary>
        /// Maximum consecutive refits before rebuilding the dynamic tree.
        /// </summary>
        internal const int MaxTreeRefits = 500;

        /// <summary>
        /// Maximum tree quality growth relative to the first refit after rebuilding, before another rebuild is
        /// required.
        /// </summary>
        internal const float MaxTreeQualityGrowth = 1.5f;

        /// <summary>
        /// Enable dynamic tree reuse. RETOOLED_TREE_REUSE=0 disables it.
        /// Stored in SharedStatic for Burst; the always-refit mode also bypasses count and quality limits.
        /// </summary>
        internal static bool TreeReuseEnabled
        {
            get => TreeReuseMode.Data != 1;
            set => TreeReuseMode.Data = value ? 0 : 1;
        }

        /// <summary>
        /// Whether every build refits the dynamic tree when the dynamic bodies are the same, however bad the tree has
        /// become. RETOOLED_TREE_REUSE=always turns it on, to check that a tree of any shape gives the same results.
        /// </summary>
        internal static bool TreeAlwaysRefitted
        {
            get => TreeReuseMode.Data == 2;
            set => TreeReuseMode.Data = value ? 2 : 0;
        }

        private sealed class TreeReuseModeKey
        {
        }

        private static readonly Unity.Burst.SharedStatic<int> TreeReuseMode = Unity.Burst.SharedStatic<int>.GetOrCreate<TreeState, TreeReuseModeKey>();

        /// <summary>
        /// Reads RETOOLED_TREE_REUSE from the environment: 0 turns off the reuse of the dynamic tree, and "always" makes
        /// every build refit the tree when it can (<see cref="TreeAlwaysRefitted"/>).
        /// </summary>
        internal static void ReadTreeReuseSwitch()
        {
            string reuse = Environment.GetEnvironmentVariable("RETOOLED_TREE_REUSE");
            TreeReuseMode.Data = reuse == "0" ? 1 : reuse == "always" ? 2 : 0;
            if (reuse == "0" || reuse == "always")
            {
                UnityEngine.Debug.Log($"[StoreStepper] RETOOLED_TREE_REUSE={reuse}: {(reuse == "0" ? "every build builds the dynamic tree from scratch" : "every build refits the dynamic tree when the dynamic bodies are the same")}");
            }
        }

        /// <summary>The builds of the dynamic tree from scratch since this state was made. Used by tests.</summary>
        internal readonly long TreeBuildCount => treeBuildCount;

        /// <summary>The builds that refitted the dynamic tree instead of building it. Used by tests.</summary>
        internal readonly long TreeRefitCount => treeRefitCount;

        /// <summary>The refits that built the nodes' filters again because a body's filter had changed. Used by tests.</summary>
        internal readonly long TreeFilterRefitCount => treeFilterRefitCount;

        /// <summary>
        /// Reuse bounds cached by the previous refit. Tests can disable this for comparison.
        /// </summary>
        internal static bool BoundsCacheEnabled
        {
            get => BoundsCacheOff.Data == 0;
            set => BoundsCacheOff.Data = value ? 0 : 1;
        }

        private sealed class BoundsCacheOffKey
        {
        }

        private static readonly Unity.Burst.SharedStatic<int> BoundsCacheOff = Unity.Burst.SharedStatic<int>.GetOrCreate<TreeState, BoundsCacheOffKey>();

        /// <summary>
        /// Read bounds cache switches: RETOOLED_BOUNDS_CACHE=0 disables caching;
        /// RETOOLED_BOUNDS_CHECK=1 validates cached bounds; RETOOLED_SHAPE_BOUNDS=0 disables cached collider
        /// geometry.
        /// </summary>
        internal static void ReadBoundsCacheSwitches()
        {
            string cache = Environment.GetEnvironmentVariable("RETOOLED_BOUNDS_CACHE");
            BoundsCacheEnabled = cache != "0";
            string check = Environment.GetEnvironmentVariable("RETOOLED_BOUNDS_CHECK");
            DynamicBoundsCache.CheckEnabled = check == "1";
            string shapes = Environment.GetEnvironmentVariable("RETOOLED_SHAPE_BOUNDS");
            DynamicBoundsCache.ShapesEnabled = shapes != "0";
            if (cache == "0" || check == "1" || shapes == "0")
            {
                UnityEngine.Debug.Log($"[StoreStepper] RETOOLED_BOUNDS_CACHE={cache}, RETOOLED_BOUNDS_CHECK={check}, RETOOLED_SHAPE_BOUNDS={shapes}: bounds cache {(BoundsCacheEnabled ? "on" : "off")}, check {(DynamicBoundsCache.CheckEnabled ? "on" : "off")}, shape bounds {(DynamicBoundsCache.ShapesEnabled ? "on" : "off")}");
            }
        }

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
        /// Call outside jobs; query jobs use HelpPendingRefit instead.
        /// </summary>
        internal void FinishPendingRefit(WorldData* store)
        {
            if (!refitPending)
            {
                return;
            }

            using (StepWaitForTreeMarker.Auto())
            {
                // A job whose refit threw is complete too: the thread that threw marked the refit failed, and the
                // other threads stopped waiting for its piece.
                refit.Complete();

                // The jobs have done all of the work unless a test told them to leave it.
                WaitForRefit(store);
            }

            refit = default;
            refitPending = false;
            NoteFailedRefit(store);
        }

        /// <summary>
        /// Invalidate the published world after a refit fails. The next main-thread query rebuilds it;
        /// workers may continue querying pre-step poses until then.
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
        /// Logs the check's mismatches of the bounds cache when there are new ones, and the counts every 1,000 steps.
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
                else if (quality > treeBaseQuality * MaxTreeQualityGrowth && !TreeAlwaysRefitted)
                {
                    return false;
                }
            }

            return TreeReuseEnabled && treeMatchesHeld && partition.dynamicIndices.Length > 0
                && (treeRefits < MaxTreeRefits || TreeAlwaysRefitted) && BodyPartition.SameRows(treeRows, partition.dynamicIndices);
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
            boundsCacheBuilt = BoundsCacheEnabled;
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
            return BoundsCacheEnabled ? boundsCache.Resize(partition.dynamicIndices.Length) : default;
        }

        internal readonly bool CanCompareBounds(ref PhysicsWorld world, ref BodyPartition partition)
        {
            return boundsCacheBuilt && BoundsCacheEnabled && world.NumDynamicBodies == partition.dynamicIndices.Length;
        }

        internal readonly NativeReference<int>.ReadOnly BuildStaticTree(bool staticsChanged) => (staticsChanged ? rebuildStatics : keepStatics).AsReadOnly();
    }
}
