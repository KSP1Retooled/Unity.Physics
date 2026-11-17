using System.Threading;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;

namespace Unity.Physics
{
    /// <summary>
    /// Shared dynamic tree refit that queries can help while scheduled workers run.
    /// Claim body batches, subtrees and the final upper-tree stage atomically; wait only for work already running.
    /// Keep at a fixed address. Schedule on the main thread with no helpers active, after the previous refit
    /// finishes.
    /// </summary>
    internal unsafe struct SharedDynamicTreeRefit
    {
        const int BodiesPerBatch = 32;

        MotionData* m_MotionDatas;
        MotionVelocity* m_MotionVelocities;
        RigidBody* m_Bodies;
        int m_Count;
        float m_AabbMargin;
        float m_TimeStep;
        float3 m_Gravity;
        DynamicBoundsCache.View m_Cache;
        BoundingVolumeHierarchy m_Tree;

        // The bounds of the bodies for the tree's leaves. The struct owns them.
        Aabb* m_Aabbs;
        int m_AabbCapacity;

        // 0 while the refit is pending, and 1 once it has finished. The owner gives its address.
        int* m_Ready;

        // The next batch of bodies and the next subtree that nobody has taken, the bodies and subtrees that are done,
        // and whether a thread has taken the nodes above the cut.
        int m_NextBatch;
        int m_DoneBodies;
        int m_NextSubtree;
        int m_DoneSubtrees;
        int m_TopTaken;

        // 1 when a thread failed to finish its piece of work, because of an exception (see Help).
        int m_Failed;

        /// <summary>
        /// Clear the ready word and schedule helpers after inputDeps to refit the same dynamic bodies in the same
        /// order.
        /// Optionally cache collider bounds. With no dynamic bodies, mark ready and return inputDeps.
        /// Only inputDeps may write motions while waiting, and helpers must not run before it completes.
        /// </summary>
        internal static JobHandle Schedule(SharedDynamicTreeRefit* refit, ref PhysicsWorld world, float timeStep, float3 gravity,
            JobHandle inputDeps, DynamicBoundsCache boundsCache, int* ready, int helpers)
        {
            int count = world.NumDynamicBodies;
            refit->m_Ready = ready;
            if (count == 0)
            {
                refit->m_Count = 0;
                Interlocked.Exchange(ref *ready, 1);
                return inputDeps;
            }

            if (refit->m_AabbCapacity < count)
            {
                if (refit->m_Aabbs != null)
                {
                    UnsafeUtility.Free(refit->m_Aabbs, Allocator.Persistent);
                }

                refit->m_AabbCapacity = math.max(count, refit->m_AabbCapacity * 2);
                refit->m_Aabbs = (Aabb*)UnsafeUtility.Malloc((long)refit->m_AabbCapacity * sizeof(Aabb), UnsafeUtility.AlignOf<Aabb>(), Allocator.Persistent);
            }

            // The solve's jobs still write the motions, so the pointers are taken without the safety checks.
            refit->m_MotionDatas = (MotionData*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionDatas);
            refit->m_MotionVelocities = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities);
            refit->m_Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.DynamicBodies);
            refit->m_Count = count;
            refit->m_AabbMargin = world.CollisionWorld.CollisionTolerance * 0.5f; // each body contributes half, as in the build
            refit->m_TimeStep = timeStep;
            refit->m_Gravity = gravity;
            refit->m_Cache = boundsCache.IsCreated && boundsCache.CurrentView.Length == count ? boundsCache.CurrentView : default;
            refit->m_Tree = world.CollisionWorld.Broadphase.DynamicTree.BoundingVolumeHierarchy;
            refit->m_DoneBodies = 0;
            refit->m_DoneSubtrees = 0;
            refit->m_TopTaken = 0;
            refit->m_NextSubtree = 0;
            refit->m_Failed = 0;
            Interlocked.Exchange(ref *ready, 0);

            // Last, so that a thread that takes a batch sees the fields above.
            Interlocked.Exchange(ref refit->m_NextBatch, 0);
            return new HelpJob { Refit = refit }.Schedule(math.max(helpers, 1), 1, inputDeps);
        }

        /// <summary>Frees the bounds. Call it only when no refit is pending.</summary>
        internal void Dispose()
        {
            if (m_Aabbs != null)
            {
                UnsafeUtility.Free(m_Aabbs, Allocator.Persistent);
                m_Aabbs = null;
                m_AabbCapacity = 0;
            }
        }

        /// <summary>
        /// Claim remaining refit work. Return when finished or when the last piece is running elsewhere;
        /// in the latter case, the caller waits for the ready word.
        /// </summary>
        internal void Help()
        {
            // A refit that was never scheduled, or that had no bodies, has no work.
            if (m_Ready == null || m_Count == 0)
            {
                return;
            }

            // On exception, mark the refit failed and set ready so that other helpers and queries stop waiting.
            bool returned = false;
            try
            {
                DoWork();
                returned = true;
            }
            finally
            {
                if (!returned)
                {
                    Fail();
                }
            }
        }

        /// <summary>
        /// Whether a helper failed. Read only after ready is set, then rebuild before using potentially stale
        /// bounds.
        /// </summary>
        internal bool Failed => Interlocked.CompareExchange(ref m_Failed, 0, -1) != 0;

        void Fail()
        {
            Interlocked.Exchange(ref m_Failed, 1);
            Interlocked.Exchange(ref *m_Ready, 1);
            AddressWait.WakeAll(m_Ready);
        }

        void DoWork()
        {
            int batches = (m_Count + BodiesPerBatch - 1) / BodiesPerBatch;
            BodiesMarker.Begin();
            for (int batch = Interlocked.Increment(ref m_NextBatch) - 1; batch < batches; batch = Interlocked.Increment(ref m_NextBatch) - 1)
            {
                int start = batch * BodiesPerBatch;
                int end = math.min(start + BodiesPerBatch, m_Count);
                if (batch == 0 && s_ThrowForTests.Data != 0)
                {
                    ThrowForTests();
                }

                for (int i = start; i < end; i++)
                {
                    CollisionWorld.UpdateDynamicBoundsJob.UpdateBody(i, m_MotionDatas, m_MotionVelocities, m_Bodies, m_Aabbs, m_AabbMargin,
                        m_Gravity, m_TimeStep, m_Cache);
                }

                Interlocked.Add(ref m_DoneBodies, end - start);
            }

            BodiesMarker.End();
            WaitMarker.Begin();
            while (Interlocked.CompareExchange(ref m_DoneBodies, 0, -1) < m_Count && Interlocked.CompareExchange(ref m_Failed, 0, -1) == 0)
            {
                Common.Pause();
            }

            WaitMarker.End();
            if (Interlocked.CompareExchange(ref m_Failed, 0, -1) != 0)
            {
                return;
            }

            const int subtrees = BoundingVolumeHierarchy.RefitSubtreeCount;
            NodesMarker.Begin();
            for (int slot = Interlocked.Increment(ref m_NextSubtree) - 1; slot < subtrees; slot = Interlocked.Increment(ref m_NextSubtree) - 1)
            {
                int root = m_Tree.FindRefitSubtree(slot);
                if (root != 0)
                {
                    m_Tree.RefitSubtree(root, m_Aabbs);
                }

                Interlocked.Increment(ref m_DoneSubtrees);
            }

            NodesMarker.End();
            WaitMarker.Begin();
            while (Interlocked.CompareExchange(ref m_DoneSubtrees, 0, -1) < subtrees && Interlocked.CompareExchange(ref m_Failed, 0, -1) == 0)
            {
                Common.Pause();
            }

            WaitMarker.End();
            if (Interlocked.CompareExchange(ref m_Failed, 0, -1) == 0 && Interlocked.CompareExchange(ref m_TopTaken, 1, 0) == 0)
            {
                TopMarker.Begin();
                m_Tree.RefitAboveCut();
                Interlocked.Exchange(ref *m_Ready, 1);
                AddressWait.WakeAll(m_Ready);
                TopMarker.End();
            }
        }

        static readonly ProfilerMarker BodiesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Refit.Bodies");
        static readonly ProfilerMarker NodesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Refit.Subtrees");
        static readonly ProfilerMarker TopMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Refit.Top");
        static readonly ProfilerMarker WaitMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Refit.Wait");

        /// <summary>
        /// Test option leaving all refit work to queries or owner helpers instead of scheduled workers.
        /// </summary>
        internal static bool JobsLeaveTheWork
        {
            get => s_JobsLeaveTheWork.Data != 0;
            set => s_JobsLeaveTheWork.Data = value ? 1 : 0;
        }

        sealed class JobsLeaveTheWorkKey
        {
        }

        static readonly SharedStatic<int> s_JobsLeaveTheWork = SharedStatic<int>.GetOrCreate<SharedDynamicTreeRefit, JobsLeaveTheWorkKey>();

        /// <summary>
        /// Inject an exception in the first body batch for Mono tests. Burst omits it; ThrowsForTests counts
        /// executions.
        /// </summary>
        internal static bool ThrowForTestsEnabled
        {
            get => s_ThrowForTests.Data != 0;
            set => s_ThrowForTests.Data = value ? 1 : 0;
        }

        /// <summary>For tests: the number of injected throws since the program started.</summary>
        internal static int ThrowsForTests => s_ThrowsForTests;

        static int s_ThrowsForTests;

        [BurstDiscard]
        static void ThrowForTests()
        {
            Interlocked.Increment(ref s_ThrowsForTests);
            throw new System.InvalidOperationException("a throw that a test injected into the refit of the dynamic tree");
        }

        sealed class ThrowForTestsKey
        {
        }

        static readonly SharedStatic<int> s_ThrowForTests = SharedStatic<int>.GetOrCreate<SharedDynamicTreeRefit, ThrowForTestsKey>();

        [BurstCompile]
        struct HelpJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public SharedDynamicTreeRefit* Refit;

            public void Execute(int index)
            {
                if (s_JobsLeaveTheWork.Data == 0)
                {
                    Refit->Help();
                }
            }
        }
    }
}
