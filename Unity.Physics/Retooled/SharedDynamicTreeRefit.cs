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
    /// A refit of the dynamic tree to the solved poses, as <see cref="CollisionWorld.ScheduleRefitDynamicTree"/> does it,
    /// that any thread can help to finish. A step can then return while the refit still runs, and a thread that needs
    /// the tree before the refit has finished does the work that is left instead of waiting for jobs that may not run.
    /// <para>
    /// The work comes in three stages: moving the bodies and computing their bounds, in batches of bodies; fitting the
    /// subtrees below the cut, one subtree at a time; and fitting the nodes above the cut. A thread in <see cref="Help"/>
    /// takes the batches and subtrees that nobody has taken yet, one by one, through a counter. It then waits until the
    /// threads that took the stage's last pieces have finished them, which takes at most one piece's time, because
    /// those threads are running. The first thread past the second stage fits the nodes above the cut and sets the
    /// ready word. So no thread in Help ever waits for a thread that is not running.
    /// </para>
    /// <para>
    /// The owner keeps the struct at a fixed address. <see cref="Schedule"/> fills it on the main thread while no thread
    /// helps, and the owner must not call it again before the refit that it last scheduled has finished.
    /// </para>
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
        /// Clears the ready word and schedules jobs that refit the dynamic tree of <paramref name="world"/> after
        /// <paramref name="inputDeps"/>: <paramref name="helpers"/> jobs, each of which calls <see cref="Help"/>. A world
        /// without dynamic bodies needs no refit, so the word is set at once and the returned handle is
        /// <paramref name="inputDeps"/>. The tree must have been built for the same dynamic bodies in the same order,
        /// as for <see cref="CollisionWorld.ScheduleRefitDynamicTree"/>. With <paramref name="boundsCache"/>, the
        /// refit also leaves the bodies' bounds in the cache.
        /// <para>
        /// Only <paramref name="inputDeps"/> may write the world's motions while the jobs wait, because the pointers
        /// are taken now. A thread may call Help only after inputDeps has completed.
        /// </para>
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
        /// Does the refit's work that nobody has taken yet, and returns when the refit has finished, or when the last
        /// piece of work is with another thread that is running it. In the second case the caller waits for the
        /// ready word. Returns at once when the refit has already finished.
        /// </summary>
        internal void Help()
        {
            // A refit that was never scheduled, or that had no bodies, has no work.
            if (m_Ready == null || m_Count == 0)
            {
                return;
            }

            // An exception that leaves a piece of work unfinished would leave the other threads waiting for that
            // piece forever. So the thread marks the refit failed, which sets the ready word, and the exception goes
            // on to this thread's caller. The other threads stop waiting, and the owner sees the failure in Failed.
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
        /// Whether a thread failed to finish its part of the last refit, because of an exception. The tree can then
        /// hold bounds of the poses before the step, and the owner must build it again before a query reads it. Read
        /// it only after the ready word is set.
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
        /// For tests: whether the scheduled jobs leave all of the refit's work to the threads that call
        /// <see cref="Help"/> themselves, so that the refit stays pending until a query or the owner finishes it.
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
        /// For tests: whether the thread that takes the first batch of bodies throws, as a bug in the refit would. Only
        /// code that runs in Mono throws; Burst leaves the throw out, and <see cref="ThrowsForTests"/> counts the throws.
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
