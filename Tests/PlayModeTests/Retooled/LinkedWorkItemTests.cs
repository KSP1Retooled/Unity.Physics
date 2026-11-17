using NUnit.Framework;
using Unity.Collections;
using Unity.Physics;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Verify narrow phase and Jacobian construction partition linked pairs into identical work items.
    /// </summary>
    public class LinkedWorkItemTests
    {
        private const int NumDynamicBodies = 10;

        [Test]
        public void InvalidatingAPairLeavesTheWorkItemsWhereTheyWere()
        {
            // A batch holds 8 pairs, and body 3's link to the ground runs from pair 6 to pair 9, across the boundary.
            var pairs = new NativeArray<DispatchPairSequencer.DispatchPair>(12, Allocator.Temp);
            for (int i = 0; i < 6; i++)
            {
                pairs[i] = Pair(1, 2);
            }

            for (int i = 6; i < 10; i++)
            {
                pairs[i] = Pair(3, NumDynamicBodies + i);
            }

            pairs[10] = Pair(4, 5);
            pairs[11] = Pair(4, 5);

            var info = new DispatchPairSequencer.IterativeSolverSchedulerInfo(1, Allocator.Temp);
            info.PhaseInfo[0] = new DispatchPairSequencer.IterativeSolverSchedulerInfo.SolvePhaseInfo
            {
                DispatchPairCount = pairs.Length,
                BatchSize = 8,
                NumWorkItems = 2,
                LinkedNumDynamicBodies = NumDynamicBodies,
            };
            info.NumActivePhases[0] = 1;
            info.NumWorkItems[0] = 2;
            info.FindLinkedWorkItemRanges(pairs);

            // Simulate the first work item finding no contacts for two linked pairs while the second work item starts.
            // Invalidating those pairs must not change the ranges of either work item.
            pairs[8] = DispatchPairSequencer.DispatchPair.Invalid;
            pairs[9] = DispatchPairSequencer.DispatchPair.Invalid;

            int firstStart = info.GetLinkedWorkItemReadOffset(0, out int firstCount);
            int secondStart = info.GetLinkedWorkItemReadOffset(1, out int secondCount);
            Assert.That(new[] { firstStart, firstCount, secondStart, secondCount }, Is.EqualTo(new[] { 0, 10, 10, 2 }));

            info.Dispose();
            pairs.Dispose();
        }

        private static DispatchPairSequencer.DispatchPair Pair(int bodyA, int bodyB)
        {
            return DispatchPairSequencer.DispatchPair.CreateCollisionPair(new Broadphase.OverlapResult(bodyA, bodyB, SolverType.Iterative));
        }
    }
}
