using NUnit.Framework;
using Unity.Collections;
using Unity.Physics;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The work items of a phase of links. The narrow phase and the Jacobian build must split the pairs the same way, or the
    /// build reads another work item's contacts and the stream throws when it ends with some unread.
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

            // The first work item builds the link's pairs and finds no contacts in two of them while the second starts.
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
