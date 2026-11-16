using NUnit.Framework;
using Unity.Collections;
using Unity.Physics;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// How the island cache gives its slots to islands. A slot follows the identity of the island's first body, which
    /// the backend makes from the body's serial number, not the body's row in the world: a craft keeps its slot when its
    /// bodies move to other rows, and a new craft in a destroyed craft's rows doesn't take over its slot.
    /// </summary>
    public unsafe class DirectSolverCacheTests
    {
        private static ulong Identity(uint serial) => (ulong)serial << 32;

        private static void Assign(ref DirectSolverCache cache, int[] firstBodies, ulong[] identities)
        {
            using var bodies = new NativeArray<int>(firstBodies, Allocator.Temp);
            using var ids = new NativeArray<ulong>(identities, Allocator.Temp);
            cache.Assign(bodies, ids);
        }

        [Test]
        public void AnIslandKeepsItsSlotWhenItsFirstBodyMovesToAnotherRow()
        {
            DirectSolverCache cache = DirectSolverCache.Create();
            try
            {
                Assign(ref cache, new[] { 4, 9 }, new[] { Identity(11), Identity(12) });
                ForestIsland* first = cache.ForestOf(0);
                ForestIsland* second = cache.ForestOf(1);

                // Another body goes, and both crafts move down a row. The second craft is now listed first.
                Assign(ref cache, new[] { 3, 8 }, new[] { Identity(12), Identity(11) });
                Assert.That((ulong)cache.ForestOf(0), Is.EqualTo((ulong)second), "the second craft keeps its slot");
                Assert.That((ulong)cache.ForestOf(1), Is.EqualTo((ulong)first), "the first craft keeps its slot");
            }
            finally
            {
                cache.Dispose();
            }
        }

        [Test]
        public void ANewCraftInADestroyedCraftsRowDoesNotTakeItsSlotAsItsOwn()
        {
            DirectSolverCache cache = DirectSolverCache.Create();
            try
            {
                Assign(ref cache, new[] { 4 }, new[] { Identity(11) });
                ForestIsland* destroyed = cache.ForestOf(0);

                // The craft is destroyed, and a new craft is made in its row. The cache has slots that were never used,
                // and the new craft gets one of those rather than the destroyed craft's.
                Assign(ref cache, new[] { 4 }, new[] { Identity(20) });
                Assert.That((ulong)cache.ForestOf(0), Is.Not.EqualTo((ulong)destroyed));
            }
            finally
            {
                cache.Dispose();
            }
        }

        /// <summary>Starts the cache's next step, as the job that lists the islands does.</summary>
        private static void NextStep(ref DirectSolverCache cache) => Assign(ref cache, new int[0], new ulong[0]);

        /// <summary>
        /// A step has more fused vessel islands than the table of fused slots starts with. The table grows, and every
        /// vessel finds the same forest in the next step. When the table had a fixed size, every island past its end
        /// solved with a forest for one step only and built its structure again in every step.
        /// </summary>
        [Test]
        public void EveryFusedVesselKeepsItsForestWhenThereAreMoreThanTheTableStartsWith()
        {
            const int vessels = 600;
            DirectSolverCache cache = DirectSolverCache.Create();
            try
            {
                var forests = new ulong[vessels];
                NextStep(ref cache);
                for (int v = 0; v < vessels; v++)
                {
                    forests[v] = (ulong)cache.FusedForestOf((ulong)(100 + v));
                    Assert.That(forests[v], Is.Not.EqualTo(0ul), $"vessel {v} has a forest");
                }

                Assert.That(cache.FusedSlotCount, Is.EqualTo(vessels), "one slot for each vessel");
                NextStep(ref cache);
                for (int v = vessels - 1; v >= 0; v--)
                {
                    Assert.That((ulong)cache.FusedForestOf((ulong)(100 + v)), Is.EqualTo(forests[v]), $"vessel {v} keeps its forest");
                }

                Assert.That(cache.FusedSlotCount, Is.EqualTo(vessels), "the table doesn't grow for vessels that it holds");
            }
            finally
            {
                cache.Dispose();
            }
        }

        /// <summary>
        /// A new vessel takes the slot of a vessel that no step has solved for <see cref="DirectSolverCache.FusedKeepSteps"/>
        /// steps, and before that, it gets a new slot.
        /// </summary>
        [Test]
        public void ANewFusedVesselTakesOnlyTheSlotOfAVesselThatStayedAwayLong()
        {
            DirectSolverCache cache = DirectSolverCache.Create();
            try
            {
                NextStep(ref cache);
                ulong first = (ulong)cache.FusedForestOf(1);
                for (int i = 0; i < 10; i++)
                {
                    NextStep(ref cache);
                }

                ulong second = (ulong)cache.FusedForestOf(2);
                Assert.That(second, Is.Not.EqualTo(first), "the first vessel was solved 10 steps ago and keeps its slot");
                Assert.That(cache.FusedSlotCount, Is.EqualTo(2));
                for (int i = 0; i < DirectSolverCache.FusedKeepSteps; i++)
                {
                    NextStep(ref cache);
                }

                cache.FusedForestOf(3);
                Assert.That(cache.FusedSlotCount, Is.EqualTo(2), "the third vessel takes the first vessel's slot");
                Assert.That((ulong)cache.FindFusedForest(1), Is.EqualTo(0ul), "the first vessel has no slot now");
                Assert.That((ulong)cache.FindFusedForest(2), Is.EqualTo(second), "the second vessel keeps its slot");
            }
            finally
            {
                cache.Dispose();
            }
        }

        [Test]
        public void TwoIslandsNeverShareASlot()
        {
            DirectSolverCache cache = DirectSolverCache.Create();
            try
            {
                Assign(ref cache, new[] { 2, 5 }, new[] { Identity(7), Identity(7) });
                Assert.That((ulong)cache.ForestOf(0), Is.Not.EqualTo(0ul));
                Assert.That((ulong)cache.ForestOf(1), Is.Not.EqualTo((ulong)cache.ForestOf(0)));
            }
            finally
            {
                cache.Dispose();
            }
        }
    }
}
