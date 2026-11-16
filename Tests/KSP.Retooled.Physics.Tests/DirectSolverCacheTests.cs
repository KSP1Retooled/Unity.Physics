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

        /// <summary>
        /// A step has 200 islands, more than the 64 slots that the substeps' table starts with. The table grows, and in
        /// the next step every island finds its slot again, even when the islands come in the other order. When the
        /// table had a fixed size of 64, the islands past the 64th solved without a slot in every step.
        /// </summary>
        [Test]
        public void EveryIslandKeepsItsSlotWhenThereAreMoreThanTheTableStartsWith()
        {
            const int islands = 200;
            DirectSolverCache cache = DirectSolverCache.Create();
            try
            {
                var bodies = new int[islands];
                var identities = new ulong[islands];
                for (int i = 0; i < islands; i++)
                {
                    bodies[i] = 3 * i;
                    identities[i] = Identity((uint)(10 + i));
                }

                Assign(ref cache, bodies, identities);
                var forests = new ulong[islands];
                for (int i = 0; i < islands; i++)
                {
                    forests[i] = (ulong)cache.ForestOf(i);
                    Assert.That(forests[i], Is.Not.EqualTo(0ul), $"island {i} has a slot");
                }

                Assert.That(cache.SlotCount, Is.EqualTo(islands), "the table grows to one slot for each island");
                System.Array.Reverse(bodies);
                System.Array.Reverse(identities);
                Assign(ref cache, bodies, identities);
                for (int i = 0; i < islands; i++)
                {
                    Assert.That((ulong)cache.ForestOf(i), Is.EqualTo(forests[islands - 1 - i]), $"island {islands - 1 - i} keeps its slot");
                }

                Assert.That(cache.SlotCount, Is.EqualTo(islands), "the table doesn't grow for islands that it holds");
            }
            finally
            {
                cache.Dispose();
            }
        }

        /// <summary>
        /// The fused job keeps the state of its islands in slots of its own, so the job that marks the fused islands
        /// gives them the identity <see cref="DirectSolverCache.NoSlot"/>. Such an island gets no slot, and it takes
        /// no slot from the other islands.
        /// </summary>
        [Test]
        public void AnIslandThatNeedsNoSlotGetsNone()
        {
            DirectSolverCache cache = DirectSolverCache.Create();
            try
            {
                Assign(ref cache, new[] { 1, 4 }, new[] { Identity(11), Identity(12) });
                ulong second = (ulong)cache.ForestOf(1);
                Assign(ref cache, new[] { 1, 4 }, new[] { DirectSolverCache.NoSlot, Identity(12) });
                Assert.That((ulong)cache.ForestOf(0), Is.EqualTo(0ul), "the fused island has no forest here");
                Assert.That((ulong)cache.Solver(0), Is.EqualTo(0ul), "the fused island has no solver here");
                Assert.That((ulong)cache.ForestOf(1), Is.EqualTo(second), "the other island keeps its slot");
            }
            finally
            {
                cache.Dispose();
            }
        }

        /// <summary>
        /// Once every slot of the substeps' table has been used, a new island takes the slot of an island that no step
        /// has used for <see cref="DirectSolverCache.SlotKeepSteps"/> steps. Before that, the new island gets a new
        /// slot, so that an island that sat out a few steps finds its slot again.
        /// </summary>
        [Test]
        public void ANewIslandTakesOnlyTheSlotOfAnIslandThatStayedAwayLong()
        {
            const int first = 64;
            DirectSolverCache cache = DirectSolverCache.Create();
            try
            {
                var bodies = new int[first];
                var identities = new ulong[first];
                for (int i = 0; i < first; i++)
                {
                    bodies[i] = i;
                    identities[i] = Identity((uint)(100 + i));
                }

                Assign(ref cache, bodies, identities);
                ulong last = (ulong)cache.ForestOf(first - 1);
                Assert.That(cache.SlotCount, Is.EqualTo(first), "the first islands fill the slots that the table starts with");
                for (int i = 0; i < 10; i++)
                {
                    Assign(ref cache, new[] { 0 }, new[] { identities[0] });
                }

                Assign(ref cache, new[] { 0, 500 }, new[] { identities[0], Identity(500) });
                Assert.That(cache.SlotCount, Is.EqualTo(first + 1), "the other islands were solved 11 steps ago and keep their slots");
                for (int i = 0; i < DirectSolverCache.SlotKeepSteps; i++)
                {
                    Assign(ref cache, new[] { 0, 500 }, new[] { identities[0], Identity(500) });
                }

                Assign(ref cache, new[] { 0, 500, 600 }, new[] { identities[0], Identity(500), Identity(600) });
                Assert.That(cache.SlotCount, Is.EqualTo(first + 1), "the new island takes a slot that no step used for a long time");

                // The new island took the slot that was used least recently, the second island's. The last island's slot
                // is still its own.
                Assign(ref cache, new[] { first - 1 }, new[] { identities[first - 1] });
                Assert.That((ulong)cache.ForestOf(0), Is.EqualTo(last), "the last island finds its slot again");
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
