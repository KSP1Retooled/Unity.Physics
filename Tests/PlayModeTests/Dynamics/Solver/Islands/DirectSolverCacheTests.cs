using NUnit.Framework;
using Unity.Collections;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Islands;

namespace Unity.Physics.Tests.Dynamics.Solver.Islands
{
    /// <summary>
    /// Test island cache slots keyed by body serial so that row movement preserves state and row reuse does not.
    /// </summary>
    public unsafe class DirectSolverCacheTests
    {
        private static ulong Identity(uint serial) => (ulong)serial << 32;

        private static void Assign(ref IslandSolverCache cache, int[] firstBodies, ulong[] identities)
        {
            using var bodies = new NativeArray<int>(firstBodies, Allocator.Temp);
            using var ids = new NativeArray<ulong>(identities, Allocator.Temp);
            cache.Assign(bodies, ids);
        }

        [Test]
        public void AnIslandKeepsItsSlotWhenItsFirstBodyMovesToAnotherRow()
        {
            IslandSolverCache cache = IslandSolverCache.Create();
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
            IslandSolverCache cache = IslandSolverCache.Create();
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
        private static void NextStep(ref IslandSolverCache cache) => Assign(ref cache, new int[0], new ulong[0]);

        /// <summary>
        /// Verify fused slot table growth preserves every vessel's cached forest across steps.
        /// </summary>
        [Test]
        public void EveryFusedVesselKeepsItsForestWhenThereAreMoreThanTheTableStartsWith()
        {
            const int vessels = 600;
            IslandSolverCache cache = IslandSolverCache.Create();
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
        /// A new vessel takes the slot of a vessel that no step has solved for <see cref="IslandSolverCache.FusedKeepSteps"/>
        /// steps, and before that, it gets a new slot.
        /// </summary>
        [Test]
        public void ANewFusedVesselTakesOnlyTheSlotOfAVesselThatStayedAwayLong()
        {
            IslandSolverCache cache = IslandSolverCache.Create();
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
                for (int i = 0; i < IslandSolverCache.FusedKeepSteps; i++)
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
        /// Verify substep slot table growth preserves 200 island slots when their order reverses.
        /// </summary>
        [Test]
        public void EveryIslandKeepsItsSlotWhenThereAreMoreThanTheTableStartsWith()
        {
            const int islands = 200;
            IslandSolverCache cache = IslandSolverCache.Create();
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
        /// Verify fused islands marked NoSlot do not consume or displace substep cache slots.
        /// </summary>
        [Test]
        public void AnIslandThatNeedsNoSlotGetsNone()
        {
            IslandSolverCache cache = IslandSolverCache.Create();
            try
            {
                Assign(ref cache, new[] { 1, 4 }, new[] { Identity(11), Identity(12) });
                ulong second = (ulong)cache.ForestOf(1);
                Assign(ref cache, new[] { 1, 4 }, new[] { IslandSolverCache.NoSlot, Identity(12) });
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
        /// Verify inactive substep slots are reused only after SlotKeepSteps, preserving recently absent islands.
        /// </summary>
        [Test]
        public void ANewIslandTakesOnlyTheSlotOfAnIslandThatStayedAwayLong()
        {
            const int first = 64;
            IslandSolverCache cache = IslandSolverCache.Create();
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
                var previous = cache.ForestOf(1);
                previous->StructureBuilds = 42;
                cache.SetOwner(123, previous);
                ulong last = (ulong)cache.ForestOf(first - 1);
                Assert.That(cache.SlotCount, Is.EqualTo(first), "the first islands fill the slots that the table starts with");
                for (int i = 0; i < 10; i++)
                {
                    Assign(ref cache, new[] { 0 }, new[] { identities[0] });
                }

                Assign(ref cache, new[] { 0, 500 }, new[] { identities[0], Identity(500) });
                Assert.That(cache.SlotCount, Is.EqualTo(first + 1), "the other islands were solved 11 steps ago and keep their slots");
                for (int i = 0; i < IslandSolverCache.SlotKeepSteps; i++)
                {
                    Assign(ref cache, new[] { 0, 500 }, new[] { identities[0], Identity(500) });
                }

                Assign(ref cache, new[] { 0, 500, 600 }, new[] { identities[0], Identity(500), Identity(600) });
                Assert.That(cache.SlotCount, Is.EqualTo(first + 1), "the new island takes a slot that no step used for a long time");
                Assert.That(cache.ForestOf(2)->StructureBuilds, Is.Zero);
                Assert.That((ulong)cache.OwnerOf(123), Is.Zero);

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
            IslandSolverCache cache = IslandSolverCache.Create();
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
