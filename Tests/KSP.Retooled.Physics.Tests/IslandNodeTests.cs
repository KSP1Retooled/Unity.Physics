using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics;
using Math = System.Math;
using DispatchPair =Unity.Physics.DispatchPairSequencer.DispatchPair;
using SolverIsland = Unity.Physics.DispatchPairSequencer.SolverIsland;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The job that finds the solver's islands treats each vessel as one node. These tests run the job alone on made-up
    /// pairs and vessels, and check which islands it finds, how it numbers them, and in what order it lists their
    /// bodies and vessels. They also check that the job's cache follows a change of the vessels' bodies.
    /// </summary>
    public unsafe class IslandNodeTests
    {
        /// <summary>The islands, pairs, bodies and vessels that one run of the job found.</summary>
        private sealed class Found
        {
            public SolverIsland[] Islands;
            public DispatchPair[] Pairs;
            public int[] Bodies;
            public int[] Vessels;

            public int[] BodiesOf(int island) => Bodies.Skip(Islands[island].FirstBody).Take(Islands[island].BodyCount).ToArray();

            public int[] VesselsOf(int island) => Vessels.Skip(Islands[island].FirstVessel).Take(Islands[island].VesselCount).ToArray();

            public (int, int)[] PairsOf(int island) => Pairs.Skip(Islands[island].FirstPair).Take(Islands[island].PairCount)
                .Select(p => (p.BodyIndexA, p.BodyIndexB)).ToArray();
        }

        /// <summary>
        /// Runs the job on contact pairs between the given bodies, with the given vessels. Each vessel is a list of part
        /// bodies, in part order.
        /// </summary>
        private static Found Run(int dynamicBodies, (int, int)[] contacts, int[][] vesselParts, SolverIslandCache cache = default)
        {
            var pairs = new NativeList<DispatchPair>(Allocator.TempJob);
            var joints = new NativeArray<Joint>(0, Allocator.TempJob);
            var records = new NativeArray<VesselRecord>(Math.Max(1, vesselParts.Length), Allocator.TempJob);
            var partBodies = new NativeArray<int>(Math.Max(1, vesselParts.Sum(v => v.Length)), Allocator.TempJob);
            var islands = new NativeList<SolverIsland>(Allocator.TempJob);
            var islandPairs = new NativeList<DispatchPair>(Allocator.TempJob);
            var islandBodies = new NativeList<int>(Allocator.TempJob);
            var islandVessels = new NativeList<int>(Allocator.TempJob);
            var phased = new NativeList<DispatchPair>(Allocator.TempJob);
            var phasesUsed = new NativeList<byte>(Allocator.TempJob);
            var links = new NativeList<int>(Allocator.TempJob);
            try
            {
                foreach ((int a, int b) in contacts)
                {
                    pairs.Add(DispatchPair.CreateCollisionPair(new Broadphase.OverlapResult(a, b)));
                }

                DispatchPairSequencer.SortDispatchPairs(pairs);
                int part = 0;
                for (int v = 0; v < vesselParts.Length; v++)
                {
                    records[v] = new VesselRecord { Id = (ulong)(v + 1), FirstPart = part, PartCount = vesselParts[v].Length };
                    foreach (int body in vesselParts[v])
                    {
                        partBodies[part++] = body;
                    }
                }

                var set = new VesselSet
                {
                    VesselCount = vesselParts.Length,
                    Vessels = (VesselRecord*)records.GetUnsafePtr(),
                    PartBodies = (int*)partBodies.GetUnsafePtr(),
                };

                new DispatchPairSequencer.FindSolverIslandsJob
                {
                    DispatchPairs = pairs.AsArray(),
                    Joints = joints,
                    NumDynamicBodies = dynamicBodies,
                    Islands = islands,
                    IslandPairs = islandPairs,
                    IslandBodies = islandBodies,
                    IslandVessels = islandVessels,
                    PhasedDispatchPairs = phased,
                    PhasesUsed = phasesUsed,
                    LinksInPhases = links,
                    Cache = cache,
                    Vessels = set,
                }.Run();

                return new Found
                {
                    Islands = islands.AsArray().ToArray(),
                    Pairs = islandPairs.AsArray().ToArray(),
                    Bodies = islandBodies.AsArray().ToArray(),
                    Vessels = islandVessels.AsArray().ToArray(),
                };
            }
            finally
            {
                pairs.Dispose();
                joints.Dispose();
                records.Dispose();
                partBodies.Dispose();
                islands.Dispose();
                islandPairs.Dispose();
                islandBodies.Dispose();
                islandVessels.Dispose();
                phased.Dispose();
                phasesUsed.Dispose();
                links.Dispose();
            }
        }

        [Test]
        public void VesselsJoinTheirIslandsAsWholes()
        {
            // Bodies 0 to 9 are dynamic and body 10 is static. Body 6 touches a part of vessel 0 and a part of vessel 1,
            // so the two vessels share an island. Body 9 touches vessel 2, and body 7 touches only the static body.
            Found found = Run(10, new[] { (1, 6), (6, 4), (7, 10), (9, 8) },
                new[] { new[] { 2, 0, 1 }, new[] { 5, 3, 4 }, new[] { 8 } });

            Assert.AreEqual(3, found.Islands.Length);

            // The islands are numbered by their lowest body: 0, then 7, then 8.
            CollectionAssert.AreEqual(new[] { 0, 1 }, found.VesselsOf(0));
            CollectionAssert.AreEqual(new[] { 2, 0, 1, 5, 3, 4, 6 }, found.BodiesOf(0));
            CollectionAssert.AreEqual(new[] { (1, 6), (4, 6) }, found.PairsOf(0));
            Assert.AreEqual(6, found.Islands[0].VesselParts);
            Assert.AreEqual(1, found.Islands[0].OtherBodies);
            Assert.IsTrue(found.Islands[0].HasVessel);

            CollectionAssert.IsEmpty(found.VesselsOf(1));
            CollectionAssert.AreEqual(new[] { 7 }, found.BodiesOf(1));
            CollectionAssert.AreEqual(new[] { (7, 10) }, found.PairsOf(1));
            Assert.IsFalse(found.Islands[1].HasVessel);

            CollectionAssert.AreEqual(new[] { 2 }, found.VesselsOf(2));
            CollectionAssert.AreEqual(new[] { 8, 9 }, found.BodiesOf(2));
            Assert.AreEqual(1, found.Islands[2].OtherBodies);
        }

        [Test]
        public void AVesselWithoutPairsIsAnIslandOfItsOwn()
        {
            Found found = Run(4, new (int, int)[0], new[] { new[] { 3, 1, 2 } });

            Assert.AreEqual(1, found.Islands.Length);
            CollectionAssert.AreEqual(new[] { 3, 1, 2 }, found.BodiesOf(0));
            CollectionAssert.AreEqual(new[] { 0 }, found.VesselsOf(0));
            Assert.AreEqual(0, found.Islands[0].PairCount);
        }

        [Test]
        public void TheCacheFollowsAChangeOfTheVesselsBodies()
        {
            SolverIslandCache cache = SolverIslandCache.Create();
            try
            {
                (int, int)[] contacts = { (0, 5) };
                Found first = Run(6, contacts, new[] { new[] { 0, 1, 2 } }, cache);
                Found again = Run(6, contacts, new[] { new[] { 0, 1, 2 } }, cache);
                CollectionAssert.AreEqual(first.Bodies, again.Bodies);
                CollectionAssert.AreEqual(new[] { 0, 1, 2, 5 }, again.BodiesOf(0));

                // The vessel and the pairs stay the same, but the vessel's parts now sit on other bodies.
                Found moved = Run(6, contacts, new[] { new[] { 0, 3, 4 } }, cache);
                CollectionAssert.AreEqual(new[] { 0, 3, 4, 5 }, moved.BodiesOf(0));
                Assert.AreEqual(1, moved.Islands.Length);

                // Without vessels, the same pair makes an island of two bodies.
                Found none = Run(6, contacts, new int[0][], cache);
                Assert.AreEqual(1, none.Islands.Length);
                CollectionAssert.AreEqual(new[] { 0, 5 }, none.BodiesOf(0));
                CollectionAssert.IsEmpty(none.Vessels);
            }
            finally
            {
                cache.Dispose();
            }
        }

        /// <summary>
        /// Compares the job with a plain search over bodies, on random vessels and pairs, with and without the cache.
        /// The islands must hold the same bodies and pairs, be numbered by their lowest body, and list the vessel parts
        /// first, then the other bodies in ascending order.
        /// </summary>
        [Test]
        public void RandomScenesMatchASearchOverBodies()
        {
            var random = new System.Random(1234);
            SolverIslandCache cache = SolverIslandCache.Create();
            try
            {
                int[][] vessels = null;
                (int, int)[] contacts = null;
                for (int round = 0; round < 200; round++)
                {
                    int dynamicBodies = 40;

                    // Change the vessels in one round of three and the pairs in one of two, so that the cache sees
                    // every mix of kept and changed inputs.
                    if (vessels == null || round % 3 == 0)
                    {
                        List<int> free = Enumerable.Range(0, dynamicBodies).OrderBy(_ => random.Next()).ToList();
                        int count = random.Next(0, 5);
                        vessels = new int[count][];
                        for (int v = 0; v < count; v++)
                        {
                            int parts = random.Next(1, 7);
                            vessels[v] = free.Take(parts).ToArray();
                            free.RemoveRange(0, parts);
                        }
                    }

                    if (contacts == null || round % 2 == 0)
                    {
                        contacts = Enumerable.Range(0, random.Next(0, 25)).Select(_ =>
                        {
                            int a = random.Next(dynamicBodies);
                            int b = random.Next(dynamicBodies + 3);
                            return (a, b == a ? dynamicBodies : b);
                        }).ToArray();
                    }

                    Check(dynamicBodies, contacts, vessels, Run(dynamicBodies, contacts, vessels, round % 4 == 0 ? default : cache));
                }
            }
            finally
            {
                cache.Dispose();
            }
        }

        private static void Check(int dynamicBodies, (int, int)[] contacts, int[][] vessels, Found found)
        {
            var parent = Enumerable.Range(0, dynamicBodies).ToArray();
            int Root(int body) => parent[body] == body ? body : parent[body] = Root(parent[body]);
            var vesselOf = new Dictionary<int, int>();
            for (int v = 0; v < vessels.Length; v++)
            {
                foreach (int body in vessels[v])
                {
                    vesselOf[body] = v;
                    parent[Root(body)] = Root(vessels[v][0]);
                }
            }

            foreach ((int a, int b) in contacts)
            {
                if (a < dynamicBodies && b < dynamicBodies)
                {
                    parent[Root(a)] = Root(b);
                }
            }

            // The expected islands: the groups with a pair or a vessel, ordered by their lowest body.
            var members = new HashSet<int>(vessels.SelectMany(v => v));
            foreach ((int a, int b) in contacts)
            {
                members.Add(Math.Min(a, b));
                if (Math.Max(a, b) < dynamicBodies)
                {
                    members.Add(Math.Max(a, b));
                }
            }

            List<int[]> expected = members.GroupBy(Root).Select(g => g.OrderBy(b => b).ToArray()).OrderBy(g => g[0]).ToList();
            Assert.AreEqual(expected.Count, found.Islands.Length);

            // Without vessels, the job finds the islands as it always did, numbered in the order of their roots, so
            // match them by their lowest body.
            int[] islandOfGroup = Enumerable.Range(0, expected.Count).ToArray();
            if (vessels.Length == 0)
            {
                islandOfGroup = expected.Select(g => Enumerable.Range(0, found.Islands.Length).Single(i => found.BodiesOf(i).Min() == g[0])).ToArray();
            }

            for (int group = 0; group < expected.Count; group++)
            {
                int i = islandOfGroup[group];
                int[] bodies = found.BodiesOf(i);
                CollectionAssert.AreEqual(expected[group], bodies.OrderBy(b => b).ToArray());

                int[] islandVessels = found.VesselsOf(i);
                CollectionAssert.AreEqual(islandVessels.OrderBy(v => v).ToArray(), islandVessels);
                int[] parts = islandVessels.SelectMany(v => vessels[v]).ToArray();
                CollectionAssert.AreEqual(parts, bodies.Take(parts.Length).ToArray());
                int[] others = bodies.Skip(parts.Length).ToArray();
                CollectionAssert.AreEqual(others.OrderBy(b => b).ToArray(), others);
                Assert.IsTrue(others.All(b => !vesselOf.ContainsKey(b)));
                Assert.AreEqual(others.Length, found.Islands[i].OtherBodies);
                Assert.AreEqual(parts.Length, found.Islands[i].VesselParts);

                int root = Root(expected[group][0]);
                int expectedPairs = contacts.Count(c => Root(Math.Min(c.Item1, c.Item2)) == root);
                Assert.AreEqual(expectedPairs, found.Islands[i].PairCount);
                Assert.IsTrue(found.PairsOf(i).All(p => Root(p.Item1) == root));
            }
        }
    }
}
