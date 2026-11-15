using System;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.Physics
{
    partial struct DispatchPairSequencer
    {
        /// <summary>
        /// The maximum number of phases for an island that uses the iterative solver. The team job processes this fixed
        /// number of phases in each iteration, including empty phases.
        /// </summary>
        internal const int NumIslandPhases = 4;

        /// <summary> One bucket of pairs for each phase, then one for the direct solver. </summary>
        const int NumIslandBuckets = NumIslandPhases + 1;

        /// <summary>
        /// Finds the islands in one job, then sorts their pairs into buckets in parallel across islands. The array of
        /// phased pairs contains <see cref="NumIslandBuckets"/> equally sized buckets, each large enough to hold every
        /// pair. Each job can therefore reserve space in its bucket independently, without a later pass to combine the
        /// results.
        /// </summary>
        internal static JobHandle ScheduleSortPairsByIsland(in SimulationStepInput stepInput, NativeList<DispatchPair> sortedPairs,
            NativeList<DispatchPair> phasedPairs, in SolverSchedulerInfo solverSchedulerInfo, JobHandle inputDeps)
        {
            int numDynamicBodies = stepInput.World.NumDynamicBodies;
            var islands = new NativeList<SolverIsland>(Allocator.TempJob);
            var islandPairs = new NativeList<DispatchPair>(Allocator.TempJob);
            var islandBodies = new NativeList<int>(Allocator.TempJob);
            var phasesUsed = new NativeList<byte>(Allocator.TempJob);
            var linksInPhases = new NativeList<int>(Allocator.TempJob);
            var bucketCounts = new NativeArray<int>(NumIslandBuckets, Allocator.TempJob);

            JobHandle handle = new FindSolverIslandsJob
            {
                DispatchPairs = sortedPairs.AsDeferredJobArray(),
                Joints = stepInput.World.DynamicsWorld.InternalJointsList.AsDeferredJobArray(),
                NumDynamicBodies = numDynamicBodies,
                Islands = islands,
                IslandPairs = islandPairs,
                IslandBodies = islandBodies,
                PhasedDispatchPairs = phasedPairs,
                PhasesUsed = phasesUsed,
                LinksInPhases = linksInPhases,
                Cache = stepInput.SolverIslandCache,
                Vessels = VesselSet.OrTestSet(stepInput.VesselSet),
            }.Schedule(inputDeps);

            handle = new SortIslandIntoBucketsJob
            {
                IslandPairs = islandPairs.AsDeferredJobArray(),
                Islands = islands.AsDeferredJobArray(),
                PhasesUsed = phasesUsed.AsDeferredJobArray(),
                LinksInPhases = linksInPhases.AsDeferredJobArray(),
                PhasedDispatchPairs = phasedPairs.AsDeferredJobArray(),
                BucketCounts = bucketCounts,
                NumDynamicBodies = numDynamicBodies
            }.Schedule(islands, 1, handle);

            handle = new PublishIslandBucketsJob
            {
                Islands = islands.AsDeferredJobArray(),
                IslandBodies = islandBodies.AsDeferredJobArray(),
                BucketCounts = bucketCounts,
                SolverSchedulerInfo = solverSchedulerInfo,
                PhasedDispatchPairs = phasedPairs.AsDeferredJobArray(),
                NumDynamicBodies = numDynamicBodies
            }.Schedule(handle);

            return JobHandle.CombineDependencies(
                JobHandle.CombineDependencies(islands.Dispose(handle), islandPairs.Dispose(handle), islandBodies.Dispose(handle)),
                JobHandle.CombineDependencies(phasesUsed.Dispose(handle), linksInPhases.Dispose(handle)),
                bucketCounts.Dispose(handle));
        }

        /// <summary>
        /// One connected group of bodies, as a range of <see cref="FindSolverIslandsJob.IslandPairs"/> and of
        /// <see cref="FindSolverIslandsJob.IslandBodies"/>.
        /// </summary>
        internal struct SolverIsland
        {
            public int FirstPair;
            public int PairCount;
            public int FirstBody;
            public int BodyCount;

            /// <summary>
            /// The offset of the island's pairs in the direct bucket, or -1 if the island uses the iterative phases.
            /// </summary>
            public int FirstDirectPair;

            /// <summary>
            /// The number of joint pairs in the direct bucket for this island. Joints precede contacts.
            /// </summary>
            public int NumDirectJoints;

            /// <summary>
            /// Whether the island holds a vessel. Its bodies are joined by the vessel's edges, which are not pairs, so
            /// the
            /// island exists even with no pairs at all, and it always goes to the direct solver.
            /// </summary>
            public bool HasVessel;
        }

        /// <summary>
        /// Groups pairs into islands connected by joints, contacts or vessel parts. Only dynamic bodies connect
        /// islands. The solver does not modify static bodies, so islands that share only a static body can be solved
        /// independently.
        /// </summary>
        [BurstCompile]
        unsafe struct FindSolverIslandsJob : IJob
        {
            [ReadOnly]
            public NativeArray<DispatchPair> DispatchPairs;

            [ReadOnly]
            public NativeArray<Joint> Joints;

            public int NumDynamicBodies;

            public NativeList<SolverIsland> Islands;

            /// <summary>
            /// The pairs to solve, grouped by island and preserving their sorted order within each island.
            /// </summary>
            public NativeList<DispatchPair> IslandPairs;

            public NativeList<int> IslandBodies;

            /// <summary> Sized here for the buckets, once it is known how many pairs each must be able to hold.
            /// </summary>
            public NativeList<DispatchPair> PhasedDispatchPairs;

            /// <summary>
            /// The phases already assigned to links that involve each dynamic body.
            /// </summary>
            public NativeList<byte> PhasesUsed;

            /// <summary>
            /// Allocated here, then filled for an island's bodies only if links need to be recoloured.
            /// </summary>
            public NativeList<int> LinksInPhases;

            /// <summary>
            /// The previous step's islands, reusable if the solved pairs, dynamic body count and vessel key are
            /// unchanged.
            /// </summary>
            public SolverIslandCache Cache;

            /// <summary> The step's vessels. All the bodies of one vessel are in one island. </summary>
            public VesselSet Vessels;

            /// <summary>
            /// Computes a key from the vessel count, vessel IDs, structure versions and body indices of their parts.
            /// These values determine whether cached islands can be reused when vessels are present.
            /// </summary>
            ulong VesselKey()
            {
                ulong key = (ulong)Vessels.VesselCount;
                for (int v = 0; v < Vessels.VesselCount; v++)
                {
                    VesselRecord vessel = Vessels.Vessels[v];
                    key = Mix(key, vessel.Id);
                    key = Mix(key, vessel.StructureVersion);
                    for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                    {
                        key = Mix(key, (ulong)Vessels.PartBodies[part]);
                    }
                }

                return key;
            }

            bool InDynamics(int body) => body >= 0 && body < NumDynamicBodies;

            static ulong Mix(ulong key, ulong value) => (key ^ value) * 0x100000001B3UL + (key >> 29);

            static int FindRoot(NativeArray<int> parents, int body)
            {
                while (parents[body] != body)
                {
                    parents[body] = parents[parents[body]];
                    body = parents[body];
                }

                return body;
            }

            // A joint with no constraints has already affected whether its bodies can collide, but produces no
            // Jacobians to solve. Read its constraint count in place to avoid copying the large Joint struct.
            bool IsSolved(DispatchPair pair, Joint* joints)
            {
                return pair.IsValid && (pair.IsJoint ? joints[pair.JointIndex].Constraints.Length != 0 : pair.IsCollisionEnabled);
            }

            public void Execute()
            {
                var parents = new NativeArray<int>(NumDynamicBodies, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < parents.Length; i++)
                {
                    parents[i] = i;
                }

                // Most overlaps in a craft involve parts with collisions disabled between them. Filter those pairs here
                // so later passes process only pairs that need to be solved.
                var solved = new NativeArray<DispatchPair>(DispatchPairs.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var joints = (Joint*)Joints.GetUnsafeReadOnlyPtr();
                int numPairs = 0;
                for (int i = 0; i < DispatchPairs.Length; i++)
                {
                    DispatchPair pair = DispatchPairs[i];
                    if (IsSolved(pair, joints))
                    {
                        solved[numPairs++] = pair;
                    }
                }

                // Reuse the islands only if the solved pairs, dynamic body count and vessel key match the cached
                // values.
                NativeArray<DispatchPair> solvedPairs = solved.GetSubArray(0, numPairs);
                ulong vesselKey = Vessels.IsCreated ? VesselKey() : 0;
                if (Cache.IsCreated && Cache.TryReuse(solvedPairs, NumDynamicBodies, vesselKey, Islands, IslandPairs, IslandBodies))
                {
                    SizeBuckets(numPairs);
                    return;
                }

                for (int i = 0; i < numPairs; i++)
                {
                    DispatchPair pair = solved[i];
                    if (pair.BodyIndexB < NumDynamicBodies)
                    {
                        parents[FindRoot(parents, pair.BodyIndexA)] = FindRoot(parents, pair.BodyIndexB);
                    }
                }

                // A vessel's parts are joined by its edges, which are not pairs.
                var vesselRoot = new NativeArray<bool>(Vessels.IsCreated ? NumDynamicBodies : 0, Allocator.Temp);
                for (int v = 0; v < Vessels.VesselCount && Vessels.IsCreated; v++)
                {
                    VesselRecord vessel = Vessels.Vessels[v];
                    int first = Vessels.PartBodies[vessel.FirstPart];
                    for (int part = vessel.FirstPart + 1; part < vessel.FirstPart + vessel.PartCount; part++)
                    {
                        int body = Vessels.PartBodies[part];
                        if (InDynamics(first) && InDynamics(body))
                        {
                            parents[FindRoot(parents, body)] = FindRoot(parents, first);
                        }
                    }
                }

                // Store each body's root directly so later passes do not need to follow parent chains.
                for (int body = 0; body < NumDynamicBodies; body++)
                {
                    parents[body] = FindRoot(parents, body);
                }

                for (int v = 0; v < Vessels.VesselCount && Vessels.IsCreated; v++)
                {
                    VesselRecord vessel = Vessels.Vessels[v];
                    if (vessel.PartCount > 0 && InDynamics(Vessels.PartBodies[vessel.FirstPart]))
                    {
                        vesselRoot[parents[Vessels.PartBodies[vessel.FirstPart]]] = true;
                    }
                }

                // Roots with pairs or a vessel become islands, numbered in body order so that the numbers don't depend
                // on the
                // pairs' order.
                var islandOfRoot = new NativeArray<int>(NumDynamicBodies, Allocator.Temp);
                for (int i = 0; i < numPairs; i++)
                {
                    islandOfRoot[parents[solved[i].BodyIndexA]]++;
                }

                Islands.Clear();
                int firstPair = 0;
                for (int root = 0; root < NumDynamicBodies; root++)
                {
                    int pairCount = islandOfRoot[root];
                    bool hasVessel = vesselRoot.Length > 0 && vesselRoot[root];
                    if (pairCount == 0 && !hasVessel)
                    {
                        islandOfRoot[root] = -1;
                        continue;
                    }

                    islandOfRoot[root] = Islands.Length;
                    Islands.Add(new SolverIsland { FirstPair = firstPair, PairCount = pairCount, FirstDirectPair = -1, HasVessel = hasVessel });
                    firstPair += pairCount;
                }

                for (int body = 0; body < NumDynamicBodies; body++)
                {
                    int island = islandOfRoot[parents[body]];
                    if (island >= 0)
                    {
                        Islands.ElementAt(island).BodyCount++;
                    }
                }

                int firstBody = 0;
                for (int i = 0; i < Islands.Length; i++)
                {
                    ref SolverIsland island = ref Islands.ElementAt(i);
                    island.FirstBody = firstBody;
                    firstBody += island.BodyCount;
                }

                IslandPairs.ResizeUninitialized(numPairs);
                IslandBodies.ResizeUninitialized(firstBody);
                var nextPair = new NativeArray<int>(Islands.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var nextBody = new NativeArray<int>(Islands.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < Islands.Length; i++)
                {
                    nextPair[i] = Islands[i].FirstPair;
                    nextBody[i] = Islands[i].FirstBody;
                }

                for (int i = 0; i < numPairs; i++)
                {
                    DispatchPair pair = solved[i];
                    int island = islandOfRoot[parents[pair.BodyIndexA]];
                    IslandPairs[nextPair[island]++] = pair;
                }

                for (int body = 0; body < NumDynamicBodies; body++)
                {
                    int island = islandOfRoot[parents[body]];
                    if (island >= 0)
                    {
                        IslandBodies[nextBody[island]++] = body;
                    }
                }

                if (Cache.IsCreated)
                {
                    Cache.Keep(solvedPairs, NumDynamicBodies, vesselKey, Islands, IslandPairs, IslandBodies);
                }

                SizeBuckets(numPairs);
            }

            void SizeBuckets(int numPairs)
            {
                PhasedDispatchPairs.ResizeUninitialized(numPairs * NumIslandBuckets);
                PhasesUsed.Resize(NumDynamicBodies, NativeArrayOptions.ClearMemory);
                LinksInPhases.ResizeUninitialized(NumDynamicBodies * NumIslandPhases);
            }
        }

        /// <summary>
        /// Assigns phases to an island's links and puts its pairs in the corresponding buckets. An island uses
        /// iterative phases only if it has no joints or vessels and its links fit in <see cref="NumIslandPhases"/>
        /// phases. Otherwise, all its pairs go into the direct bucket, so each body is handled by one solver. A colour
        /// represents a phase assigned to a link (see <see cref="DispatchPair.IsSameLink"/>). Pairs on the same link
        /// run sequentially on one thread and therefore do not conflict with each other. Islands have no dynamic bodies
        /// in common, so jobs can use the shared <see cref="PhasesUsed"/> and <see cref="LinksInPhases"/> arrays. Each
        /// thread reserves space in a bucket atomically. Island order within a bucket can vary between runs, but pair
        /// order within each island is preserved.
        /// </summary>
        [BurstCompile]
        unsafe struct SortIslandIntoBucketsJob : IJobParallelForDefer
        {
            const int AllPhases = (1 << NumIslandPhases) - 1;

            /// <summary>
            /// The number of chain traversal steps allowed per link. This usually permits recolouring a craft's links
            /// while bounding the total work for large islands.
            /// </summary>
            const int MaxChainStepsPerLink = 8;

            [ReadOnly]
            public NativeArray<DispatchPair> IslandPairs;

            [NativeDisableParallelForRestriction]
            public NativeArray<SolverIsland> Islands;

            [NativeDisableParallelForRestriction]
            public NativeArray<byte> PhasesUsed;

            /// <summary>
            /// The link assigned to each dynamic body and phase, or -1 if none is assigned. Filled only when the
            /// initial phase assignment needs recolouring.
            /// </summary>
            [NativeDisableParallelForRestriction]
            public NativeArray<int> LinksInPhases;

            [NativeDisableParallelForRestriction]
            public NativeArray<DispatchPair> PhasedDispatchPairs;

            /// <summary> How many pairs each bucket holds so far. </summary>
            [NativeDisableParallelForRestriction]
            public NativeArray<int> BucketCounts;

            public int NumDynamicBodies;

            public void Execute(int islandIndex)
            {
                SolverIsland island = Islands[islandIndex];
                NativeArray<DispatchPair> pairs = IslandPairs.GetSubArray(island.FirstPair, island.PairCount);

                // The pairs are sorted by body, so a link's pairs are next to each other, and they stay so in a bucket.
                var firstPairs = new NativeArray<int>(pairs.Length + 1, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                int numLinks = 0;
                bool hasJoint = false;
                for (int i = 0; i < pairs.Length; i++)
                {
                    hasJoint |= pairs[i].IsJoint;
                    if (i == 0 || !pairs[i].IsSameLink(pairs[i - 1], NumDynamicBodies))
                    {
                        firstPairs[numLinks++] = i;
                    }
                }

                firstPairs[numLinks] = pairs.Length;

                // Use the direct solver for every island with a joint, even if its links fit in the available phases.
                // Iterative sweeps alone would make a stiff joint appear soft, including on small craft.
                var phases = new NativeArray<byte>(numLinks, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                bool fits = !island.HasVessel && !hasJoint && CountCrowdedBodies(pairs, firstPairs, numLinks) == 0
                    && ColourLinks(pairs, firstPairs, numLinks, phases);

                int bucketSize = PhasedDispatchPairs.Length / NumIslandBuckets;
                if (!fits)
                {
                    // Build Jacobians of the direct solver's type only for joints. Keep the iterative type for contact
                    // Jacobians: the sparse solver uses them to build contact rows, and the event export reads them
                    // afterwards.
                    island.FirstDirectPair = Reserve(NumIslandPhases, pairs.Length);
                    int nextPair = NumIslandPhases * bucketSize + island.FirstDirectPair;
                    for (int pass = 0; pass < 2; pass++)
                    {
                        for (int i = 0; i < pairs.Length; i++)
                        {
                            DispatchPair pair = pairs[i];
                            if (pair.IsJoint == (pass == 0))
                            {
                                pair.SolverType = pair.IsJoint ? SolverType.Direct : SolverType.Iterative;
                                PhasedDispatchPairs[nextPair++] = pair;
                                island.NumDirectJoints += pass == 0 ? 1 : 0;
                            }
                        }
                    }

                    Islands[islandIndex] = island;
                    return;
                }

                int* pairsInPhase = stackalloc int[NumIslandPhases];
                UnsafeUtility.MemClear(pairsInPhase, NumIslandPhases * sizeof(int));
                for (int link = 0; link < numLinks; link++)
                {
                    pairsInPhase[phases[link]] += firstPairs[link + 1] - firstPairs[link];
                }

                int* nextPairInPhase = stackalloc int[NumIslandPhases];
                for (int phase = 0; phase < NumIslandPhases; phase++)
                {
                    nextPairInPhase[phase] = phase * bucketSize + Reserve(phase, pairsInPhase[phase]);
                }

                // A pair with a static body comes after the others on its body, as it does in the other layouts.
                for (int pass = 0; pass < 2; pass++)
                {
                    for (int link = 0; link < numLinks; link++)
                    {
                        if (pairs[firstPairs[link]].BodyIndexB >= NumDynamicBodies == (pass == 1))
                        {
                            for (int i = firstPairs[link]; i < firstPairs[link + 1]; i++)
                            {
                                PhasedDispatchPairs[nextPairInPhase[phases[link]]++] = pairs[i];
                            }
                        }
                    }
                }
            }

            /// <summary>
            /// Counts bodies that participate in more links than there are phases. Links that share a body need
            /// different phases, so any such body requires the direct solver without attempting to assign phases.
            /// Clears <see cref="PhasesUsed"/> before returning.
            /// </summary>
            int CountCrowdedBodies(NativeArray<DispatchPair> pairs, NativeArray<int> firstPairs, int numLinks)
            {
                int crowdedBodies = 0;
                for (int link = 0; link < numLinks; link++)
                {
                    DispatchPair pair = pairs[firstPairs[link]];
                    crowdedBodies += CountLink(pair.BodyIndexA);
                    if (pair.BodyIndexB < NumDynamicBodies)
                    {
                        crowdedBodies += CountLink(pair.BodyIndexB);
                    }
                }

                for (int link = 0; link < numLinks; link++)
                {
                    DispatchPair pair = pairs[firstPairs[link]];
                    PhasesUsed[pair.BodyIndexA] = 0;
                    if (pair.BodyIndexB < NumDynamicBodies)
                    {
                        PhasesUsed[pair.BodyIndexB] = 0;
                    }
                }

                return crowdedBodies;
            }

            /// <summary>
            /// Returns 1 when the body's link count first exceeds the number of phases; otherwise returns 0.
            /// </summary>
            int CountLink(int body)
            {
                int links = PhasesUsed[body];
                if (links > NumIslandPhases)
                {
                    return 0;
                }

                PhasesUsed[body] = (byte)(links + 1);
                return links == NumIslandPhases ? 1 : 0;
            }

            /// <summary>
            /// Assigns a phase to every link, or returns false. First tries an available phase for each link. If
            /// none is available, fills <see cref="LinksInPhases"/> and attempts to free a phase by swapping colours
            /// along chains. The total number of chain traversal steps is limited in proportion to the number of links.
            /// If no swap succeeds or the limit is reached, the island uses the direct solver. A failed attempt can
            /// result from an odd cycle, but reaching the limit alone does not prove that the island has one.
            /// </summary>
            bool ColourLinks(NativeArray<DispatchPair> pairs, NativeArray<int> firstPairs, int numLinks, NativeArray<byte> phases)
            {
                int stepsLeft = numLinks * MaxChainStepsPerLink;
                bool hasTable = false;
                var chain = default(NativeArray<int>);

                for (int pass = 0; pass < 2; pass++)
                {
                    for (int link = 0; link < numLinks; link++)
                    {
                        DispatchPair pair = pairs[firstPairs[link]];
                        bool isBodyBStatic = pair.BodyIndexB >= NumDynamicBodies;
                        if (isBodyBStatic != (pass == 1))
                        {
                            continue;
                        }

                        int phase = FirstFreePhase(pair);
                        if (phase < 0)
                        {
                            if (!hasTable)
                            {
                                hasTable = true;
                                chain = new NativeArray<int>(numLinks, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                                FillLinksInPhases(pairs, firstPairs, phases, link, pass);
                            }

                            phase = FreePhaseWithChain(pair, pairs, firstPairs, phases, chain, ref stepsLeft);
                            if (phase < 0)
                            {
                                return false;
                            }
                        }

                        phases[link] = (byte)phase;
                        SetPhase(pair, phase, link, hasTable);
                    }
                }

                return true;
            }

            /// <summary>
            /// Finds a phase that is available on both dynamic bodies of a link, or returns -1 if none is available.
            /// Static bodies are not modified, so any number of links can share a static body within a phase. For a
            /// link to a static body, chooses the last available phase so constraints against the world are processed
            /// later.
            /// </summary>
            int FirstFreePhase(DispatchPair pair)
            {
                bool isBodyBStatic = pair.BodyIndexB >= NumDynamicBodies;
                int used = PhasesUsed[pair.BodyIndexA] | (isBodyBStatic ? 0 : PhasesUsed[pair.BodyIndexB]);
                int free = ~used & AllPhases;
                if (free == 0)
                {
                    return -1;
                }

                return isBodyBStatic ? 31 - math.lzcnt(free) : math.tzcnt(free);
            }

            void SetPhase(DispatchPair pair, int phase, int link, bool hasTable)
            {
                PhasesUsed[pair.BodyIndexA] = (byte)(PhasesUsed[pair.BodyIndexA] | (1 << phase));
                if (hasTable)
                {
                    LinksInPhases[pair.BodyIndexA * NumIslandPhases + phase] = link;
                }

                if (pair.BodyIndexB < NumDynamicBodies)
                {
                    PhasesUsed[pair.BodyIndexB] = (byte)(PhasesUsed[pair.BodyIndexB] | (1 << phase));
                    if (hasTable)
                    {
                        LinksInPhases[pair.BodyIndexB * NumIslandPhases + phase] = link;
                    }
                }
            }

            void ClearPhase(DispatchPair pair, int phase)
            {
                PhasesUsed[pair.BodyIndexA] = (byte)(PhasesUsed[pair.BodyIndexA] & ~(1 << phase));
                LinksInPhases[pair.BodyIndexA * NumIslandPhases + phase] = -1;
                if (pair.BodyIndexB < NumDynamicBodies)
                {
                    PhasesUsed[pair.BodyIndexB] = (byte)(PhasesUsed[pair.BodyIndexB] & ~(1 << phase));
                    LinksInPhases[pair.BodyIndexB * NumIslandPhases + phase] = -1;
                }
            }

            /// <summary>
            /// Records links that already have a phase: all links from earlier passes and links before the current link
            /// in this pass.
            /// </summary>
            void FillLinksInPhases(NativeArray<DispatchPair> pairs, NativeArray<int> firstPairs, NativeArray<byte> phases, int stuckLink, int stuckPass)
            {
                int numLinks = phases.Length;
                for (int link = 0; link < numLinks; link++)
                {
                    DispatchPair pair = pairs[firstPairs[link]];
                    for (int phase = 0; phase < NumIslandPhases; phase++)
                    {
                        LinksInPhases[pair.BodyIndexA * NumIslandPhases + phase] = -1;
                        if (pair.BodyIndexB < NumDynamicBodies)
                        {
                            LinksInPhases[pair.BodyIndexB * NumIslandPhases + phase] = -1;
                        }
                    }
                }

                for (int link = 0; link < numLinks; link++)
                {
                    DispatchPair pair = pairs[firstPairs[link]];
                    int pass = pair.BodyIndexB >= NumDynamicBodies ? 1 : 0;
                    if (pass < stuckPass || (pass == stuckPass && link < stuckLink))
                    {
                        SetPhase(pair, phases[link], link, true);
                    }
                }
            }

            /// <summary>
            /// Attempts to free the same phase at both ends of a link that could not be assigned a phase. Chooses a
            /// phase free at each end, then follows a path of links with alternating phases from one end. Swapping
            /// those phases along the path frees a common phase unless the path reaches the other end. Reaching the
            /// other end forms an odd cycle with the unassigned link. This obstruction cannot occur in a tree, but can
            /// occur in a craft with cycles formed by struts or contacts. Returns -1 if every pair of phases fails or
            /// the traversal limit is reached, including if that limit is reached on a tree.
            /// </summary>
            int FreePhaseWithChain(DispatchPair stuck, NativeArray<DispatchPair> pairs, NativeArray<int> firstPairs,
                NativeArray<byte> phases, NativeArray<int> chain, ref int stepsLeft)
            {
                int bodyA = stuck.BodyIndexA;
                int bodyB = stuck.BodyIndexB;
                int freeOnA = ~PhasesUsed[bodyA] & AllPhases;
                int freeOnB = ~PhasesUsed[bodyB] & AllPhases;

                for (int phaseA = 0; phaseA < NumIslandPhases; phaseA++)
                {
                    if ((freeOnA & (1 << phaseA)) == 0)
                    {
                        continue;
                    }

                    for (int phaseB = 0; phaseB < NumIslandPhases; phaseB++)
                    {
                        if ((freeOnB & (1 << phaseB)) == 0)
                        {
                            continue;
                        }

                        int length = 0;
                        int body = bodyB;
                        int wanted = phaseA;
                        bool reachedA = false;
                        while (true)
                        {
                            if (--stepsLeft < 0)
                            {
                                return -1;
                            }

                            int next = LinksInPhases[body * NumIslandPhases + wanted];
                            if (next < 0)
                            {
                                break;
                            }

                            chain[length++] = next;
                            DispatchPair pair = pairs[firstPairs[next]];
                            body = pair.BodyIndexA == body ? pair.BodyIndexB : pair.BodyIndexA;
                            reachedA = body == bodyA;
                            if (reachedA || body >= NumDynamicBodies)
                            {
                                break;
                            }

                            wanted = wanted == phaseA ? phaseB : phaseA;
                        }

                        if (reachedA)
                        {
                            continue;
                        }

                        // Clear all assignments on the path before setting new ones, because adjacent links exchange
                        // the same two phase slots.
                        for (int i = 0; i < length; i++)
                        {
                            ClearPhase(pairs[firstPairs[chain[i]]], phases[chain[i]]);
                        }

                        for (int i = 0; i < length; i++)
                        {
                            int swapped = phases[chain[i]] == phaseA ? phaseB : phaseA;
                            phases[chain[i]] = (byte)swapped;
                            SetPhase(pairs[firstPairs[chain[i]]], swapped, chain[i], true);
                        }

                        return phaseA;
                    }
                }

                return -1;
            }

            /// <summary>
            /// Atomically reserves space in a bucket and returns its starting offset.
            /// </summary>
            int Reserve(int bucket, int count)
            {
                ref int bucketCount = ref UnsafeUtility.ArrayElementAsRef<int>(BucketCounts.GetUnsafePtr(), bucket);
                return Interlocked.Add(ref bucketCount, count) - count;
            }

        }

        /// <summary>
        /// Publishes the pairs and work items in each bucket to the solver's scheduling data. Empty buckets are
        /// allowed. <see cref="IterativeSolverSchedulerInfo.FindPhaseId"/> handles consecutive phases with the same
        /// offset in the work items.
        /// </summary>
        [BurstCompile]
        struct PublishIslandBucketsJob : IJob
        {
            [ReadOnly]
            public NativeArray<SolverIsland> Islands;

            [ReadOnly]
            public NativeArray<int> IslandBodies;

            [ReadOnly]
            public NativeArray<int> BucketCounts;

            [NativeDisableContainerSafetyRestriction]
            public SolverSchedulerInfo SolverSchedulerInfo;

            [ReadOnly]
            public NativeArray<DispatchPair> PhasedDispatchPairs;

            public int NumDynamicBodies;

            static IterativeSolverSchedulerInfo.SolvePhaseInfo Phase(int numPairs, int firstPair, int firstWorkItem, int linkedNumDynamicBodies)
            {
                int batchSize = math.min(CreateDispatchPairPhasesJob.kMaxBatchSize, numPairs);
                return new IterativeSolverSchedulerInfo.SolvePhaseInfo
                {
                    DispatchPairCount = numPairs,
                    BatchSize = batchSize,
                    NumWorkItems = batchSize == 0 ? 0 : (numPairs + batchSize - 1) / batchSize,
                    FirstDispatchPairOffset = firstPair,
                    FirstWorkItemIndexOffset = firstWorkItem,
                    LinkedNumDynamicBodies = linkedNumDynamicBodies,
                };
            }

            public void Execute()
            {
                int bucketSize = PhasedDispatchPairs.Length / NumIslandBuckets;

                IterativeSolverSchedulerInfo iterative = SolverSchedulerInfo.IterativePairsIterativeScheduling;
                int numIterativePairs = 0;
                int numIterativeWorkItems = 0;
                for (int phase = 0; phase < NumIslandPhases; phase++)
                {
                    iterative.PhaseInfo[phase] = Phase(BucketCounts[phase], phase * bucketSize, numIterativeWorkItems, NumDynamicBodies);
                    numIterativePairs += BucketCounts[phase];
                    numIterativeWorkItems += iterative.PhaseInfo[phase].NumWorkItems;
                }

                iterative.NumActivePhases[0] = NumIslandPhases;
                iterative.NumWorkItems[0] = numIterativeWorkItems;
                iterative.NumDispatchPairs.Value = numIterativePairs;
                iterative.FindLinkedWorkItemRanges(PhasedDispatchPairs);

                int numDirectPairs = BucketCounts[NumIslandPhases];
                IterativeSolverSchedulerInfo direct = SolverSchedulerInfo.DirectPairsIterativeScheduling;
                direct.PhaseInfo[0] = Phase(numDirectPairs, 0, 0, 0);
                direct.NumActivePhases[0] = 1;
                direct.NumWorkItems[0] = direct.PhaseInfo[0].NumWorkItems;
                direct.FirstDispatchPairIndex.Value = NumIslandPhases * bucketSize;
                direct.NumDispatchPairs.Value = numDirectPairs;

                DirectSolverSchedulerInfo islands = SolverSchedulerInfo.DirectPairsDirectScheduling;
                islands.FirstDispatchPairIndex.Value = direct.FirstDispatchPairIndex.Value;
                islands.ResizeMaps(numDirectPairs);
                bool anyDirect = numDirectPairs > 0;
                for (int i = 0; i < Islands.Length && !anyDirect; i++)
                {
                    anyDirect = Islands[i].FirstDirectPair >= 0;
                }

                islands.BodiesInIslands.Resize(anyDirect ? NumDynamicBodies : 0, NativeArrayOptions.ClearMemory);
                for (int i = 0; i < numDirectPairs; i++)
                {
                    islands.UnphasedToPhasedDispatchPairMap[i] = i;
                }

                for (int i = 0; i < Islands.Length; i++)
                {
                    SolverIsland island = Islands[i];
                    if (island.FirstDirectPair < 0)
                    {
                        continue;
                    }

                    int directIsland = islands.DispatchPairIslandInfoCounts.Length;
                    islands.DispatchPairIslandInfoCounts.Add(island.NumDirectJoints);
                    islands.IslandContactCounts.Add(island.PairCount - island.NumDirectJoints);
                    islands.FirstDispatchPairIslandInfoIndices.Add(islands.DispatchPairIslandInfos.Length);
                    for (int pair = 0; pair < island.PairCount; pair++)
                    {
                        islands.DispatchPairIslandInfos.Add(new DispatchPairIslandInfo
                        {
                            UnphasedDispatchPairIndex = island.FirstDirectPair + pair,
                            IslandIndex = directIsland
                        });
                    }

                    islands.FirstIslandBodyIndices.Add(islands.IslandBodies.Length);
                    for (int body = 0; body < island.BodyCount; body++)
                    {
                        int bodyIndex = IslandBodies[island.FirstBody + body];
                        islands.IslandBodies.Add(bodyIndex);
                        islands.BodiesInIslands[bodyIndex] = true;
                    }
                }

                SolverSchedulerInfo.FinalizeIterativeWorkItemInfo();
            }
        }
    }

    /// <summary>
    /// Caches the previous step's islands and the pairs used to find them. A craft often has the same joints across
    /// steps, and a floating craft may have no other constraints, so the islands can often be reused. Reuse requires
    /// matching pairs, dynamic body count and vessel key. The caller owns this cache and passes it in
    /// <see cref="SimulationStepInput"/>. Without a cache, islands are found again every step.
    /// </summary>
    public unsafe struct SolverIslandCache : IDisposable
    {
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<DispatchPairSequencer.DispatchPair>* pairs;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<DispatchPairSequencer.SolverIsland>* islands;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<DispatchPairSequencer.DispatchPair>* islandPairs;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* islandBodies;

        /// <summary>
        /// The number of dynamic bodies used to find the cached islands, or -1 if no islands have been cached.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        int* numDynamicBodies;

        /// <summary>
        /// The key computed from the vessels used to find the cached islands.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        ulong* vesselKey;

        public bool IsCreated => pairs != null;

        public static SolverIslandCache Create()
        {
            var cache = new SolverIslandCache
            {
                pairs = UnsafeList<DispatchPairSequencer.DispatchPair>.Create(0, Allocator.Persistent),
                islands = UnsafeList<DispatchPairSequencer.SolverIsland>.Create(0, Allocator.Persistent),
                islandPairs = UnsafeList<DispatchPairSequencer.DispatchPair>.Create(0, Allocator.Persistent),
                islandBodies = UnsafeList<int>.Create(0, Allocator.Persistent),
                numDynamicBodies = (int*)UnsafeUtility.MallocTracked(sizeof(int), UnsafeUtility.AlignOf<int>(), Allocator.Persistent, 0),
                vesselKey = (ulong*)UnsafeUtility.MallocTracked(sizeof(ulong), UnsafeUtility.AlignOf<ulong>(), Allocator.Persistent, 0),
            };
            *cache.numDynamicBodies = -1;
            *cache.vesselKey = 0;
            return cache;
        }

        public void Dispose()
        {
            if (!IsCreated)
            {
                return;
            }

            UnsafeList<DispatchPairSequencer.DispatchPair>.Destroy(pairs);
            UnsafeList<DispatchPairSequencer.SolverIsland>.Destroy(islands);
            UnsafeList<DispatchPairSequencer.DispatchPair>.Destroy(islandPairs);
            UnsafeList<int>.Destroy(islandBodies);
            UnsafeUtility.FreeTracked(numDynamicBodies, Allocator.Persistent);
            UnsafeUtility.FreeTracked(vesselKey, Allocator.Persistent);
            this = default;
        }

        internal bool TryReuse(NativeArray<DispatchPairSequencer.DispatchPair> solved, int dynamicBodies, ulong vessels,
            NativeList<DispatchPairSequencer.SolverIsland> outIslands, NativeList<DispatchPairSequencer.DispatchPair> outPairs,
            NativeList<int> outBodies)
        {
            if (*numDynamicBodies != dynamicBodies || *vesselKey != vessels || pairs->Length != solved.Length
                || UnsafeUtility.MemCmp(pairs->Ptr, solved.GetUnsafeReadOnlyPtr(), (long)solved.Length * sizeof(DispatchPairSequencer.DispatchPair)) != 0)
            {
                return false;
            }

            Copy(islands, outIslands);
            Copy(islandPairs, outPairs);
            Copy(islandBodies, outBodies);
            return true;
        }

        internal void Keep(NativeArray<DispatchPairSequencer.DispatchPair> solved, int dynamicBodies, ulong vessels,
            NativeList<DispatchPairSequencer.SolverIsland> found, NativeList<DispatchPairSequencer.DispatchPair> foundPairs,
            NativeList<int> foundBodies)
        {
            *numDynamicBodies = dynamicBodies;
            *vesselKey = vessels;
            Store(pairs, solved);
            Store(islands, found.AsArray());
            Store(islandPairs, foundPairs.AsArray());
            Store(islandBodies, foundBodies.AsArray());
        }

        static void Copy<T>(UnsafeList<T>* from, NativeList<T> to)
            where T : unmanaged
        {
            to.ResizeUninitialized(from->Length);
            UnsafeUtility.MemCpy(to.GetUnsafePtr(), from->Ptr, (long)from->Length * sizeof(T));
        }

        static void Store<T>(UnsafeList<T>* to, NativeArray<T> from)
            where T : unmanaged
        {
            to->Resize(from.Length);
            UnsafeUtility.MemCpy(to->Ptr, from.GetUnsafeReadOnlyPtr(), (long)from.Length * sizeof(T));
        }
    }
}
