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
        /// How many phases an island may need and stay with the iterative solver. A step schedules this many jobs for each
        /// iteration whatever it contains, so this is what keeps their number fixed and small.
        /// </summary>
        internal const int NumIslandPhases = 4;

        /// <summary> One bucket of pairs for each phase, then one for the direct solver. </summary>
        const int NumIslandBuckets = NumIslandPhases + 1;

        /// <summary>
        /// Finds the islands one job at a time, then has each island sort itself into the buckets at once. The phased pairs hold
        /// <see cref="NumIslandBuckets"/> buckets of the same size, each big enough for every pair, so that no bucket has to
        /// know how much the others got and nothing has to be copied together afterwards.
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
                LinksInPhases = linksInPhases
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
        struct SolverIsland
        {
            public int FirstPair;
            public int PairCount;
            public int FirstBody;
            public int BodyCount;

            /// <summary> Where its pairs are in the direct bucket, or -1 if they fit in phases. </summary>
            public int FirstDirectPair;

            /// <summary> How many of those are joints, which come before the contacts. </summary>
            public int NumDirectJoints;
        }

        /// <summary>
        /// Groups the pairs by island. Islands connect through joints and contacts alike, and through dynamic bodies only: a
        /// static body is never written to, so what happens on either side of one is independent.
        /// </summary>
        [BurstCompile]
        struct FindSolverIslandsJob : IJob
        {
            [ReadOnly]
            public NativeArray<DispatchPair> DispatchPairs;

            [ReadOnly]
            public NativeArray<Joint> Joints;

            public int NumDynamicBodies;

            public NativeList<SolverIsland> Islands;

            /// <summary> The pairs to solve, grouped by island and in their sorted order within one. </summary>
            public NativeList<DispatchPair> IslandPairs;

            public NativeList<int> IslandBodies;

            /// <summary> Sized here for the buckets, once it is known how many pairs each must be able to hold. </summary>
            public NativeList<DispatchPair> PhasedDispatchPairs;

            /// <summary> The phases each dynamic body is in so far. </summary>
            public NativeList<byte> PhasesUsed;

            /// <summary> Only sized here: an island fills in its bodies' rows if it has to recolour. </summary>
            public NativeList<int> LinksInPhases;

            static int FindRoot(NativeArray<int> parents, int body)
            {
                while (parents[body] != body)
                {
                    parents[body] = parents[parents[body]];
                    body = parents[body];
                }

                return body;
            }

            // A joint with no constraints has had its say on whether its bodies collide, and writes no Jacobians to solve.
            bool IsSolved(DispatchPair pair)
            {
                return pair.IsValid && (pair.IsJoint ? Joints[pair.JointIndex].Constraints.Length != 0 : pair.IsCollisionEnabled);
            }

            public void Execute()
            {
                var parents = new NativeArray<int>(NumDynamicBodies, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < parents.Length; i++)
                {
                    parents[i] = i;
                }

                int numPairs = 0;
                for (int i = 0; i < DispatchPairs.Length; i++)
                {
                    DispatchPair pair = DispatchPairs[i];
                    if (IsSolved(pair))
                    {
                        numPairs++;
                        if (pair.BodyIndexB < NumDynamicBodies)
                        {
                            parents[FindRoot(parents, pair.BodyIndexA)] = FindRoot(parents, pair.BodyIndexB);
                        }
                    }
                }

                // Roots with pairs become islands, numbered in body order so that the numbers don't depend on the pairs' order.
                var islandOfRoot = new NativeArray<int>(NumDynamicBodies, Allocator.Temp);
                for (int i = 0; i < DispatchPairs.Length; i++)
                {
                    DispatchPair pair = DispatchPairs[i];
                    if (IsSolved(pair))
                    {
                        islandOfRoot[FindRoot(parents, pair.BodyIndexA)]++;
                    }
                }

                Islands.Clear();
                int firstPair = 0;
                for (int root = 0; root < NumDynamicBodies; root++)
                {
                    int pairCount = islandOfRoot[root];
                    if (pairCount == 0)
                    {
                        islandOfRoot[root] = -1;
                        continue;
                    }

                    islandOfRoot[root] = Islands.Length;
                    Islands.Add(new SolverIsland { FirstPair = firstPair, PairCount = pairCount, FirstDirectPair = -1 });
                    firstPair += pairCount;
                }

                for (int body = 0; body < NumDynamicBodies; body++)
                {
                    int island = islandOfRoot[FindRoot(parents, body)];
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

                for (int i = 0; i < DispatchPairs.Length; i++)
                {
                    DispatchPair pair = DispatchPairs[i];
                    if (IsSolved(pair))
                    {
                        int island = islandOfRoot[FindRoot(parents, pair.BodyIndexA)];
                        IslandPairs[nextPair[island]++] = pair;
                    }
                }

                for (int body = 0; body < NumDynamicBodies; body++)
                {
                    int island = islandOfRoot[FindRoot(parents, body)];
                    if (island >= 0)
                    {
                        IslandBodies[nextBody[island]++] = body;
                    }
                }

                PhasedDispatchPairs.ResizeUninitialized(numPairs * NumIslandBuckets);
                PhasesUsed.Resize(NumDynamicBodies, NativeArrayOptions.ClearMemory);
                LinksInPhases.ResizeUninitialized(NumDynamicBodies * NumIslandPhases);
            }
        }

        /// <summary>
        /// Colours one island, and sorts its pairs into the bucket of each phase if it has no joints and fits in
        /// <see cref="NumIslandPhases"/> of them, or all into the direct bucket if not, so that no body is left between the two
        /// solvers. What gets a colour is a
        /// link (see <see cref="DispatchPair.IsSameLink"/>): its pairs run one after another on one thread, so they conflict
        /// with nothing among themselves. Islands share no dynamic body, which lets them share <see cref="PhasesUsed"/> and
        /// <see cref="LinksInPhases"/>, and a thread takes room in a bucket atomically. The order of islands in a bucket
        /// therefore varies from run to run, and the order of pairs within an island doesn't.
        /// </summary>
        [BurstCompile]
        unsafe struct SortIslandIntoBucketsJob : IJobParallelForDefer
        {
            const int AllPhases = (1 << NumIslandPhases) - 1;

            /// <summary> Plenty for the rare stuck link in a craft, and it bounds the worst case of a very large island. </summary>
            const int MaxChainStepsPerLink = 8;

            [ReadOnly]
            public NativeArray<DispatchPair> IslandPairs;

            [NativeDisableParallelForRestriction]
            public NativeArray<SolverIsland> Islands;

            [NativeDisableParallelForRestriction]
            public NativeArray<byte> PhasesUsed;

            /// <summary> For each dynamic body and phase, the link there, or -1. Valid only once an island has got stuck. </summary>
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

                // A joint goes direct however few colours its island needs: sweeps alone leave a stiff joint soft, and a craft
                // shouldn't behave differently for being small.
                var phases = new NativeArray<byte>(numLinks, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                bool fits = !hasJoint && CountCrowdedBodies(pairs, firstPairs, numLinks) == 0
                    && ColourLinks(pairs, firstPairs, numLinks, phases);

                int bucketSize = PhasedDispatchPairs.Length / NumIslandBuckets;
                if (!fits)
                {
                    // Only joints get the direct solver's Jacobians: its coupled solve doesn't cope with contacts on them, so
                    // DirectIslandJob gives those to the iterative solver around it.
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
            /// Counts the bodies on more links than there are phases. Links sharing a body can't share a phase, so one such body
            /// settles where the island goes before any colouring is tried. Leaves <see cref="PhasesUsed"/> clear.
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

            /// <summary> Returns 1 the moment a body goes past the number of phases. </summary>
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
            /// Gives every link a phase, or returns false. First fit does nearly all of it in one pass over the links. Only when
            /// it sticks does the island pay for <see cref="LinksInPhases"/> and swap colours along chains, on a budget that
            /// keeps the whole thing linear in the links: an island that uses it up has odd cycles and goes direct.
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
            /// Finds a phase free on the dynamic bodies a link touches, or returns -1 if there is none. A static body doesn't
            /// count: nothing writes to one, so any number of links can share it in a phase. A link with a static body takes the
            /// last phase it can, which leaves the last word on a body to what holds it against the world.
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

            /// <summary> Records the links coloured so far: those of earlier passes, and of this pass up to the stuck one. </summary>
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
            /// Frees a phase on both ends of a stuck link. Each end has a phase the other uses; the links alternating between those
            /// two phases from one end form a path, and swapping the two along it frees one on that end unless the path comes
            /// back to the other end, which takes an odd cycle. So this can't fail on a tree, but can on a craft closed by struts
            /// or contacts. Returns -1 if every choice of the two phases fails or the steps run out.
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

                        // Everything comes off before anything goes on: neighbours on the path trade the same two slots.
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

            /// <summary> Takes room in a bucket, and returns where the room starts. </summary>
            int Reserve(int bucket, int count)
            {
                ref int bucketCount = ref UnsafeUtility.ArrayElementAsRef<int>(BucketCounts.GetUnsafePtr(), bucket);
                return Interlocked.Add(ref bucketCount, count) - count;
            }

        }

        /// <summary>
        /// Tells the solver what ended up in the buckets. A bucket can be empty: <see cref="IterativeSolverSchedulerInfo.FindPhaseId"/>
        /// copes, as the next phase has the same offset.
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
                islands.BodiesInIslands.Resize(numDirectPairs > 0 ? NumDynamicBodies : 0, NativeArrayOptions.ClearMemory);
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
}
