using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Dynamics.Vessels.Structure;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    /// <summary>
    /// Persistent island forest pairs, locks, structure and factor arrays, owned by IslandSolverCache.
    /// Number moving bodies parent-first; BodyPart and PartBody translate indices.
    /// </summary>
    internal unsafe partial struct ForestIsland
    {
        public VesselStructure Structure;

        /// <summary>The island this was built for: its body count, then each forest block's bodies and role in order.</summary>
        public UnsafeList<ulong> Signature;

        public UnsafeList<VesselPart> Parts;
        public UnsafeList<VesselEdge> Edges;

        // Remembered final boundary states for the next solve; see BoundaryMemory.
        public BoundaryMemory BoundaryMemory;
        public UnsafeList<VesselClosure> Closures;

        /// <summary>Each forest block's link as link * 2 + 1 when its body A is the link's parent side, or one of the kinds below.</summary>
        public UnsafeList<int> BlockLink;

        /// <summary>A forest block with one moving side, whose rows go to its section's diagonal.</summary>
        public const int OneSided = -1;

        /// <summary>A forest block between two bodies of one section: a locked pair's, or an inert closure's.</summary>
        public const int InSection = -2;

        public UnsafeList<int> PartBody;
        public UnsafeList<int> BodyPart;

        /// <summary>Each moving body's section.</summary>
        public UnsafeList<int> BodySection;

        /// <summary>Each part's pair with its parent, or -1 for a root.</summary>
        public UnsafeList<int> PartPair;

        // Distinct forest body pairs with spanning roles, blocks, lock states and predicted spring deflections.
        public UnsafeList<int> BlockPair;
        public UnsafeList<int2> PairBodies;

        // Ordered body identities for matching prior pair lock states, stored as low/high halves.
        public UnsafeList<int4> PairWorldKey;
        public UnsafeList<byte> PairTree;
        public UnsafeList<byte> PairEdge;
        public UnsafeList<byte> PairLocked;
        public UnsafeList<float> PairDeflection;

        // Deflections summed over the current step's substeps; their mean represents the full step load.
        public UnsafeList<float> PairDeflectionSum;

        /// <summary>Each pair's deflection over the last whole step alone, which <see cref="PairDeflection"/> smooths.</summary>
        public UnsafeList<float> PairDeflectionStep;

        /// <summary>Whether either body of each pair had a contact row in the last solve, for diagnostics.</summary>
        public UnsafeList<byte> PairContact;

        /// <summary>Steps each pair has kept its lock state.</summary>
        public UnsafeList<int> PairHeld;

        /// <summary>The solves summed in <see cref="PairDeflectionSum"/>.</summary>
        public int DeflectionCount;
        public UnsafeList<int> PairBlockStart;

        // Cached locked-pair inverse in the child's frame, with compliance sum and row count.
        // Use NaN until computed and after unlocking.
        public UnsafeList<double> PairInverse;
        public UnsafeList<float> PairInverseKey;
        public UnsafeList<int> PairInverseRows;
        public UnsafeList<int> PairBlocks;

        public UnsafeList<VesselSection> Sections;
        public UnsafeList<double> Blocks;
        public UnsafeList<double> Inverse;
        public UnsafeList<double> Rhs;
        public UnsafeList<double> Solution;
        public UnsafeList<double> ExtraDiagonal;
        public UnsafeList<int> LinkRowStart;
        public UnsafeList<int> LinkRowList;
        public UnsafeList<int> Counters;

        public VesselContext Context;

        /// <summary>How many times the pairs, and how many times the structure, were built, for tests.</summary>
        public int Builds;

        public int StructureBuilds;

        // A vessel island's own storage: the kernel's per-link rows and per-part work arrays, and what the island keeps of
        // its vessels from one step to the next.
        public UnsafeList<VesselRows> LinkRows;
        public UnsafeList<double3> PartForce;
        public UnsafeList<double3> PartTorque;
        public UnsafeList<double3> ChildPoint;
        public UnsafeList<double> SectionResidual;
        public UnsafeList<VesselEdgeLoad> KernelEdgeLoads;
        public UnsafeList<VesselEdgeLoad> KernelClosureLoads;
        public UnsafeList<VesselEdgeState> NextState;
        public UnsafeList<ulong> VesselSignature;
        public UnsafeList<int> VesselPartStart;
        public UnsafeList<int> VesselClosureStart;
        public UnsafeList<byte> PartLocked;
        public UnsafeList<int> PartHeld;
        public UnsafeList<float> PartRatio;

        // Part and closure indices in the vessel set, with -1 for non-vessel bodies.
        public UnsafeList<int> PartSetPart;

        public UnsafeList<int> ClosureSetIndex;

        // First vessel-part offset for each island part and closure, or -1 outside vessels.
        public UnsafeList<int> PartVesselStart;

        public UnsafeList<int> ClosureVesselStart;

        // Per-section tasks and ranges for shared sections, with depth-first indices and force/torque prefix
        // sums.
        public UnsafeList<int> SmallSections;

        public UnsafeList<int> SharedSections;
        public UnsafeList<int> DepthFirstPart;
        public UnsafeList<int> DepthFirstPosition;
        public UnsafeList<int> SubtreeLast;
        public UnsafeList<double3> PrefixForce;
        public UnsafeList<double3> PrefixTorque;

        // Stable fused-island body numbering while vessels and bodies match.
        // Write its inverse into the shared compact index only while solving.
        public UnsafeList<int> FusedWorldBodies;

        /// <summary>
        /// The equilibrium systems of the sections that have closures inside them, kept while a section keeps its parts. The
        /// vessel kernel's context points here.
        /// </summary>
        public VesselEquilibrium Equilibrium;

        /// <summary>
        /// The forest path's equilibrium systems, one for each multi-part section with closures inside it, built from the rows
        /// for the structure build <see cref="ForestEquilibriumBuiltFor"/>.
        /// </summary>
        public UnsafeList<SectionEquilibrium> ForestEquilibrium;

        public int ForestEquilibriumBuiltFor;

        /// <summary>The time step and the switch that <see cref="Equilibrium"/> were last prepared with.</summary>
        public double EquilibriumTimeStep;

        public bool EquilibriumEnabled;

        // Imported vessel IDs, structure versions and lock states. On the next layout, prefer imported states
        // over this forest's older copies when TakesStates is set.
        public UnsafeList<ulong> TakenIds;

        public UnsafeList<uint> TakenVersions;
        public UnsafeList<int> TakenStarts;
        public UnsafeList<byte> TakenLocked;
        public UnsafeList<int> TakenHeld;
        public UnsafeList<float> TakenRatio;
        public bool TakesStates;

        /// <summary>Whether the structure was last built for vessels, rather than from handed-in rows.</summary>
        public bool VesselMode;

        /// <summary>The locked pairs and the sections with more than one body, as of the last structure.</summary>
        public int LockedPairs;

        public int MultiPartSections;

        public static ForestIsland* Create()
        {
            var forest = (ForestIsland*)UnsafeUtility.Malloc(sizeof(ForestIsland), 16, Allocator.Persistent);
            *forest = new ForestIsland
            {
                Structure = new VesselStructure(Allocator.Persistent),
                Signature = new UnsafeList<ulong>(16, Allocator.Persistent),
                Parts = new UnsafeList<VesselPart>(16, Allocator.Persistent),
                Edges = new UnsafeList<VesselEdge>(16, Allocator.Persistent),
                Closures = new UnsafeList<VesselClosure>(16, Allocator.Persistent),
                BlockLink = new UnsafeList<int>(16, Allocator.Persistent),
                PartBody = new UnsafeList<int>(16, Allocator.Persistent),
                BodyPart = new UnsafeList<int>(16, Allocator.Persistent),
                BodySection = new UnsafeList<int>(16, Allocator.Persistent),
                PartPair = new UnsafeList<int>(16, Allocator.Persistent),
                BlockPair = new UnsafeList<int>(16, Allocator.Persistent),
                PairBodies = new UnsafeList<int2>(16, Allocator.Persistent),
                PairWorldKey = new UnsafeList<int4>(16, Allocator.Persistent),
                PairTree = new UnsafeList<byte>(16, Allocator.Persistent),
                PairEdge = new UnsafeList<byte>(16, Allocator.Persistent),
                PairLocked = new UnsafeList<byte>(16, Allocator.Persistent),
                PairDeflection = new UnsafeList<float>(16, Allocator.Persistent),
                PairDeflectionSum = new UnsafeList<float>(16, Allocator.Persistent),
                PairDeflectionStep = new UnsafeList<float>(16, Allocator.Persistent),
                PairContact = new UnsafeList<byte>(16, Allocator.Persistent),
                PairHeld = new UnsafeList<int>(16, Allocator.Persistent),
                PairBlockStart = new UnsafeList<int>(16, Allocator.Persistent),
                PairInverse = new UnsafeList<double>(16, Allocator.Persistent),
                PairInverseKey = new UnsafeList<float>(16, Allocator.Persistent),
                PairInverseRows = new UnsafeList<int>(16, Allocator.Persistent),
                PairBlocks = new UnsafeList<int>(16, Allocator.Persistent),
                Sections = new UnsafeList<VesselSection>(16, Allocator.Persistent),
                Blocks = new UnsafeList<double>(16, Allocator.Persistent),
                Inverse = new UnsafeList<double>(16, Allocator.Persistent),
                Rhs = new UnsafeList<double>(16, Allocator.Persistent),
                Solution = new UnsafeList<double>(16, Allocator.Persistent),
                ExtraDiagonal = new UnsafeList<double>(16, Allocator.Persistent),
                LinkRowStart = new UnsafeList<int>(16, Allocator.Persistent),
                LinkRowList = new UnsafeList<int>(16, Allocator.Persistent),
                Counters = new UnsafeList<int>(VesselContext.CounterCount, Allocator.Persistent),
                LinkRows = new UnsafeList<VesselRows>(16, Allocator.Persistent),
                PartForce = new UnsafeList<double3>(16, Allocator.Persistent),
                PartTorque = new UnsafeList<double3>(16, Allocator.Persistent),
                ChildPoint = new UnsafeList<double3>(16, Allocator.Persistent),
                SectionResidual = new UnsafeList<double>(16, Allocator.Persistent),
                KernelEdgeLoads = new UnsafeList<VesselEdgeLoad>(16, Allocator.Persistent),
                KernelClosureLoads = new UnsafeList<VesselEdgeLoad>(16, Allocator.Persistent),
                NextState = new UnsafeList<VesselEdgeState>(16, Allocator.Persistent),
                VesselSignature = new UnsafeList<ulong>(16, Allocator.Persistent),
                VesselPartStart = new UnsafeList<int>(16, Allocator.Persistent),
                VesselClosureStart = new UnsafeList<int>(16, Allocator.Persistent),
                PartLocked = new UnsafeList<byte>(16, Allocator.Persistent),
                PartHeld = new UnsafeList<int>(16, Allocator.Persistent),
                PartRatio = new UnsafeList<float>(16, Allocator.Persistent),
                PartSetPart = new UnsafeList<int>(16, Allocator.Persistent),
                ClosureSetIndex = new UnsafeList<int>(16, Allocator.Persistent),
                PartVesselStart = new UnsafeList<int>(16, Allocator.Persistent),
                ClosureVesselStart = new UnsafeList<int>(16, Allocator.Persistent),
                SmallSections = new UnsafeList<int>(16, Allocator.Persistent),
                SharedSections = new UnsafeList<int>(16, Allocator.Persistent),
                DepthFirstPart = new UnsafeList<int>(16, Allocator.Persistent),
                DepthFirstPosition = new UnsafeList<int>(16, Allocator.Persistent),
                SubtreeLast = new UnsafeList<int>(16, Allocator.Persistent),
                PrefixForce = new UnsafeList<double3>(16, Allocator.Persistent),
                PrefixTorque = new UnsafeList<double3>(16, Allocator.Persistent),
                FusedWorldBodies = new UnsafeList<int>(16, Allocator.Persistent),
                Equilibrium = VesselEquilibrium.Create(Allocator.Persistent),
                ForestEquilibrium = new UnsafeList<SectionEquilibrium>(0, Allocator.Persistent),
                ForestEquilibriumBuiltFor = -1,
                TakenIds = new UnsafeList<ulong>(4, Allocator.Persistent),
                TakenVersions = new UnsafeList<uint>(4, Allocator.Persistent),
                TakenStarts = new UnsafeList<int>(4, Allocator.Persistent),
                TakenLocked = new UnsafeList<byte>(16, Allocator.Persistent),
                TakenHeld = new UnsafeList<int>(16, Allocator.Persistent),
                TakenRatio = new UnsafeList<float>(16, Allocator.Persistent),
            };
            forest->Counters.Resize(VesselContext.CounterCount, NativeArrayOptions.ClearMemory);
            forest->Signature.Add(ulong.MaxValue);
            return forest;
        }

        public static void Destroy(ForestIsland* forest)
        {
            if (forest == null)
            {
                return;
            }

            forest->Structure.Dispose();
            forest->Signature.Dispose();
            forest->Parts.Dispose();
            forest->Edges.Dispose();
            forest->Closures.Dispose();
            forest->BlockLink.Dispose();
            forest->PartBody.Dispose();
            forest->BodyPart.Dispose();
            forest->BodySection.Dispose();
            forest->PartPair.Dispose();
            forest->BlockPair.Dispose();
            forest->PairBodies.Dispose();
            forest->PairWorldKey.Dispose();
            forest->PairTree.Dispose();
            forest->PairEdge.Dispose();
            forest->PairLocked.Dispose();
            forest->PairDeflection.Dispose();
            forest->PairDeflectionSum.Dispose();
            forest->PairDeflectionStep.Dispose();
            forest->PairContact.Dispose();
            forest->PairHeld.Dispose();
            forest->PairBlockStart.Dispose();
            forest->BoundaryMemory.Dispose();
            forest->PairInverse.Dispose();
            forest->PairInverseKey.Dispose();
            forest->PairInverseRows.Dispose();
            forest->PairBlocks.Dispose();
            forest->Sections.Dispose();
            forest->Blocks.Dispose();
            forest->Inverse.Dispose();
            forest->Rhs.Dispose();
            forest->Solution.Dispose();
            forest->ExtraDiagonal.Dispose();
            forest->LinkRowStart.Dispose();
            forest->LinkRowList.Dispose();
            forest->Counters.Dispose();
            forest->LinkRows.Dispose();
            forest->PartForce.Dispose();
            forest->PartTorque.Dispose();
            forest->ChildPoint.Dispose();
            forest->SectionResidual.Dispose();
            forest->KernelEdgeLoads.Dispose();
            forest->KernelClosureLoads.Dispose();
            forest->NextState.Dispose();
            forest->VesselSignature.Dispose();
            forest->VesselPartStart.Dispose();
            forest->VesselClosureStart.Dispose();
            forest->PartLocked.Dispose();
            forest->PartHeld.Dispose();
            forest->PartRatio.Dispose();
            forest->PartSetPart.Dispose();
            forest->ClosureSetIndex.Dispose();
            forest->PartVesselStart.Dispose();
            forest->ClosureVesselStart.Dispose();
            forest->SmallSections.Dispose();
            forest->SharedSections.Dispose();
            forest->DepthFirstPart.Dispose();
            forest->DepthFirstPosition.Dispose();
            forest->SubtreeLast.Dispose();
            forest->PrefixForce.Dispose();
            forest->PrefixTorque.Dispose();
            forest->FusedWorldBodies.Dispose();
            forest->Equilibrium.Dispose();
            for (int e = 0; e < forest->ForestEquilibrium.Length; e++)
            {
                forest->ForestEquilibrium[e].Dispose();
            }

            forest->ForestEquilibrium.Dispose();
            forest->TakenIds.Dispose();
            forest->TakenVersions.Dispose();
            forest->TakenStarts.Dispose();
            forest->TakenLocked.Dispose();
            forest->TakenHeld.Dispose();
            forest->TakenRatio.Dispose();
            UnsafeUtility.Free(forest, Allocator.Persistent);
        }

        public int SectionCount => Structure.Header[VesselStructure.SectionCount];
        public int LinkCount => Structure.Header[VesselStructure.LinkCount];
        public int PairCount => PairBodies.Length;

        /// <summary>
        /// Build distinct forest body pairs and a spanning forest, then number parts parent-first.
        /// Preserve prior lock states by supplied body identities, never by reusable world indices.
        /// </summary>
        public void BuildPairs(in SparseConstraintSolver.Problem problem, int* forestBlocks, bool* treeBlocks, int forestBlockCount,
            ulong* identities)
        {
            Builds++;
            int bodies = problem.BodyCount;

            var previous = new NativeHashMap<int4, int>(PairWorldKey.Length, Allocator.Temp);
            for (int p = 0; p < PairWorldKey.Length; p++)
            {
                previous.TryAdd(PairWorldKey[p], p);
            }

            var oldLocked = new NativeArray<byte>(PairLocked.Length, Allocator.Temp);
            var oldDeflection = new NativeArray<float>(PairDeflection.Length, Allocator.Temp);
            var oldHeld = new NativeArray<int>(PairHeld.Length, Allocator.Temp);
            for (int p = 0; p < PairLocked.Length; p++)
            {
                oldLocked[p] = PairLocked[p];
                oldDeflection[p] = PairDeflection[p];
                oldHeld[p] = PairHeld[p];
            }

            // Distinct pairs, in block order; tree joints first take part in the spanning forest.
            var pairOf = new NativeHashMap<long, int>(forestBlockCount, Allocator.Temp);
            PairBodies.Clear();
            PairTree.Clear();
            BlockPair.Resize(forestBlockCount);
            for (int f = 0; f < forestBlockCount; f++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[forestBlocks[f]];
                if (block.BodyA < 0 || block.BodyB < 0 || block.BodyA == block.BodyB)
                {
                    BlockPair[f] = -1;
                    continue;
                }

                int low = math.min(block.BodyA, block.BodyB), high = math.max(block.BodyA, block.BodyB);
                long key = ((long)low << 32) | (uint)high;
                if (!pairOf.TryGetValue(key, out int pair))
                {
                    pair = PairBodies.Length;
                    pairOf.Add(key, pair);
                    PairBodies.Add(new int2(low, high));
                    PairTree.Add(0);
                }

                PairTree[pair] |= (byte)(treeBlocks[f] ? 1 : 0);
                BlockPair[f] = pair;
            }

            int pairCount = PairBodies.Length;
            PairBlockStart.Resize(pairCount + 1);
            UnsafeUtility.MemClear(PairBlockStart.Ptr, (pairCount + 1) * (long)sizeof(int));
            for (int f = 0; f < forestBlockCount; f++)
            {
                if (BlockPair[f] >= 0)
                {
                    PairBlockStart[BlockPair[f] + 1]++;
                }
            }

            for (int p = 0; p < pairCount; p++)
            {
                PairBlockStart[p + 1] += PairBlockStart[p];
            }

            PairBlocks.Resize(math.max(1, PairBlockStart[pairCount]));
            var pairFill = new NativeArray<int>(math.max(1, pairCount), Allocator.Temp);
            for (int f = 0; f < forestBlockCount; f++)
            {
                int pair = BlockPair[f];
                if (pair >= 0)
                {
                    PairBlocks[PairBlockStart[pair] + pairFill[pair]++] = f;
                }
            }

            // Kruskal over the tree pairs in order, then over the rest, so that a loop of tree joints still gives a tree.
            var root = new NativeArray<int>(bodies, Allocator.Temp);
            for (int i = 0; i < bodies; i++)
            {
                root[i] = i;
            }

            PairEdge.Resize(pairCount);
            for (int pass = 0; pass < 2; pass++)
            {
                for (int p = 0; p < pairCount; p++)
                {
                    if (pass == 0)
                    {
                        PairEdge[p] = 0;
                    }

                    if ((PairTree[p] != 0) != (pass == 0))
                    {
                        continue;
                    }

                    int a = Find(root, PairBodies[p].x), b = Find(root, PairBodies[p].y);
                    if (a != b)
                    {
                        root[a] = b;
                        PairEdge[p] = 1;
                    }
                }
            }

            // Orient the trees and number the parts parents first: each body's parent is its neighbour toward the first
            // body reached.
            var adjacencyStart = new NativeArray<int>(bodies + 1, Allocator.Temp);
            for (int p = 0; p < pairCount; p++)
            {
                if (PairEdge[p] != 0)
                {
                    adjacencyStart[PairBodies[p].x + 1]++;
                    adjacencyStart[PairBodies[p].y + 1]++;
                }
            }

            for (int i = 0; i < bodies; i++)
            {
                adjacencyStart[i + 1] += adjacencyStart[i];
            }

            var adjacency = new NativeArray<int>(math.max(1, adjacencyStart[bodies]), Allocator.Temp);
            var fill = new NativeArray<int>(bodies, Allocator.Temp);
            for (int p = 0; p < pairCount; p++)
            {
                if (PairEdge[p] != 0)
                {
                    int2 pb = PairBodies[p];
                    adjacency[adjacencyStart[pb.x] + fill[pb.x]++] = pb.y;
                    adjacency[adjacencyStart[pb.y] + fill[pb.y]++] = pb.x;
                }
            }

            PartBody.Resize(bodies);
            BodyPart.Resize(bodies);
            Parts.Resize(bodies);
            Edges.Resize(bodies);
            var parentBody = new NativeArray<int>(bodies, Allocator.Temp);
            var seen = new NativeArray<bool>(bodies, Allocator.Temp);
            int parts = 0;
            for (int start = 0; start < bodies; start++)
            {
                if (seen[start])
                {
                    continue;
                }

                seen[start] = true;
                parentBody[start] = -1;
                int first = parts;
                PartBody[parts] = start;
                BodyPart[start] = parts++;
                for (int k = first; k < parts; k++)
                {
                    int v = PartBody[k];
                    for (int e = adjacencyStart[v]; e < adjacencyStart[v + 1]; e++)
                    {
                        int w = adjacency[e];
                        if (!seen[w])
                        {
                            seen[w] = true;
                            parentBody[w] = v;
                            PartBody[parts] = w;
                            BodyPart[w] = parts++;
                        }
                    }
                }
            }

            for (int k = 0; k < bodies; k++)
            {
                int body = PartBody[k];
                Parts[k] = new VesselPart { Mass = problem.Bodies[body].Mass, Inertia = 1.0 };
                Edges[k] = new VesselEdge
                {
                    Parent = parentBody[body] >= 0 ? BodyPart[parentBody[body]] : -1,
                    State = VesselEdgeState.Yielding,
                };
            }

            PartPair.Resize(bodies);
            for (int k = 0; k < bodies; k++)
            {
                PartPair[k] = -1;
            }

            for (int p = 0; p < pairCount; p++)
            {
                if (PairEdge[p] != 0)
                {
                    int2 pb = PairBodies[p];
                    PartPair[parentBody[pb.x] == pb.y ? BodyPart[pb.x] : BodyPart[pb.y]] = p;
                }
            }

            // Lock states carried over by the identities of the pairs' bodies.
            PairWorldKey.Resize(pairCount);
            PairLocked.Resize(pairCount);
            PairDeflection.Resize(pairCount);
            PairDeflectionSum.Resize(pairCount);
            PairDeflectionStep.Resize(pairCount);
            PairContact.Resize(pairCount);
            for (int p = 0; p < pairCount; p++)
            {
                PairDeflectionStep[p] = float.PositiveInfinity;
                PairContact[p] = 0;
            }
            PairHeld.Resize(pairCount);
            UnsafeUtility.MemClear(PairDeflectionSum.Ptr, pairCount * (long)sizeof(float));
            DeflectionCount = 0;
            PairInverse.Resize(36 * math.max(1, pairCount));
            PairInverseKey.Resize(math.max(1, pairCount));
            PairInverseRows.Resize(math.max(1, pairCount));
            for (int p = 0; p < pairCount; p++)
            {
                PairInverseKey[p] = float.NaN;
            }

            for (int p = 0; p < pairCount; p++)
            {
                int2 pb = PairBodies[p];
                ulong low = math.min(identities[pb.x], identities[pb.y]), high = math.max(identities[pb.x], identities[pb.y]);
                var key = new int4((int)(uint)low, (int)(uint)(low >> 32), (int)(uint)high, (int)(uint)(high >> 32));
                PairWorldKey[p] = key;
                if (previous.TryGetValue(key, out int old))
                {
                    PairLocked[p] = oldLocked[old];
                    PairDeflection[p] = oldDeflection[old];
                    PairHeld[p] = oldHeld[old];
                }
                else
                {
                    PairLocked[p] = 0;
                    PairDeflection[p] = float.PositiveInfinity;
                    PairHeld[p] = int.MaxValue / 2;
                }
            }

            previous.Dispose();
            oldLocked.Dispose();
            oldDeflection.Dispose();
            oldHeld.Dispose();
            pairOf.Dispose();
            pairFill.Dispose();
            root.Dispose();
            adjacencyStart.Dispose();
            adjacency.Dispose();
            fill.Dispose();
            parentBody.Dispose();
            seen.Dispose();
        }

        /// <summary>
        /// Rebuild rigid sections from pairs and locks. Merge locked children into parents and omit internal
        /// closures.
        /// </summary>
        public void BuildStructure(in SparseConstraintSolver.Problem problem, int* forestBlocks, int forestBlockCount)
        {
            StructureBuilds++;
            int bodies = problem.BodyCount;
            int pairCount = PairCount;
            for (int k = 0; k < bodies; k++)
            {
                VesselEdge edge = Edges[k];
                edge.State = VesselEdgeState.Yielding;
                Edges[k] = edge;
            }

            LockedPairs = 0;
            Closures.Clear();
            for (int p = 0; p < pairCount; p++)
            {
                if (PairLocked[p] == 0)
                {
                    PairInverseKey[p] = float.NaN;
                }
            }

            var pairClosure = new NativeArray<int>(math.max(1, pairCount), Allocator.Temp);
            for (int p = 0; p < pairCount; p++)
            {
                pairClosure[p] = -1;
                int2 pb = PairBodies[p];
                if (PairEdge[p] != 0)
                {
                    if (PairLocked[p] != 0)
                    {
                        int child = Edges[BodyPart[pb.x]].Parent == BodyPart[pb.y] ? BodyPart[pb.x] : BodyPart[pb.y];
                        VesselEdge edge = Edges[child];
                        edge.State = VesselEdgeState.Locked;
                        Edges[child] = edge;
                        LockedPairs++;
                    }
                }
                else
                {
                    pairClosure[p] = Closures.Length;
                    Closures.Add(new VesselClosure { PartA = BodyPart[pb.x], PartB = BodyPart[pb.y] });
                }
            }

            fixed (VesselStructure* structure = &Structure)
            {
                new BuildVesselStructureJob
                {
                    Parts = View(Parts.Ptr, Parts.Length),
                    Edges = View(Edges.Ptr, Edges.Length),
                    Closures = View(Closures.Ptr, Closures.Length),
                    HubClosures = 16,
                    ChunkTarget = 96,
                    MinimumDegree = true,
                    Structure = structure,
                }.Execute();
            }

            BodySection.Resize(bodies);
            for (int i = 0; i < bodies; i++)
            {
                BodySection[i] = Structure.PartSection[BodyPart[i]];
            }

            // Each forest block's link, and which of its bodies is the link's parent side.
            BlockLink.Resize(forestBlockCount);
            for (int f = 0; f < forestBlockCount; f++)
            {
                int pair = BlockPair[f];
                if (pair < 0)
                {
                    BlockLink[f] = OneSided;
                    continue;
                }

                int2 pb = PairBodies[pair];
                int link;
                if (PairEdge[pair] != 0)
                {
                    int child = Edges[BodyPart[pb.x]].Parent == BodyPart[pb.y] ? BodyPart[pb.x] : BodyPart[pb.y];
                    link = Structure.PartLink[child];
                }
                else
                {
                    link = Structure.ClosureLink[pairClosure[pair]];
                }

                if (link < 0)
                {
                    BlockLink[f] = InSection;
                    continue;
                }

                VesselLink linked = Structure.Links[link];
                BlockLink[f] = link * 2 + (BodyPart[problem.Blocks[forestBlocks[f]].BodyA] == linked.ParentPart ? 1 : 0);
            }

            pairClosure.Dispose();

            int sections = SectionCount;
            MultiPartSections = 0;
            for (int s = 0; s < sections; s++)
            {
                MultiPartSections += Structure.SectionPartStart[s + 1] - Structure.SectionPartStart[s] > 1 ? 1 : 0;
            }

            int blocks = sections + Structure.Header[VesselStructure.OffDiagonalCount]
                + Structure.Header[VesselStructure.BufferBlockCount];
            int rhs = sections + Structure.Header[VesselStructure.BufferRhsCount];
            Sections.Resize(sections);
            Blocks.Resize(Block6x6.Size * blocks);
            Inverse.Resize(Block6x6.Size * sections);
            Rhs.Resize(6 * rhs);
            Solution.Resize(6 * sections);
            ExtraDiagonal.Resize(Block6x6.Size * sections);
            LinkRowStart.Resize(LinkCount + 1);

            Context = new VesselContext
            {
                Sections = Sections.Ptr,
                Blocks = Blocks.Ptr,
                Inverse = Inverse.Ptr,
                Rhs = Rhs.Ptr,
                Solution = Solution.Ptr,
                ExtraDiagonal = ExtraDiagonal.Ptr,
                Counters = Counters.Ptr,
                PartSection = Structure.PartSection.Ptr,
                SectionPartStart = Structure.SectionPartStart.Ptr,
                SectionParts = Structure.SectionParts.Ptr,
                ColumnStart = Structure.ColumnStart.Ptr,
                Rows = Structure.Rows.Ptr,
                UpdateStart = Structure.UpdateStart.Ptr,
                UpdateTarget = Structure.UpdateTarget.Ptr,
                RhsTarget = Structure.RhsTarget.Ptr,
                LinkStart = Structure.LinkStart.Ptr,
                Links = Structure.Links.Ptr,
                IncidentStart = Structure.IncidentStart.Ptr,
                Incident = Structure.Incident.Ptr,
                PartLink = Structure.PartLink.Ptr,
                ChunkStart = Structure.ChunkStart.Ptr,
                ChunkColumns = Structure.ChunkColumns.Ptr,
                TopColumns = Structure.TopColumns.Ptr,
                ChunkBufferStart = Structure.ChunkBufferStart.Ptr,
                BufferTarget = Structure.BufferTarget.Ptr,
                ChunkRhsStart = Structure.ChunkRhsStart.Ptr,
                RhsBufferTarget = Structure.RhsBufferTarget.Ptr,
                EliminationParent = Structure.EliminationParent.Ptr,
                SectionCount = sections,
                OffDiagonalCount = Structure.Header[VesselStructure.OffDiagonalCount],
                ChunkCount = Structure.Header[VesselStructure.ChunkCount],
                TopCount = Structure.Header[VesselStructure.TopCount],
                MaxStructure = Structure.Header[VesselStructure.MaxStructure],
                FactorOnly = true,
            };
        }

        static int Find(NativeArray<int> root, int v)
        {
            while (root[v] != v)
            {
                root[v] = root[root[v]];
                v = root[v];
            }

            return v;
        }

        static NativeArray<T> View<T>(T* data, int length) where T : unmanaged
        {
            var array = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>(data, length, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref array, AtomicSafetyHandle.GetTempMemoryHandle());
#endif
            return array;
        }

        /// <summary>
        /// Return a stable backend body identity using its unique entity version (the body serial).
        /// Ignore entity indices, which move with store rows. Fall back to world indices when no entity is
        /// available.
        /// </summary>
        internal static ulong BodyIdentity(RigidBody* bodies, int bodyCount, int worldBody)
        {
            if (bodies == null || (uint)worldBody >= (uint)bodyCount)
            {
                return (uint)worldBody;
            }

            int version = bodies[worldBody].Entity.Version;
            return version != 0 ? (ulong)(uint)version << 32 : (uint)worldBody;
        }
    }
}
