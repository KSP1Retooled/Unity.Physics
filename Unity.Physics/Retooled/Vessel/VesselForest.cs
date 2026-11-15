using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics;

namespace Unity.Physics
{
    /// <summary>How the direct solve may treat a joint's rows, see <see cref="Joint.ForestRole"/>.</summary>
    public enum JointForestRole : byte
    {
        /// <summary>A row like any other: solved with the island's contacts and limits.</summary>
        None,

        /// <summary>An edge of the vessel forest's spanning tree.</summary>
        Tree,

        /// <summary>A forest joint that closes a loop, such as a strut.</summary>
        Closure,
    }

    /// <summary>
    /// Settings for solving direct islands as a forest plus a boundary: the unbounded rows of forest joints (see
    /// <see cref="Joint.ForestRole"/>) are eliminated structurally by the vessel section solver, and every other row of the
    /// island (contacts, limits, motors with a force limit, other joints) forms a dense boundary block solved against that
    /// factor. An island with no forest rows, or with more boundary rows than <see cref="BoundaryRowCap"/>, is solved by
    /// <see cref="SparseConstraintSolver"/> as before.
    /// </summary>
    public struct ForestSettings
    {
        public bool Enabled;
        public int BoundaryRowCap;

        public static ForestSettings Default => new ForestSettings { Enabled = false, BoundaryRowCap = 256 };
    }

    /// <summary>
    /// What one direct island keeps of its forest from one solve to the next: the structure, rebuilt only when the island's
    /// forest joints or bodies change, and the arrays the factor works in. Lives in unmanaged memory owned by
    /// <see cref="DirectSolverCache"/>, so that the jobs can build and use it.
    /// </summary>
    internal unsafe struct ForestIsland
    {
        public VesselStructure Structure;

        /// <summary>The island this was built for: its body count, then each forest block's bodies and role in order.</summary>
        public UnsafeList<ulong> Signature;

        public UnsafeList<VesselPart> Parts;
        public UnsafeList<VesselEdge> Edges;
        public UnsafeList<VesselClosure> Closures;

        /// <summary>Each forest block's link as link * 2 + 1 when its body A is the link's parent side, or -1 for none.</summary>
        public UnsafeList<int> BlockLink;

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

        /// <summary>How many times the structure was built, for tests.</summary>
        public int Builds;

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
                Sections = new UnsafeList<VesselSection>(16, Allocator.Persistent),
                Blocks = new UnsafeList<double>(16, Allocator.Persistent),
                Inverse = new UnsafeList<double>(16, Allocator.Persistent),
                Rhs = new UnsafeList<double>(16, Allocator.Persistent),
                Solution = new UnsafeList<double>(16, Allocator.Persistent),
                ExtraDiagonal = new UnsafeList<double>(16, Allocator.Persistent),
                LinkRowStart = new UnsafeList<int>(16, Allocator.Persistent),
                LinkRowList = new UnsafeList<int>(16, Allocator.Persistent),
                Counters = new UnsafeList<int>(VesselContext.CounterCount, Allocator.Persistent),
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
            forest->Sections.Dispose();
            forest->Blocks.Dispose();
            forest->Inverse.Dispose();
            forest->Rhs.Dispose();
            forest->Solution.Dispose();
            forest->ExtraDiagonal.Dispose();
            forest->LinkRowStart.Dispose();
            forest->LinkRowList.Dispose();
            forest->Counters.Dispose();
            UnsafeUtility.Free(forest, Allocator.Persistent);
        }

        public int SectionCount => Structure.Header[VesselStructure.SectionCount];
        public int LinkCount => Structure.Header[VesselStructure.LinkCount];

        /// <summary>
        /// Rebuilds the structure for an island whose forest blocks are <paramref name="forestBlocks"/>: one part per moving
        /// body, a tree edge for each body pair that a tree joint joins first without closing a loop, and a closure for every
        /// other pair. Pairs are counted once however many joints and rows they have.
        /// </summary>
        public void Build(in SparseConstraintSolver.Problem problem, int* forestBlocks, bool* treeBlocks, int forestBlockCount)
        {
            Builds++;
            int bodies = problem.BodyCount;
            Parts.Resize(bodies);
            Edges.Resize(bodies);
            for (int i = 0; i < bodies; i++)
            {
                Parts[i] = new VesselPart { Mass = problem.Bodies[i].Mass, Inertia = 1.0 };
                Edges[i] = new VesselEdge { Parent = -1, State = VesselEdgeState.Yielding };
            }

            // Distinct pairs, in block order; tree joints first take part in the spanning forest.
            var pairOf = new NativeHashMap<long, int>(forestBlockCount, Allocator.Temp);
            var pairA = new NativeList<int>(forestBlockCount, Allocator.Temp);
            var pairB = new NativeList<int>(forestBlockCount, Allocator.Temp);
            var pairTree = new NativeList<bool>(forestBlockCount, Allocator.Temp);
            var blockPair = new NativeArray<int>(forestBlockCount, Allocator.Temp);
            for (int f = 0; f < forestBlockCount; f++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[forestBlocks[f]];
                if (block.BodyA < 0 || block.BodyB < 0 || block.BodyA == block.BodyB)
                {
                    blockPair[f] = -1;
                    continue;
                }

                int low = math.min(block.BodyA, block.BodyB), high = math.max(block.BodyA, block.BodyB);
                long key = ((long)low << 32) | (uint)high;
                if (!pairOf.TryGetValue(key, out int pair))
                {
                    pair = pairA.Length;
                    pairOf.Add(key, pair);
                    pairA.Add(low);
                    pairB.Add(high);
                    pairTree.Add(false);
                }

                pairTree[pair] |= treeBlocks[f];
                blockPair[f] = pair;
            }

            // Kruskal over the tree pairs in order, then over the rest, so that a loop of tree joints still gives a tree.
            var root = new NativeArray<int>(bodies, Allocator.Temp);
            for (int i = 0; i < bodies; i++)
            {
                root[i] = i;
            }

            int pairCount = pairA.Length;
            var isEdge = new NativeArray<bool>(pairCount, Allocator.Temp);
            for (int pass = 0; pass < 2; pass++)
            {
                for (int p = 0; p < pairCount; p++)
                {
                    if (pairTree[p] != (pass == 0))
                    {
                        continue;
                    }

                    int a = Find(root, pairA[p]), b = Find(root, pairB[p]);
                    if (a != b)
                    {
                        root[a] = b;
                        isEdge[p] = true;
                    }
                }
            }

            // Orient the trees: each body's parent is its neighbour toward the first body reached.
            var adjacencyStart = new NativeArray<int>(bodies + 1, Allocator.Temp);
            for (int p = 0; p < pairCount; p++)
            {
                if (isEdge[p])
                {
                    adjacencyStart[pairA[p] + 1]++;
                    adjacencyStart[pairB[p] + 1]++;
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
                if (isEdge[p])
                {
                    adjacency[adjacencyStart[pairA[p]] + fill[pairA[p]]++] = pairB[p];
                    adjacency[adjacencyStart[pairB[p]] + fill[pairB[p]]++] = pairA[p];
                }
            }

            var seen = new NativeArray<bool>(bodies, Allocator.Temp);
            var queue = new NativeList<int>(bodies, Allocator.Temp);
            for (int start = 0; start < bodies; start++)
            {
                if (seen[start])
                {
                    continue;
                }

                seen[start] = true;
                queue.Clear();
                queue.Add(start);
                for (int k = 0; k < queue.Length; k++)
                {
                    int v = queue[k];
                    for (int e = adjacencyStart[v]; e < adjacencyStart[v + 1]; e++)
                    {
                        int w = adjacency[e];
                        if (!seen[w])
                        {
                            seen[w] = true;
                            VesselEdge edge = Edges[w];
                            edge.Parent = v;
                            Edges[w] = edge;
                            queue.Add(w);
                        }
                    }
                }
            }

            Closures.Clear();
            var pairClosure = new NativeArray<int>(pairCount, Allocator.Temp);
            for (int p = 0; p < pairCount; p++)
            {
                pairClosure[p] = -1;
                if (!isEdge[p])
                {
                    pairClosure[p] = Closures.Length;
                    Closures.Add(new VesselClosure { PartA = pairA[p], PartB = pairB[p] });
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

            // Each forest block's link, and which of its bodies is the link's parent side.
            BlockLink.Resize(forestBlockCount);
            for (int f = 0; f < forestBlockCount; f++)
            {
                int pair = blockPair[f];
                if (pair < 0)
                {
                    BlockLink[f] = -1;
                    continue;
                }

                int link;
                if (isEdge[pair])
                {
                    int child = Edges[pairA[pair]].Parent == pairB[pair] ? pairA[pair] : pairB[pair];
                    link = Structure.PartLink[child];
                }
                else
                {
                    link = Structure.ClosureLink[pairClosure[pair]];
                }

                VesselLink linked = Structure.Links[link];
                BlockLink[f] = link * 2 + (problem.Blocks[forestBlocks[f]].BodyA == linked.ParentPart ? 1 : 0);
            }

            int sections = SectionCount;
            int blocks = sections + Structure.Header[VesselStructure.OffDiagonalCount]
                + Structure.Header[VesselStructure.BufferBlockCount];
            int rhs = sections + Structure.Header[VesselStructure.BufferRhsCount];
            Sections.Resize(sections);
            Blocks.Resize(VesselBlocks.Size * blocks);
            Inverse.Resize(VesselBlocks.Size * sections);
            Rhs.Resize(6 * rhs);
            Solution.Resize(6 * sections);
            ExtraDiagonal.Resize(VesselBlocks.Size * sections);
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
    }
}
