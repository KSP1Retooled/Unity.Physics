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
    /// What the joint's owner allows the forest to do with a forest joint's body pair, see <see cref="Joint.ForestLock"/>.
    /// The owner reads the loads that count against the joint's break limits; the forest adds its own test of how far the
    /// joint's springs would deflect.
    /// </summary>
    [Flags]
    public enum JointForestLock : byte
    {
        None = 0,

        /// <summary>The joint's load is low enough for its pair to lock.</summary>
        Lockable = 1 << 0,

        /// <summary>The joint's load is low enough for a locked pair to stay locked.</summary>
        Holdable = 1 << 1,
    }

    /// <summary>
    /// Settings for solving direct islands as a forest plus a boundary: the unbounded rows of forest joints (see
    /// <see cref="Joint.ForestRole"/>) are eliminated structurally by the vessel section solver, and every other row of the
    /// island (contacts, limits, motors with a force limit, other joints) forms a dense boundary block solved against that
    /// factor. An island with no forest rows, or with more boundary rows than <see cref="BoundaryRowCap"/>, is solved by
    /// <see cref="SparseConstraintSolver"/> as before.
    ///
    /// With <see cref="Locking"/>, a tree edge whose joints allow it (<see cref="JointForestLock"/>) and whose springs
    /// would deflect less than <see cref="LockAngle"/> under its last load locks: its rows leave the solve and its two
    /// bodies move as one rigid section, its load coming from inverse dynamics. It yields again past
    /// <see cref="YieldAngle"/>.
    /// </summary>
    public struct ForestSettings
    {
        public bool Enabled;
        public int BoundaryRowCap;
        public bool Locking;

        /// <summary>Radians: an edge whose springs would deflect less than this under its last load may lock.</summary>
        public float LockAngle;

        /// <summary>Radians: a locked edge whose load would deflect its springs more than this yields.</summary>
        public float YieldAngle;

        public const float DefaultLockAngle = 0.5e-3f;
        public const float DefaultYieldAngle = 2e-3f;

        public static ForestSettings Default => new ForestSettings
        {
            Enabled = false,
            BoundaryRowCap = 256,
            Locking = false,
            LockAngle = DefaultLockAngle,
            YieldAngle = DefaultYieldAngle,
        };
    }

    /// <summary>
    /// What one direct island keeps of its forest from one solve to the next: its body pairs and their lock states, the
    /// structure, rebuilt only when the island's forest joints, bodies or lock set change, and the arrays the factor works
    /// in. Lives in unmanaged memory owned by <see cref="DirectSolverCache"/>, so that the jobs can build and use it.
    ///
    /// Parts are the island's moving bodies, numbered parents first along the spanning trees (the structure needs that
    /// order); <see cref="BodyPart"/> and <see cref="PartBody"/> translate.
    /// </summary>
    internal unsafe struct ForestIsland
    {
        public VesselStructure Structure;

        /// <summary>The island this was built for: its body count, then each forest block's bodies and role in order.</summary>
        public UnsafeList<ulong> Signature;

        public UnsafeList<VesselPart> Parts;
        public UnsafeList<VesselEdge> Edges;
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

        // The island's distinct body pairs among its forest blocks, kept while the signature holds: their bodies, whether
        // a tree joint joins them, whether they are an edge of the spanning forest (else a closure), the forest blocks of
        // each, their lock state and the deflection their springs would take under their last load.
        public UnsafeList<int> BlockPair;
        public UnsafeList<int2> PairBodies;
        public UnsafeList<ulong> PairWorldKey;
        public UnsafeList<byte> PairTree;
        public UnsafeList<byte> PairEdge;
        public UnsafeList<byte> PairLocked;
        public UnsafeList<float> PairDeflection;

        /// <summary>
        /// Each pair's deflections summed over the substeps of the step so far. A step's external forces reach the
        /// velocities once, before its first substep, so a locked pair carries them all in that substep and none after; the
        /// mean over the step is the deflection of the load the step applied.
        /// </summary>
        public UnsafeList<float> PairDeflectionSum;

        /// <summary>The solves summed in <see cref="PairDeflectionSum"/>.</summary>
        public int DeflectionCount;
        public UnsafeList<int> PairBlockStart;
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
                PairWorldKey = new UnsafeList<ulong>(16, Allocator.Persistent),
                PairTree = new UnsafeList<byte>(16, Allocator.Persistent),
                PairEdge = new UnsafeList<byte>(16, Allocator.Persistent),
                PairLocked = new UnsafeList<byte>(16, Allocator.Persistent),
                PairDeflection = new UnsafeList<float>(16, Allocator.Persistent),
                PairDeflectionSum = new UnsafeList<float>(16, Allocator.Persistent),
                PairBlockStart = new UnsafeList<int>(16, Allocator.Persistent),
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
            forest->PairBlockStart.Dispose();
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
            UnsafeUtility.Free(forest, Allocator.Persistent);
        }

        public int SectionCount => Structure.Header[VesselStructure.SectionCount];
        public int LinkCount => Structure.Header[VesselStructure.LinkCount];
        public int PairCount => PairBodies.Length;

        /// <summary>
        /// Finds the island's body pairs among its forest blocks <paramref name="forestBlocks"/> and the spanning forest over
        /// them: a tree edge for each pair that a tree joint joins first without closing a loop, and a closure for every
        /// other pair. Pairs are counted once however many joints and rows they have. Numbers the parts parents first. A
        /// pair that was here before keeps its lock state, found by its bodies' numbers in the world.
        /// </summary>
        public void BuildPairs(in SparseConstraintSolver.Problem problem, int* forestBlocks, bool* treeBlocks, int forestBlockCount,
            int* worldBodies)
        {
            Builds++;
            int bodies = problem.BodyCount;

            var previous = new NativeHashMap<ulong, int>(PairWorldKey.Length, Allocator.Temp);
            for (int p = 0; p < PairWorldKey.Length; p++)
            {
                previous.TryAdd(PairWorldKey[p], p);
            }

            var oldLocked = new NativeArray<byte>(PairLocked.Length, Allocator.Temp);
            var oldDeflection = new NativeArray<float>(PairDeflection.Length, Allocator.Temp);
            for (int p = 0; p < PairLocked.Length; p++)
            {
                oldLocked[p] = PairLocked[p];
                oldDeflection[p] = PairDeflection[p];
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

            // Lock states carried over by the pairs' bodies in the world.
            PairWorldKey.Resize(pairCount);
            PairLocked.Resize(pairCount);
            PairDeflection.Resize(pairCount);
            PairDeflectionSum.Resize(pairCount);
            UnsafeUtility.MemClear(PairDeflectionSum.Ptr, pairCount * (long)sizeof(float));
            DeflectionCount = 0;
            for (int p = 0; p < pairCount; p++)
            {
                int2 pb = PairBodies[p];
                uint wa = (uint)worldBodies[pb.x], wb = (uint)worldBodies[pb.y];
                ulong key = ((ulong)math.min(wa, wb) << 32) | math.max(wa, wb);
                PairWorldKey[p] = key;
                if (previous.TryGetValue(key, out int old))
                {
                    PairLocked[p] = oldLocked[old];
                    PairDeflection[p] = oldDeflection[old];
                }
                else
                {
                    PairLocked[p] = 0;
                    PairDeflection[p] = float.PositiveInfinity;
                }
            }

            previous.Dispose();
            oldLocked.Dispose();
            oldDeflection.Dispose();
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
        /// Rebuilds the structure for the pairs and their lock states: a locked pair's edge folds its child into its
        /// parent's section, and a closure whose ends share a section drops out.
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
