using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics;
using Unity.Profiling;

namespace Unity.Physics
{
    public static partial class Solver
    {
        static readonly ProfilerMarker ForestFactorMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Factor");
        static readonly ProfilerMarker ForestBoundaryMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Boundary");
        static readonly ProfilerMarker ForestSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Solve");
        static readonly ProfilerMarker ForestStructureMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Structure");

        const byte ForestRowClass = 0;
        const byte BoundaryRowClass = 1;
        const byte FixedRowClass = 2;

        /// <summary>A forest row between two bodies of one section: a locked pair's, whose load comes from inverse dynamics, or an inert closure's.</summary>
        const byte SectionRowClass = 3;

        /// <summary>
        /// One island's solve as a forest plus a boundary, see <see cref="ForestSettings"/>. Holds what a second solve with
        /// a changed right-hand side reuses: the factor (in the island's <see cref="ForestIsland"/>), the classes of the
        /// rows and the boundary block W = J_b A^-1 J_b^T + E_b; and, for sections of more than one body, what turns the
        /// sections' answer back into each body's.
        /// </summary>
        unsafe struct ForestRun
        {
            public ForestIsland* Forest;
            public SparseConstraintSolver.Problem Problem;
            public bool Bounded;

            /// <summary>
            /// The rows as the sections see them: a row on a body of a section of several acts on the section's velocity at
            /// the body's offset from its centre. The problem's own rows when every section is one body.
            /// </summary>
            public float* Jacobian;

            public NativeArray<float> SectionJacobian;

            /// <summary>Each row's class, and each block's two sections, or -1 for a side that doesn't move.</summary>
            public NativeArray<byte> RowClass;
            public NativeArray<int2> BlockSections;
            public NativeArray<int> RowBlock;

            /// <summary>The problem's forest blocks, in order, as the island's pairs number them.</summary>
            public NativeList<int> ForestBlocks;

            /// <summary>The boundary rows, the dense block over them, and the bounds and couplings they take.</summary>
            public NativeArray<int> Boundary;
            public NativeArray<int> BoundaryOf;
            public NativeArray<double> W;
            public int BoundaryCount;

            /// <summary>The change of velocity of every section, and a vector of the same size to work in.</summary>
            public NativeArray<double> Velocity;
            public NativeArray<double> Work;

            /// <summary>
            /// For sections of several bodies: each section's centre and the velocity its bodies had together before the
            /// solve (momentum-conserving), each body's offset from its section's centre, and the bodies' own velocities.
            /// Empty when every section is one body.
            /// </summary>
            public NativeArray<double3> SectionCenter;
            public NativeArray<double3> SectionLinear;
            public NativeArray<double3> SectionAngular;
            public NativeArray<double3> Arm;
            public NativeArray<double3> OwnLinear;
            public NativeArray<double3> OwnAngular;
            public bool MultiPart;

            /// <summary>
            /// For a settle: each section's velocity after the solve and the turn it makes in a substep, and what each body of
            /// a section of several still needs from its locked edges (an impulse and its moment about the body's centre).
            /// </summary>
            public NativeArray<double3> SettleLinear;
            public NativeArray<double3> SettleAngular;
            public NativeArray<double3x3> SettleTurn;
            public NativeArray<double6> Need;

            /// <summary>Each block whose bodies share a section, whose rows have no drift to aim at; empty without such.</summary>
            public NativeArray<byte> BlockInSection;

            public int* WorldBodies;
            public MotionVelocity* MotionVelocities;
            public MotionData* MotionDatas;
            public float Timestep;

            /// <summary>The lanes that share the island's solve, or null.</summary>
            public SparseSolveTeam* Team;

            /// <summary>How long the factor, the boundary block and any structure rebuild took, in profiler timestamp units.</summary>
            public long FactorTicks;
            public long BoundaryTicks;
            public long StructureTicks;

            /// <summary>Whether the structure was rebuilt for this solve, and how many pairs changed lock state.</summary>
            public bool Rebuilt;
            public int LockChanges;

            /// <summary>The largest momentum left over at a section's first body by its inverse dynamics, in N s.</summary>
            public double Residual;

            public void Dispose()
            {
                RowClass.Dispose();
                BlockSections.Dispose();
                RowBlock.Dispose();
                ForestBlocks.Dispose();
                Boundary.Dispose();
                BoundaryOf.Dispose();
                W.Dispose();
                Velocity.Dispose();
                Work.Dispose();
                if (SectionJacobian.IsCreated)
                {
                    SectionJacobian.Dispose();
                }

                if (SectionCenter.IsCreated)
                {
                    SectionCenter.Dispose();
                    SectionLinear.Dispose();
                    SectionAngular.Dispose();
                    Arm.Dispose();
                    OwnLinear.Dispose();
                    OwnAngular.Dispose();
                }

                if (BlockInSection.IsCreated)
                {
                    BlockInSection.Dispose();
                    SettleLinear.Dispose();
                    SettleAngular.Dispose();
                    SettleTurn.Dispose();
                    Need.Dispose();
                }
            }
        }

        /// <summary>
        /// Solves an island's rows as a forest plus a boundary and writes every row's impulse, or returns false, changing
        /// nothing, when the island isn't one for it: no forest rows, or more boundary rows than the cap.
        /// </summary>
        static unsafe bool TryStartForest(in SparseConstraintSolver.Problem problem, JacobianHeader** headers, bool bounded,
            ForestIsland* forest, in ForestSettings settings, SparseSolveTeam* team, int* worldBodies,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, float timestep,
            out ForestRun run)
        {
            run = default;
            int rows = problem.RowCount;
            var coupled = new NativeArray<bool>(rows, Allocator.Temp);
            for (int c = 0; bounded && c < problem.CouplingCount; c++)
            {
                SparseConstraintSolver.Coupling coupling = problem.Couplings[c];
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    coupled[row] = true;
                }
            }

            var rowClass = new NativeArray<byte>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var rowBlock = new NativeArray<int>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var forestBlocks = new NativeList<int>(problem.BlockCount, Allocator.Temp);
            var treeBlocks = new NativeList<bool>(problem.BlockCount, Allocator.Temp);

            // What each forest block allows its pair: bit 0 lockable, bit 1 holdable, bit 2 every row a forest row or idle.
            var blockAllows = new NativeList<byte>(problem.BlockCount, Allocator.Temp);
            int forestRows = 0, boundaryRows = 0;
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                JacobianFlags flags = headers[b]->Flags;
                bool forestBlock = headers[b]->Type != JacobianType.Contact && (flags & JacobianFlags.ForestRow) != 0
                    && (block.BodyA >= 0 || block.BodyB >= 0);
                bool allForest = true;
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    rowBlock[row] = b;
                    float lower = problem.Lower[row], upper = problem.Upper[row];
                    bool free = !bounded || (lower <= -float.MaxValue && upper >= float.MaxValue);
                    if (bounded && lower == upper && !coupled[row])
                    {
                        // A limit out of reach is a fixed row with no impulse, which doesn't keep its pair from locking.
                        rowClass[row] = FixedRowClass;
                        allForest &= lower == 0.0f;
                    }
                    else if (forestBlock && free && problem.Epsilon[row] > 0.0f)
                    {
                        rowClass[row] = ForestRowClass;
                        forestRows++;
                    }
                    else
                    {
                        rowClass[row] = BoundaryRowClass;
                        boundaryRows++;
                        allForest = false;
                    }
                }

                if (forestBlock)
                {
                    forestBlocks.Add(b);
                    treeBlocks.Add((flags & JacobianFlags.ForestTree) != 0);
                    blockAllows.Add((byte)(((flags & JacobianFlags.ForestLockable) != 0 ? 1 : 0)
                        | ((flags & JacobianFlags.ForestHoldable) != 0 ? 2 : 0) | (allForest ? 4 : 0)));
                }
            }

            // A coupling's rows and its sources have to be boundary rows, which contacts' always are.
            bool couplingsInBoundary = true;
            for (int c = 0; bounded && c < problem.CouplingCount; c++)
            {
                SparseConstraintSolver.Coupling coupling = problem.Couplings[c];
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    couplingsInBoundary &= rowClass[row] == BoundaryRowClass;
                }

                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    couplingsInBoundary &= rowClass[row] == BoundaryRowClass;
                }
            }

            if (forestRows == 0 || boundaryRows > settings.BoundaryRowCap || !couplingsInBoundary)
            {
                coupled.Dispose();
                rowClass.Dispose();
                rowBlock.Dispose();
                forestBlocks.Dispose();
                treeBlocks.Dispose();
                blockAllows.Dispose();
                return false;
            }

            // The pairs, kept while the island's bodies and forest blocks stay the same.
            ref UnsafeList<ulong> signature = ref forest->Signature;
            bool same = signature.Length == forestBlocks.Length + 1 && signature[0] == (ulong)problem.BodyCount;
            for (int f = 0; same && f < forestBlocks.Length; f++)
            {
                same = signature[f + 1] == BlockKey(problem.Blocks[forestBlocks[f]], treeBlocks[f]);
            }

            long structureStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            bool rebuild = !same || forest->StructureBuilds == 0;
            if (!same)
            {
                signature.Clear();
                signature.Add((ulong)problem.BodyCount);
                for (int f = 0; f < forestBlocks.Length; f++)
                {
                    signature.Add(BlockKey(problem.Blocks[forestBlocks[f]], treeBlocks[f]));
                }

                forest->BuildPairs(problem, (int*)forestBlocks.GetUnsafeReadOnlyPtr(), (bool*)treeBlocks.GetUnsafeReadOnlyPtr(),
                    forestBlocks.Length, worldBodies);
            }

            // Which pairs are locked this solve: a tree edge whose joints all allow it, with nothing but forest rows, locks
            // when its springs would deflect less than the lock angle under its last load, and stays locked until its load
            // would deflect them past the yield angle.
            int lockChanges = 0;
            int pairCount = forest->PairCount;
            for (int p = 0; p < pairCount; p++)
            {
                bool locked = forest->PairLocked[p] != 0;
                bool next = false;
                if (settings.Locking && forest->PairEdge[p] != 0 && forest->PairTree[p] != 0)
                {
                    int allows = 7;
                    for (int k = forest->PairBlockStart[p]; k < forest->PairBlockStart[p + 1]; k++)
                    {
                        allows &= blockAllows[forest->PairBlocks[k]];
                    }

                    float deflection = forest->PairDeflection[p];
                    next = locked
                        ? (allows & 6) == 6 && deflection <= settings.YieldAngle
                        : (allows & 5) == 5 && deflection < settings.LockAngle;
                }

                if (next != locked)
                {
                    forest->PairLocked[p] = (byte)(next ? 1 : 0);
                    lockChanges++;
                }
            }

            rebuild |= lockChanges > 0;
            if (rebuild)
            {
                using (ForestStructureMarker.Auto())
                {
                    forest->BuildStructure(problem, (int*)forestBlocks.GetUnsafeReadOnlyPtr(), forestBlocks.Length);
                }
            }

            long structureTicks = rebuild ? Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - structureStarted : 0;
            treeBlocks.Dispose();
            blockAllows.Dispose();

            int sections = forest->SectionCount;
            ref VesselContext context = ref forest->Context;
            int* bodySection = forest->BodySection.Ptr;
            var blockSections = new NativeArray<int2>(problem.BlockCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                blockSections[b] = new int2(block.BodyA >= 0 ? bodySection[block.BodyA] : -1, block.BodyB >= 0 ? bodySection[block.BodyB] : -1);
            }

            // A forest row between two bodies of one section leaves the solve.
            for (int f = 0; f < forestBlocks.Length; f++)
            {
                if (forest->BlockLink[f] != ForestIsland.InSection)
                {
                    continue;
                }

                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[forestBlocks[f]];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (rowClass[row] == ForestRowClass)
                    {
                        rowClass[row] = SectionRowClass;
                    }
                }
            }

            run = new ForestRun
            {
                Forest = forest,
                Problem = problem,
                Bounded = bounded,
                Jacobian = problem.Jacobian,
                RowClass = rowClass,
                BlockSections = blockSections,
                RowBlock = rowBlock,
                ForestBlocks = forestBlocks,
                Velocity = new NativeArray<double>(6 * sections, Allocator.Temp),
                Work = new NativeArray<double>(6 * sections, Allocator.Temp),
                MultiPart = forest->MultiPartSections > 0,
                WorldBodies = worldBodies,
                MotionVelocities = (MotionVelocity*)motionVelocities.GetUnsafeReadOnlyPtr(),
                MotionDatas = (MotionData*)motionDatas.GetUnsafeReadOnlyPtr(),
                Timestep = timestep,
                Team = team,
                StructureTicks = structureTicks,
                Rebuilt = rebuild,
                LockChanges = lockChanges,
            };

            for (int i = 0; i < problem.BodyCount; i++)
            {
                forest->Sections[bodySection[i]] = new VesselSection
                {
                    Mass = problem.Bodies[i].Mass,
                    Inertia = problem.Bodies[i].Inertia,
                };
            }

            if (run.MultiPart)
            {
                PrepareSections(ref run);
            }

            float* jacobian = run.Jacobian;
            long factorStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            using (ForestFactorMarker.Auto())
            {
                // Each link's rows, and the rows on one moving body, which go to its section's diagonal.
                int links = forest->LinkCount;
                int* linkRowStart = forest->LinkRowStart.Ptr;
                UnsafeUtility.MemClear(linkRowStart, (links + 1) * (long)sizeof(int));
                UnsafeUtility.MemClear(forest->ExtraDiagonal.Ptr, forest->ExtraDiagonal.Length * (long)sizeof(double));
                var rowEntry = new NativeArray<int>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int f = 0; f < forestBlocks.Length; f++)
                {
                    int b = forestBlocks[f];
                    SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                    int linked = forest->BlockLink[f];
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        rowEntry[row] = -1;
                        if (rowClass[row] != ForestRowClass)
                        {
                            continue;
                        }

                        if (linked >= 0)
                        {
                            rowEntry[row] = row * 2 + (linked & 1);
                            linkRowStart[(linked >> 1) + 1]++;
                        }
                        else
                        {
                            // One moving side: w j j^T on its section's diagonal.
                            float* columns = jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                            int2 sides = blockSections[b];
                            int s = sides.x >= 0 ? sides.x : sides.y;
                            double6 j = VesselContext.LoadRow(sides.x >= 0 ? columns : columns + 6);
                            VesselContext.AddOuter(forest->ExtraDiagonal.Ptr + VesselBlocks.Size * s, 1.0 / problem.Epsilon[row], j, j);
                        }
                    }
                }

                for (int k = 0; k < links; k++)
                {
                    linkRowStart[k + 1] += linkRowStart[k];
                }

                forest->LinkRowList.Resize(math.max(1, linkRowStart[links]));
                int* linkRowList = forest->LinkRowList.Ptr;
                var linkFill = new NativeArray<int>(math.max(1, links), Allocator.Temp);
                for (int f = 0; f < forestBlocks.Length; f++)
                {
                    int linked = forest->BlockLink[f];
                    if (linked < 0)
                    {
                        continue;
                    }

                    SparseConstraintSolver.ConstraintBlock block = problem.Blocks[forestBlocks[f]];
                    int link = linked >> 1;
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        if (rowClass[row] == ForestRowClass)
                        {
                            linkRowList[linkRowStart[link] + linkFill[link]++] = rowEntry[row];
                        }
                    }
                }

                linkFill.Dispose();
                rowEntry.Dispose();

                context.RowJacobian = jacobian;
                context.RowEpsilon = problem.Epsilon;
                context.RowRhs = problem.RightHandSide;
                context.LinkRowStart = linkRowStart;
                context.LinkRowList = linkRowList;
                context.ExtraDiagonal = forest->ExtraDiagonal.Ptr;
                context.ExtraRhs = null;

                VesselContext* contextPtr = (VesselContext*)UnsafeUtility.AddressOf(ref context);
                SparseSolveTeam.RunAll(team, contextPtr, SparseSolveTeam.ForestFactorKind, context.ChunkCount, null);
                var scratch = new NativeArray<double>(VesselBlocks.Size * math.max(1, context.MaxStructure), Allocator.Temp,
                    NativeArrayOptions.UninitializedMemory);
                context.SolveTop((double*)scratch.GetUnsafePtr());
                scratch.Dispose();
            }

            // The boundary block, from forward substitutions along each boundary row's elimination-tree path.
            var boundary = new NativeArray<int>(math.max(1, boundaryRows), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var boundaryOf = new NativeArray<int>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            int nb = 0;
            for (int row = 0; row < rows; row++)
            {
                boundaryOf[row] = -1;
                if (rowClass[row] == BoundaryRowClass)
                {
                    boundaryOf[row] = nb;
                    boundary[nb++] = row;
                }
            }

            long boundaryStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            var w = new NativeArray<double>(math.max(1, nb * nb), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            using (ForestBoundaryMarker.Auto())
            {
                var y = new NativeArray<double>(6 * sections, Allocator.Temp);
                var mark = new NativeArray<int>(sections, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var path = new NativeArray<int>(sections, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int s = 0; s < sections; s++)
                {
                    mark[s] = -1;
                }

                var pathStart = new NativeArray<int>(nb + 1, Allocator.Temp);
                var pathSection = new NativeList<int>(nb * 8, Allocator.Temp);
                var pathY = new NativeList<double6>(nb * 8, Allocator.Temp);
                var pathZ = new NativeList<double6>(nb * 8, Allocator.Temp);
                int* seeds = stackalloc int[2];
                double* yPtr = (double*)y.GetUnsafePtr();
                for (int i = 0; i < nb; i++)
                {
                    int row = boundary[i];
                    int2 sides = blockSections[rowBlock[row]];
                    float* columns = jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    int seedCount = 0;
                    if (sides.x >= 0)
                    {
                        seeds[seedCount++] = sides.x;
                        (double6.Load(yPtr + 6 * sides.x) + VesselContext.LoadRow(columns)).Store(yPtr + 6 * sides.x);
                    }

                    if (sides.y >= 0)
                    {
                        if (sides.y != sides.x)
                        {
                            seeds[seedCount++] = sides.y;
                        }

                        (double6.Load(yPtr + 6 * sides.y) + VesselContext.LoadRow(columns + 6)).Store(yPtr + 6 * sides.y);
                    }

                    int count = context.CollectPath(seeds, seedCount, (int*)path.GetUnsafePtr(), (int*)mark.GetUnsafePtr(), i);
                    context.ForwardPath(yPtr, (int*)path.GetUnsafePtr(), count);
                    pathStart[i] = pathSection.Length;
                    for (int k = 0; k < count; k++)
                    {
                        int s = path[k];
                        double6 ys = double6.Load(yPtr + 6 * s);
                        pathSection.Add(s);
                        pathY.Add(ys);
                        pathZ.Add(context.InverseDiagonal(s, ys));
                        default(double6).Store(yPtr + 6 * s);
                    }
                }

                pathStart[nb] = pathSection.Length;
                double* wPtr = (double*)w.GetUnsafePtr();
                for (int i = 0; i < nb; i++)
                {
                    for (int j = i; j < nb; j++)
                    {
                        double sum = 0.0;
                        int p = pathStart[i], pEnd = pathStart[i + 1], q = pathStart[j], qEnd = pathStart[j + 1];
                        while (p < pEnd && q < qEnd)
                        {
                            int sp = pathSection[p], sq = pathSection[q];
                            if (sp == sq)
                            {
                                sum += double6.Dot(pathY[p], pathZ[q]);
                                p++;
                                q++;
                            }
                            else if (sp < sq)
                            {
                                p++;
                            }
                            else
                            {
                                q++;
                            }
                        }

                        wPtr[i * nb + j] = sum;
                        wPtr[j * nb + i] = sum;
                    }

                    wPtr[i * nb + i] += problem.Epsilon[boundary[i]];
                }

                y.Dispose();
                mark.Dispose();
                path.Dispose();
                pathStart.Dispose();
                pathSection.Dispose();
                pathY.Dispose();
                pathZ.Dispose();
            }

            coupled.Dispose();
            run.Boundary = boundary;
            run.BoundaryOf = boundaryOf;
            run.W = w;
            run.BoundaryCount = nb;
            run.FactorTicks = boundaryStarted - factorStarted;
            run.BoundaryTicks = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - boundaryStarted;
            return true;
        }

        static ulong BlockKey(in SparseConstraintSolver.ConstraintBlock block, bool tree)
        {
            return (ulong)(uint)(block.BodyA + 1) | ((ulong)(uint)(block.BodyB + 1) << 31) | (tree ? 1UL << 62 : 0UL);
        }

        /// <summary>
        /// For sections of several bodies: each section's mass, centre and inertia from its bodies, the momentum-conserving
        /// velocity its bodies have together, the rows as the sections see them, and each row's right-hand side moved from
        /// the bodies' own velocities to the section's. A section's bodies need not move rigidly when it starts, after a
        /// lock or a force on one of them; the solve takes them as one, and each body's answer is the section's.
        /// </summary>
        static unsafe void PrepareSections(ref ForestRun run)
        {
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            ForestIsland* forest = run.Forest;
            int sections = forest->SectionCount;
            int bodies = problem.BodyCount;
            int* sectionPartStart = forest->Structure.SectionPartStart.Ptr;
            int* sectionParts = forest->Structure.SectionParts.Ptr;
            int* partBody = forest->PartBody.Ptr;
            int* bodySection = forest->BodySection.Ptr;
            run.SectionCenter = new NativeArray<double3>(sections, Allocator.Temp);
            run.SectionLinear = new NativeArray<double3>(sections, Allocator.Temp);
            run.SectionAngular = new NativeArray<double3>(sections, Allocator.Temp);
            run.Arm = new NativeArray<double3>(bodies, Allocator.Temp);
            run.OwnLinear = new NativeArray<double3>(bodies, Allocator.Temp);
            run.OwnAngular = new NativeArray<double3>(bodies, Allocator.Temp);
            run.BlockInSection = new NativeArray<byte>(problem.BlockCount, Allocator.Temp);
            run.SettleLinear = new NativeArray<double3>(sections, Allocator.Temp);
            run.SettleAngular = new NativeArray<double3>(sections, Allocator.Temp);
            run.SettleTurn = new NativeArray<double3x3>(sections, Allocator.Temp);
            run.Need = new NativeArray<double6>(bodies, Allocator.Temp);

            for (int s = 0; s < sections; s++)
            {
                int start = sectionPartStart[s], end = sectionPartStart[s + 1];
                if (end - start < 2)
                {
                    continue;
                }

                double mass = 0.0;
                double3 moment = 0.0, momentum = 0.0;
                for (int k = start; k < end; k++)
                {
                    int i = partBody[sectionParts[k]];
                    int world = run.WorldBodies[i];
                    double m = problem.Bodies[i].Mass;
                    mass += m;
                    moment += m * (double3)run.MotionDatas[world].WorldFromMotion.pos;
                    momentum += m * (double3)run.MotionVelocities[world].LinearVelocity;
                }

                double3 center = moment / mass;
                double3x3 inertia = double3x3.zero;
                double3 angularMomentum = 0.0;
                for (int k = start; k < end; k++)
                {
                    int i = partBody[sectionParts[k]];
                    int world = run.WorldBodies[i];
                    double m = problem.Bodies[i].Mass;
                    double3 r = (double3)run.MotionDatas[world].WorldFromMotion.pos - center;
                    run.Arm[i] = r;
                    run.OwnLinear[i] = run.MotionVelocities[world].LinearVelocity;
                    run.OwnAngular[i] = BodyAngular(run, world);
                    double3x3 own = problem.Bodies[i].Inertia;
                    inertia += own + m * (math.dot(r, r) * double3x3.identity - new double3x3(r * r.x, r * r.y, r * r.z));
                    angularMomentum += math.mul(own, run.OwnAngular[i]) + m * math.cross(r, run.OwnLinear[i]);
                }

                run.SectionCenter[s] = center;
                run.SectionLinear[s] = momentum / mass;
                run.SectionAngular[s] = math.mul(math.inverse(inertia), angularMomentum);
                run.Forest->Sections[s] = new VesselSection { Mass = mass, Center = center, Inertia = inertia };
            }

            // The rows on a body of a section of several act on the section at the body's offset: j_angular + r x j_linear.
            // Each block's rows are copied and turned by one task, on the team's lanes.
            int rows = problem.RowCount;
            long floats = SparseConstraintSolver.ColumnsPerRow * (long)rows;
            run.SectionJacobian = new NativeArray<float>((int)floats, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            run.Jacobian = (float*)run.SectionJacobian.GetUnsafePtr();
            var tasks = new SectionTasks
            {
                Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                Problem = problem,
                Phase = SectionTasks.PrepareRows,
            };
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestSectionsKind,
                (problem.BlockCount + BlocksPerSectionTask - 1) / BlocksPerSectionTask, null);
        }

        const int BlocksPerSectionTask = 128;
        const int PartsPerSectionTask = 64;

        /// <summary>What the team's section tasks share: the run, its problem, and the arrays a settle writes.</summary>
        unsafe struct SectionTasks
        {
            public const int PrepareRows = 0;
            public const int SettleBodies = 1;
            public const int SplitLoads = 2;

            public ForestRun* Run;
            public SparseConstraintSolver.Problem Problem;
            public float* Impulses;
            public float3* Linear;
            public float3* Angular;
            public bool PushOut;
            public int Phase;
        }

        /// <summary>One task of a section stage, for a team.</summary>
        internal static unsafe void ForestSectionTask(void* context, int task)
        {
            var tasks = (SectionTasks*)context;
            if (tasks->Phase == SectionTasks.PrepareRows)
            {
                PrepareRowsTask(ref *tasks->Run, tasks->Problem, task);
            }
            else if (tasks->Phase == SectionTasks.SettleBodies)
            {
                SettleBodiesTask(ref *tasks->Run, tasks, task);
            }
            else
            {
                SplitLoadsTask(ref *tasks->Run, tasks, task);
            }
        }

        /// <summary>One task's blocks of <see cref="PrepareSections"/>: their rows copied, turned to the sections, and moved.</summary>
        static unsafe void PrepareRowsTask(ref ForestRun run, in SparseConstraintSolver.Problem problem, int task)
        {
            ForestIsland* forest = run.Forest;
            int* bodySection = forest->BodySection.Ptr;
            float* jacobian = run.Jacobian;
            int end = math.min((task + 1) * BlocksPerSectionTask, problem.BlockCount);
            for (int b = task * BlocksPerSectionTask; b < end; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                long first = SparseConstraintSolver.ColumnsPerRow * (long)block.FirstRow;
                UnsafeUtility.MemCpy(jacobian + first, problem.Jacobian + first, SparseConstraintSolver.ColumnsPerRow * (long)block.RowCount * sizeof(float));
                bool multiA = block.BodyA >= 0 && Multi(forest, bodySection[block.BodyA]);
                bool multiB = block.BodyB >= 0 && Multi(forest, bodySection[block.BodyB]);
                if (!multiA && !multiB)
                {
                    continue;
                }

                run.BlockInSection[b] = (byte)(block.BodyA >= 0 && block.BodyB >= 0 && bodySection[block.BodyA] == bodySection[block.BodyB] ? 1 : 0);
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    // A row inside a section isn't solved, and an idle fixed row adds nothing: neither needs turning.
                    byte rowClass = run.RowClass[row];
                    if (rowClass == SectionRowClass || (rowClass == FixedRowClass && problem.Lower[row] == 0.0f))
                    {
                        continue;
                    }

                    float* columns = jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    double change = 0.0;
                    if (multiA)
                    {
                        change += ToSection(ref run, columns, block.BodyA);
                    }

                    if (multiB)
                    {
                        change += ToSection(ref run, columns + 6, block.BodyB);
                    }

                    if (rowClass != FixedRowClass)
                    {
                        problem.RightHandSide[row] += (float)change;
                    }
                }
            }
        }

        /// <summary>Radians per second below which a section's spin is taken as none.</summary>
        const double MinimumSpin = 1e-9;

        /// <summary>
        /// A value for single precision        /// <summary>
        /// A value for single precision with anything below float's normal range made zero. A locked section at rest leaves
        /// differences of order 1e-40 that would otherwise turn into denormal velocities and impulses, which every job
        /// reading them afterwards (row and Jacobian builds, integration) takes tens of times longer to compute with.
        /// </summary>
        static float Flushed(double value) => math.abs(value) < 1e-30 ? 0.0f : (float)value;

        static float3 Flushed(double3 value) => new float3(Flushed(value.x), Flushed(value.y), Flushed(value.z));

        /// <summary>Whether section s has more than one body.</summary>
        static unsafe bool Multi(ForestIsland* forest, int s) =>
            forest->Structure.SectionPartStart[s + 1] - forest->Structure.SectionPartStart[s] > 1;

        /// <summary>A body's angular velocity in world space.</summary>
        static unsafe double3 BodyAngular(in ForestRun run, int world) =>
            math.rotate(run.MotionDatas[world].WorldFromMotion.rot, run.MotionVelocities[world].AngularVelocity);

        /// <summary>
        /// Turns one side of a row from a body's columns to its section's, and returns how much the row's right-hand side
        /// moves when the body's velocity is taken as the section's: j . u_body - j_section . u_section.
        /// </summary>
        static unsafe double ToSection(ref ForestRun run, float* columns, int body)
        {
            int s = run.Forest->BodySection[body];
            double3 linear = new double3(columns[0], columns[1], columns[2]);
            double3 angular = new double3(columns[3], columns[4], columns[5]);
            double own = math.dot(linear, run.OwnLinear[body]) + math.dot(angular, run.OwnAngular[body]);
            double3 moved = angular + math.cross(run.Arm[body], linear);
            columns[3] = (float)moved.x;
            columns[4] = (float)moved.y;
            columns[5] = (float)moved.z;
            return own - (math.dot(linear, run.SectionLinear[s]) + math.dot(moved, run.SectionAngular[s]));
        }

        /// <summary>
        /// Solves the run's island for the right-hand side <paramref name="rhs"/> against the factor and boundary block it
        /// holds, and writes every row's impulse; a row inside a section gets none here. Returns false if the boundary's
        /// pivoting ran out of iterations, leaving its impulses clamped to their bounds.
        /// </summary>
        static unsafe bool SolveForest(ref ForestRun run, float* rhs, float* impulses)
        {
            using ProfilerMarker.AutoScope span = ForestSolveMarker.Auto();

            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            float* jacobian = run.Jacobian;
            int rows = problem.RowCount;
            double* v = (double*)run.Velocity.GetUnsafePtr();
            double* work = (double*)run.Work.GetUnsafePtr();
            UnsafeUtility.MemClear(v, run.Velocity.Length * (long)sizeof(double));

            // b = J_f^T E_f^-1 r_f, plus the known impulses of fixed rows; then the forest's own answer.
            for (int row = 0; row < rows; row++)
            {
                byte rowClass = run.RowClass[row];
                double scale;
                if (rowClass == ForestRowClass)
                {
                    scale = rhs[row] / (double)problem.Epsilon[row];
                }
                else if (rowClass == FixedRowClass)
                {
                    scale = problem.Lower[row];
                }
                else
                {
                    continue;
                }

                AddRow(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row, scale);
            }

            SolveAgainstFactor(ref run, v);

            // The boundary: (J_b A^-1 J_b^T + E_b) z = r_b - J_b v0, bounded.
            int nb = run.BoundaryCount;
            bool solved = true;
            if (nb > 0)
            {
                var q = new NativeArray<double>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var lower = new NativeArray<double>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var upper = new NativeArray<double>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var states = new NativeArray<byte>(nb, Allocator.Temp, NativeArrayOptions.ClearMemory);
                var z = new NativeArray<double>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < nb; i++)
                {
                    int row = run.Boundary[i];
                    q[i] = rhs[row] - RowSpeed(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
                    lower[i] = run.Bounded ? problem.Lower[row] : double.NegativeInfinity;
                    upper[i] = run.Bounded ? problem.Upper[row] : double.PositiveInfinity;
                    if (run.Bounded && problem.InitialStates != null)
                    {
                        states[i] = problem.InitialStates[row];
                    }
                }

                var couplings = new NativeList<SparseConstraintSolver.Coupling>(problem.CouplingCount, Allocator.Temp);
                for (int c = 0; run.Bounded && c < problem.CouplingCount; c++)
                {
                    SparseConstraintSolver.Coupling coupling = problem.Couplings[c];
                    coupling.FirstRow = run.BoundaryOf[coupling.FirstRow];
                    coupling.FirstSourceRow = run.BoundaryOf[coupling.FirstSourceRow];
                    couplings.Add(coupling);
                }

                solved = BoundarySolver.Solve(nb, (double*)run.W.GetUnsafeReadOnlyPtr(), (double*)q.GetUnsafePtr(),
                    (double*)lower.GetUnsafePtr(), (double*)upper.GetUnsafePtr(), (byte*)states.GetUnsafePtr(),
                    couplings.GetUnsafePtr(), couplings.Length, run.Bounded, (double*)z.GetUnsafePtr());

                // v += A^-1 J_b^T z.
                UnsafeUtility.MemClear(work, run.Work.Length * (long)sizeof(double));
                for (int i = 0; i < nb; i++)
                {
                    int row = run.Boundary[i];
                    AddRow(work, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row, z[i]);
                    impulses[row] = (float)z[i];
                }

                SolveAgainstFactor(ref run, work);
                for (int i = 0; i < run.Velocity.Length; i++)
                {
                    v[i] += work[i];
                }

                q.Dispose();
                lower.Dispose();
                upper.Dispose();
                states.Dispose();
                z.Dispose();
                couplings.Dispose();
            }

            // Every forest row's impulse from the velocities: z = E^-1 (r - J v).
            for (int row = 0; row < rows; row++)
            {
                byte rowClass = run.RowClass[row];
                if (rowClass == ForestRowClass)
                {
                    double speed = RowSpeed(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
                    impulses[row] = (float)((rhs[row] - speed) / problem.Epsilon[row]);
                }
                else if (rowClass == FixedRowClass)
                {
                    impulses[row] = problem.Lower[row];
                }
                else if (rowClass == SectionRowClass)
                {
                    impulses[row] = 0.0f;
                }
            }

            return solved;
        }

        /// <summary>
        /// Each body's impulse once the run's sections have their change of velocity, as <see cref="SumBodyImpulses"/>
        /// gives it, written to <paramref name="linear"/> and <paramref name="angular"/>. A body of a section of several
        /// takes the impulse that gives it its section's motion, rigid about the section's centre; for a solve, the
        /// locked pairs' rows then get the impulses that carry those bodies' share, by inverse dynamics. For a push-out,
        /// the sections start from rest and nothing else is written.
        /// </summary>
        static unsafe void SettleForest(ref ForestRun run, in SparseConstraintSolver.Problem problem, float* impulses,
            float3* linear, float3* angular, bool pushOut)
        {
            SumBodyImpulses(problem, impulses, linear, angular);
            if (!run.MultiPart)
            {
                return;
            }

            ForestIsland* forest = run.Forest;
            int sections = forest->SectionCount;
            int* sectionPartStart = forest->Structure.SectionPartStart.Ptr;
            int* sectionParts = forest->Structure.SectionParts.Ptr;
            int* partBody = forest->PartBody.Ptr;
            double h = run.Timestep;
            double* v = (double*)run.Velocity.GetUnsafePtr();
            for (int s = 0; s < sections; s++)
            {
                if (!Multi(forest, s))
                {
                    continue;
                }

                double3 dLinear = new double3(v[6 * s], v[6 * s + 1], v[6 * s + 2]);
                double3 dAngular = new double3(v[6 * s + 3], v[6 * s + 4], v[6 * s + 5]);
                double3 sectionLinear = pushOut ? dLinear : run.SectionLinear[s] + dLinear;
                double3 sectionAngular = pushOut ? dAngular : run.SectionAngular[s] + dAngular;

                // A locked section at rest keeps a spin of order 1e-19 rad/s from rounding, whose squares the next step's
                // row and Jacobian builds and the integration carry into denormal floats, several times slower to compute
                // with. Nothing that small is a real turn.
                if (math.lengthsq(sectionAngular) < MinimumSpin * MinimumSpin)
                {
                    sectionAngular = 0.0;
                }

                // Each body lands where the section's rotation takes it, not along the tangent, so that a turning section
                // stays rigid however long it stays locked.
                double angle = math.length(sectionAngular) * h;
                double4 turn = angle > 1e-12 ? new double4(math.normalize(sectionAngular) * math.sin(0.5 * angle), math.cos(0.5 * angle))
                    : new double4(0.0, 0.0, 0.0, 1.0);
                run.SettleLinear[s] = sectionLinear;
                run.SettleAngular[s] = sectionAngular;
                run.SettleTurn[s] = VesselContext.Rotation(turn);
            }

            int parts = forest->Structure.SectionParts.Length;
            int partTasks = (parts + PartsPerSectionTask - 1) / PartsPerSectionTask;
            var tasks = new SectionTasks
            {
                Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                Problem = problem,
                Impulses = impulses,
                Linear = linear,
                Angular = angular,
                PushOut = pushOut,
                Phase = SectionTasks.SettleBodies,
            };
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestSectionsKind, partTasks, null);
            if (pushOut)
            {
                return;
            }

            // Leaves inward: each locked edge carries what its child's subtree still needs, moved to the parent's centre.
            // Short work on one lane; the split of each edge's load over its rows goes back to the team.
            run.Residual = 0.0;
            double6* need = (double6*)run.Need.GetUnsafePtr();
            for (int s = 0; s < sections; s++)
            {
                int start = sectionPartStart[s], end = sectionPartStart[s + 1];
                if (end - start < 2)
                {
                    continue;
                }

                for (int k = end - 1; k > start; k--)
                {
                    int part = sectionParts[k];
                    int child = partBody[part];
                    int parent = partBody[forest->Edges[part].Parent];
                    double6 carried = need[child];
                    double3 force = new double3(carried.Low.x, carried.Low.y, carried.Low.z);
                    double3 torque = new double3(carried.Low.w, carried.High.x, carried.High.y) + math.cross(run.Arm[child] - run.Arm[parent], force);
                    need[parent] = need[parent] + new double6(new double4(force, torque.x), new double2(torque.y, torque.z));
                }

                double6 left = need[partBody[sectionParts[start]]];
                run.Residual = math.max(run.Residual, math.length(new double3(left.Low.x, left.Low.y, left.Low.z)));
            }

            tasks.Phase = SectionTasks.SplitLoads;
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestSectionsKind, partTasks, null);
        }

        /// <summary>
        /// One task's parts of a settle: each body of a section of several takes the impulse that gives it the section's
        /// motion, and for a solve, what it still needs from its locked edges is noted.
        /// </summary>
        static unsafe void SettleBodiesTask(ref ForestRun run, SectionTasks* tasks, int task)
        {
            ForestIsland* forest = run.Forest;
            int* sectionParts = forest->Structure.SectionParts.Ptr;
            int* partSection = forest->Structure.PartSection.Ptr;
            int* partBody = forest->PartBody.Ptr;
            double h = run.Timestep;
            bool pushOut = tasks->PushOut;
            int end = math.min((task + 1) * PartsPerSectionTask, forest->Structure.SectionParts.Length);
            for (int k = task * PartsPerSectionTask; k < end; k++)
            {
                int part = sectionParts[k];
                int s = partSection[part];
                if (!Multi(forest, s))
                {
                    continue;
                }

                int i = partBody[part];
                int world = run.WorldBodies[i];
                MotionVelocity motion = run.MotionVelocities[world];
                quaternion rotation = run.MotionDatas[world].WorldFromMotion.rot;
                double3 r = run.Arm[i];
                double3 sectionLinear = run.SettleLinear[s], sectionAngular = run.SettleAngular[s];
                double3 targetLinear = pushOut ? sectionLinear + math.cross(sectionAngular, r)
                    : sectionLinear + (math.mul(run.SettleTurn[s], r) - r) / h;
                double3 changeLinear = pushOut ? targetLinear : targetLinear - run.OwnLinear[i];
                double3 changeAngular = pushOut ? sectionAngular : sectionAngular - run.OwnAngular[i];

                // The impulses that give exactly these changes as the bodies take impulses: by their own mass, and by their
                // principal inertia in their own frame.
                double3 inverseInertia = math.max((double3)motion.InverseInertia, 1e-12);
                double3 local = math.rotate(math.conjugate(rotation), (float3)changeAngular);
                double3 linearImpulse = changeLinear / math.max((double)motion.InverseMass, 1e-12);
                double3 angularImpulse = math.rotate(rotation, (float3)(local / inverseInertia));
                if (!pushOut)
                {
                    float3 known = tasks->Linear[i], knownAngular = tasks->Angular[i];
                    run.Need[i] = new double6(new double4(linearImpulse - (double3)known, angularImpulse.x - knownAngular.x),
                        new double2(angularImpulse.y - knownAngular.y, angularImpulse.z - knownAngular.z));
                }

                tasks->Linear[i] = Flushed(linearImpulse);
                tasks->Angular[i] = Flushed(angularImpulse);
            }
        }

        /// <summary>One task's parts of a settle: the load each locked edge to a parent carries, split over its rows.</summary>
        static unsafe void SplitLoadsTask(ref ForestRun run, SectionTasks* tasks, int task)
        {
            ForestIsland* forest = run.Forest;
            int* sectionParts = forest->Structure.SectionParts.Ptr;
            int* sectionPartStart = forest->Structure.SectionPartStart.Ptr;
            int* partSection = forest->Structure.PartSection.Ptr;
            int end = math.min((task + 1) * PartsPerSectionTask, forest->Structure.SectionParts.Length);
            for (int k = task * PartsPerSectionTask; k < end; k++)
            {
                int part = sectionParts[k];
                int s = partSection[part];
                if (k == sectionPartStart[s] || !Multi(forest, s))
                {
                    continue;
                }

                int child = forest->PartBody[part];
                LockedPairImpulses(ref run, tasks->Problem, forest->PartPair[part], child, run.Need[child], tasks->Impulses);
            }
        }

        /// <summary>
        /// The impulses of a locked pair's rows that carry <paramref name="need"/> (an impulse and its moment about the
        /// child's centre) into the child: the least-squares split weighted by each row's stiffness, z = W J (J^T W J)^-1
        /// need, which is how the rows would share it if the pair were a very stiff spring. (J^T W J)^-1 is kept in the
        /// child's own frame, where a locked pair's rows stay put, from one solve to the next while the pair stays locked
        /// and its rows' compliances don't change.
        /// </summary>
        static unsafe void LockedPairImpulses(ref ForestRun run, in SparseConstraintSolver.Problem problem, int pair, int child,
            double6 need, float* impulses)
        {
            if (pair < 0)
            {
                return;
            }

            ForestIsland* forest = run.Forest;
            double3x3 rotation = new double3x3(new float3x3(run.MotionDatas[run.WorldBodies[child]].WorldFromMotion.rot));
            double3x3 back = math.transpose(rotation);
            float key = 0.0f;
            int count = 0;
            for (int k = forest->PairBlockStart[pair]; k < forest->PairBlockStart[pair + 1]; k++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (run.RowClass[row] == SectionRowClass)
                    {
                        key += problem.Epsilon[row];
                        count++;
                    }
                }
            }

            double* inverse = forest->PairInverse.Ptr + 36 * pair;
            float kept = forest->PairInverseKey[pair];
            if (!(math.abs(kept - key) <= 1e-6f * key) || forest->PairInverseRows[pair] != count)
            {
                PairLeastSquares(ref run, problem, pair, child, back, inverse);
                forest->PairInverseKey[pair] = key;
                forest->PairInverseRows[pair] = count;
            }

            double6 local = new double6(new double4(math.mul(back, new double3(need.Low.x, need.Low.y, need.Low.z)), 0.0), default);
            double3 torque = math.mul(back, new double3(need.Low.w, need.High.x, need.High.y));
            local = new double6(new double4(local.Low.xyz, torque.x), new double2(torque.y, torque.z));
            double6 y = ((double6x6*)inverse)->Mul(local);
            double3 xLinear = math.mul(rotation, new double3(y.Low.x, y.Low.y, y.Low.z));
            double3 xAngular = math.mul(rotation, new double3(y.Low.w, y.High.x, y.High.y));
            double6 x = new double6(new double4(xLinear, xAngular.x), new double2(xAngular.y, xAngular.z));
            for (int k = forest->PairBlockStart[pair]; k < forest->PairBlockStart[pair + 1]; k++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                int side = block.BodyA == child ? 0 : 6;
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (run.RowClass[row] == SectionRowClass)
                    {
                        double6 j = VesselContext.LoadRow(problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row + side);
                        impulses[row] = Flushed(double6.Dot(j, x) / problem.Epsilon[row]);
                    }
                }
            }
        }

        /// <summary>
        /// (J^T W J)^-1 over a locked pair's rows on the child's side, W the rows' stiffness 1 / epsilon, with the rows
        /// taken into the child's frame by <paramref name="back"/>.
        /// </summary>
        static unsafe void PairLeastSquares(ref ForestRun run, in SparseConstraintSolver.Problem problem, int pair, int child,
            double3x3 back, double* inverse)
        {
            ForestIsland* forest = run.Forest;
            double* m = stackalloc double[36];
            UnsafeUtility.MemClear(m, 36 * sizeof(double));
            for (int k = forest->PairBlockStart[pair]; k < forest->PairBlockStart[pair + 1]; k++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                int side = block.BodyA == child ? 0 : 6;
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (run.RowClass[row] == SectionRowClass)
                    {
                        float* c = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row + side;
                        double3 l = math.mul(back, new double3(c[0], c[1], c[2]));
                        double3 a = math.mul(back, new double3(c[3], c[4], c[5]));
                        double6 j = new double6(new double4(l, a.x), new double2(a.y, a.z));
                        VesselContext.AddOuter(m, 1.0 / problem.Epsilon[row], j, j);
                    }
                }
            }

            // Rows that don't reach every direction leave the rest of the load to rounding, not to a singular matrix.
            double trace = m[0] + m[7] + m[14] + m[21] + m[28] + m[35];
            for (int d = 0; d < 6; d++)
            {
                m[7 * d] += 1e-12 * trace + 1e-300;
            }

            VesselBlocks.InvertSymmetric(inverse, m);
        }

        /// <summary>
        /// Each tree pair's deflection under this solve's load: the largest over its rotation rows of how far the row's
        /// spring would turn to carry its impulse, z / (h k) with h^2 k = 1 / epsilon. Summed over a step's substeps; the
        /// next step's first substep turns the sum into their mean, the deflection the lock tests read from then on (see
        /// <see cref="ForestIsland.PairDeflectionSum"/>). Returns this solve's largest.
        /// </summary>
        static unsafe float RecordDeflections(ref ForestRun run, float* impulses, int substep)
        {
            float largest = 0.0f;
            ForestIsland* forest = run.Forest;
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            float h = run.Timestep;
            for (int p = 0; p < forest->PairCount; p++)
            {
                if (forest->PairEdge[p] == 0 || forest->PairTree[p] == 0)
                {
                    continue;
                }

                float deflection = 0.0f;
                for (int k = forest->PairBlockStart[p]; k < forest->PairBlockStart[p + 1]; k++)
                {
                    SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        byte rowClass = run.RowClass[row];
                        if (rowClass != ForestRowClass && rowClass != SectionRowClass)
                        {
                            continue;
                        }

                        float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                        float linear = columns[0] * columns[0] + columns[1] * columns[1] + columns[2] * columns[2];
                        float angular = columns[3] * columns[3] + columns[4] * columns[4] + columns[5] * columns[5];
                        if (linear <= 1e-8f * angular)
                        {
                            deflection = math.max(deflection, math.abs(impulses[row]) * problem.Epsilon[row] * h);
                        }
                    }
                }

                // A step's first substep closes the step before it: its mean becomes the deflection the lock tests read.
                float sum = forest->PairDeflectionSum[p];
                if (substep <= 0)
                {
                    if (forest->DeflectionCount > 0)
                    {
                        forest->PairDeflection[p] = sum / forest->DeflectionCount;
                    }

                    sum = 0.0f;
                }

                forest->PairDeflectionSum[p] = sum + deflection;

                largest = math.max(largest, deflection);
            }

            forest->DeflectionCount = substep <= 0 ? 1 : forest->DeflectionCount + 1;
            return largest;
        }

        /// <summary>x = A^-1 x against the forest's factor, by chunks on the team's lanes.</summary>
        static unsafe void SolveAgainstFactor(ref ForestRun run, double* x)
        {
            ref VesselContext context = ref run.Forest->Context;
            VesselContext* contextPtr = (VesselContext*)UnsafeUtility.AddressOf(ref context);
            long bytes = 6L * context.SectionCount * sizeof(double);
            UnsafeUtility.MemCpy(context.Rhs, x, bytes);
            SparseSolveTeam.RunAll(run.Team, contextPtr, SparseSolveTeam.ForestForwardKind, context.ChunkCount, null);
            context.SubstituteTop();
            SparseSolveTeam.RunAll(run.Team, contextPtr, SparseSolveTeam.ForestSubstituteKind, context.ChunkCount, null);
            UnsafeUtility.MemCpy(x, context.Solution, bytes);
        }

        /// <summary>One chunk of a forest's factor, for a team.</summary>
        internal static unsafe void ForestFactorTask(void* context, int chunk)
        {
            var vessel = (VesselContext*)context;
            var scratch = new NativeArray<double>(VesselBlocks.Size * math.max(1, vessel->MaxStructure), Allocator.Temp,
                NativeArrayOptions.UninitializedMemory);
            vessel->FactorChunk(chunk, (double*)scratch.GetUnsafePtr());
            scratch.Dispose();
        }

        static unsafe void AddRow(double* vector, int2 sections, float* columns, double scale)
        {
            if (sections.x >= 0)
            {
                double* to = vector + 6 * sections.x;
                (double6.Load(to) + scale * VesselContext.LoadRow(columns)).Store(to);
            }

            if (sections.y >= 0)
            {
                double* to = vector + 6 * sections.y;
                (double6.Load(to) + scale * VesselContext.LoadRow(columns + 6)).Store(to);
            }
        }

        static unsafe double RowSpeed(double* vector, int2 sections, float* columns)
        {
            double speed = 0.0;
            if (sections.x >= 0)
            {
                speed += double6.Dot(double6.Load(vector + 6 * sections.x), VesselContext.LoadRow(columns));
            }

            if (sections.y >= 0)
            {
                speed += double6.Dot(double6.Load(vector + 6 * sections.y), VesselContext.LoadRow(columns + 6));
            }

            return speed;
        }

        /// <summary>
        /// Each moving body's share of J^T z, in the frame of the Jacobian's columns, as
        /// <see cref="SparseConstraintSolver.BodyImpulse"/> gives it.
        /// </summary>
        static unsafe void SumBodyImpulses(in SparseConstraintSolver.Problem problem, float* impulses, float3* linear, float3* angular)
        {
            UnsafeUtility.MemClear(linear, problem.BodyCount * (long)sizeof(float3));
            UnsafeUtility.MemClear(angular, problem.BodyCount * (long)sizeof(float3));
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    float impulse = impulses[row];
                    if (block.BodyA >= 0)
                    {
                        linear[block.BodyA] += impulse * new float3(columns[0], columns[1], columns[2]);
                        angular[block.BodyA] += impulse * new float3(columns[3], columns[4], columns[5]);
                    }

                    if (block.BodyB >= 0)
                    {
                        linear[block.BodyB] += impulse * new float3(columns[6], columns[7], columns[8]);
                        angular[block.BodyB] += impulse * new float3(columns[9], columns[10], columns[11]);
                    }
                }
            }
        }
    }
}
