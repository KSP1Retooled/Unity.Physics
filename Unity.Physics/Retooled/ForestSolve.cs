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

        const byte ForestRowClass = 0;
        const byte BoundaryRowClass = 1;
        const byte FixedRowClass = 2;

        /// <summary>
        /// One island's solve as a forest plus a boundary, see <see cref="ForestSettings"/>. Holds what a second solve with
        /// a changed right-hand side reuses: the factor (in the island's <see cref="ForestIsland"/>), the classes of the
        /// rows and the boundary block W = J_b A^-1 J_b^T + E_b.
        /// </summary>
        unsafe struct ForestRun
        {
            public ForestIsland* Forest;
            public SparseConstraintSolver.Problem Problem;
            public bool Bounded;

            /// <summary>Each row's class, and each block's two sections, or -1 for a side that doesn't move.</summary>
            public NativeArray<byte> RowClass;
            public NativeArray<int2> BlockSections;
            public NativeArray<int> RowBlock;

            /// <summary>The boundary rows, the dense block over them, and the bounds and couplings they take.</summary>
            public NativeArray<int> Boundary;
            public NativeArray<int> BoundaryOf;
            public NativeArray<double> W;
            public int BoundaryCount;

            /// <summary>The change of velocity of every section, and a vector of the same size to work in.</summary>
            public NativeArray<double> Velocity;
            public NativeArray<double> Work;

            /// <summary>The lanes that share the island's solve, or null.</summary>
            public SparseSolveTeam* Team;

            /// <summary>How long the factor and the boundary block took, in profiler timestamp units.</summary>
            public long FactorTicks;
            public long BoundaryTicks;

            public void Dispose()
            {
                RowClass.Dispose();
                BlockSections.Dispose();
                RowBlock.Dispose();
                Boundary.Dispose();
                BoundaryOf.Dispose();
                W.Dispose();
                Velocity.Dispose();
                Work.Dispose();
            }
        }

        /// <summary>
        /// Solves an island's rows as a forest plus a boundary and writes every row's impulse, or returns false, changing
        /// nothing, when the island isn't one for it: no forest rows, or more boundary rows than the cap.
        /// </summary>
        static unsafe bool TryStartForest(in SparseConstraintSolver.Problem problem, JacobianHeader** headers, bool bounded,
            ForestIsland* forest, int boundaryCap, SparseSolveTeam* team, out ForestRun run)
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
            int forestRows = 0, boundaryRows = 0;
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                JacobianFlags flags = headers[b]->Flags;
                bool forestBlock = headers[b]->Type != JacobianType.Contact && (flags & JacobianFlags.ForestRow) != 0
                    && (block.BodyA >= 0 || block.BodyB >= 0);
                if (forestBlock)
                {
                    forestBlocks.Add(b);
                    treeBlocks.Add((flags & JacobianFlags.ForestTree) != 0);
                }

                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    rowBlock[row] = b;
                    float lower = problem.Lower[row], upper = problem.Upper[row];
                    bool free = !bounded || (lower <= -float.MaxValue && upper >= float.MaxValue);
                    if (bounded && lower == upper && !coupled[row])
                    {
                        rowClass[row] = FixedRowClass;
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
                    }
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

            if (forestRows == 0 || boundaryRows > boundaryCap || !couplingsInBoundary)
            {
                coupled.Dispose();
                rowClass.Dispose();
                rowBlock.Dispose();
                forestBlocks.Dispose();
                treeBlocks.Dispose();
                return false;
            }

            // The structure, kept while the island's bodies and forest blocks stay the same.
            ref UnsafeList<ulong> signature = ref forest->Signature;
            bool same = signature.Length == forestBlocks.Length + 1 && signature[0] == (ulong)problem.BodyCount;
            for (int f = 0; same && f < forestBlocks.Length; f++)
            {
                same = signature[f + 1] == BlockKey(problem.Blocks[forestBlocks[f]], treeBlocks[f]);
            }

            if (!same)
            {
                signature.Clear();
                signature.Add((ulong)problem.BodyCount);
                for (int f = 0; f < forestBlocks.Length; f++)
                {
                    signature.Add(BlockKey(problem.Blocks[forestBlocks[f]], treeBlocks[f]));
                }

                forest->Build(problem, (int*)forestBlocks.GetUnsafeReadOnlyPtr(), (bool*)treeBlocks.GetUnsafeReadOnlyPtr(),
                    forestBlocks.Length);
            }

            int sections = forest->SectionCount;
            ref VesselContext context = ref forest->Context;
            int* partSection = forest->Structure.PartSection.Ptr;
            var blockSections = new NativeArray<int2>(problem.BlockCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                blockSections[b] = new int2(block.BodyA >= 0 ? partSection[block.BodyA] : -1, block.BodyB >= 0 ? partSection[block.BodyB] : -1);
            }

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
                            float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
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

                for (int i = 0; i < problem.BodyCount; i++)
                {
                    forest->Sections[partSection[i]] = new VesselSection
                    {
                        Mass = problem.Bodies[i].Mass,
                        Inertia = problem.Bodies[i].Inertia,
                    };
                }

                context.RowJacobian = problem.Jacobian;
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
                    float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    int seedCount = 0;
                    if (sides.x >= 0)
                    {
                        seeds[seedCount++] = sides.x;
                        (double6.Load(yPtr + 6 * sides.x) + VesselContext.LoadRow(columns)).Store(yPtr + 6 * sides.x);
                    }

                    if (sides.y >= 0)
                    {
                        seeds[seedCount++] = sides.y;
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
            forestBlocks.Dispose();
            treeBlocks.Dispose();
            run = new ForestRun
            {
                Forest = forest,
                Problem = problem,
                Bounded = bounded,
                RowClass = rowClass,
                BlockSections = blockSections,
                RowBlock = rowBlock,
                Boundary = boundary,
                BoundaryOf = boundaryOf,
                W = w,
                BoundaryCount = nb,
                Velocity = new NativeArray<double>(6 * sections, Allocator.Temp),
                Work = new NativeArray<double>(6 * sections, Allocator.Temp),
                Team = team,
                FactorTicks = boundaryStarted - factorStarted,
                BoundaryTicks = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - boundaryStarted,
            };
            return true;
        }

        static ulong BlockKey(in SparseConstraintSolver.ConstraintBlock block, bool tree)
        {
            return (ulong)(uint)(block.BodyA + 1) | ((ulong)(uint)(block.BodyB + 1) << 31) | (tree ? 1UL << 62 : 0UL);
        }

        /// <summary>
        /// Solves the run's island for the right-hand side <paramref name="rhs"/> against the factor and boundary block it
        /// holds, and writes every row's impulse. Returns false if the boundary's pivoting ran out of iterations, leaving
        /// its impulses clamped to their bounds.
        /// </summary>
        static unsafe bool SolveForest(ref ForestRun run, float* rhs, float* impulses)
        {
            using ProfilerMarker.AutoScope span = ForestSolveMarker.Auto();

            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            ref VesselContext context = ref run.Forest->Context;
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

                AddRow(v, run.BlockSections[run.RowBlock[row]], problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row, scale);
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
                    q[i] = rhs[row] - RowSpeed(v, run.BlockSections[run.RowBlock[row]], problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
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
                    AddRow(work, run.BlockSections[run.RowBlock[row]], problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row, z[i]);
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
                    double speed = RowSpeed(v, run.BlockSections[run.RowBlock[row]], problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
                    impulses[row] = (float)((rhs[row] - speed) / problem.Epsilon[row]);
                }
                else if (rowClass == FixedRowClass)
                {
                    impulses[row] = problem.Lower[row];
                }
            }

            return solved;
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
