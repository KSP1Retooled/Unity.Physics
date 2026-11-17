using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// The forest path's strut loads: inside a locked section, the locked pairs on a closure's loop and the closure's
        /// own rows share the section's loads by the rows' stiffness, as the vessel kernel's sections do (see
        /// <see cref="SectionStatics"/>). The pairs that no closure's loop crosses keep the least-squares split of their
        /// subtree's need. Runs after <see cref="SplitLoadsTask"/> and replaces the impulses of the looped pairs' rows and of
        /// the closures' rows, which are zero before.
        /// </summary>
        static unsafe void ForestStrutLoads(ref ForestRun run, in SparseConstraintSolver.Problem problem, float* impulses)
        {
            ForestIsland* forest = run.Forest;
            if (!StrutLoads.Enabled)
            {
                return;
            }

            if (forest->ForestStaticsBuiltFor != forest->StructureBuilds)
            {
                using (StrutLoads.FactorMarker.Auto())
                {
                    BuildForestStatics(ref run, problem);
                }

                forest->ForestStaticsBuiltFor = forest->StructureBuilds;
            }

            if (forest->ForestStatics.Length == 0)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = StrutLoads.SolveMarker.Auto();
            double6* need = (double6*)run.Need.GetUnsafePtr();
            for (int e = 0; e < forest->ForestStatics.Length; e++)
            {
                SectionStatics* entry = forest->ForestStatics.Ptr + e;
                int count = entry->Count;
                double3x3 rotation = new double3x3(new float3x3(run.MotionDatas[run.WorldBodies[entry->FirstBody]].WorldFromMotion.rot));
                double3x3 back = math.transpose(rotation);

                // What the links must give each unknown body: the need of its subtree, less the needs of the subtrees of
                // its children on loops, moved to its centre as the subtree pass moves them.
                var buffer = new NativeArray<double>(6 * math.max(1, count), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                double* rhs = (double*)buffer.GetUnsafePtr();
                for (int o = 0; o < count; o++)
                {
                    need[entry->Bodies[o]].Store(rhs + 6 * entry->Position[o]);
                }

                for (int l = 0; l < entry->PairLinks.Length; l++)
                {
                    int4 link = entry->PairLinks[l];
                    if (link.w != 0 || link.z < 0)
                    {
                        continue;
                    }

                    int child = entry->Bodies[link.y], parent = entry->Bodies[link.z];
                    double6 carried = need[child];
                    double3 force = new double3(carried.Low.x, carried.Low.y, carried.Low.z);
                    double3 torque = new double3(carried.Low.w, carried.High.x, carried.High.y) + math.cross(run.Arm[child] - run.Arm[parent], force);
                    double* to = rhs + 6 * entry->Position[link.z];
                    (double6.Load(to) - new double6(new double4(force, torque.x), new double2(torque.y, torque.z))).Store(to);
                }

                // In the frame of the section's first body.
                for (int j = 0; j < count; j++)
                {
                    VectorMath.Store3(rhs + 6 * j, math.mul(back, VectorMath.Load3(rhs + 6 * j)));
                    VectorMath.Store3(rhs + 6 * j + 3, math.mul(back, VectorMath.Load3(rhs + 6 * j + 3)));
                }

                SectionStaticsSolve.Solve(entry, rhs);

                // Each row's impulse is its stiffness times its stretch: then the rows give every unknown body its need.
                for (int l = 0; l < entry->PairLinks.Length; l++)
                {
                    int4 link = entry->PairLinks[l];
                    int pair = link.x;
                    for (int k = forest->PairBlockStart[pair]; k < forest->PairBlockStart[pair + 1]; k++)
                    {
                        SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                        int unknownA = block.BodyA >= 0 ? entry->BodyUnknown(block.BodyA) : -1;
                        int unknownB = block.BodyB >= 0 ? entry->BodyUnknown(block.BodyB) : -1;
                        for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                        {
                            if (run.RowClass[row] != SectionRowClass)
                            {
                                continue;
                            }

                            float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                            double stretch = 0.0;
                            if (unknownA >= 0)
                            {
                                stretch += double6.Dot(LocalColumns(columns, back), double6.Load(rhs + 6 * entry->Position[unknownA]));
                            }

                            if (unknownB >= 0)
                            {
                                stretch += double6.Dot(LocalColumns(columns + 6, back), double6.Load(rhs + 6 * entry->Position[unknownB]));
                            }

                            impulses[row] = VectorMath.Flushed(stretch * RowStiffness(ref run, problem, pair, k, row));
                        }
                    }
                }
            }
        }

        /// <summary>One side's six columns of a row, turned into a section's frame.</summary>
        static unsafe double6 LocalColumns(float* columns, in double3x3 back)
        {
            double3 linear = math.mul(back, new double3(columns[0], columns[1], columns[2]));
            double3 angular = math.mul(back, new double3(columns[3], columns[4], columns[5]));
            return new double6(new double4(linear, angular.x), new double2(angular.y, angular.z));
        }

        /// <summary>
        /// Builds and factors the static system of each multi-part section that has closures inside it, from the rows of
        /// the locked pairs on the closures' loops and of the closures, in the frame of the section's first body.
        /// </summary>
        static unsafe void BuildForestStatics(ref ForestRun run, in SparseConstraintSolver.Problem problem)
        {
            ForestIsland* forest = run.Forest;
            for (int e = 0; e < forest->ForestStatics.Length; e++)
            {
                forest->ForestStatics[e].Dispose();
            }

            forest->ForestStatics.Clear();
            ref VesselStructure structure = ref forest->Structure;
            int sections = forest->SectionCount;
            int parts = forest->Parts.Length;

            // The pair of each closure: the structure numbers the closures by the order of the pairs that close loops.
            var closurePair = new NativeArray<int>(math.max(1, forest->Closures.Length), Allocator.Temp);
            for (int p = 0, c = 0; p < forest->PairCount; p++)
            {
                if (forest->PairEdge[p] == 0)
                {
                    closurePair[c++] = p;
                }
            }

            var sectionClosures = new NativeArray<int>(math.max(1, sections), Allocator.Temp);
            for (int c = 0; c < forest->Closures.Length; c++)
            {
                int a = forest->Closures[c].PartA, b = forest->Closures[c].PartB;
                if (a != b && structure.PartSection[a] == structure.PartSection[b])
                {
                    sectionClosures[structure.PartSection[a]]++;
                }
            }

            var place = new NativeArray<int>(math.max(1, parts), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int s = 0; s < sections; s++)
            {
                if (sectionClosures[s] == 0 || !Multi(forest, s))
                {
                    continue;
                }

                int start = structure.SectionPartStart[s], end = structure.SectionPartStart[s + 1];
                int n = end - start;
                int* sectionParts = structure.SectionParts.Ptr + start;
                var depth = new NativeArray<int>(n, Allocator.Temp);
                var parentPlace = new NativeArray<int>(n, Allocator.Temp);
                var looped = new NativeArray<bool>(n, Allocator.Temp);
                for (int k = 0; k < n; k++)
                {
                    place[sectionParts[k]] = k;
                }

                parentPlace[0] = -1;
                for (int k = 1; k < n; k++)
                {
                    parentPlace[k] = place[forest->Edges[sectionParts[k]].Parent];
                    depth[k] = depth[parentPlace[k]] + 1;
                }

                var closures = new NativeList<int>(Allocator.Temp);
                for (int c = 0; c < forest->Closures.Length; c++)
                {
                    int partA = forest->Closures[c].PartA, partB = forest->Closures[c].PartB;
                    if (partA == partB || structure.PartSection[partA] != s || structure.PartSection[partB] != s)
                    {
                        continue;
                    }

                    closures.Add(c);
                    int a = place[partA], b = place[partB];
                    while (a != b)
                    {
                        if (depth[a] >= depth[b])
                        {
                            looped[a] = true;
                            a = parentPlace[a];
                        }
                        else
                        {
                            looped[b] = true;
                            b = parentPlace[b];
                        }
                    }
                }

                var entry = new SectionStatics
                {
                    KeyParts = new UnsafeList<int>(0, Allocator.Persistent),
                    Node = new UnsafeList<int>(0, Allocator.Persistent),
                    Position = new UnsafeList<int>(0, Allocator.Persistent),
                    ColumnStart = new UnsafeList<int>(0, Allocator.Persistent),
                    Rows = new UnsafeList<int>(0, Allocator.Persistent),
                    Inverse = new UnsafeList<double>(0, Allocator.Persistent),
                    Blocks = new UnsafeList<double>(0, Allocator.Persistent),
                    Links = new UnsafeList<StaticLink>(0, Allocator.Persistent),
                    Bodies = new UnsafeList<int>(0, Allocator.Persistent),
                    PairLinks = new UnsafeList<int4>(0, Allocator.Persistent),
                    FirstBody = forest->PartBody[sectionParts[0]],
                };

                var unknown = new NativeArray<int>(n, Allocator.Temp);
                for (int k = 0; k < n; k++)
                {
                    unknown[k] = -1;
                    if (looped[k])
                    {
                        unknown[k] = entry.Bodies.Length;
                        entry.Bodies.Add(forest->PartBody[sectionParts[k]]);
                    }
                }

                entry.Count = entry.Bodies.Length;
                entry.BodyMap = new UnsafeList<int>(math.max(1, problem.BodyCount), Allocator.Persistent);
                entry.BodyMap.Resize(math.max(1, problem.BodyCount));
                for (int body = 0; body < entry.BodyMap.Length; body++)
                {
                    entry.BodyMap[body] = -1;
                }

                for (int o = 0; o < entry.Count; o++)
                {
                    entry.BodyMap[entry.Bodies[o]] = o;
                }
                for (int k = 1; k < n; k++)
                {
                    if (looped[k] && forest->PartPair[sectionParts[k]] >= 0)
                    {
                        entry.PairLinks.Add(new int4(forest->PartPair[sectionParts[k]], unknown[k], unknown[parentPlace[k]], 0));
                    }
                }

                for (int i = 0; i < closures.Length; i++)
                {
                    VesselClosure closure = forest->Closures[closures[i]];
                    entry.PairLinks.Add(new int4(closurePair[closures[i]], unknown[place[closure.PartB]], unknown[place[closure.PartA]], 1));
                }

                // Assemble each link's blocks from its rows: w j j^T on each moving side, and w j_a j_b^T between them.
                double3x3 back = math.transpose(new double3x3(new float3x3(run.MotionDatas[run.WorldBodies[entry.FirstBody]].WorldFromMotion.rot)));
                int count = entry.Count;
                var diagonal = new NativeArray<double>(VesselBlocks.Size * math.max(1, count), Allocator.Temp);
                var couplings = new NativeList<int2>(Allocator.Temp);
                var couplingBlocks = new NativeList<double>(Allocator.Temp);
                double* coupling = stackalloc double[VesselBlocks.Size];
                for (int l = 0; l < entry.PairLinks.Length; l++)
                {
                    int pair = entry.PairLinks[l].x;
                    for (int k = forest->PairBlockStart[pair]; k < forest->PairBlockStart[pair + 1]; k++)
                    {
                        SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                        int unknownA = block.BodyA >= 0 ? entry.BodyUnknown(block.BodyA) : -1;
                        int unknownB = block.BodyB >= 0 ? entry.BodyUnknown(block.BodyB) : -1;
                        VesselBlocks.Zero(coupling);
                        for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                        {
                            if (run.RowClass[row] != SectionRowClass)
                            {
                                continue;
                            }

                            float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                            double weight = RowStiffness(ref run, problem, pair, k, row);
                            double6 a = LocalColumns(columns, back), b = LocalColumns(columns + 6, back);
                            if (unknownA >= 0)
                            {
                                VesselBlocks.AddOuter((double*)diagonal.GetUnsafePtr() + VesselBlocks.Size * unknownA, weight, a, a);
                            }

                            if (unknownB >= 0)
                            {
                                VesselBlocks.AddOuter((double*)diagonal.GetUnsafePtr() + VesselBlocks.Size * unknownB, weight, b, b);
                            }

                            VesselBlocks.AddOuter(coupling, weight, a, b);
                        }

                        if (unknownA >= 0 && unknownB >= 0 && unknownA != unknownB)
                        {
                            couplings.Add(new int2(unknownA, unknownB));
                            for (int i = 0; i < VesselBlocks.Size; i++)
                            {
                                couplingBlocks.Add(coupling[i]);
                            }
                        }
                    }
                }

                // Rows that don't reach every direction leave the rest to rounding, not to a singular matrix.
                for (int o = 0; o < count; o++)
                {
                    double* d = (double*)diagonal.GetUnsafePtr() + VesselBlocks.Size * o;
                    double trace = d[0] + d[7] + d[14] + d[21] + d[28] + d[35];
                    for (int i = 0; i < 6; i++)
                    {
                        d[7 * i] += 1e-12 * trace + 1e-300;
                    }
                }

                StaticFactor.Factor(&entry, count, (double*)diagonal.GetUnsafePtr(), couplings.GetUnsafePtr(),
                    couplingBlocks.GetUnsafePtr(), couplings.Length);
                System.Threading.Interlocked.Increment(ref StrutLoads.Counts.Data.Factored);
                System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.FactoredUnknowns, count);
                forest->ForestStatics.Add(entry);
            }
        }
    }
}
