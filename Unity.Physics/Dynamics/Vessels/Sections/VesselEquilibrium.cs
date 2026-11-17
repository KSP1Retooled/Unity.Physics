using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Numerics.Linear.Sparse;
using Unity.Physics.Dynamics.Vessels.Model;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>
    /// The equilibrium systems of a vessel solver's sections that have closures inside them, kept from one structure to the
    /// next while a section keeps its parts.
    /// </summary>
    internal unsafe struct VesselEquilibrium : IDisposable
    {
        public UnsafeList<SectionEquilibrium> Entries;

        /// <summary>Each section's system in <see cref="Entries"/>, or -1 when the section takes the tree method.</summary>
        public UnsafeList<int> SectionEntry;

        /// <summary>The closures inside each section, in ascending order, from SectionClosureStart[s].</summary>
        public UnsafeList<int> SectionClosureStart;

        public UnsafeList<int> SectionClosures;

        /// <summary>The structure build that the systems were matched to, or -1 before the first.</summary>
        public int BuiltFor;

        /// <summary>How many systems were factored, and how many were kept for a new structure, for tests.</summary>
        public int Builds;

        public int Reuses;

        /// <summary>The time spent matching and factoring the systems, in profiler ticks, for the cost report.</summary>
        public long PrepareTicks;

        public static VesselEquilibrium Create(Allocator allocator) => new VesselEquilibrium
        {
            Entries = new UnsafeList<SectionEquilibrium>(4, allocator),
            SectionEntry = new UnsafeList<int>(16, allocator),
            SectionClosureStart = new UnsafeList<int>(16, allocator),
            SectionClosures = new UnsafeList<int>(16, allocator),
            BuiltFor = -1,
        };

        public void Dispose()
        {
            for (int e = 0; e < Entries.Length; e++)
            {
                Entries[e].Dispose();
            }

            Entries.Dispose();
            SectionEntry.Dispose();
            SectionClosureStart.Dispose();
            SectionClosures.Dispose();
        }

        /// <summary>
        /// Matches every section that has a closure inside it with a kept system, and factors a new system for each section
        /// that has none. A kept system is used again only when the vessel, its structure version, the time step and the
        /// section's list of parts are all the same.
        /// </summary>
        /// <param name="context">The solver's context, after a structure build, with the parts' current poses.</param>
        /// <param name="partCount">The number of parts.</param>
        /// <param name="closureCount">The number of closures.</param>
        /// <param name="keyId">For each part, the Id of its vessel, which is never reused.</param>
        /// <param name="keyVersion">For each part, its vessel's structure version.</param>
        /// <param name="keyIndex">For each part, its index in its vessel.</param>
        /// <param name="timeStep">The step that the rows' weights h^2 k + h c are for.</param>
        public void Prepare(in VesselContext context, int partCount, int closureCount, ulong* keyId, uint* keyVersion,
            int* keyIndex, double timeStep)
        {
            int sections = context.SectionCount;
            SectionEntry.Resize(math.max(1, sections));
            SectionClosureStart.Resize(sections + 1);
            UnsafeUtility.MemClear(SectionClosureStart.Ptr, (long)(sections + 1) * sizeof(int));
            for (int s = 0; s < sections; s++)
            {
                SectionEntry[s] = -1;
            }

            for (int e = 0; e < Entries.Length; e++)
            {
                Entries.Ptr[e].Claimed = false;
            }

            if (StrutLoads.Enabled)
            {
                // The closures inside each section, in the order of their indices.
                for (int c = 0; c < closureCount; c++)
                {
                    int s = InsideSection(context, c);
                    if (s >= 0)
                    {
                        SectionClosureStart[s + 1]++;
                    }
                }
            }

            for (int s = 0; s < sections; s++)
            {
                SectionClosureStart[s + 1] += SectionClosureStart[s];
            }

            SectionClosures.Resize(math.max(1, SectionClosureStart[sections]));
            if (SectionClosureStart[sections] > 0)
            {
                var fill = new NativeArray<int>(sections, Allocator.Temp);
                for (int c = 0; c < closureCount; c++)
                {
                    int s = InsideSection(context, c);
                    if (s >= 0)
                    {
                        SectionClosures[SectionClosureStart[s] + fill[s]++] = c;
                    }
                }
            }

            // Match each section that needs a system with a kept one.
            var fresh = new NativeList<int>(Allocator.Temp);
            for (int s = 0; s < sections; s++)
            {
                if (SectionClosureStart[s + 1] == SectionClosureStart[s])
                {
                    continue;
                }

                int start = context.SectionPartStart[s], end = context.SectionPartStart[s + 1];
                int first = context.SectionParts[start];
                int match = -1;
                for (int e = 0; e < Entries.Length && match < 0; e++)
                {
                    SectionEquilibrium* entry = Entries.Ptr + e;
                    if (entry->Claimed || entry->Id != keyId[first] || entry->Version != keyVersion[first]
                        || entry->TimeStep != timeStep || entry->KeyParts.Length != end - start)
                    {
                        continue;
                    }

                    bool same = true;
                    for (int k = start; same && k < end; k++)
                    {
                        same = entry->KeyParts[k - start] == keyIndex[context.SectionParts[k]];
                    }

                    match = same ? e : -1;
                }

                if (match >= 0)
                {
                    Entries.Ptr[match].Claimed = true;
                    SectionEntry[s] = match;
                    Reuses++;
                    continue;
                }

                var created = new SectionEquilibrium
                {
                    Id = keyId[first],
                    Version = keyVersion[first],
                    TimeStep = timeStep,
                    KeyParts = new UnsafeList<int>(end - start, Allocator.Persistent),
                    Node = new UnsafeList<int>(0, Allocator.Persistent),
                    ColumnStart = new UnsafeList<int>(0, Allocator.Persistent),
                    Rows = new UnsafeList<int>(0, Allocator.Persistent),
                    Inverse = new UnsafeList<double>(0, Allocator.Persistent),
                    Blocks = new UnsafeList<double>(0, Allocator.Persistent),
                    Links = new UnsafeList<EquilibriumLink>(0, Allocator.Persistent),
                    Claimed = true,
                };
                for (int k = start; k < end; k++)
                {
                    created.KeyParts.Add(keyIndex[context.SectionParts[k]]);
                }

                created.Sum = new UnsafeList<double>(0, Allocator.Persistent);

                SectionEntry[s] = Entries.Length;
                Entries.Add(created);
                fresh.Add(s);
            }

            // Drop the systems that no section uses now, and renumber the others.
            var renumber = new NativeArray<int>(math.max(1, Entries.Length), Allocator.Temp);
            int kept = 0;
            for (int e = 0; e < Entries.Length; e++)
            {
                if (Entries[e].Claimed)
                {
                    renumber[e] = kept;
                    Entries[kept++] = Entries[e];
                }
                else
                {
                    Entries[e].Dispose();
                    renumber[e] = -1;
                }
            }

            Entries.Resize(kept);
            for (int s = 0; s < sections; s++)
            {
                SectionEntry[s] = SectionEntry[s] >= 0 ? renumber[SectionEntry[s]] : -1;
            }

            for (int i = 0; i < fresh.Length; i++)
            {
                int s = fresh[i];
                using (StrutLoads.FactorMarker.Auto())
                {
                    long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    Build(context, s, Entries.Ptr + SectionEntry[s], timeStep, partCount);
                    SectionEquilibrium* built = Entries.Ptr + SectionEntry[s];
                    built->Sum.Resize(6 * math.max(1, built->Count), NativeArrayOptions.ClearMemory);
                    built->Summed = 0;
                    long ticks = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - started;
                    System.Threading.Interlocked.Increment(ref StrutLoads.Counts.Data.Factored);
                    System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.FactoredUnknowns, Entries[SectionEntry[s]].Count);
                    System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.FactoredBlocks, Entries[SectionEntry[s]].Rows.Length);
                    System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.FactorTicks, ticks);
                }

                Builds++;
            }
        }

        /// <summary>Returns the section that closure c lies inside, or -1 when its parts are in different sections.</summary>
        static int InsideSection(in VesselContext context, int c)
        {
            VesselClosure* closure = context.Closures + c;
            int a = context.PartSection[closure->PartA], b = context.PartSection[closure->PartB];
            return a == b && closure->PartA != closure->PartB ? a : -1;
        }

        /// <summary>Builds and factors section s's equilibrium system.</summary>
        void Build(in VesselContext context, int s, SectionEquilibrium* entry, double h, int partCount)
        {
            int start = context.SectionPartStart[s], end = context.SectionPartStart[s + 1];
            int n = end - start;
            int* sectionParts = context.SectionParts + start;
            VesselPart* parts = context.Parts;
            VesselEdge* edges = context.Edges;

            // Each part's place in the section's list, and its depth below the section's first part.
            var place = new NativeArray<int>(partCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var depth = new NativeArray<int>(n, Allocator.Temp);
            var parentPlace = new NativeArray<int>(n, Allocator.Temp);
            for (int k = 0; k < n; k++)
            {
                place[sectionParts[k]] = k;
            }

            parentPlace[0] = -1;
            for (int k = 1; k < n; k++)
            {
                parentPlace[k] = place[edges[sectionParts[k]].Parent];
                depth[k] = depth[parentPlace[k]] + 1;
            }

            // An edge on the tree path between a closure's two parts lies on that closure's loop.
            var looped = new NativeArray<bool>(n, Allocator.Temp);
            int closureStart = SectionClosureStart[s], closureEnd = SectionClosureStart[s + 1];
            for (int c = closureStart; c < closureEnd; c++)
            {
                VesselClosure* closure = context.Closures + SectionClosures[c];
                int a = place[closure->PartA], b = place[closure->PartB];
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

            // The unknowns: the parts whose own edge is on a loop, in the order of the section's list for now.
            var unknown = new NativeArray<int>(n, Allocator.Temp);
            int count = 0;
            for (int k = 0; k < n; k++)
            {
                unknown[k] = looped[k] ? count++ : -1;
            }

            entry->Count = count;
            entry->Node.Resize(count);
            for (int k = 0; k < n; k++)
            {
                if (unknown[k] >= 0)
                {
                    entry->Node[unknown[k]] = k;
                }
            }

            // The section's frame: the first part's rotation, from its centre.
            VesselPart* firstPart = parts + sectionParts[0];
            double3x3 back = math.transpose(DoubleMath.Rotation(firstPart->Orientation));
            double3 origin = firstPart->Position;
            double omega = context.HardFrequency;
            double dampingPerStiffness = 2.0 * context.HardDampingRatio / omega;

            entry->Links.Clear();
            for (int k = 1; k < n; k++)
            {
                if (!looped[k])
                {
                    continue;
                }

                VesselPart* child = parts + sectionParts[k];
                VesselPart* parent = parts + sectionParts[parentPlace[k]];
                VesselEdge* edge = edges + sectionParts[k];
                double hardLinear = HardLinear(omega, child, parent);
                double hardAngular = HardAngular(omega, child, parent);

                // The locked edge as a yielding one: hard linear rows, and its twist and swing springs, hard when infinite.
                double3 linearStiffness = hardLinear, linearDamping = hardLinear * dampingPerStiffness;
                double3 angularStiffness = new double3(edge->TwistStiffness, edge->SwingStiffness, edge->SwingStiffness);
                double3 angularDamping = new double3(edge->TwistDamping, edge->SwingDamping, edge->SwingDamping);
                Harden(ref angularStiffness, ref angularDamping, hardAngular, dampingPerStiffness);
                double3x3 childRotation = DoubleMath.Rotation(child->Orientation);
                double3x3 parentRotation = DoubleMath.Rotation(parent->Orientation);
                double3 point = child->Position + math.mul(childRotation, edge->ChildAnchor);
                double3x3 axes = math.mul(parentRotation, edge->Frame);
                double3 linearWeight = h * h * linearStiffness + h * linearDamping;
                double3 angularWeight = h * h * angularStiffness + h * angularDamping;

                // A locked edge holds every axis. An axis with no spring at all gets a tiny share of the hard weight, so
                // that the edge still carries what nothing else can.
                double floor = 1e-9 * (h * h * hardAngular + h * hardAngular * dampingPerStiffness);
                angularWeight = math.max(angularWeight, floor);
                entry->Links.Add(MakeLink(unknown[parentPlace[k]], unknown[k], k, point, axes, linearWeight, angularWeight,
                    back, origin));
            }

            for (int c = closureStart; c < closureEnd; c++)
            {
                VesselClosure* closure = context.Closures + SectionClosures[c];
                VesselPart* a = parts + closure->PartA;
                VesselPart* b = parts + closure->PartB;
                double3 linearStiffness = closure->LinearStiffness, linearDamping = closure->LinearDamping;
                double3 angularStiffness = closure->AngularStiffness, angularDamping = closure->AngularDamping;
                Harden(ref linearStiffness, ref linearDamping, HardLinear(omega, b, a), dampingPerStiffness);
                Harden(ref angularStiffness, ref angularDamping, HardAngular(omega, b, a), dampingPerStiffness);
                double3 linearWeight = h * h * linearStiffness + h * linearDamping;
                double3 angularWeight = h * h * angularStiffness + h * angularDamping;
                if (!(math.cmax(linearWeight) > 0.0) && !(math.cmax(angularWeight) > 0.0))
                {
                    // A tombstone has no stiffness and carries nothing.
                    continue;
                }

                double3 point = b->Position + math.mul(DoubleMath.Rotation(b->Orientation), closure->AnchorB);
                double3x3 axes = math.mul(DoubleMath.Rotation(a->Orientation), closure->Frame);
                entry->Links.Add(MakeLink(unknown[place[closure->PartA]], unknown[place[closure->PartB]], -1 - (c - closureStart),
                    point, axes, linearWeight, angularWeight, back, origin));
            }

            // The ordering and the factor's structure, from the links between two unknowns.
            var adjacencyStart = new NativeArray<int>(count + 1, Allocator.Temp);
            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink link = entry->Links[l];
                if (link.A >= 0 && link.B >= 0)
                {
                    adjacencyStart[link.A + 1]++;
                    adjacencyStart[link.B + 1]++;
                }
            }

            for (int v = 0; v < count; v++)
            {
                adjacencyStart[v + 1] += adjacencyStart[v];
            }

            var adjacency = new NativeArray<int>(math.max(1, adjacencyStart[count]), Allocator.Temp);
            var adjacencyFill = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink link = entry->Links[l];
                if (link.A >= 0 && link.B >= 0)
                {
                    adjacency[adjacencyStart[link.A] + adjacencyFill[link.A]++] = link.B;
                    adjacency[adjacencyStart[link.B] + adjacencyFill[link.B]++] = link.A;
                }
            }

            var order = new NativeList<int>(math.max(1, count), Allocator.Temp);
            MinimumDegreeOrdering.Compute(count, adjacency, adjacencyStart, order);
            var position = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            var columnStart = new NativeArray<int>(count + 1, Allocator.Temp);
            var parentOf = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            var firstChild = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            var nextSibling = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            var rows = new NativeList<int>(math.max(1, 2 * count), Allocator.Temp);
            SymbolicCholesky.FactorStructure(order, position, adjacency, adjacencyStart, columnStart, rows, parentOf,
                firstChild, nextSibling);

            entry->ColumnStart.Clear();
            entry->ColumnStart.AddRange(columnStart.GetUnsafeReadOnlyPtr(), count + 1);
            entry->Rows.Clear();
            entry->Rows.AddRange(rows.GetUnsafeReadOnlyPtr(), rows.Length);

            // Number everything by elimination position from here on.
            var node = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            for (int k = 0; k < n; k++)
            {
                if (unknown[k] >= 0)
                {
                    node[position[unknown[k]]] = k;
                }
            }

            for (int j = 0; j < count; j++)
            {
                entry->Node[j] = node[j];
            }

            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink* link = entry->Links.Ptr + l;
                link->A = link->A >= 0 ? position[link->A] : -1;
                link->B = link->B >= 0 ? position[link->B] : -1;
            }

            // Assemble: each link adds its block to both sides' diagonals and takes it from the block between them.
            int offDiagonal = rows.Length;
            entry->Inverse.Resize(Block6x6.Size * math.max(1, count));
            entry->Blocks.Resize(Block6x6.Size * (count + offDiagonal));
            UnsafeUtility.MemClear(entry->Blocks.Ptr, (long)entry->Blocks.Length * sizeof(double));
            double* blocks = entry->Blocks.Ptr;
            double* off = blocks + Block6x6.Size * count;
            double* block = stackalloc double[Block6x6.Size];
            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink link = entry->Links[l];
                LinkBlock(link, block);
                if (link.A >= 0)
                {
                    Block6x6.Add(blocks + Block6x6.Size * link.A, block);
                }

                if (link.B >= 0)
                {
                    Block6x6.Add(blocks + Block6x6.Size * link.B, block);
                }

                if (link.A >= 0 && link.B >= 0 && link.A != link.B)
                {
                    int lower = math.min(link.A, link.B), upper = math.max(link.A, link.B);
                    int e = Find(entry->Rows.Ptr, columnStart[lower], columnStart[lower + 1], upper);
                    double* coupling = off + Block6x6.Size * e;
                    for (int i = 0; i < Block6x6.Size; i++)
                    {
                        coupling[i] -= block[i];
                    }
                }
            }

            // Eliminate the columns in order, as VesselContext.EliminateColumn does, with the targets found by search.
            entry->ClampedPivots = 0;
            double* scratch = stackalloc double[Block6x6.Size];
            int* structure = entry->Rows.Ptr;
            for (int j = 0; j < count; j++)
            {
                double* inverse = entry->Inverse.Ptr + Block6x6.Size * j;
                if (!Block6x6.InvertSymmetric(inverse, blocks + Block6x6.Size * j))
                {
                    entry->ClampedPivots++;
                }

                int columnFirst = columnStart[j], columnCount = columnStart[j + 1] - columnFirst;
                double* column = off + Block6x6.Size * columnFirst;
                for (int p = columnCount - 1; p >= 0; p--)
                {
                    double* up = column + Block6x6.Size * p;
                    int row = structure[columnFirst + p];
                    Block6x6.Mul(scratch, inverse, up);
                    Block6x6.SubtractTransposeMul(blocks + Block6x6.Size * row, up, scratch);
                    for (int q = p + 1; q < columnCount; q++)
                    {
                        int e = Find(structure, columnStart[row], columnStart[row + 1], structure[columnFirst + q]);
                        Block6x6.SubtractTransposeMul(off + Block6x6.Size * e, up, column + Block6x6.Size * q);
                    }

                    Block6x6.Copy(up, scratch);
                }
            }

            // The diagonal blocks are not needed after the factor: keep only the inverses and the scaled column blocks.
            UnsafeUtility.MemMove(blocks, off, (long)Block6x6.Size * offDiagonal * sizeof(double));
            entry->Blocks.Resize(Block6x6.Size * math.max(1, offDiagonal));
        }

        static double HardLinear(double omega, VesselPart* a, VesselPart* b) => omega * omega / (1.0 / a->Mass + 1.0 / b->Mass);

        static double HardAngular(double omega, VesselPart* a, VesselPart* b)
        {
            double traceA = math.csum(1.0 / a->Inertia), traceB = math.csum(1.0 / b->Inertia);
            return omega * omega / ((traceA + traceB) / 3.0);
        }

        /// <summary>Gives each axis whose stiffness is infinite the hard stiffness, as VesselContext.BuildLink does.</summary>
        static void Harden(ref double3 stiffness, ref double3 damping, double hard, double dampingPerStiffness)
        {
            for (int i = 0; i < 3; i++)
            {
                if (double.IsPositiveInfinity(stiffness[i]))
                {
                    stiffness[i] = hard;
                    damping[i] = hard * dampingPerStiffness;
                }
            }
        }

        static EquilibriumLink MakeLink(int a, int b, int target, double3 point, in double3x3 axes, double3 linearWeight,
            double3 angularWeight, in double3x3 back, double3 origin)
        {
            double3x3 localAxes = math.mul(back, axes);
            return new EquilibriumLink
            {
                A = a,
                B = b,
                Target = target,
                Point = math.mul(back, point - origin),
                Linear = DoubleMath.Weighted(localAxes, linearWeight),
                Angular = DoubleMath.Weighted(localAxes, angularWeight),
            };
        }

        /// <summary>
        /// The 6 by 6 block that a link adds to each side's diagonal. With X the cross-product matrix of the joint point,
        /// the relative motion at the point is [I, -X] times the difference of the two sides' motions, so the block is
        /// [[L, -L X], [X L, R - X L X]] for L and R the linear and angular stiffness.
        /// </summary>
        static void LinkBlock(in EquilibriumLink link, double* block)
        {
            double3x3 cross = DoubleMath.Cross(link.Point);
            double3x3 linearCross = math.mul(link.Linear, cross);
            Block6x6.Zero(block);
            Block6x6.SetQuarter(block, 0, 0, link.Linear);
            Block6x6.SetQuarter(block, 0, 1, -linearCross);
            Block6x6.SetQuarter(block, 1, 0, math.transpose(-linearCross));
            Block6x6.SetQuarter(block, 1, 1, link.Angular - math.mul(cross, linearCross));
        }

        static int Find(int* rows, int from, int to, int row)
        {
            int low = from, high = to - 1;
            while (low <= high)
            {
                int middle = (low + high) >> 1;
                int value = rows[middle];
                if (value == row)
                {
                    return middle;
                }

                if (value < row)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle - 1;
                }
            }

            return -1;
        }

        /// <summary>What <see cref="Apply"/> does with a section's loads.</summary>
        public enum ApplyMode
        {
            /// <summary>Solve with this substep's loads and write the looped edges' and closures' loads.</summary>
            Now,

            /// <summary>Add this substep's loads to the section's sum, and leave the tree method's loads in place.</summary>
            Add,

            /// <summary>Solve with the mean of the summed loads and write the looped edges' and closures' loads.</summary>
            Finish,
        }

        /// <summary>Clears every section's sum of loads, before a step's first substep.</summary>
        public void BeginStep()
        {
            for (int e = 0; e < Entries.Length; e++)
            {
                SectionEquilibrium* entry = Entries.Ptr + e;
                if (entry->Sum.IsCreated)
                {
                    UnsafeUtility.MemClear(entry->Sum.Ptr, (long)entry->Sum.Length * sizeof(double));
                }

                entry->Summed = 0;
            }
        }

        /// <summary>
        /// Takes the loads of section s's looped edges and closures from its equilibrium system. The tree method must have
        /// stored every locked edge's load first: an edge's load then is its child's whole subtree, which is also what
        /// the bridges keep. Writes the looped edges' loads and every closure inside the section, unless the mode only
        /// adds this substep's loads to the section's sum. The system is linear, so the solve of the mean of the substeps'
        /// loads is the mean of their solves, and one solve in each step gives the mean loads that the step reports.
        /// </summary>
        public void Apply(ref VesselContext context, int s, ApplyMode mode = ApplyMode.Now)
        {
            int entryIndex = SectionEntry[s];
            int closureStart = SectionClosureStart[s], closureEnd = SectionClosureStart[s + 1];
            for (int c = closureStart; c < closureEnd; c++)
            {
                context.ClosureLoads[SectionClosures[c]] = default;
            }

            if (entryIndex < 0)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = StrutLoads.SolveMarker.Auto();
            long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            SectionEquilibrium* entry = Entries.Ptr + entryIndex;
            int count = entry->Count;
            int start = context.SectionPartStart[s];
            int* sectionParts = context.SectionParts + start;
            VesselPart* firstPart = context.Parts + sectionParts[0];
            double3x3 rotation = DoubleMath.Rotation(firstPart->Orientation);
            double3x3 back = math.transpose(rotation);
            double3 origin = firstPart->Position;
            var buffer = new NativeArray<double>(12 * math.max(1, count), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            double* subtree = (double*)buffer.GetUnsafePtr();
            double* rhs = subtree + 6 * count;
            if (mode != ApplyMode.Finish)
            {
                // Each unknown's subtree load, about the first part's centre, in its frame.
                for (int j = 0; j < count; j++)
                {
                    int part = sectionParts[entry->Node[j]];
                    VesselEdgeLoad load = context.EdgeLoads[part];
                    double3 torque = load.Torque + math.cross(context.ChildPoint[part] - origin, load.Force);
                    DoubleMath.Store3(subtree + 6 * j, math.mul(back, load.Force));
                    DoubleMath.Store3(subtree + 6 * j + 3, math.mul(back, torque));
                }

                // What the links must give each unknown: its own subtree, less the subtrees of its children on loops,
                // which hang from it through links of the system.
                UnsafeUtility.MemCpy(rhs, subtree, 6L * count * sizeof(double));
                for (int l = 0; l < entry->Links.Length; l++)
                {
                    EquilibriumLink link = entry->Links[l];
                    if (link.Target >= 0 && link.A >= 0)
                    {
                        (double6.Load(rhs + 6 * link.A) - double6.Load(subtree + 6 * link.B)).Store(rhs + 6 * link.A);
                    }
                }
            }

            if (mode == ApplyMode.Add)
            {
                double* sum = entry->Sum.Ptr;
                for (int i = 0; i < 6 * count; i++)
                {
                    sum[i] += rhs[i];
                }

                entry->Summed++;
                System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LoadTicks, Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - started);
                return;
            }

            if (mode == ApplyMode.Finish)
            {
                if (entry->Summed == 0)
                {
                    return;
                }

                double mean = 1.0 / entry->Summed;
                double* sum = entry->Sum.Ptr;
                for (int i = 0; i < 6 * count; i++)
                {
                    rhs[i] = mean * sum[i];
                }
            }

            long loaded = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            SectionEquilibriumSolve.Solve(entry, rhs);
            long substituted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;

            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink link = entry->Links[l];
                double6 motion = link.B >= 0 ? double6.Load(rhs + 6 * link.B) : default;
                if (link.A >= 0)
                {
                    motion -= double6.Load(rhs + 6 * link.A);
                }

                double3 shift = new double3(motion.Low.x, motion.Low.y, motion.Low.z);
                double3 turn = new double3(motion.Low.w, motion.High.x, motion.High.y);
                double3 force = math.mul(link.Linear, shift + math.cross(turn, link.Point));
                double3 torque = math.mul(link.Angular, turn);
                var load = new VesselEdgeLoad { Force = math.mul(rotation, force), Torque = math.mul(rotation, torque) };
                if (link.Target >= 0)
                {
                    context.EdgeLoads[sectionParts[link.Target]] = load;
                }
                else
                {
                    context.ClosureLoads[SectionClosures[closureStart - 1 - link.Target]] = load;
                }
            }

            long finished = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            System.Threading.Interlocked.Increment(ref StrutLoads.Counts.Data.Solves);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.SolveTicks, finished - started);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LoadTicks, loaded - started);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.SubstituteTicks, substituted - loaded);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LinkTicks, finished - substituted);
        }
    }
}
