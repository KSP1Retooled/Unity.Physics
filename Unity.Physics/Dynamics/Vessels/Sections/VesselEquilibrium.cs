using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Configuration;
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

        /// <summary>
        /// How many systems that no section uses are kept, and for how many structures at most. An edge that yields
        /// and locks again within that brings back its section's system without a new factor.
        /// </summary>
        public const int KeepUnused = 8;

        public const int KeepMatches = 1000;

        /// <summary>How many times the systems were matched to a structure.</summary>
        public long Matches;

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
            System.Threading.Interlocked.Increment(ref StrutLoads.Counts.Data.Prepares);
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

            if (VesselSwitches.StrutLoadsEnabled)
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
                    System.Threading.Interlocked.Increment(ref StrutLoads.Counts.Data.Reused);
                    if (Entries[match].LastUsed < Matches)
                    {
                        System.Threading.Interlocked.Increment(ref StrutLoads.Counts.Data.Reactivated);
                    }
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
                    Position = new UnsafeList<int>(0, Allocator.Persistent),
                    Claimed = true,
                };
                created.CreateSolveLists();
                for (int k = start; k < end; k++)
                {
                    created.KeyParts.Add(keyIndex[context.SectionParts[k]]);
                }

                created.Sum = new UnsafeList<double>(0, Allocator.Persistent);

                SectionEntry[s] = Entries.Length;
                Entries.Add(created);
                fresh.Add(s);
            }

            // Keep the systems that no section uses now for a while, the most recently used first: an edge that yields
            // and locks again brings its section back with the same parts, and then its system needs no new factor.
            // Decide what to keep before moving entries so that every comparison sees the same list.
            Matches++;
            var renumber = new NativeArray<int>(math.max(1, Entries.Length), Allocator.Temp);
            int kept = 0;
            for (int e = 0; e < Entries.Length; e++)
            {
                if (Entries[e].Claimed)
                {
                    Entries.Ptr[e].LastUsed = Matches;
                }
            }

            for (int e = 0; e < Entries.Length; e++)
            {
                bool keep = Entries[e].Claimed;
                if (!keep)
                {
                    // Count the unused systems that were used more recently than this one, ties going to the lower index.
                    int newer = 0;
                    for (int f = 0; f < Entries.Length; f++)
                    {
                        newer += !Entries[f].Claimed && (Entries[f].LastUsed > Entries[e].LastUsed
                            || (Entries[f].LastUsed == Entries[e].LastUsed && f < e)) ? 1 : 0;
                    }

                    keep = newer < KeepUnused && Matches - Entries[e].LastUsed <= KeepMatches;
                }

                renumber[e] = keep ? 0 : -1;
            }

            for (int e = 0; e < Entries.Length; e++)
            {
                if (renumber[e] >= 0)
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
                    System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.FactoredUnknowns, built->Count);
                    System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.FactoredBlocks, built->Rows.Length);
                    System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.FactoredChunks, built->Chunks.ChunkCount);
                    System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.FactoredTop, built->Chunks.TopColumns.Length);
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
            var ends = new NativeArray<int2>(math.max(1, closureEnd - closureStart), Allocator.Temp);
            for (int c = closureStart; c < closureEnd; c++)
            {
                VesselClosure* closure = context.Closures + SectionClosures[c];
                ends[c - closureStart] = new int2(place[closure->PartA], place[closure->PartB]);
            }

            EquilibriumFactor.MarkLoops(n, (int*)parentPlace.GetUnsafePtr(), (int2*)ends.GetUnsafePtr(), closureEnd - closureStart,
                (bool*)looped.GetUnsafePtr());

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

            // Assemble: each link adds its block to both sides' diagonals and takes it from the block between them.
            var diagonal = new NativeArray<double>(Block6x6.Size * math.max(1, count), Allocator.Temp);
            var couplings = new NativeList<int2>(Allocator.Temp);
            var couplingBlocks = new NativeList<double>(Allocator.Temp);
            double* block = stackalloc double[Block6x6.Size];
            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink link = entry->Links[l];
                LinkBlock(link, block);
                if (link.A >= 0)
                {
                    Block6x6.Add((double*)diagonal.GetUnsafePtr() + Block6x6.Size * link.A, block);
                }

                if (link.B >= 0)
                {
                    Block6x6.Add((double*)diagonal.GetUnsafePtr() + Block6x6.Size * link.B, block);
                }

                if (link.A >= 0 && link.B >= 0 && link.A != link.B)
                {
                    couplings.Add(new int2(link.A, link.B));
                    for (int i = 0; i < Block6x6.Size; i++)
                    {
                        couplingBlocks.Add(-block[i]);
                    }
                }
            }

            EquilibriumFactor.Factor(entry, count, (double*)diagonal.GetUnsafePtr(), couplings.GetUnsafePtr(), couplingBlocks.GetUnsafePtr(),
                couplings.Length);

            // Number everything by elimination position from here on.
            for (int k = 0; k < n; k++)
            {
                if (unknown[k] >= 0)
                {
                    entry->Node[entry->Position[unknown[k]]] = k;
                }
            }

            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink* link = entry->Links.Ptr + l;
                link->A = link->A >= 0 ? entry->Position[link->A] : -1;
                link->B = link->B >= 0 ? entry->Position[link->B] : -1;
            }

            // Each unknown's children on loops, whose subtrees hang from it through links of the system.
            entry->ChildStart.Clear();
            entry->ChildStart.Resize(count + 1, NativeArrayOptions.ClearMemory);
            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink link = entry->Links[l];
                if (link.Target >= 0 && link.A >= 0)
                {
                    entry->ChildStart[link.A + 1]++;
                }
            }

            for (int j = 0; j < count; j++)
            {
                entry->ChildStart[j + 1] += entry->ChildStart[j];
            }

            entry->Children.Resize(math.max(1, entry->ChildStart[count]));
            var childFill = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            for (int l = 0; l < entry->Links.Length; l++)
            {
                EquilibriumLink link = entry->Links[l];
                if (link.Target >= 0 && link.A >= 0)
                {
                    entry->Children[entry->ChildStart[link.A] + childFill[link.A]++] = link.B;
                }
            }
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
        /// Takes the loads of section s's looped edges and closures from its equilibrium system, on the calling thread. The
        /// tree method must have stored every locked edge's load first: an edge's load then is its child's whole subtree,
        /// which is also what the bridges keep. Writes the looped edges' loads and every closure inside the section, unless
        /// the mode only adds this substep's loads to the section's sum. The system is linear, so the solve of the mean of
        /// the substeps' loads is the mean of their solves, and one solve in each step gives the mean loads that the step
        /// reports. A team runs the same stages on ranges, see <see cref="Loads"/>.
        /// </summary>
        public void Apply(ref VesselContext context, int s, ApplyMode mode = ApplyMode.Now)
        {
            ClearClosures(ref context, s);

            int entryIndex = SectionEntry[s];
            if (entryIndex < 0 || SkipSolve(mode))
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = StrutLoads.SolveMarker.Auto();
            long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            SectionEquilibrium* entry = Entries.Ptr + entryIndex;
            if (mode != ApplyMode.Finish)
            {
                Loads(ref context, s, 0, entry->Count, mode == ApplyMode.Add);
            }

            long loaded = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            if (mode == ApplyMode.Add)
            {
                entry->Summed++;
                System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LoadTicks, loaded - started);
                System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.AccumulateTicks, loaded - started);
                return;
            }

            if (mode == ApplyMode.Finish && entry->Summed == 0)
            {
                return;
            }

            bool fromSum = mode == ApplyMode.Finish;
            for (int c = 0; c < entry->Chunks.ChunkCount; c++)
            {
                SectionEquilibriumSolve.ForwardChunk(entry, c, fromSum);
            }

            long topStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            SectionEquilibriumSolve.Top(entry, fromSum);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.TopTicks, Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - topStarted);
            for (int c = 0; c < entry->Chunks.ChunkCount; c++)
            {
                SectionEquilibriumSolve.BackChunk(entry, c);
            }

            long substituted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            LinkLoads(ref context, s, 0, entry->Links.Length);
            long finished = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            System.Threading.Interlocked.Increment(ref StrutLoads.Counts.Data.Solves);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.SolveTicks, finished - started);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LoadTicks, loaded - started);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.SubstituteTicks, substituted - loaded);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LinkTicks, finished - substituted);
        }

        internal static bool SkipSolve(ApplyMode mode) => VesselSwitches.StrutExperiment == 1
            || (VesselSwitches.StrutExperiment == 2 && mode == ApplyMode.Finish);

        /// <summary>Sets the loads of every closure inside section s to zero, before its system writes those that it holds.</summary>
        public void ClearClosures(ref VesselContext context, int s)
        {
            for (int c = SectionClosureStart[s]; c < SectionClosureStart[s + 1]; c++)
            {
                context.ClosureLoads[SectionClosures[c]] = default;
            }
        }

        /// <summary>
        /// For the unknowns at positions from .. to of section s's system: what the links must give each one, which is its
        /// own subtree load less the subtree loads of its children on loops, about the first part's centre and in its
        /// frame. Writes it to the solve's numbers, or adds it to the section's sum.
        /// </summary>
        public void Loads(ref VesselContext context, int s, int from, int to, bool add)
        {
            SectionEquilibrium* entry = Entries.Ptr + SectionEntry[s];
            int* sectionParts = context.SectionParts + context.SectionPartStart[s];
            VesselPart* firstPart = context.Parts + sectionParts[0];
            double3x3 back = math.transpose(DoubleMath.Rotation(firstPart->Orientation));
            double3 origin = firstPart->Position;
            double* target = add ? entry->Sum.Ptr : entry->Work.Ptr;
            for (int j = from; j < to; j++)
            {
                double6 load = Subtree(ref context, sectionParts[entry->Node[j]], back, origin);
                for (int e = entry->ChildStart[j]; e < entry->ChildStart[j + 1]; e++)
                {
                    load -= Subtree(ref context, sectionParts[entry->Node[entry->Children[e]]], back, origin);
                }

                if (add)
                {
                    load += double6.Load(target + 6 * j);
                }

                load.Store(target + 6 * j);
            }
        }

        static double6 Subtree(ref VesselContext context, int part, in double3x3 back, double3 origin)
        {
            VesselEdgeLoad load = context.EdgeLoads[part];
            double3 force = math.mul(back, load.Force);
            double3 torque = math.mul(back, load.Torque + math.cross(context.ChildPoint[part] - origin, load.Force));
            return new double6(new double4(force, torque.x), new double2(torque.y, torque.z));
        }

        /// <summary>The loads of links from .. to of section s's system, from the solved motions, written where they belong.</summary>
        public void LinkLoads(ref VesselContext context, int s, int from, int to)
        {
            SectionEquilibrium* entry = Entries.Ptr + SectionEntry[s];
            int* sectionParts = context.SectionParts + context.SectionPartStart[s];
            double3x3 rotation = DoubleMath.Rotation(context.Parts[sectionParts[0]].Orientation);
            double* x = entry->Work.Ptr;
            int closureStart = SectionClosureStart[s];
            for (int l = from; l < to; l++)
            {
                EquilibriumLink link = entry->Links[l];
                double6 motion = link.B >= 0 ? double6.Load(x + 6 * link.B) : default;
                if (link.A >= 0)
                {
                    motion -= double6.Load(x + 6 * link.A);
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
        }
    }
}
