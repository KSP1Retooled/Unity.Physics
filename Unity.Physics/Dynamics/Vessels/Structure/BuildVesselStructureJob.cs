using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Numerics.Linear.Sparse;
using Unity.Physics.Dynamics.Vessels.Model;

namespace Unity.Physics.Dynamics.Vessels.Structure
{
    /// <summary>Rebuilds a <see cref="VesselStructure"/> from the edges' states. It runs on one thread.</summary>
    [BurstCompile(DisableSafetyChecks = true, CompileSynchronously = true)]
    internal unsafe struct BuildVesselStructureJob : IJob
    {
        [ReadOnly] public NativeArray<VesselPart> Parts;
        [ReadOnly] public NativeArray<VesselEdge> Edges;
        [ReadOnly] public NativeArray<VesselClosure> Closures;
        public int HubClosures;
        public int ChunkTarget;
        public bool MinimumDegree;
        [NativeDisableUnsafePtrRestriction] public VesselStructure* Structure;

        // A link while the sections are still labels in tree order.
        struct LabelLink
        {
            public int Child;
            public int Parent;
            public int Edge;
            public int Closure;
        }

        public void Execute()
        {
            int partCount = Edges.Length;

            // Sections: parents come first, so a part on a locked edge takes its parent's label and any other part starts one.
            var label = new NativeArray<int>(partCount, Allocator.Temp);
            int labelCount = 0;
            for (int i = 0; i < partCount; i++)
            {
                int parent = Edges[i].Parent;
                label[i] = parent >= 0 && Edges[i].State == VesselEdgeState.Locked ? label[parent] : labelCount++;
            }

            var labelMass = new NativeArray<double>(labelCount, Allocator.Temp);
            var labelParts = new NativeArray<int>(labelCount, Allocator.Temp);
            for (int i = 0; i < partCount; i++)
            {
                labelMass[label[i]] += Parts[i].Mass;
                labelParts[label[i]]++;
            }

            // Links between sections. A broken edge leaves its subtree as a separate tree of the forest; a closure inside
            // one section is inert.
            var links = new NativeList<LabelLink>(partCount, Allocator.Temp);
            for (int i = 0; i < partCount; i++)
            {
                VesselEdgeState state = Edges[i].State;
                if (Edges[i].Parent >= 0 && (state == VesselEdgeState.Yielding || state == VesselEdgeState.FreeAxis))
                {
                    links.Add(new LabelLink { Child = label[i], Parent = label[Edges[i].Parent], Edge = i, Closure = -1 });
                }
            }

            int treeLinkCount = links.Length;
            for (int c = 0; c < Closures.Length; c++)
            {
                int a = label[Closures[c].PartA], b = label[Closures[c].PartB];
                if (a != b)
                {
                    links.Add(new LabelLink { Child = b, Parent = a, Edge = -1, Closure = c });
                }
            }

            // Adjacency over every link, and over tree links alone, which give the order.
            var adjacencyStart = new NativeArray<int>(labelCount + 1, Allocator.Temp);
            var treeStart = new NativeArray<int>(labelCount + 1, Allocator.Temp);
            var closureDegree = new NativeArray<int>(labelCount, Allocator.Temp);
            for (int k = 0; k < links.Length; k++)
            {
                adjacencyStart[links[k].Child + 1]++;
                adjacencyStart[links[k].Parent + 1]++;
                if (k < treeLinkCount)
                {
                    treeStart[links[k].Child + 1]++;
                    treeStart[links[k].Parent + 1]++;
                }
                else
                {
                    closureDegree[links[k].Child]++;
                    closureDegree[links[k].Parent]++;
                }
            }

            for (int l = 0; l < labelCount; l++)
            {
                adjacencyStart[l + 1] += adjacencyStart[l];
                treeStart[l + 1] += treeStart[l];
            }

            var adjacency = new NativeArray<int>(adjacencyStart[labelCount], Allocator.Temp);
            var tree = new NativeArray<int>(treeStart[labelCount], Allocator.Temp);
            var fill = new NativeArray<int>(labelCount, Allocator.Temp);
            var treeFill = new NativeArray<int>(labelCount, Allocator.Temp);
            for (int k = 0; k < links.Length; k++)
            {
                int a = links[k].Child, b = links[k].Parent;
                adjacency[adjacencyStart[a] + fill[a]++] = b;
                adjacency[adjacencyStart[b] + fill[b]++] = a;
                if (k < treeLinkCount)
                {
                    tree[treeStart[a] + treeFill[a]++] = b;
                    tree[treeStart[b] + treeFill[b]++] = a;
                }
            }

            // The first order: each tree of the forest leaves-inward from its heaviest section, with hubs held back to
            // go last, lightest-connected first.
            var hub = new NativeArray<bool>(labelCount, Allocator.Temp);
            int hubCount = 0;
            for (int l = 0; l < labelCount; l++)
            {
                hub[l] = closureDegree[l] >= HubClosures;
                hubCount += hub[l] ? 1 : 0;
            }

            var order = new NativeList<int>(labelCount, Allocator.Temp);
            var seen = new NativeArray<bool>(labelCount, Allocator.Temp);
            var queue = new NativeList<int>(Allocator.Temp);
            var stackNode = new NativeList<int>(Allocator.Temp);
            var stackNext = new NativeList<int>(Allocator.Temp);
            var placed = new NativeArray<bool>(labelCount, Allocator.Temp);
            for (int start = 0; start < labelCount; start++)
            {
                if (seen[start])
                {
                    continue;
                }

                queue.Clear();
                queue.Add(start);
                seen[start] = true;
                int root = start;
                for (int k = 0; k < queue.Length; k++)
                {
                    int l = queue[k];
                    if (labelMass[l] > labelMass[root])
                    {
                        root = l;
                    }

                    for (int e = treeStart[l]; e < treeStart[l + 1]; e++)
                    {
                        if (!seen[tree[e]])
                        {
                            seen[tree[e]] = true;
                            queue.Add(tree[e]);
                        }
                    }
                }

                stackNode.Clear();
                stackNext.Clear();
                stackNode.Add(root);
                stackNext.Add(treeStart[root]);
                placed[root] = true;
                while (stackNode.Length > 0)
                {
                    int top = stackNode.Length - 1;
                    int l = stackNode[top];
                    int e = stackNext[top];
                    if (e < treeStart[l + 1])
                    {
                        stackNext[top] = e + 1;
                        int next = tree[e];
                        if (!placed[next])
                        {
                            placed[next] = true;
                            stackNode.Add(next);
                            stackNext.Add(treeStart[next]);
                        }
                    }
                    else
                    {
                        if (!hub[l])
                        {
                            order.Add(l);
                        }

                        stackNode.RemoveAt(top);
                        stackNext.RemoveAt(top);
                    }
                }
            }

            int nonHubs = order.Length;
            for (int l = 0; l < labelCount; l++)
            {
                if (hub[l])
                {
                    order.Add(l);
                }
            }

            // Hubs by closure count, insertion sort: there are few.
            for (int i = nonHubs + 1; i < order.Length; i++)
            {
                int l = order[i];
                int j = i - 1;
                while (j >= nonHubs && closureDegree[order[j]] > closureDegree[l])
                {
                    order[j + 1] = order[j];
                    j--;
                }

                order[j + 1] = l;
            }

            if (MinimumDegree)
            {
                order.Clear();
                MinimumDegreeOrdering.Compute(labelCount, adjacency, adjacencyStart, order);
            }

            // The symbolic factor in that order gives the elimination tree. A postorder of that tree eliminates the same
            // way with the same fill, and puts every subtree in one run of columns, which is what the chunks need.
            var position = new NativeArray<int>(labelCount, Allocator.Temp);
            var columnStart = new NativeArray<int>(labelCount + 1, Allocator.Temp);
            var parentOf = new NativeArray<int>(labelCount, Allocator.Temp);
            var firstChild = new NativeArray<int>(labelCount, Allocator.Temp);
            var nextSibling = new NativeArray<int>(labelCount, Allocator.Temp);
            var rows = new NativeList<int>(labelCount * 2, Allocator.Temp);
            SymbolicCholesky.FactorStructure(order, position, adjacency, adjacencyStart, columnStart, rows, parentOf,
                firstChild, nextSibling);

            var postorder = new NativeList<int>(labelCount, Allocator.Temp);
            for (int r = 0; r < labelCount; r++)
            {
                if (parentOf[r] >= 0)
                {
                    continue;
                }

                stackNode.Clear();
                stackNext.Clear();
                stackNode.Add(r);
                stackNext.Add(firstChild[r]);
                while (stackNode.Length > 0)
                {
                    int top = stackNode.Length - 1;
                    int child = stackNext[top];
                    if (child >= 0)
                    {
                        stackNext[top] = nextSibling[child];
                        stackNode.Add(child);
                        stackNext.Add(firstChild[child]);
                    }
                    else
                    {
                        postorder.Add(order[stackNode[top]]);
                        stackNode.RemoveAt(top);
                        stackNext.RemoveAt(top);
                    }
                }
            }

            rows.Clear();
            SymbolicCholesky.FactorStructure(postorder, position, adjacency, adjacencyStart, columnStart, rows, parentOf,
                firstChild, nextSibling);
            int sectionCount = labelCount;
            int offDiagonal = rows.Length;
            int fillCount = offDiagonal - (links.Length > 0 ? CountDistinctPairs(links, position) : 0);

            // Chunks. A column's weight is roughly its factor work in block products plus its parts; a subtree lighter
            // than the target is a candidate chunk, and everything heavier is the top.
            var weight = new NativeArray<double>(sectionCount, Allocator.Temp);
            var subtree = new NativeArray<int>(sectionCount, Allocator.Temp);
            int maxStructure = 0;
            double total = 0.0;
            for (int s = 0; s < sectionCount; s++)
            {
                int count = columnStart[s + 1] - columnStart[s];
                maxStructure = count > maxStructure ? count : maxStructure;
                weight[s] += 2.0 * (3 + count + count * (count + 1) / 2) + labelParts[postorder[s]];
                subtree[s] += 1;
                total += 2.0 * (3 + count + count * (count + 1) / 2) + labelParts[postorder[s]];
                if (parentOf[s] >= 0)
                {
                    weight[parentOf[s]] += weight[s];
                    subtree[parentOf[s]] += subtree[s];
                }
            }

            double target = total / (ChunkTarget > 0 ? ChunkTarget : 1);
            var chunkOf = new NativeArray<int>(sectionCount, Allocator.Temp);
            Structure->ChunkStart.Clear();
            Structure->ChunkColumns.Clear();
            Structure->TopColumns.Clear();
            double accumulated = 0.0;
            bool open = false;
            for (int s = 0; s < sectionCount; s++)
            {
                chunkOf[s] = -1;
            }

            for (int s = 0; s < sectionCount; s++)
            {
                bool top = weight[s] > target;
                if (top)
                {
                    Structure->TopColumns.Add(s);
                    continue;
                }

                int parent = parentOf[s];
                if (parent >= 0 && weight[parent] <= target)
                {
                    continue;
                }

                // s roots a chunk subtree: columns s - subtree + 1 to s.
                if (!open)
                {
                    Structure->ChunkStart.Add(Structure->ChunkColumns.Length);
                    open = true;
                    accumulated = 0.0;
                }

                int chunk = Structure->ChunkStart.Length - 1;
                for (int c = s - subtree[s] + 1; c <= s; c++)
                {
                    Structure->ChunkColumns.Add(c);
                    chunkOf[c] = chunk;
                }

                accumulated += weight[s];
                if (accumulated >= target)
                {
                    open = false;
                }
            }

            int chunkCount = Structure->ChunkStart.Length;
            Structure->ChunkStart.Add(Structure->ChunkColumns.Length);

            // Copy out the structure.
            Copy(ref Structure->ColumnStart, columnStart);
            Structure->Rows.Clear();
            Structure->Rows.AddRange(rows.GetUnsafeReadOnlyPtr(), rows.Length);
            Copy(ref Structure->EliminationParent, parentOf);

            // Update targets, with a chunk's shares of top blocks sent to its buffer.
            Structure->UpdateStart.Resize(sectionCount + 1);
            int updates = 0;
            for (int s = 0; s < sectionCount; s++)
            {
                Structure->UpdateStart[s] = updates;
                int count = columnStart[s + 1] - columnStart[s];
                updates += count * (count + 1) / 2;
            }

            Structure->UpdateStart[sectionCount] = updates;
            Structure->UpdateTarget.Resize(updates);
            Structure->RhsTarget.Resize(offDiagonal);
            Structure->ChunkBufferStart.Clear();
            Structure->ChunkRhsStart.Clear();
            Structure->BufferTarget.Clear();
            Structure->RhsBufferTarget.Clear();
            int blockBase = sectionCount + offDiagonal;
            var blockStamp = new NativeArray<int>(blockBase, Allocator.Temp);
            var blockSlot = new NativeArray<int>(blockBase, Allocator.Temp);
            var rhsStamp = new NativeArray<int>(sectionCount, Allocator.Temp);
            var rhsSlot = new NativeArray<int>(sectionCount, Allocator.Temp);
            for (int i = 0; i < blockBase; i++)
            {
                blockStamp[i] = -1;
            }

            for (int i = 0; i < sectionCount; i++)
            {
                rhsStamp[i] = -1;
            }

            for (int chunk = 0; chunk <= chunkCount; chunk++)
            {
                bool isTop = chunk == chunkCount;
                Structure->ChunkBufferStart.Add(Structure->BufferTarget.Length);
                Structure->ChunkRhsStart.Add(Structure->RhsBufferTarget.Length);
                if (isTop)
                {
                    break;
                }

                for (int k = Structure->ChunkStart[chunk]; k < Structure->ChunkStart[chunk + 1]; k++)
                {
                    SetTargets(Structure->ChunkColumns[k], chunk, chunkOf, columnStart, rows, blockStamp, blockSlot, rhsStamp,
                        rhsSlot, sectionCount, blockBase);
                }
            }

            for (int k = 0; k < Structure->TopColumns.Length; k++)
            {
                SetTargets(Structure->TopColumns[k], -1, chunkOf, columnStart, rows, blockStamp, blockSlot, rhsStamp, rhsSlot,
                    sectionCount, blockBase);
            }

            // Links, owned by their lower section; the upper section's share goes wherever that column's update for the
            // upper row goes.
            var sectionOf = new NativeArray<int>(labelCount, Allocator.Temp);
            for (int s = 0; s < sectionCount; s++)
            {
                sectionOf[postorder[s]] = s;
            }

            Structure->LinkStart.Clear();
            Structure->LinkStart.Resize(sectionCount + 1, NativeArrayOptions.ClearMemory);
            for (int k = 0; k < links.Length; k++)
            {
                int a = sectionOf[links[k].Child], b = sectionOf[links[k].Parent];
                Structure->LinkStart[(a < b ? a : b) + 1]++;
            }

            for (int s = 0; s < sectionCount; s++)
            {
                Structure->LinkStart[s + 1] += Structure->LinkStart[s];
            }

            Structure->Links.Resize(links.Length);
            Structure->PartLink.Resize(partCount);
            Structure->ClosureLink.Resize(Closures.Length);
            for (int i = 0; i < partCount; i++)
            {
                Structure->PartLink[i] = -1;
            }

            for (int c = 0; c < Closures.Length; c++)
            {
                Structure->ClosureLink[c] = -1;
            }

            var linkFill = new NativeArray<int>(sectionCount, Allocator.Temp);
            for (int k = 0; k < links.Length; k++)
            {
                LabelLink source = links[k];
                int child = sectionOf[source.Child], parent = sectionOf[source.Parent];
                int lower = child < parent ? child : parent, upper = child < parent ? parent : child;
                int start = columnStart[lower], count = columnStart[lower + 1] - start;
                int p = Find(rows, start, count, upper);
                int diagonalEntry = Structure->UpdateStart[lower] + p * count - p * (p - 1) / 2;
                var link = new VesselLink
                {
                    Edge = source.Edge,
                    Closure = source.Closure,
                    ChildPart = source.Edge >= 0 ? source.Edge : Closures[source.Closure].PartB,
                    ParentPart = source.Edge >= 0 ? Edges[source.Edge].Parent : Closures[source.Closure].PartA,
                    ChildSection = child,
                    ParentSection = parent,
                    Lower = lower,
                    Upper = upper,
                    Coupling = sectionCount + start + p,
                    UpperDiagonal = Structure->UpdateTarget[diagonalEntry],
                    UpperRhs = Structure->RhsTarget[start + p],
                };
                int index = Structure->LinkStart[lower] + linkFill[lower]++;
                Structure->Links[index] = link;
                if (source.Edge >= 0)
                {
                    Structure->PartLink[source.Edge] = index;
                }
                else
                {
                    Structure->ClosureLink[source.Closure] = index;
                }
            }

            // Every link at each section, for the loads.
            Structure->IncidentStart.Clear();
            Structure->IncidentStart.Resize(sectionCount + 1, NativeArrayOptions.ClearMemory);
            for (int k = 0; k < links.Length; k++)
            {
                Structure->IncidentStart[Structure->Links[k].ChildSection + 1]++;
                Structure->IncidentStart[Structure->Links[k].ParentSection + 1]++;
            }

            for (int s = 0; s < sectionCount; s++)
            {
                Structure->IncidentStart[s + 1] += Structure->IncidentStart[s];
            }

            Structure->Incident.Resize(2 * links.Length);
            for (int s = 0; s < sectionCount; s++)
            {
                linkFill[s] = 0;
            }

            for (int k = 0; k < links.Length; k++)
            {
                int a = Structure->Links[k].ChildSection, b = Structure->Links[k].ParentSection;
                Structure->Incident[Structure->IncidentStart[a] + linkFill[a]++] = k;
                Structure->Incident[Structure->IncidentStart[b] + linkFill[b]++] = k;
            }

            // Each section's parts, parents first since the parts already are.
            Structure->PartSection.Resize(partCount);
            Structure->SectionPartStart.Clear();
            Structure->SectionPartStart.Resize(sectionCount + 1, NativeArrayOptions.ClearMemory);
            for (int i = 0; i < partCount; i++)
            {
                int s = sectionOf[label[i]];
                Structure->PartSection[i] = s;
                Structure->SectionPartStart[s + 1]++;
            }

            for (int s = 0; s < sectionCount; s++)
            {
                Structure->SectionPartStart[s + 1] += Structure->SectionPartStart[s];
                linkFill[s] = 0;
            }

            Structure->SectionParts.Resize(partCount);
            for (int i = 0; i < partCount; i++)
            {
                int s = Structure->PartSection[i];
                Structure->SectionParts[Structure->SectionPartStart[s] + linkFill[s]++] = i;
            }

            Structure->Header[VesselStructure.SectionCount] = sectionCount;
            Structure->Header[VesselStructure.OffDiagonalCount] = offDiagonal;
            Structure->Header[VesselStructure.BufferBlockCount] = Structure->BufferTarget.Length;
            Structure->Header[VesselStructure.BufferRhsCount] = Structure->RhsBufferTarget.Length;
            Structure->Header[VesselStructure.ChunkCount] = chunkCount;
            Structure->Header[VesselStructure.TopCount] = Structure->TopColumns.Length;
            Structure->Header[VesselStructure.MaxStructure] = maxStructure;
            Structure->Header[VesselStructure.LinkCount] = links.Length;
            Structure->Header[VesselStructure.HubCount] = hubCount;
            Structure->Header[VesselStructure.FillCount] = fillCount;
        }

        void SetTargets(int s, int chunk, NativeArray<int> chunkOf, NativeArray<int> columnStart, NativeList<int> rows,
            NativeArray<int> blockStamp, NativeArray<int> blockSlot, NativeArray<int> rhsStamp, NativeArray<int> rhsSlot,
            int sectionCount, int blockBase)
        {
            int start = columnStart[s], count = columnStart[s + 1] - start;
            int entry = Structure->UpdateStart[s];
            for (int p = 0; p < count; p++)
            {
                int row = rows[start + p];

                // A row outside this chunk is in the top, and so is every later row of the column.
                bool buffered = chunk >= 0 && chunkOf[row] != chunk;
                Structure->UpdateTarget[entry++] = buffered ? Buffer(row, chunk, blockStamp, blockSlot, blockBase) : row;
                int rowStart = columnStart[row], rowCount = columnStart[row + 1] - rowStart;
                for (int q = p + 1; q < count; q++)
                {
                    int target = sectionCount + rowStart + Find(rows, rowStart, rowCount, rows[start + q]);
                    Structure->UpdateTarget[entry++] = buffered ? Buffer(target, chunk, blockStamp, blockSlot, blockBase) : target;
                }

                if (buffered)
                {
                    if (rhsStamp[row] != chunk)
                    {
                        rhsStamp[row] = chunk;
                        rhsSlot[row] = Structure->RhsBufferTarget.Length;
                        Structure->RhsBufferTarget.Add(row);
                    }

                    Structure->RhsTarget[start + p] = sectionCount + rhsSlot[row];
                }
                else
                {
                    Structure->RhsTarget[start + p] = row;
                }
            }
        }

        int Buffer(int target, int chunk, NativeArray<int> blockStamp, NativeArray<int> blockSlot, int blockBase)
        {
            if (blockStamp[target] != chunk)
            {
                blockStamp[target] = chunk;
                blockSlot[target] = Structure->BufferTarget.Length;
                Structure->BufferTarget.Add(target);
            }

            return blockBase + blockSlot[target];
        }

        static int Find(NativeList<int> rows, int start, int count, int row)
        {
            int low = 0, high = count - 1;
            while (low <= high)
            {
                int middle = (low + high) >> 1;
                int value = rows[start + middle];
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

        static int CountDistinctPairs(NativeList<LabelLink> links, NativeArray<int> position)
        {
            var pairs = new NativeHashSet<long>(links.Length, Allocator.Temp);
            for (int k = 0; k < links.Length; k++)
            {
                long a = position[links[k].Child], b = position[links[k].Parent];
                pairs.Add(a < b ? (a << 32) | b : (b << 32) | a);
            }

            return pairs.Count;
        }

        static void Copy(ref UnsafeList<int> to, NativeArray<int> from)
        {
            to.Clear();
            to.AddRange(from.GetUnsafeReadOnlyPtr(), from.Length);
        }
    }
}
