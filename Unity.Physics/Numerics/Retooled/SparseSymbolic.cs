using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Numerics.Retooled
{
    /// <summary>
    /// Holds the symbolic steps of a sparse Cholesky factorization. These steps work on the graph of a symmetric matrix
    /// and don't need its values. One routine picks an order in which to eliminate the vertices, and the other routine
    /// works out which entries of the factor are nonzero in that order. Each graph uses a compressed layout: the
    /// neighbours of vertex v are adjacency[adjacencyStart[v]] up to, but not including, adjacency[adjacencyStart[v + 1]].
    /// </summary>
    public static class SparseSymbolic
    {
        /// <summary>
        /// Computes a minimum-degree elimination order for a graph with <paramref name="count"/> vertices, and appends
        /// it to <paramref name="order"/>. The routine repeatedly eliminates the vertex that has the fewest remaining
        /// neighbours. It then connects that vertex's neighbours to each other, as eliminating the vertex in a
        /// factorization would. On a tree, this order takes the leaves first. On a tree with a few extra edges that close
        /// loops, the order avoids filling the whole tree path between the two ends of each extra edge. It also leaves
        /// the vertices that have many neighbours until last.
        /// </summary>
        public static void MinimumDegreeOrder(int count, NativeArray<int> adjacency, NativeArray<int> adjacencyStart,
            NativeList<int> order)
        {
            var neighbours = new NativeArray<UnsafeList<int>>(count, Allocator.Temp);
            var heap = new NativeList<long>(count * 2, Allocator.Temp);
            for (int v = 0; v < count; v++)
            {
                var list = new UnsafeList<int>(adjacencyStart[v + 1] - adjacencyStart[v], Allocator.Temp);
                for (int e = adjacencyStart[v]; e < adjacencyStart[v + 1]; e++)
                {
                    int w = adjacency[e];
                    if (w != v && !Contains(list, w))
                    {
                        list.Add(w);
                    }
                }

                neighbours[v] = list;
                Push(heap, ((long)list.Length << 32) | (uint)v);
            }

            var eliminated = new NativeArray<bool>(count, Allocator.Temp);
            while (heap.Length > 0)
            {
                long key = Pop(heap);
                int v = (int)(key & 0xffffffff);
                int degree = (int)(key >> 32);
                if (eliminated[v] || neighbours[v].Length != degree)
                {
                    continue;
                }

                eliminated[v] = true;
                order.Add(v);
                UnsafeList<int> around = neighbours[v];
                for (int i = 0; i < around.Length; i++)
                {
                    int a = around[i];
                    UnsafeList<int> list = neighbours[a];
                    int at = IndexOf(list, v);
                    list.RemoveAtSwapBack(at);
                    for (int j = 0; j < around.Length; j++)
                    {
                        int b = around[j];
                        if (b != a && !Contains(list, b))
                        {
                            list.Add(b);
                        }
                    }

                    neighbours[a] = list;
                }

                for (int i = 0; i < around.Length; i++)
                {
                    int a = around[i];
                    Push(heap, ((long)neighbours[a].Length << 32) | (uint)a);
                }
            }
        }

        static bool Contains(UnsafeList<int> list, int value) => IndexOf(list, value) >= 0;

        static int IndexOf(UnsafeList<int> list, int value)
        {
            for (int i = 0; i < list.Length; i++)
            {
                if (list[i] == value)
                {
                    return i;
                }
            }

            return -1;
        }

        static void Push(NativeList<long> heap, long key)
        {
            heap.Add(key);
            int i = heap.Length - 1;
            while (i > 0)
            {
                int parent = (i - 1) >> 1;
                if (heap[parent] <= key)
                {
                    break;
                }

                heap[i] = heap[parent];
                i = parent;
            }

            heap[i] = key;
        }

        static long Pop(NativeList<long> heap)
        {
            long top = heap[0];
            long last = heap[heap.Length - 1];
            heap.RemoveAt(heap.Length - 1);
            int count = heap.Length;
            if (count == 0)
            {
                return top;
            }

            int i = 0;
            while (true)
            {
                int child = 2 * i + 1;
                if (child >= count)
                {
                    break;
                }

                if (child + 1 < count && heap[child + 1] < heap[child])
                {
                    child++;
                }

                if (heap[child] >= last)
                {
                    break;
                }

                heap[i] = heap[child];
                i = child;
            }

            heap[i] = last;
            return top;
        }

        /// <summary>
        /// Computes which entries of the factor are nonzero when the vertices are eliminated in the given order. Column j
        /// of the factor has a row for each neighbour of j that comes later in the order. It also has every row of each
        /// of its children in the elimination tree, when that row comes after j. The routine writes the position of each
        /// vertex in the order to <paramref name="position"/>. It appends the rows of each column to
        /// <paramref name="rows"/>, sorted, and writes where each column's rows start to <paramref name="columnStart"/>,
        /// which needs one entry more than there are vertices. It also writes the elimination tree: the parent of each
        /// column (or -1 for a root), its first child, and its next sibling (or -1 for none).
        /// </summary>
        public static void FactorStructure(NativeList<int> order, NativeArray<int> position, NativeArray<int> adjacency,
            NativeArray<int> adjacencyStart, NativeArray<int> columnStart, NativeList<int> rows, NativeArray<int> parentOf,
            NativeArray<int> firstChild, NativeArray<int> nextSibling)
        {
            int count = order.Length;
            for (int j = 0; j < count; j++)
            {
                position[order[j]] = j;
                firstChild[j] = -1;
                nextSibling[j] = -1;
            }

            var mark = new NativeArray<int>(count, Allocator.Temp);
            for (int j = 0; j < count; j++)
            {
                mark[j] = -1;
            }

            var column = new NativeList<int>(Allocator.Temp);
            for (int j = 0; j < count; j++)
            {
                // Set the start of column j before reading its children. The rows of column j - 1 end at this index.
                // If column j - 1 is a child of column j, the loop below reads that child's rows up to this index.
                columnStart[j] = rows.Length;
                column.Clear();
                int label = order[j];
                for (int e = adjacencyStart[label]; e < adjacencyStart[label + 1]; e++)
                {
                    int r = position[adjacency[e]];
                    if (r > j && mark[r] != j)
                    {
                        mark[r] = j;
                        column.Add(r);
                    }
                }

                for (int child = firstChild[j]; child >= 0; child = nextSibling[child])
                {
                    for (int e = columnStart[child]; e < columnStart[child + 1]; e++)
                    {
                        int r = rows[e];
                        if (r > j && mark[r] != j)
                        {
                            mark[r] = j;
                            column.Add(r);
                        }
                    }
                }

                column.Sort();
                rows.AddRange(column.AsArray());
                int parent = column.Length > 0 ? column[0] : -1;
                parentOf[j] = parent;
                if (parent >= 0)
                {
                    nextSibling[j] = firstChild[parent];
                    firstChild[parent] = j;
                }
            }

            columnStart[count] = rows.Length;
        }
    }
}
