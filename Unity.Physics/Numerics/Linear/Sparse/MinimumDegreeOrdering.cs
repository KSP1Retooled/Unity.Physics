using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Numerics.Linear.Sparse
{
    /// <summary>
    /// Picks the order in which a sparse Cholesky factorization eliminates the vertices of a symmetric matrix's graph,
    /// by the minimum-degree rule. The order needs only the graph, not the values of the matrix. A good order keeps the
    /// factor sparse. <see cref="SymbolicCholesky"/> then works out which entries of the factor are nonzero in that
    /// order. The graph uses a compressed layout: the neighbours of vertex v are adjacency[adjacencyStart[v]] up to, but
    /// not including, adjacency[adjacencyStart[v + 1]]. The vessel solver can order the sections of a vessel with it.
    /// </summary>
    public static class MinimumDegreeOrdering
    {
        /// <summary>
        /// Computes a minimum-degree elimination order for a graph with <paramref name="count"/> vertices, and appends
        /// it to <paramref name="order"/>. The routine repeatedly eliminates the vertex that has the fewest remaining
        /// neighbours. It then connects that vertex's neighbours to each other, as eliminating the vertex in a
        /// factorization would. On a tree, this order takes the leaves first. On a tree with a few extra edges that close
        /// loops, the order avoids filling the whole tree path between the two ends of each extra edge. It also leaves
        /// the vertices that have many neighbours until last.
        /// </summary>
        public static void Compute(int count, NativeArray<int> adjacency, NativeArray<int> adjacencyStart,
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
    }
}
