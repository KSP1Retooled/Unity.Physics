using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Numerics.Linear.Sparse
{
    /// <summary>
    /// Minimum-degree ordering for sparse symmetric factorization, using graph structure only.
    /// Vertex v's neighbours occupy adjacency[adjacencyStart[v] .. adjacencyStart[v + 1]).
    /// </summary>
    public static class MinimumDegreeOrdering
    {
        /// <summary>
        /// Append a minimum-degree elimination order for count vertices.
        /// Eliminate the vertex with fewest remaining neighbours, connecting those neighbours to account for fill.
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
