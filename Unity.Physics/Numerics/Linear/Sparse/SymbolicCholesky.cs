using Unity.Collections;

namespace Unity.Numerics.Linear.Sparse
{
    /// <summary>
    /// Symbolic Cholesky factorization of a symmetric sparse matrix.
    /// Compute the factor's sparsity pattern and elimination tree without using matrix values.
    /// </summary>
    public static class SymbolicCholesky
    {
        /// <summary>
        /// Compute the strictly lower triangular sparsity pattern and elimination tree for the given elimination order.
        /// </summary>
        /// <param name="order">Vertex labels in elimination order, usually from <see cref="MinimumDegreeOrdering"/>.</param>
        /// <param name="position">Output position of each vertex in order.</param>
        /// <param name="adjacency">Neighbour labels for each vertex, in the layout described by <see cref="MinimumDegreeOrdering"/>.</param>
        /// <param name="adjacencyStart">Start of each vertex's neighbours in adjacency, plus the final end offset.</param>
        /// <param name="columnStart">Output offsets into rows, with one entry per column plus the final end offset.</param>
        /// <param name="rows">Sorted row indices appended for each column, excluding the diagonal. Indices refer to elimination order.</param>
        /// <param name="parentOf">Output parent of each column in the elimination tree, or -1 for a root.</param>
        /// <param name="firstChild">Output first child of each column, or -1 if none.</param>
        /// <param name="nextSibling">Output next sibling of each column, or -1 if none.</param>
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
                // Set the end offset of column j - 1 before reading it as a possible child of j.
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
