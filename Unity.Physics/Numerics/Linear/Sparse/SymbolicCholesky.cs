using Unity.Collections;

namespace Unity.Numerics.Linear.Sparse
{
    /// <summary>
    /// Works out the structure of a sparse Cholesky factor from the graph of a symmetric matrix and an elimination
    /// order, without the values of the matrix: which entries of the factor are nonzero, and the elimination tree. The
    /// order usually comes from <see cref="MinimumDegreeOrdering"/>. The graph uses the compressed layout that
    /// <see cref="MinimumDegreeOrdering"/> describes. The vessel solver lays out the factor of a vessel's sections with
    /// it.
    /// </summary>
    public static class SymbolicCholesky
    {
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
