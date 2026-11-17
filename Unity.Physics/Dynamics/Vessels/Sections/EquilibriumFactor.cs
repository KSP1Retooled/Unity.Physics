using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Numerics.Linear.Sparse;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>Orders and factors an equilibrium system that its caller assembled.</summary>
    internal static unsafe class EquilibriumFactor
    {
        /// <summary>
        /// Orders a system of <paramref name="count"/> unknowns by minimum degree and factors it, as SectionEquilibrium's own
        /// build does. The diagonal holds one 6 by 6 block for each unknown in the caller's order; coupling c joins unknowns
        /// couplings[c].x and .y with the block of row x and column y, 36 doubles from couplingBlocks + 36 c. Fills the
        /// entry's Count, Position (each unknown's place in the elimination order), ColumnStart, Rows, Inverse and Blocks.
        /// </summary>
        public static void Factor(SectionEquilibrium* entry, int count, double* diagonal, int2* couplings, double* couplingBlocks,
            int couplingCount)
        {
            var adjacencyStart = new NativeArray<int>(count + 1, Allocator.Temp);
            for (int c = 0; c < couplingCount; c++)
            {
                adjacencyStart[couplings[c].x + 1]++;
                adjacencyStart[couplings[c].y + 1]++;
            }

            for (int v = 0; v < count; v++)
            {
                adjacencyStart[v + 1] += adjacencyStart[v];
            }

            var adjacency = new NativeArray<int>(math.max(1, adjacencyStart[count]), Allocator.Temp);
            var fill = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            for (int c = 0; c < couplingCount; c++)
            {
                int a = couplings[c].x, b = couplings[c].y;
                adjacency[adjacencyStart[a] + fill[a]++] = b;
                adjacency[adjacencyStart[b] + fill[b]++] = a;
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

            entry->Count = count;
            entry->Position.Clear();
            entry->Position.AddRange(position.GetUnsafeReadOnlyPtr(), count);
            entry->ColumnStart.Clear();
            entry->ColumnStart.AddRange(columnStart.GetUnsafeReadOnlyPtr(), count + 1);
            entry->Rows.Clear();
            entry->Rows.AddRange(rows.GetUnsafeReadOnlyPtr(), rows.Length);

            int offDiagonal = rows.Length;
            entry->Inverse.Resize(Block6x6.Size * math.max(1, count));
            var work = new NativeArray<double>(Block6x6.Size * (count + offDiagonal), Allocator.Temp);
            double* blocks = (double*)work.GetUnsafePtr();
            double* off = blocks + Block6x6.Size * count;
            for (int v = 0; v < count; v++)
            {
                Block6x6.Copy(blocks + Block6x6.Size * position[v], diagonal + Block6x6.Size * v);
            }

            double* turned = stackalloc double[Block6x6.Size];
            for (int c = 0; c < couplingCount; c++)
            {
                int pa = position[couplings[c].x], pb = position[couplings[c].y];
                double* block = couplingBlocks + Block6x6.Size * c;
                if (pa > pb)
                {
                    Block6x6.Transpose(turned, block);
                    block = turned;
                }

                int lower = math.min(pa, pb), upper = math.max(pa, pb);
                int e = Find(entry->Rows.Ptr, columnStart[lower], columnStart[lower + 1], upper);
                Block6x6.Add(off + Block6x6.Size * e, block);
            }

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

            entry->Blocks.Resize(Block6x6.Size * math.max(1, offDiagonal));
            UnsafeUtility.MemCpy(entry->Blocks.Ptr, off, (long)Block6x6.Size * offDiagonal * sizeof(double));
        }

        static int Find(int* rows, int from, int to, int row)
        {
            int low = from, high = to - 1;
            while (low <= high)
            {
                int middle = (low + high) >> 1;
                if (rows[middle] == row)
                {
                    return middle;
                }

                if (rows[middle] < row)
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
    }
}
