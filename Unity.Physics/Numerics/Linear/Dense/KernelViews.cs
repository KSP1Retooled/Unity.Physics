using Unity.Numerics.Kernels;

namespace Unity.Numerics.Linear.Dense.Primitives
{
    // Adapt Matrix and Vector storage to numerical kernels. Keep wrapper loops for unsupported layouts.

    internal unsafe partial struct Vector
    {
        /// <summary>
        /// Logical entry-zero address for BLAS loops. With negative stride, start at the far end so that
        /// entry i is at <c>KernelStart + i * Stride</c>.
        /// </summary>
        internal float* KernelStart => Stride >= 0 ? data : data + (long)(1 - Dimension) * Stride;

        /// <summary>The vector as a matrix with one column, for a kernel that takes a matrix.</summary>
        internal MatrixView<float> AsColumn => new MatrixView<float>(KernelStart, Dimension, 1, Stride, Dimension * System.Math.Abs(Stride));

        /// <summary>The vector as a matrix with one row.</summary>
        internal MatrixView<float> AsRow => AsColumn.Transpose;
    }

    internal unsafe partial struct Matrix
    {
        /// <summary>
        /// Create a kernel view when columns have equal positive strides, correct row counts and non-overlapping
        /// storage.
        /// </summary>
        internal static bool TryView(in Matrix matrix, out MatrixView<float> view)
        {
            view = default;
            int rows = matrix.NumRows;
            int columns = matrix.NumCols;
            if (rows <= 0 || columns <= 0)
            {
                return false;
            }

            float* first = matrix.Cols[0].ReinterpretCast<float>();
            int rowStride = matrix.Cols[0].Stride;
            if (rowStride <= 0)
            {
                return false;
            }

            long columnStride = columns > 1 ? matrix.Cols[1].ReinterpretCast<float>() - first : (long)rows * rowStride;
            if (columnStride < (long)(rows - 1) * rowStride + 1 || columnStride > int.MaxValue)
            {
                return false;
            }

            for (int j = 0; j < columns; j++)
            {
                if (matrix.Cols[j].Stride != rowStride || matrix.Cols[j].Dimension != rows ||
                    matrix.Cols[j].ReinterpretCast<float>() != first + j * columnStride)
                {
                    return false;
                }
            }

            view = new MatrixView<float>(first, rows, columns, rowStride, (int)columnStride);
            return true;
        }

        /// <summary>Whether the storage of two views can share an entry: whether their address ranges overlap.</summary>
        internal static bool Overlaps(MatrixView<float> x, MatrixView<float> y)
        {
            Range(x, out float* xFirst, out float* xLast);
            Range(y, out float* yFirst, out float* yLast);
            return xFirst <= yLast && yFirst <= xLast;
        }

        static void Range(MatrixView<float> v, out float* first, out float* last)
        {
            long rowSpan = (long)(v.Rows - 1) * v.RowStride;
            long columnSpan = (long)(v.Columns - 1) * v.ColumnStride;
            first = v.Data + System.Math.Min(rowSpan, 0) + System.Math.Min(columnSpan, 0);
            last = v.Data + System.Math.Max(rowSpan, 0) + System.Math.Max(columnSpan, 0);
        }

        /// <summary>
        /// Return the effective triangle view: op(A) for left operations, op(A)^T for right operations.
        /// Return true if the resulting view is lower triangular.
        /// </summary>
        internal static bool TriangleForKernels(MatrixView<float> a, int n, Side side, TriangularType type, Op op,
            out MatrixView<float> triangle)
        {
            triangle = a.Block(0, 0, n, n);
            bool lower = type == TriangularType.Lower;
            if (op == Op.Transpose)
            {
                triangle = triangle.Transpose;
                lower = !lower;
            }

            if (side == Side.Right)
            {
                triangle = triangle.Transpose;
                lower = !lower;
            }

            return lower;
        }
    }
}
