// Ported from faer 0.19.4 src/linalg/matmul (a matrix product with one column), as the triangular solves call it.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers, with double4 in place of the SIMD types.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// y += alpha A x, where A is m by n, x has n entries that are <paramref name="incx"/> apart, and y has m
        /// entries that are <paramref name="incy"/> apart. faer picks the loop by A's layout, and so does this port:
        /// <list type="bullet">
        /// <item>If the columns of A and y are contiguous, the routine adds alpha x_j times each column j of A to y.</item>
        /// <item>If the rows of A and x are contiguous, entry i of y takes the dot product of row i of A with x.</item>
        /// <item>Any other layout takes a plain loop.</item>
        /// </list>
        /// The transposed product y += alpha A^T x is this routine on <see cref="MatrixView{T}.Transpose"/>.
        /// </summary>
        public static void Gemv(double alpha, MatrixView<double> a, double* x, int incx, double* y, int incy)
        {
            int m = a.Rows;
            int n = a.Columns;
            if (m <= 0 || n <= 0)
            {
                return;
            }

            if (a.ColumnsAreContiguous && incy == 1)
            {
                for (int j = 0; j < n; j++)
                {
                    Axpy(m, alpha * x[(long)j * incx], a.At(0, j), y);
                }
            }
            else if (a.RowsAreContiguous && incx == 1)
            {
                for (int i = 0; i < m; i++)
                {
                    double* yi = y + (long)i * incy;
                    *yi = FusedMath.Fma(alpha, Dot(n, a.At(i, 0), x), *yi);
                }
            }
            else
            {
                for (int i = 0; i < m; i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < n; j++)
                    {
                        sum = FusedMath.Fma(a[i, j], x[(long)j * incx], sum);
                    }

                    double* yi = y + (long)i * incy;
                    *yi = FusedMath.Fma(alpha, sum, *yi);
                }
            }
        }
    }
}
