// Ported from faer 0.19.4 src/linalg/matmul (a matrix product with one column), as the triangular solves call it.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The float version: the loops are the same as for doubles, and the float Axpy and Dot that it calls run in vectors of
// eight floats.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// y += alpha A x for floats, where A is m by n, x has n entries that are <paramref name="incx"/> apart, and y
        /// has m entries that are <paramref name="incy"/> apart. As for doubles: if the columns of A and y are
        /// contiguous, the routine adds alpha x_j times each column j of A to y; if the rows of A and x are contiguous,
        /// entry i of y takes the dot product of row i of A with x; any other layout takes a plain loop.
        /// </summary>
        public static void Gemv(float alpha, MatrixView<float> a, float* x, int incx, float* y, int incy)
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
                    float* yi = y + (long)i * incy;
                    *yi = FusedMath.Fma(alpha, Dot(n, a.At(i, 0), x), *yi);
                }
            }
            else
            {
                for (int i = 0; i < m; i++)
                {
                    float sum = 0f;
                    for (int j = 0; j < n; j++)
                    {
                        sum = FusedMath.Fma(a[i, j], x[(long)j * incx], sum);
                    }

                    float* yi = y + (long)i * incy;
                    *yi = FusedMath.Fma(alpha, sum, *yi);
                }
            }
        }
    }
}
