// Ported from faer 0.19.4 src/linalg/matmul (matrix product with one column).
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Compute y += alpha*A*x, where A is m by n, x has n entries and y has m entries.</summary>
        /// <param name="incx">Stride between entries of x, in elements.</param>
        /// <param name="incy">Stride between entries of y, in elements.</param>
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
