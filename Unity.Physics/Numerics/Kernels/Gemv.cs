// Ported from faer 0.19.4 src/linalg/matmul (a matrix product with one column), as the triangular solves call it.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers in column-major order, with double4 in place of the SIMD types.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>y += alpha A x, where A is m by n. The routine adds alpha x_j times each column j of A to y.</summary>
        public static void GemvN(int m, int n, double alpha, double* a, int lda, double* x, double* y)
        {
            for (int j = 0; j < n; j++)
            {
                Axpy(m, alpha * x[j], a + (long)j * lda, y);
            }
        }

        /// <summary>y += alpha A^T x, where A is m by n. Entry j of y takes the dot product of column j of A with x.</summary>
        public static void GemvT(int m, int n, double alpha, double* a, int lda, double* x, double* y)
        {
            for (int j = 0; j < n; j++)
            {
                y[j] += alpha * Dot(m, a + (long)j * lda, x);
            }
        }
    }
}
