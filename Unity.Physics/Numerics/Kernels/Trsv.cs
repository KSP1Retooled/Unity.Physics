// Ported from faer 0.19.4 src/linalg/triangular_solve.rs, with one right-hand side.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers in column-major order, with double4 in place of the SIMD types.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>x := L^-1 x, for one right-hand side. This is <see cref="TrsmLeftLower"/> with a matrix-vector product.</summary>
        public static void TrsvLower(int n, double* l, int ldl, double* x)
        {
            if (n <= 4)
            {
                TrsmLeftLower(n, 1, l, ldl, x, n);
                return;
            }

            int bs = TriangularBlock(n);
            TrsvLower(bs, l, ldl, x);
            GemvN(n - bs, bs, -1.0, l + bs, ldl, x, x + bs);
            TrsvLower(n - bs, l + bs + (long)bs * ldl, ldl, x + bs);
        }

        /// <summary>
        /// x := L^-T x, for one right-hand side. faer solves with the upper triangle L^T by reversing the rows and
        /// columns into a lower solve. So this routine solves the lower block of rows first, takes its share out of
        /// the upper block with x_0 -= L_10^T x_1, which reads the columns of L_10, and solves the upper block.
        /// </summary>
        public static void TrsvLowerTranspose(int n, double* l, int ldl, double* x)
        {
            if (n <= 4)
            {
                for (int i = n - 1; i >= 0; i--)
                {
                    double y = x[i];
                    for (int p = i + 1; p < n; p++)
                    {
                        y = FusedMath.Fma(-l[p + (long)i * ldl], x[p], y);
                    }

                    x[i] = y / l[i + (long)i * ldl];
                }

                return;
            }

            int bs = TriangularBlock(n);
            TrsvLowerTranspose(n - bs, l + bs + (long)bs * ldl, ldl, x + bs);
            GemvT(n - bs, bs, -1.0, l + bs, ldl, x + bs, x);
            TrsvLowerTranspose(bs, l, ldl, x);
        }
    }
}
