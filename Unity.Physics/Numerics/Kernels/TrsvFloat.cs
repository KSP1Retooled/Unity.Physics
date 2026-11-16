// Ported from faer 0.19.4 src/linalg/triangular_solve.rs, with one right-hand side.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The float version, with the float blocks of TrsmFloat.cs and the float matrix-vector product.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// x := L^-1 x for floats, where L is n by n and lower triangular, and x has n entries that are
        /// <paramref name="incx"/> apart. The recursion is the one of the double version, with the float blocks.
        /// </summary>
        public static void TrsvLower(MatrixView<float> l, float* x, int incx, bool unitDiagonal = false)
        {
            int n = l.Rows;
            if (n <= FloatTriangleBaseRows)
            {
                for (int i = 0; i < n; i++)
                {
                    float y = x[(long)i * incx];
                    for (int p = 0; p < i; p++)
                    {
                        y = FusedMath.Fma(-l[i, p], x[(long)p * incx], y);
                    }

                    x[(long)i * incx] = unitDiagonal ? y : y / l[i, i];
                }

                return;
            }

            int bs = TriangularBlockFloat(n);
            TrsvLower(l.Block(0, 0, bs, bs), x, incx, unitDiagonal);
            Gemv(-1f, l.Block(bs, 0, n - bs, bs), x, incx, x + (long)bs * incx, incx);
            TrsvLower(l.Block(bs, bs, n - bs, n - bs), x + (long)bs * incx, incx, unitDiagonal);
        }

        /// <summary>
        /// x := U^-1 x for floats, where U is n by n and upper triangular: the lower block of rows first, then
        /// x_0 -= U_01 x_1, then the upper block. The solve with L^T is this routine on L's transpose.
        /// </summary>
        public static void TrsvUpper(MatrixView<float> u, float* x, int incx, bool unitDiagonal = false)
        {
            int n = u.Rows;
            if (n <= FloatTriangleBaseRows)
            {
                for (int i = n - 1; i >= 0; i--)
                {
                    float y = x[(long)i * incx];
                    for (int p = i + 1; p < n; p++)
                    {
                        y = FusedMath.Fma(-u[i, p], x[(long)p * incx], y);
                    }

                    x[(long)i * incx] = unitDiagonal ? y : y / u[i, i];
                }

                return;
            }

            int bs = TriangularBlockFloat(n);
            TrsvUpper(u.Block(bs, bs, n - bs, n - bs), x + (long)bs * incx, incx, unitDiagonal);
            Gemv(-1f, u.Block(0, bs, bs, n - bs), x + (long)bs * incx, incx, x, incx);
            TrsvUpper(u.Block(0, 0, bs, bs), x, incx, unitDiagonal);
        }
    }
}
