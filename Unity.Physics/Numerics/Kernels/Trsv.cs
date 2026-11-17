// Ported from faer 0.19.4 src/linalg/triangular_solve.rs.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Solve L*x = b, overwriting b with x.</summary>
        /// <param name="l">Lower triangular n by n matrix.</param>
        /// <param name="x">Right-hand side on entry; solution on return.</param>
        /// <param name="incx">Stride between entries of x, in elements.</param>
        /// <param name="unitDiagonal">Treat diagonal entries as 1 without reading them.</param>
        public static void TrsvLower(MatrixView<double> l, double* x, int incx, bool unitDiagonal = false)
        {
            int n = l.Rows;
            if (n <= 4 && l.RowStride == 1 && incx == 1)
            {
                // Use column updates for column-major L and contiguous x.
                double* d = l.Data;
                long ld = l.ColumnStride;
                for (int i = 0; i < n; i++)
                {
                    double y = x[i];
                    for (int p = 0; p < i; p++)
                    {
                        y = FusedMath.Fma(-d[i + p * ld], x[p], y);
                    }

                    x[i] = unitDiagonal ? y : y / d[i + i * ld];
                }

                return;
            }

            if (n <= 4)
            {
                for (int i = 0; i < n; i++)
                {
                    double y = x[(long)i * incx];
                    for (int p = 0; p < i; p++)
                    {
                        y = FusedMath.Fma(-l[i, p], x[(long)p * incx], y);
                    }

                    x[(long)i * incx] = unitDiagonal ? y : y / l[i, i];
                }

                return;
            }

            int bs = TriangularBlock(n);
            TrsvLower(l.Block(0, 0, bs, bs), x, incx, unitDiagonal);
            Gemv(-1.0, l.Block(bs, 0, n - bs, bs), x, incx, x + (long)bs * incx, incx);
            TrsvLower(l.Block(bs, bs, n - bs, n - bs), x + (long)bs * incx, incx, unitDiagonal);
        }

        /// <summary>Solve U*x = b, overwriting b with x.</summary>
        /// <param name="u">Upper triangular n by n matrix.</param>
        /// <param name="x">Right-hand side on entry; solution on return.</param>
        /// <param name="incx">Stride between entries of x, in elements.</param>
        /// <param name="unitDiagonal">Treat diagonal entries as 1 without reading them.</param>
        public static void TrsvUpper(MatrixView<double> u, double* x, int incx, bool unitDiagonal = false)
        {
            int n = u.Rows;
            if (n <= 4 && u.ColumnStride == 1 && incx == 1)
            {
                // Use row dot products for contiguous rows of U and contiguous x.
                double* d = u.Data;
                long ld = u.RowStride;
                for (int i = n - 1; i >= 0; i--)
                {
                    double y = x[i];
                    for (int p = i + 1; p < n; p++)
                    {
                        y = FusedMath.Fma(-d[p + i * ld], x[p], y);
                    }

                    x[i] = unitDiagonal ? y : y / d[i + i * ld];
                }

                return;
            }

            if (n <= 4)
            {
                for (int i = n - 1; i >= 0; i--)
                {
                    double y = x[(long)i * incx];
                    for (int p = i + 1; p < n; p++)
                    {
                        y = FusedMath.Fma(-u[i, p], x[(long)p * incx], y);
                    }

                    x[(long)i * incx] = unitDiagonal ? y : y / u[i, i];
                }

                return;
            }

            int bs = TriangularBlock(n);
            TrsvUpper(u.Block(bs, bs, n - bs, n - bs), x + (long)bs * incx, incx, unitDiagonal);
            Gemv(-1.0, u.Block(0, bs, bs, n - bs), x + (long)bs * incx, incx, x, incx);
            TrsvUpper(u.Block(0, 0, bs, bs), x, incx, unitDiagonal);
        }

        /// <summary>Solve L*x = b in place, with L in column-major order.</summary>
        /// <param name="ldl">Leading dimension of l, in elements.</param>
        /// <param name="x">Contiguous right-hand side on entry; solution on return.</param>
        public static void TrsvLower(int n, double* l, int ldl, double* x)
        {
            TrsvLower(MatrixView<double>.ColumnMajor(l, n, n, ldl), x, 1);
        }

        /// <summary>Solve L^T*x = b in place, with L in column-major order.</summary>
        /// <param name="ldl">Leading dimension of l, in elements.</param>
        /// <param name="x">Contiguous right-hand side on entry; solution on return.</param>
        public static void TrsvLowerTranspose(int n, double* l, int ldl, double* x)
        {
            TrsvUpper(MatrixView<double>.ColumnMajor(l, n, n, ldl).Transpose, x, 1);
        }
    }
}
