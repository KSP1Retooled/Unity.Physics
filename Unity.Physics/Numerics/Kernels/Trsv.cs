// Ported from faer 0.19.4 src/linalg/triangular_solve.rs, with one right-hand side.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers, with double4 in place of the SIMD types.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// x := L^-1 x, where L is n by n and lower triangular, and x has n entries that are <paramref name="incx"/>
        /// apart. This is <see cref="TrsmLeftLower(MatrixView{double}, MatrixView{double}, bool)"/> with a
        /// matrix-vector product: for more than 4 rows, the routine solves the upper block of rows, takes its share out
        /// of the lower block with <see cref="Gemv(double, MatrixView{double}, double*, int, double*, int)"/>, and
        /// solves the lower block. If <paramref name="unitDiagonal"/> is true, the routine takes every diagonal entry
        /// of L as 1 and never reads it.
        /// </summary>
        public static void TrsvLower(MatrixView<double> l, double* x, int incx, bool unitDiagonal = false)
        {
            int n = l.Rows;
            if (n <= 4 && l.RowStride == 1 && incx == 1)
            {
                // The fast path: L stored column by column and x contiguous.
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

        /// <summary>
        /// x := U^-1 x, where U is n by n and upper triangular. faer solves with an upper triangle by reversing the rows
        /// and columns into a lower solve. This routine writes the same recursion for U as it is: it solves the lower
        /// block of rows first, takes its share out of the upper block with x_0 -= U_01 x_1, and solves the upper
        /// block. The solve with L^T, for a lower triangular L, is this routine on L's
        /// <see cref="MatrixView{T}.Transpose"/>. Then the product reads the columns of L as dot products.
        /// </summary>
        public static void TrsvUpper(MatrixView<double> u, double* x, int incx, bool unitDiagonal = false)
        {
            int n = u.Rows;
            if (n <= 4 && u.ColumnStride == 1 && incx == 1)
            {
                // The fast path: U is the transpose of a lower triangle stored column by column, so row i of U is
                // column i of that triangle, and x is contiguous.
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

        /// <summary>x := L^-1 x, for L stored column by column with leading dimension <paramref name="ldl"/>.</summary>
        public static void TrsvLower(int n, double* l, int ldl, double* x)
        {
            TrsvLower(MatrixView<double>.ColumnMajor(l, n, n, ldl), x, 1);
        }

        /// <summary>x := L^-T x, for L stored column by column with leading dimension <paramref name="ldl"/>.</summary>
        public static void TrsvLowerTranspose(int n, double* l, int ldl, double* x)
        {
            TrsvUpper(MatrixView<double>.ColumnMajor(l, n, n, ldl).Transpose, x, 1);
        }
    }
}
