// Ported from faer 0.19.4 src/linalg/triangular_solve.rs.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// Size of the leading block for a recursive triangular solve.
        /// Round the trailing block to a multiple of the vector width.
        /// </summary>
        static int TriangularBlock(int n)
        {
            int rest = n / 2;
            if (n >= 32)
            {
                rest = (rest + 15) / 16 * 16;
            }
            else if (n >= 16)
            {
                rest = (rest + 7) / 8 * 8;
            }
            else if (n >= 8)
            {
                rest = (rest + 3) / 4 * 4;
            }

            return n - rest;
        }

        /// <summary>Solve L*X = B, overwriting B with X.</summary>
        /// <param name="l">Lower triangular n by n matrix.</param>
        /// <param name="b">An n by nrhs matrix of right-hand sides. Overwritten with the solutions.</param>
        /// <param name="unitDiagonal">Treat diagonal entries as 1 without reading them.</param>
        public static void TrsmLeftLower(MatrixView<double> l, MatrixView<double> b, bool unitDiagonal = false)
        {
            int n = l.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= 4)
            {
                TrsmLeftLowerBaseCase(l, b, unitDiagonal);
                return;
            }

            int bs = TriangularBlock(n);
            int nrhs = b.Columns;
            TrsmLeftLower(l.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
            Gemm(-1.0, l.Block(bs, 0, n - bs, bs), b.Block(0, 0, bs, nrhs), b.Block(bs, 0, n - bs, nrhs));
            TrsmLeftLower(l.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
        }

        /// <summary>Forward substitution for a triangle with at most four rows.</summary>
        static void TrsmLeftLowerBaseCase(MatrixView<double> l, MatrixView<double> b, bool unitDiagonal)
        {
            int n = l.Rows;
            int nrhs = b.Columns;
            if (!b.ColumnsAreContiguous && b.RowsAreContiguous)
            {
                TrsmLeftLowerByRows(l, b, unitDiagonal);
                return;
            }

            double4 inverse = 1.0;
            if (!unitDiagonal)
            {
                for (int i = 0; i < n; i++)
                {
                    inverse[i] = 1.0 / l[i, i];
                }
            }

            for (int j = 0; j < nrhs; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    double y = b[i, j] * inverse[i];
                    for (int p = 0; p < i; p++)
                    {
                        y = FusedMath.Fma(-(l[i, p] * inverse[i]), b[p, j], y);
                    }

                    b[i, j] = y;
                }
            }
        }

        /// <summary>Forward substitution for right-hand sides with contiguous rows.</summary>
        static void TrsmLeftLowerByRows(MatrixView<double> l, MatrixView<double> b, bool unitDiagonal)
        {
            int n = l.Rows;
            int nrhs = b.Columns;
            double* r0 = b.At(0, 0);
            double* r1 = n > 1 ? b.At(1, 0) : r0;
            double* r2 = n > 2 ? b.At(2, 0) : r0;
            double* r3 = n > 3 ? b.At(3, 0) : r0;
            double i0 = unitDiagonal ? 1.0 : 1.0 / l[0, 0];
            double i1 = n > 1 && !unitDiagonal ? 1.0 / l[1, 1] : 1.0;
            double i2 = n > 2 && !unitDiagonal ? 1.0 / l[2, 2] : 1.0;
            double i3 = n > 3 && !unitDiagonal ? 1.0 / l[3, 3] : 1.0;
            double m10 = n > 1 ? -l[1, 0] : 0.0;
            double m20 = n > 2 ? -l[2, 0] : 0.0, m21 = n > 2 ? -l[2, 1] : 0.0;
            double m30 = n > 3 ? -l[3, 0] : 0.0, m31 = n > 3 ? -l[3, 1] : 0.0, m32 = n > 3 ? -l[3, 2] : 0.0;
            int k = 0;
            for (; k + 4 <= nrhs; k += 4)
            {
                double4 x0 = *(double4*)(r0 + k) * i0;
                *(double4*)(r0 + k) = x0;
                if (n > 1)
                {
                    double4 x1 = FusedMath.Fma(m10, x0, *(double4*)(r1 + k)) * i1;
                    *(double4*)(r1 + k) = x1;
                    if (n > 2)
                    {
                        double4 x2 = FusedMath.Fma(m21, x1, FusedMath.Fma(m20, x0, *(double4*)(r2 + k))) * i2;
                        *(double4*)(r2 + k) = x2;
                        if (n > 3)
                        {
                            double4 x3 = FusedMath.Fma(m32, x2, FusedMath.Fma(m31, x1, FusedMath.Fma(m30, x0, *(double4*)(r3 + k)))) * i3;
                            *(double4*)(r3 + k) = x3;
                        }
                    }
                }
            }

            for (; k < nrhs; k++)
            {
                double x0 = r0[k] * i0;
                r0[k] = x0;
                if (n > 1)
                {
                    double x1 = FusedMath.Fma(m10, x0, r1[k]) * i1;
                    r1[k] = x1;
                    if (n > 2)
                    {
                        double x2 = FusedMath.Fma(m21, x1, FusedMath.Fma(m20, x0, r2[k])) * i2;
                        r2[k] = x2;
                        if (n > 3)
                        {
                            r3[k] = FusedMath.Fma(m32, x2, FusedMath.Fma(m31, x1, FusedMath.Fma(m30, x0, r3[k]))) * i3;
                        }
                    }
                }
            }
        }

        /// <summary>Solve U*X = B, overwriting B with X.</summary>
        /// <param name="u">Upper triangular n by n matrix.</param>
        /// <param name="b">An n by nrhs matrix of right-hand sides. Overwritten with the solutions.</param>
        /// <param name="unitDiagonal">Treat diagonal entries as 1 without reading them.</param>
        public static void TrsmLeftUpper(MatrixView<double> u, MatrixView<double> b, bool unitDiagonal = false)
        {
            int n = u.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= 4)
            {
                TrsmLeftUpperBaseCase(u, b, unitDiagonal);
                return;
            }

            int bs = TriangularBlock(n);
            int nrhs = b.Columns;
            TrsmLeftUpper(u.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
            Gemm(-1.0, u.Block(0, bs, bs, n - bs), b.Block(bs, 0, n - bs, nrhs), b.Block(0, 0, bs, nrhs));
            TrsmLeftUpper(u.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
        }

        /// <summary>Backward substitution for a triangle with at most four rows.</summary>
        static void TrsmLeftUpperBaseCase(MatrixView<double> u, MatrixView<double> b, bool unitDiagonal)
        {
            int n = u.Rows;
            int nrhs = b.Columns;
            double4 inverse = 1.0;
            if (!unitDiagonal)
            {
                for (int i = 0; i < n; i++)
                {
                    inverse[i] = 1.0 / u[i, i];
                }
            }

            if (!b.ColumnsAreContiguous && b.RowsAreContiguous)
            {
                for (int i = n - 1; i >= 0; i--)
                {
                    double* bi = b.At(i, 0);
                    for (int p = i + 1; p < n; p++)
                    {
                        Axpy(nrhs, -u[i, p], b.At(p, 0), bi);
                    }

                    if (!unitDiagonal)
                    {
                        Scale(nrhs, inverse[i], bi);
                    }
                }

                return;
            }

            for (int j = 0; j < nrhs; j++)
            {
                for (int i = n - 1; i >= 0; i--)
                {
                    double y = b[i, j] * inverse[i];
                    for (int p = i + 1; p < n; p++)
                    {
                        y = FusedMath.Fma(-(u[i, p] * inverse[i]), b[p, j], y);
                    }

                    b[i, j] = y;
                }
            }
        }

        /// <summary>Solve L*X = B in place. L is n by n and B is n by nrhs, both in column-major order.</summary>
        /// <param name="ldl">Leading dimension of l, in elements.</param>
        /// <param name="ldb">Leading dimension of b, in elements.</param>
        public static void TrsmLeftLower(int n, int nrhs, double* l, int ldl, double* b, int ldb)
        {
            TrsmLeftLower(MatrixView<double>.ColumnMajor(l, n, n, ldl), MatrixView<double>.ColumnMajor(b, n, nrhs, ldb));
        }

        /// <summary>Solve X*L^T = B in place. L is n by n and B is m by n, both in column-major order.</summary>
        /// <param name="ldl">Leading dimension of l, in elements.</param>
        /// <param name="ldb">Leading dimension of b, in elements.</param>
        public static void TrsmRightLowerTranspose(int m, int n, double* l, int ldl, double* b, int ldb)
        {
            TrsmLeftLower(MatrixView<double>.ColumnMajor(l, n, n, ldl), MatrixView<double>.ColumnMajor(b, m, n, ldb).Transpose);
        }
    }
}
