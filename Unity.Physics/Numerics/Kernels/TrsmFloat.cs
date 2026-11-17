// Ported from faer 0.19.4 src/linalg/triangular_solve.rs.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Maximum number of rows for the float base case (one float8 vector).</summary>
        const int FloatTriangleBaseRows = 8;

        /// <summary>
        /// Size of the leading block for a recursive triangular solve.
        /// Round the trailing block to a multiple of eight rows.
        /// </summary>
        static int TriangularBlockFloat(int n)
        {
            int rest = n / 2;
            if (n >= 64)
            {
                rest = (rest + 31) / 32 * 32;
            }
            else if (n >= 32)
            {
                rest = (rest + 15) / 16 * 16;
            }
            else if (n >= 16)
            {
                rest = (rest + 7) / 8 * 8;
            }

            return n - rest;
        }

        /// <summary>Solve L*X = B, overwriting B with X.</summary>
        /// <param name="l">Lower triangular n by n matrix.</param>
        /// <param name="b">An n by nrhs matrix of right-hand sides. Overwritten with the solutions.</param>
        /// <param name="unitDiagonal">Treat diagonal entries as 1 without reading them.</param>
        public static void TrsmLeftLower(MatrixView<float> l, MatrixView<float> b, bool unitDiagonal = false)
        {
            int n = l.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= FloatTriangleBaseRows)
            {
                TrsmBaseCase(l, b, unitDiagonal, false);
                return;
            }

            int bs = TriangularBlockFloat(n);
            int nrhs = b.Columns;
            TrsmLeftLower(l.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
            Gemm(-1f, l.Block(bs, 0, n - bs, bs), b.Block(0, 0, bs, nrhs), b.Block(bs, 0, n - bs, nrhs));
            TrsmLeftLower(l.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
        }

        /// <summary>Solve U*X = B, overwriting B with X.</summary>
        /// <param name="u">Upper triangular n by n matrix.</param>
        /// <param name="b">An n by nrhs matrix of right-hand sides. Overwritten with the solutions.</param>
        /// <param name="unitDiagonal">Treat diagonal entries as 1 without reading them.</param>
        public static void TrsmLeftUpper(MatrixView<float> u, MatrixView<float> b, bool unitDiagonal = false)
        {
            int n = u.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= FloatTriangleBaseRows)
            {
                TrsmBaseCase(u, b, unitDiagonal, true);
                return;
            }

            int bs = TriangularBlockFloat(n);
            int nrhs = b.Columns;
            TrsmLeftUpper(u.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
            Gemm(-1f, u.Block(0, bs, bs, n - bs), b.Block(bs, 0, n - bs, nrhs), b.Block(0, 0, bs, nrhs));
            TrsmLeftUpper(u.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
        }

        /// <summary>Forward or backward substitution for a triangle with at most eight rows.</summary>
        static void TrsmBaseCase(MatrixView<float> t, MatrixView<float> b, bool unitDiagonal, bool upper)
        {
            int n = t.Rows;
            int nrhs = b.Columns;
            float8 inverse8 = 1f;
            float* inverse = (float*)&inverse8;
            if (!unitDiagonal)
            {
                for (int i = 0; i < n; i++)
                {
                    inverse[i] = 1f / t[i, i];
                }
            }

            if (!b.ColumnsAreContiguous && b.RowsAreContiguous)
            {
                for (int s = 0; s < n; s++)
                {
                    int i = upper ? n - 1 - s : s;
                    float* bi = b.At(i, 0);
                    int from = upper ? i + 1 : 0;
                    int to = upper ? n : i;
                    for (int p = from; p < to; p++)
                    {
                        Axpy(nrhs, -t[i, p], b.At(p, 0), bi);
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
                for (int s = 0; s < n; s++)
                {
                    int i = upper ? n - 1 - s : s;
                    int from = upper ? i + 1 : 0;
                    int to = upper ? n : i;
                    float y = b[i, j] * inverse[i];
                    for (int p = from; p < to; p++)
                    {
                        y = FusedMath.Fma(-(t[i, p] * inverse[i]), b[p, j], y);
                    }

                    b[i, j] = y;
                }
            }
        }
    }
}
