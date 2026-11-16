// Ported from faer 0.19.4 src/linalg/triangular_solve.rs (the recursive triangular solves and their block sizes).
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The float version, written for vectors of eight floats (float8): the blocks are rounded to whole vectors of eight,
// so they are twice as large as for doubles, and the base case takes up to 8 rows.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>The largest triangle that the float solves handle without splitting it: one vector of eight rows.</summary>
        const int FloatTriangleBaseRows = 8;

        /// <summary>
        /// faer's block size for the recursive triangular solve of floats: the lower part of the split is about half
        /// of the rows, rounded up to a multiple of 32, 16 or 8 rows as the matrix gets smaller, so that it fills
        /// whole vectors of eight floats. Returns the size of the upper part.
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

        /// <summary>
        /// B := L^-1 B for floats, where L is n by n and lower triangular, and B is n by nrhs. The recursion is the
        /// same as for doubles, with the float blocks. If <paramref name="unitDiagonal"/> is true, the routine takes
        /// every diagonal entry of L as 1 and never reads it.
        /// </summary>
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

        /// <summary>
        /// B := U^-1 B for floats, where U is n by n and upper triangular: the lower block of rows first, then
        /// B_0 -= U_01 X_1, then the upper block, as for doubles.
        /// </summary>
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

        /// <summary>
        /// The base case of the float solves, for 8 rows or fewer, with the lower triangle or, if
        /// <paramref name="upper"/> is true, the upper one. It takes the reciprocals of the diagonal once, into one
        /// vector of eight. If the rows of B are contiguous and its columns are not, it works on whole rows of B, which
        /// runs in vectors along the row. Otherwise it runs the substitution down each column of B.
        /// </summary>
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
