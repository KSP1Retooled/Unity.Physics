// The product of a triangular matrix with a general one, B := T B, for floats. faer 0.19.4 computes it in
// src/linalg/matmul/triangular.rs by splitting the triangle in half and recursing, and this file follows that shape with
// the float blocks of TrsmFloat.cs.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The float version, written for vectors of eight floats (float8).

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// B := L B for floats, where L is n by n and lower triangular, and B is n by nrhs, with the recursion of the
        /// double version: B_1 := L_11 B_1, then B_1 += L_10 B_0, then B_0 := L_00 B_0.
        /// </summary>
        public static void TrmmLeftLower(MatrixView<float> l, MatrixView<float> b, bool unitDiagonal = false)
        {
            int n = l.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= FloatTriangleBaseRows)
            {
                TrmmBaseCase(l, b, unitDiagonal, false);
                return;
            }

            int bs = TriangularBlockFloat(n);
            int nrhs = b.Columns;
            TrmmLeftLower(l.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
            Gemm(1f, l.Block(bs, 0, n - bs, bs), b.Block(0, 0, bs, nrhs), b.Block(bs, 0, n - bs, nrhs));
            TrmmLeftLower(l.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
        }

        /// <summary>
        /// B := U B for floats, where U is n by n and upper triangular: B_0 := U_00 B_0, then B_0 += U_01 B_1, then
        /// B_1 := U_11 B_1.
        /// </summary>
        public static void TrmmLeftUpper(MatrixView<float> u, MatrixView<float> b, bool unitDiagonal = false)
        {
            int n = u.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= FloatTriangleBaseRows)
            {
                TrmmBaseCase(u, b, unitDiagonal, true);
                return;
            }

            int bs = TriangularBlockFloat(n);
            int nrhs = b.Columns;
            TrmmLeftUpper(u.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
            Gemm(1f, u.Block(0, bs, bs, n - bs), b.Block(bs, 0, n - bs, nrhs), b.Block(0, 0, bs, nrhs));
            TrmmLeftUpper(u.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
        }

        /// <summary>
        /// The float base case, for 8 rows or fewer, in the same order as the double one: whole rows of B in vectors
        /// if its rows are contiguous and its columns are not, and otherwise each column of B.
        /// </summary>
        static void TrmmBaseCase(MatrixView<float> t, MatrixView<float> b, bool unitDiagonal, bool upper)
        {
            int n = t.Rows;
            int nrhs = b.Columns;
            if (!b.ColumnsAreContiguous && b.RowsAreContiguous)
            {
                for (int s = 0; s < n; s++)
                {
                    int i = upper ? s : n - 1 - s;
                    float* bi = b.At(i, 0);
                    if (!unitDiagonal)
                    {
                        Scale(nrhs, t[i, i], bi);
                    }

                    int from = upper ? i + 1 : 0;
                    int to = upper ? n : i;
                    for (int p = from; p < to; p++)
                    {
                        Axpy(nrhs, t[i, p], b.At(p, 0), bi);
                    }
                }

                return;
            }

            for (int j = 0; j < nrhs; j++)
            {
                for (int s = 0; s < n; s++)
                {
                    int i = upper ? s : n - 1 - s;
                    int from = upper ? i + 1 : 0;
                    int to = upper ? n : i;
                    float y = unitDiagonal ? b[i, j] : t[i, i] * b[i, j];
                    for (int p = from; p < to; p++)
                    {
                        y = FusedMath.Fma(t[i, p], b[p, j], y);
                    }

                    b[i, j] = y;
                }
            }
        }
    }
}
