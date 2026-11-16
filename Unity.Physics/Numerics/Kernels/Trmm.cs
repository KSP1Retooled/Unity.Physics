// The product of a triangular matrix with a general one, B := T B. faer 0.19.4 computes it in
// src/linalg/matmul/triangular.rs by splitting the triangle in half and recursing, and this file follows that shape with
// the block sizes of the triangular solves in Trsm.cs.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers, with double4 in place of the SIMD types.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// B := L B, where L is n by n and lower triangular, and B is n by nrhs. For more than 4 rows, the routine
        /// splits L as the triangular solve does. The lower block of rows of the answer needs the upper block of B as
        /// it was, so it goes first: B_1 := L_11 B_1, then B_1 += L_10 B_0, then B_0 := L_00 B_0. If
        /// <paramref name="unitDiagonal"/> is true, the routine takes every diagonal entry of L as 1 and never reads
        /// it. The product B := B L, from the right, is this routine on the transposes: B^T := L^T B^T, with
        /// <see cref="TrmmLeftUpper(MatrixView{double}, MatrixView{double}, bool)"/>.
        /// </summary>
        public static void TrmmLeftLower(MatrixView<double> l, MatrixView<double> b, bool unitDiagonal = false)
        {
            int n = l.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= 4)
            {
                TrmmBaseCase(l, b, unitDiagonal, false);
                return;
            }

            int bs = TriangularBlock(n);
            int nrhs = b.Columns;
            TrmmLeftLower(l.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
            Gemm(1.0, l.Block(bs, 0, n - bs, bs), b.Block(0, 0, bs, nrhs), b.Block(bs, 0, n - bs, nrhs));
            TrmmLeftLower(l.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
        }

        /// <summary>
        /// B := U B, where U is n by n and upper triangular: B_0 := U_00 B_0, then B_0 += U_01 B_1, then
        /// B_1 := U_11 B_1, because the upper block of rows of the answer needs the lower block of B as it was.
        /// </summary>
        public static void TrmmLeftUpper(MatrixView<double> u, MatrixView<double> b, bool unitDiagonal = false)
        {
            int n = u.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= 4)
            {
                TrmmBaseCase(u, b, unitDiagonal, true);
                return;
            }

            int bs = TriangularBlock(n);
            int nrhs = b.Columns;
            TrmmLeftUpper(u.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
            Gemm(1.0, u.Block(0, bs, bs, n - bs), b.Block(bs, 0, n - bs, nrhs), b.Block(0, 0, bs, nrhs));
            TrmmLeftUpper(u.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
        }

        /// <summary>
        /// The base case of the triangular products, for 4 rows or fewer. Each row of the answer is a sum of rows of B
        /// that the routine has not changed yet: for a lower triangle it goes from the last row up, and for an upper
        /// one from the first row down. If the rows of B are contiguous and its columns are not, it works on whole
        /// rows, which runs in vectors along the row. Otherwise it works down each column of B.
        /// </summary>
        static void TrmmBaseCase(MatrixView<double> t, MatrixView<double> b, bool unitDiagonal, bool upper)
        {
            int n = t.Rows;
            int nrhs = b.Columns;
            if (!b.ColumnsAreContiguous && b.RowsAreContiguous)
            {
                for (int s = 0; s < n; s++)
                {
                    int i = upper ? s : n - 1 - s;
                    double* bi = b.At(i, 0);
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
                    double y = unitDiagonal ? b[i, j] : t[i, i] * b[i, j];
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
