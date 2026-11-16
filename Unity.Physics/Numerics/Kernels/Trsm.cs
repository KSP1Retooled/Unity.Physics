// Ported from faer 0.19.4 src/linalg/triangular_solve.rs (the recursive triangular solves and their block sizes).
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers, with double4 in place of the SIMD types.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// faer's block size for the recursive triangular solve: the lower part of the split is about half of the
        /// rows, rounded up to a multiple of 16, 8 or 4 rows as the matrix gets smaller, so that it fills whole
        /// vectors. Returns the size of the upper part.
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

        /// <summary>
        /// B := L^-1 B, where L is n by n and lower triangular, and B is n by nrhs. This is faer's
        /// solve_lower_triangular_in_place: for more than 4 rows it solves the upper block of rows, takes its share out
        /// of the lower block with a matrix product, and solves the lower block. faer also splits B by columns when it
        /// has more than 64; this port leaves that out, because the solver's B has few columns. If
        /// <paramref name="unitDiagonal"/> is true, the routine takes every diagonal entry of L as 1 and never reads it.
        /// <para>
        /// The solve X L^T = B, for B m by n, is this routine on the transposes: L X^T = B^T. faer solves it that way,
        /// and so does <see cref="TrsmRightLowerTranspose(int, int, double*, int, double*, int)"/>.
        /// </para>
        /// </summary>
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

        /// <summary>
        /// The base case of <see cref="TrsmLeftLower(MatrixView{double}, MatrixView{double}, bool)"/>, for 4 rows or
        /// fewer. It computes the reciprocals of the diagonal once. If the rows of B are contiguous and its columns are
        /// not, it works on whole rows of B: row i takes out the earlier rows times L's multipliers and is scaled by
        /// the reciprocal, which runs in vectors along the row. Otherwise it runs the forward substitution down each
        /// column of B.
        /// </summary>
        static void TrsmLeftLowerBaseCase(MatrixView<double> l, MatrixView<double> b, bool unitDiagonal)
        {
            int n = l.Rows;
            int nrhs = b.Columns;
            double4 inverse = 1.0;
            if (!unitDiagonal)
            {
                for (int i = 0; i < n; i++)
                {
                    inverse[i] = 1.0 / l[i, i];
                }
            }

            if (!b.ColumnsAreContiguous && b.RowsAreContiguous)
            {
                for (int i = 0; i < n; i++)
                {
                    double* bi = b.At(i, 0);
                    for (int p = 0; p < i; p++)
                    {
                        Axpy(nrhs, -l[i, p], b.At(p, 0), bi);
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

        /// <summary>
        /// B := U^-1 B, where U is n by n and upper triangular, and B is n by nrhs. faer reverses the rows and columns
        /// into a lower solve. This routine writes the same recursion for U as it is: it solves the lower block of
        /// rows first, takes its share out of the upper block with B_0 -= U_01 X_1, and solves the upper block. The
        /// solve with L^T, for a lower triangular L, is this routine on L's <see cref="MatrixView{T}.Transpose"/>.
        /// </summary>
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

        /// <summary>The base case of <see cref="TrsmLeftUpper"/>: the lower one, run from the last row up.</summary>
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

        /// <summary>
        /// B := L^-1 B, for L (n by n) and B (n by nrhs) stored column by column with leading dimensions
        /// <paramref name="ldl"/> and <paramref name="ldb"/>.
        /// </summary>
        public static void TrsmLeftLower(int n, int nrhs, double* l, int ldl, double* b, int ldb)
        {
            TrsmLeftLower(MatrixView<double>.ColumnMajor(l, n, n, ldl), MatrixView<double>.ColumnMajor(b, n, nrhs, ldb));
        }

        /// <summary>
        /// B := B L^-T, where L is n by n and lower triangular and B is m by n, both stored column by column. This is
        /// the lower solve on B^T, whose rows are contiguous, so the base case runs in vectors down the columns of B
        /// and the products take B as it is stored.
        /// </summary>
        public static void TrsmRightLowerTranspose(int m, int n, double* l, int ldl, double* b, int ldb)
        {
            TrsmLeftLower(MatrixView<double>.ColumnMajor(l, n, n, ldl), MatrixView<double>.ColumnMajor(b, m, n, ldb).Transpose);
        }
    }
}
