// Ported from faer 0.19.4 (https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md):
//   src/linalg/cholesky/llt/update.rs (rank_r_update_clobber, insert_rows_and_cols_clobber, delete_rows_and_cols_clobber)
// The algorithms are faer's. The code is written for raw double pointers, with double4 in place of faer's SIMD types,
// and calls the BLAS-style routines in Blas. What was adapted is listed in each summary.

using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class CholeskyLLT
    {
        /// <summary>
        /// Changes the factor L of A, n by n, into the factor of A + W diag(alpha) W^T, as faer's rank_r_update_clobber
        /// does. W is n by k and is overwritten, and so is alpha. If <paramref name="firstRows"/> is not null, column c of
        /// W is zero above row firstRows[c], the rows are in increasing order, and column j of the factor only takes
        /// the columns of W that start at or above row j. faer gets the same count from a function; this port takes
        /// the rows instead.
        /// <para>
        /// For each column j of L, the update takes the columns of W four at a time. For each of the four, a few
        /// scalars give the new diagonal entry, and then one pass down the column below the diagonal changes the
        /// column of L and the four columns of W together. Each row's chain goes through the four columns in turn,
        /// and the rows are independent. If the columns of L and W are contiguous, the pass runs four rows at a time
        /// in vectors; otherwise it runs one row at a time.
        /// </para>
        /// Returns how many pivots were raised to <see cref="SmallestPivot"/>; faer returns an error there instead.
        /// </summary>
        public static int RankUpdate(MatrixView<double> l, MatrixView<double> w, double* alpha, int* firstRows)
        {
            const int BlockSize = 4;
            int n = l.Rows;
            int k = w.Columns;
            bool vectors = l.ColumnsAreContiguous && w.ColumnsAreContiguous;
            int raised = 0;
            int active = firstRows == null ? k : 0;
            double4 negative = 0.0, newOverOld = 0.0, alphaOverNew = 0.0;
            for (int j = 0; j < n; j++)
            {
                while (firstRows != null && active < k && firstRows[active] <= j)
                {
                    active++;
                }

                for (int r = 0; r < active; r += BlockSize)
                {
                    int count = math.min(BlockSize, active - r);
                    // faer divides by the old and the new diagonal entry three times for each column of W. Each
                    // division waits on the square root before it, so this port takes one reciprocal of each new entry
                    // and multiplies by it, which leaves one square root and one division on the chain.
                    double ljj = l[j, j];
                    double inverse = 1.0 / ljj;
                    for (int t = 0; t < count; t++)
                    {
                        double wj = w[j, r + t];
                        double aw = alpha[r + t] * wj;
                        double sum = FusedMath.Fma(ljj, ljj, aw * wj);
                        if (!(sum > SmallestPivot))
                        {
                            sum = SmallestPivot;
                            raised++;
                        }

                        double newLjj = math.sqrt(sum);
                        double newInverse = 1.0 / newLjj;
                        negative[t] = -wj * inverse;
                        newOverOld[t] = newLjj * inverse;
                        alphaOverNew[t] = aw * newInverse;
                        alpha[r + t] -= alphaOverNew[t] * alphaOverNew[t];
                        ljj = newLjj;
                        inverse = newInverse;
                    }

                    l[j, j] = ljj;
                    int i = j + 1;
                    if (vectors)
                    {
                        double* lj = l.At(0, j);
                        double* w0 = w.At(0, r);
                        double* w1 = w0 + w.LeadingDimension;
                        double* w2 = w1 + w.LeadingDimension;
                        double* w3 = w2 + w.LeadingDimension;
                        for (; i + 4 <= n; i += 4)
                        {
                            double4 li = *(double4*)(lj + i);
                            Step(ref li, w0 + i, negative.x, alphaOverNew.x, newOverOld.x);
                            if (count > 1)
                            {
                                Step(ref li, w1 + i, negative.y, alphaOverNew.y, newOverOld.y);
                            }

                            if (count > 2)
                            {
                                Step(ref li, w2 + i, negative.z, alphaOverNew.z, newOverOld.z);
                            }

                            if (count > 3)
                            {
                                Step(ref li, w3 + i, negative.w, alphaOverNew.w, newOverOld.w);
                            }

                            *(double4*)(lj + i) = li;
                        }
                    }

                    for (; i < n; i++)
                    {
                        double li = l[i, j];
                        for (int t = 0; t < count; t++)
                        {
                            double* wt = w.At(i, r + t);
                            *wt = FusedMath.Fma(negative[t], li, *wt);
                            li = FusedMath.Fma(alphaOverNew[t], *wt, newOverOld[t] * li);
                        }

                        l[i, j] = li;
                    }
                }
            }

            return raised;
        }

        /// <summary>The rank update of a factor and a W that are both stored column by column.</summary>
        public static int RankUpdate(int n, double* l, int ldl, int k, double* w, int ldw, double* alpha, int* firstRows)
        {
            return RankUpdate(MatrixView<double>.ColumnMajor(l, n, n, ldl), MatrixView<double>.ColumnMajor(w, n, k, ldw), alpha, firstRows);
        }

        static void Step(ref double4 l, double* w, double negative, double alphaOverNew, double newOverOld)
        {
            double4 wi = FusedMath.Fma(negative, l, *(double4*)w);
            *(double4*)w = wi;
            l = FusedMath.Fma(alphaOverNew, wi, newOverOld * l);
        }

        /// <summary>
        /// The number of doubles of scratch that <see cref="InsertRowsAndColumns(int, MatrixView{double}, int, int, MatrixView{double}, double*)"/>
        /// needs.
        /// </summary>
        public static int InsertScratch(int n, int p, int r) => (n - p) * r + r;

        /// <summary>
        /// Inserts r rows and columns at place p, as faer's insert_rows_and_cols_clobber does. On input, the lower
        /// triangle of the first n rows and columns of <paramref name="l"/> holds the factor of the n by n matrix A,
        /// and B, (n + r) by r, holds the new columns of the larger matrix: row i of B is entry i of the new matrix, in
        /// the new matrix's order. <paramref name="l"/> must have room for n + r rows and columns. On output, it holds
        /// the factor of the (n + r) by (n + r) matrix, and B is overwritten.
        /// <para>
        /// The routine moves the rows and columns from p on down and to the right by r. Then it solves
        /// X = L_00^-1 B_0, which is the new rows' part in the first p columns. The new diagonal block is the factor of
        /// B_1 - X^T X. The new rows' part below it is (B_2 - L_20 X) L_11^-T. Last, the factor of the rows below takes
        /// out the product of that part with itself, which is a rank-r update with weights of -1. faer keeps the
        /// weights and the copy of that part in B's storage; this port takes a scratch array of
        /// <see cref="InsertScratch"/> doubles for them. If p equals n, nothing moves and the last step has no rows.
        /// </para>
        /// <para>
        /// The product and the factor of the new diagonal block write the whole tiles that its diagonal crosses, so
        /// some entries above the diagonal of the factor change, as they did before the factor learned to keep them.
        /// </para>
        /// Returns how many pivots were raised to <see cref="SmallestPivot"/>.
        /// </summary>
        public static int InsertRowsAndColumns(int n, MatrixView<double> l, int p, int r, MatrixView<double> b, double* scratch)
        {
            if (r == 0)
            {
                return 0;
            }

            // Move the old factor's rows and columns from p on to make room, from the right so that nothing is
            // overwritten before it moves.
            for (int j = n - 1; j >= p; j--)
            {
                for (int i = n - 1; i >= j; i--)
                {
                    l[i + r, j + r] = l[i, j];
                }
            }

            for (int j = 0; j < p; j++)
            {
                for (int i = n - 1; i >= p; i--)
                {
                    l[i + r, j] = l[i, j];
                }
            }

            int below = n - p;
            MatrixView<double> x = b.Block(0, 0, p, r);
            MatrixView<double> b1 = b.Block(p, 0, r, r);
            MatrixView<double> b2 = b.Block(p + r, 0, below, r);
            MatrixView<double> l00 = l.Block(0, 0, p, p);
            MatrixView<double> l10 = l.Block(p, 0, r, p);
            MatrixView<double> l11 = l.Block(p, p, r, r);
            MatrixView<double> l20 = l.Block(p + r, 0, below, p);
            MatrixView<double> l21 = l.Block(p + r, p, below, r);
            MatrixView<double> l22 = l.Block(p + r, p + r, below, below);

            // X = L_00^-1 B_0, and the new rows' part in the first p columns is X^T.
            Blas.TrsmLeftLower(l00, x);
            for (int j = 0; j < p; j++)
            {
                for (int i = 0; i < r; i++)
                {
                    l10[i, j] = x[j, i];
                }
            }

            // The new diagonal block: the factor of B_1 - X^T X.
            for (int j = 0; j < r; j++)
            {
                for (int i = j; i < r; i++)
                {
                    l11[i, j] = b1[i, j];
                }
            }

            Blas.SyrkLower(-1.0, l10, l11, false);
            int raised = Factor(l11, false);

            // The new rows' part below the diagonal block: (B_2 - L_20 X) L_11^-T, which is the lower solve on its
            // transpose.
            for (int j = 0; j < r; j++)
            {
                for (int i = 0; i < below; i++)
                {
                    l21[i, j] = b2[i, j];
                }
            }

            Blas.Gemm(-1.0, l20, x, l21);
            Blas.TrsmLeftLower(l11, l21.Transpose);

            // The rows below take out the product of that part with itself.
            if (below > 0)
            {
                double* weights = scratch;
                MatrixView<double> copy = MatrixView<double>.ColumnMajor(scratch + r, below, r, below);
                for (int j = 0; j < r; j++)
                {
                    weights[j] = -1.0;
                    for (int i = 0; i < below; i++)
                    {
                        copy[i, j] = l21[i, j];
                    }
                }

                raised += RankUpdate(l22, copy, weights, null);
            }

            return raised;
        }

        /// <summary>The insertion into a factor and a B that are both stored column by column.</summary>
        public static int InsertRowsAndColumns(int n, double* l, int ldl, int p, int r, double* b, int ldb, double* scratch)
        {
            return InsertRowsAndColumns(n, MatrixView<double>.ColumnMajor(l, n + r, n + r, ldl), p, r,
                MatrixView<double>.ColumnMajor(b, n + r, r, ldb), scratch);
        }

        /// <summary>
        /// The number of doubles of scratch that <see cref="DeleteRowsAndColumns(MatrixView{double}, int*, int, double*)"/>
        /// needs, with <paramref name="first"/> the first deleted index.
        /// </summary>
        public static int DeleteScratch(int n, int first, int r) => (n - first - r) * r + r + (r + 1) / 2;

        /// <summary>
        /// Deletes r rows and columns from the factor of the n by n matrix in <paramref name="l"/>, as faer's
        /// delete_rows_and_cols_clobber does. The indices must be in increasing order. On output, the lower triangle
        /// of the first n - r rows and columns holds the factor of A without them.
        /// <para>
        /// Before anything moves, the routine copies the parts of the deleted columns that lie in the kept rows below
        /// the first deleted index. Then it closes the gaps in the stored triangle. The kept rows below the first
        /// deleted index lost those columns' parts, so their factor takes a rank-r update with weights of +1 and the
        /// copied parts as its vectors. Column c of the copy starts at the first kept row after deleted index c, so
        /// each column of the factor only takes the copies that have started. faer computes that count in a function;
        /// this port keeps each copy's first row in the scratch array, which holds
        /// <see cref="DeleteScratch"/> doubles.
        /// </para>
        /// Returns how many pivots were raised to <see cref="SmallestPivot"/>.
        /// </summary>
        public static int DeleteRowsAndColumns(MatrixView<double> l, int* indices, int r, double* scratch)
        {
            if (r == 0)
            {
                return 0;
            }

            int n = l.Rows;
            int first = indices[0];
            int rows = n - first - r;
            MatrixView<double> w = MatrixView<double>.ColumnMajor(scratch, rows, r, rows);
            double* alpha = scratch + (long)rows * r;
            int* firstRows = (int*)(alpha + r);

            // Copy the deleted columns' parts in the kept rows. Row i of the old matrix, with c deleted indices
            // before it, is row i - c - first of the copy.
            for (int k = 0; k < r; k++)
            {
                alpha[k] = 1.0;
                firstRows[k] = indices[k] - k - first;
                int j = indices[k];
                for (int c = k + 1; c <= r; c++)
                {
                    int start = indices[c - 1] + 1;
                    int end = c == r ? n : indices[c];
                    for (int i = start; i < end; i++)
                    {
                        w[i - c - first, k] = l[i, j];
                    }
                }

                // The copy is zero above its first row; the update never reads those rows.
                for (int i = 0; i < firstRows[k] && i < rows; i++)
                {
                    w[i, k] = 0.0;
                }
            }

            // Close the gaps. Every entry moves up, to the left, or both, so a pass that takes the columns in order,
            // and each column from the top down, never overwrites an entry that has yet to move. In the columns before
            // the first deleted index, the rows before that index stay where they are. If the columns are contiguous,
            // each run of kept rows between two deleted indices moves as one block.
            bool contiguous = l.ColumnsAreContiguous;
            int column = 0;
            int nextDeletedColumn = 0;
            for (int j = 0; j < n; j++)
            {
                if (nextDeletedColumn < r && indices[nextDeletedColumn] == j)
                {
                    nextDeletedColumn++;
                    continue;
                }

                int start = j < first ? first : j;
                int nextDeletedRow = 0;
                while (nextDeletedRow < r && indices[nextDeletedRow] < start)
                {
                    nextDeletedRow++;
                }

                int row = start - nextDeletedRow;
                for (int i = start; i < n; nextDeletedRow++)
                {
                    int end = nextDeletedRow < r ? indices[nextDeletedRow] : n;
                    int length = end - i;
                    if (length > 0 && (row != i || column != j))
                    {
                        if (contiguous)
                        {
                            UnsafeUtility.MemMove(l.At(row, column), l.At(i, j), (long)length * sizeof(double));
                        }
                        else
                        {
                            for (int t = 0; t < length; t++)
                            {
                                l[row + t, column] = l[i + t, j];
                            }
                        }
                    }

                    row += length;
                    i = end + 1;
                }

                column++;
            }

            return RankUpdate(l.Block(first, first, rows, rows), w, alpha, firstRows);
        }

        /// <summary>The deletion from a factor that is stored column by column.</summary>
        public static int DeleteRowsAndColumns(int n, double* l, int ldl, int* indices, int r, double* scratch)
        {
            return DeleteRowsAndColumns(MatrixView<double>.ColumnMajor(l, n, n, ldl), indices, r, scratch);
        }
    }
}
