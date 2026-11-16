// Ported from faer 0.19.4 (https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md):
//   src/linalg/cholesky/llt/update.rs (rank_r_update_clobber, insert_rows_and_cols_clobber, delete_rows_and_cols_clobber)
// The algorithms are faer's. The code is written for raw double pointers in column-major order, with double4 in place
// of faer's SIMD types, and calls the BLAS-style routines in Blas. What was adapted is listed in each summary.

using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class CholeskyLLT
    {
        /// <summary>
        /// Changes the factor of A, n by n, into the factor of A + W diag(alpha) W^T, as faer's rank_r_update_clobber
        /// does. W is n by k and is overwritten, and so is alpha. If <paramref name="firstRows"/> is not null, column c of
        /// W is zero above row firstRows[c], the rows are in increasing order, and column j of the factor only takes
        /// the columns of W that start at or above row j. faer gets the same count from a function; this port takes
        /// the rows instead.
        /// <para>
        /// For each column j of L, the update takes the columns of W four at a time. For each of the four, a few
        /// scalars give the new diagonal entry, and then one pass down the column below the diagonal changes the
        /// column of L and the four columns of W together. Each row's chain goes through the four columns in turn,
        /// and the rows are independent, so the pass runs four rows at a time in vectors.
        /// </para>
        /// Returns how many pivots were raised to <see cref="SmallestPivot"/>; faer returns an error there instead.
        /// </summary>
        public static int RankUpdate(int n, double* l, int ldl, int k, double* w, int ldw, double* alpha, int* firstRows)
        {
            const int BlockSize = 4;
            int raised = 0;
            int active = firstRows == null ? k : 0;
            double4 negative = 0.0, newOverOld = 0.0, alphaOverNew = 0.0;
            for (int j = 0; j < n; j++)
            {
                while (firstRows != null && active < k && firstRows[active] <= j)
                {
                    active++;
                }

                double* lj = l + (long)j * ldl;
                for (int r = 0; r < active; r += BlockSize)
                {
                    int count = math.min(BlockSize, active - r);
                    // faer divides by the old and the new diagonal entry three times for each column of W. Each
                    // division waits on the square root before it, so this port takes one reciprocal of each new entry
                    // and multiplies by it, which leaves one square root and one division on the chain.
                    double ljj = lj[j];
                    double inverse = 1.0 / ljj;
                    for (int t = 0; t < count; t++)
                    {
                        double wj = w[j + (long)(r + t) * ldw];
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

                    lj[j] = ljj;
                    double* w0 = w + (long)r * ldw;
                    double* w1 = w0 + ldw;
                    double* w2 = w1 + ldw;
                    double* w3 = w2 + ldw;
                    int i = j + 1;
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

                    for (; i < n; i++)
                    {
                        double li = lj[i];
                        for (int t = 0; t < count; t++)
                        {
                            double* wt = w + i + (long)(r + t) * ldw;
                            *wt = FusedMath.Fma(negative[t], li, *wt);
                            li = FusedMath.Fma(alphaOverNew[t], *wt, newOverOld[t] * li);
                        }

                        lj[i] = li;
                    }
                }
            }

            return raised;
        }

        static void Step(ref double4 l, double* w, double negative, double alphaOverNew, double newOverOld)
        {
            double4 wi = FusedMath.Fma(negative, l, *(double4*)w);
            *(double4*)w = wi;
            l = FusedMath.Fma(alphaOverNew, wi, newOverOld * l);
        }

        /// <summary>
        /// The number of doubles of scratch that <see cref="InsertRowsAndColumns"/> needs.
        /// </summary>
        public static int InsertScratch(int n, int p, int r) => (n - p) * r + r;

        /// <summary>
        /// Inserts r rows and columns at place p, as faer's insert_rows_and_cols_clobber does. On input, the lower
        /// triangle of <paramref name="l"/> holds the factor of the n by n matrix A, and B, (n + r) by r, holds the new
        /// columns of the larger matrix: row i of B is entry i of the new matrix, in the new matrix's order. On output,
        /// <paramref name="l"/> holds the factor of the (n + r) by (n + r) matrix, and B is overwritten.
        /// <para>
        /// The routine moves the rows and columns from p on down and to the right by r. Then it solves
        /// X = L_00^-1 B_0, which is the new rows' part in the first p columns. The new diagonal block is the factor of
        /// B_1 - X^T X. The new rows' part below it is (B_2 - L_20 X) L_11^-T. Last, the factor of the rows below takes
        /// out the product of that part with itself, which is a rank-r update with weights of -1. faer keeps the
        /// weights and the copy of that part in B's storage; this port takes a scratch array of
        /// <see cref="InsertScratch"/> doubles for them. If p equals n, nothing moves and the last step has no rows.
        /// </para>
        /// Returns how many pivots were raised to <see cref="SmallestPivot"/>.
        /// </summary>
        public static int InsertRowsAndColumns(int n, double* l, int ldl, int p, int r, double* b, int ldb, double* scratch)
        {
            if (r == 0)
            {
                return 0;
            }

            int total = n + r;

            // Move the old factor's rows and columns from p on to make room, from the right so that nothing is
            // overwritten before it moves.
            for (int j = n - 1; j >= p; j--)
            {
                for (int i = n - 1; i >= j; i--)
                {
                    l[i + r + (long)(j + r) * ldl] = l[i + (long)j * ldl];
                }
            }

            for (int j = 0; j < p; j++)
            {
                for (int i = n - 1; i >= p; i--)
                {
                    l[i + r + (long)j * ldl] = l[i + (long)j * ldl];
                }
            }

            double* x = b;
            double* b1 = b + p;
            double* b2 = b + p + r;
            double* l10 = l + p;
            double* l11 = l + p + (long)p * ldl;
            double* l20 = l + p + r;
            double* l21 = l + p + r + (long)p * ldl;
            double* l22 = l + p + r + (long)(p + r) * ldl;

            // X = L_00^-1 B_0, and the new rows' part in the first p columns is X^T.
            Blas.TrsmLeftLower(p, r, l, ldl, x, ldb);
            for (int j = 0; j < p; j++)
            {
                for (int i = 0; i < r; i++)
                {
                    l10[i + (long)j * ldl] = x[j + (long)i * ldb];
                }
            }

            // The new diagonal block: the factor of B_1 - X^T X.
            for (int j = 0; j < r; j++)
            {
                for (int i = j; i < r; i++)
                {
                    l11[i + (long)j * ldl] = b1[i + (long)j * ldb];
                }
            }

            Blas.SyrkLowerNT(r, p, -1.0, l10, ldl, l11, ldl);
            int raised = Factor(r, l11, ldl);

            // The new rows' part below the diagonal block: (B_2 - L_20 X) L_11^-T.
            int below = n - p;
            for (int j = 0; j < r; j++)
            {
                for (int i = 0; i < below; i++)
                {
                    l21[i + (long)j * ldl] = b2[i + (long)j * ldb];
                }
            }

            Blas.GemmNN(below, r, p, -1.0, l20, ldl, x, ldb, l21, ldl);
            Blas.TrsmRightLowerTranspose(below, r, l11, ldl, l21, ldl);

            // The rows below take out the product of that part with itself.
            if (below > 0)
            {
                double* weights = scratch;
                double* copy = scratch + r;
                for (int j = 0; j < r; j++)
                {
                    weights[j] = -1.0;
                    for (int i = 0; i < below; i++)
                    {
                        copy[i + (long)j * below] = l21[i + (long)j * ldl];
                    }
                }

                raised += RankUpdate(below, l22, ldl, r, copy, below, weights, null);
            }

            return raised;
        }

        /// <summary>
        /// The number of doubles of scratch that <see cref="DeleteRowsAndColumns"/> needs, with
        /// <paramref name="first"/> the first deleted index.
        /// </summary>
        public static int DeleteScratch(int n, int first, int r) => (n - first - r) * r + r + (r + 1) / 2;

        /// <summary>
        /// Deletes r rows and columns, as faer's delete_rows_and_cols_clobber does. The indices must be in increasing
        /// order. On output, the lower triangle of the first n - r rows and columns holds the factor of A without them.
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
        public static int DeleteRowsAndColumns(int n, double* l, int ldl, int* indices, int r, double* scratch)
        {
            if (r == 0)
            {
                return 0;
            }

            int first = indices[0];
            int rows = n - first - r;
            double* w = scratch;
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
                        w[i - c - first + (long)k * rows] = l[i + (long)j * ldl];
                    }
                }

                // The copy is zero above its first row; the update never reads those rows.
                for (int i = 0; i < firstRows[k] && i < rows; i++)
                {
                    w[i + (long)k * rows] = 0.0;
                }
            }

            // Close the gaps. Every entry moves up, to the left, or both, so a pass in storage order never overwrites
            // an entry that has yet to move. In the columns before the first deleted index, the rows before that index
            // stay where they are. Each run of kept rows between two deleted indices moves as one block.
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
                double* source = l + (long)j * ldl;
                double* target = l + (long)column * ldl;
                for (int i = start; i < n; nextDeletedRow++)
                {
                    int end = nextDeletedRow < r ? indices[nextDeletedRow] : n;
                    int length = end - i;
                    if (length > 0 && target + row != source + i)
                    {
                        UnsafeUtility.MemMove(target + row, source + i, (long)length * sizeof(double));
                    }

                    row += length;
                    i = end + 1;
                }

                column++;
            }

            double* l22 = l + first + (long)first * ldl;
            return RankUpdate(rows, l22, ldl, r, w, rows, alpha, firstRows);
        }
    }
}
