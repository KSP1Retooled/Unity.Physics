// Ported from faer 0.19.4 (https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md):
//   src/linalg/cholesky/llt/update.rs (rank_r_update_clobber, insert_rows_and_cols_clobber, delete_rows_and_cols_clobber)

using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class CholeskyLLT
    {
        /// <summary>
        /// Update the Cholesky factor in place for A + W*diag(alpha)*W^T.
        /// </summary>
        /// <param name="l">Lower triangular factor of the n by n matrix A. Overwritten with the updated factor.</param>
        /// <param name="w">An n by k matrix of update vectors. Overwritten.</param>
        /// <param name="alpha">One weight per column of w. Overwritten.</param>
        /// <param name="firstRows">First nonzero row of each column of w, in nondecreasing order.
        /// Entries above these rows must be zero. Pass null to use all rows.</param>
        /// <returns>Number of pivots raised to SmallestPivot.</returns>
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
                    // Reuse each diagonal's reciprocal to avoid repeated divisions after the square root.
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

        /// <summary>Update the Cholesky factor for A + W*diag(alpha)*W^T, with L and W in column-major order.</summary>
        /// <param name="l">Lower triangular factor of the n by n matrix A. Overwritten with the updated factor.</param>
        /// <param name="ldl">Leading dimension of l, in elements.</param>
        /// <param name="w">An n by k matrix of update vectors. Overwritten.</param>
        /// <param name="ldw">Leading dimension of w, in elements.</param>
        /// <param name="alpha">One weight per column of w. Overwritten.</param>
        /// <param name="firstRows">First nonzero row of each column of w, in nondecreasing order.
        /// Entries above these rows must be zero. Pass null to use all rows.</param>
        /// <returns>Number of pivots raised to SmallestPivot.</returns>
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
        /// Required scratch space in doubles to insert r rows and columns at index p into an n by n factor.
        /// </summary>
        public static int InsertScratch(int n, int p, int r) => (n - p) * r + r;

        /// <summary>
        /// Update the Cholesky factor in place after inserting r rows and columns into A.
        /// Entries above the diagonal of L may be overwritten.
        /// </summary>
        /// <param name="n">Size of A before insertion.</param>
        /// <param name="l">Lower triangular factor of A in the first n rows and columns.
        /// Must have room for the resulting (n + r) by (n + r) factor.</param>
        /// <param name="p">Insertion index, from 0 to n. Use n to append.</param>
        /// <param name="r">Number of rows and columns to insert.</param>
        /// <param name="b">The r new columns of A, with n + r rows in the order after insertion. Overwritten.</param>
        /// <param name="scratch">Workspace of at least InsertScratch(n, p, r) doubles.</param>
        /// <returns>Number of pivots raised to SmallestPivot.</returns>
        public static int InsertRowsAndColumns(int n, MatrixView<double> l, int p, int r, MatrixView<double> b, double* scratch)
        {
            if (r == 0)
            {
                return 0;
            }

            // Make room from the bottom right to avoid overwriting entries before they move.
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

            // Solve L_00 * X = B_0, then store L_10 = X^T.
            Blas.TrsmLeftLower(l00, x);
            for (int j = 0; j < p; j++)
            {
                for (int i = 0; i < r; i++)
                {
                    l10[i, j] = x[j, i];
                }
            }

            // Factor the new diagonal block: L_11 * L_11^T = B_1 - X^T * X.
            for (int j = 0; j < r; j++)
            {
                for (int i = j; i < r; i++)
                {
                    l11[i, j] = b1[i, j];
                }
            }

            Blas.SyrkLower(-1.0, l10, l11, false);
            int raised = Factor(l11, false);

            // Compute L_21 = (B_2 - L_20 * X) * L_11^-T.
            for (int j = 0; j < r; j++)
            {
                for (int i = 0; i < below; i++)
                {
                    l21[i, j] = b2[i, j];
                }
            }

            Blas.Gemm(-1.0, l20, x, l21);
            Blas.TrsmLeftLower(l11, l21.Transpose);

            // Downdate the trailing factor for A_22 - L_21 * L_21^T.
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

        /// <summary>Insert r rows and columns at index p, with L and B in column-major order.</summary>
        /// <param name="n">Size of A before insertion.</param>
        /// <param name="l">Lower triangular factor of A, with room for the resulting (n + r) by (n + r) factor.
        /// Updated in place; entries above the diagonal may be overwritten.</param>
        /// <param name="ldl">Leading dimension of l, in elements.</param>
        /// <param name="p">Insertion index, from 0 to n. Use n to append.</param>
        /// <param name="r">Number of rows and columns to insert.</param>
        /// <param name="b">The r new columns of A, with n + r rows in the order after insertion. Overwritten.</param>
        /// <param name="ldb">Leading dimension of b, in elements.</param>
        /// <param name="scratch">Workspace of at least InsertScratch(n, p, r) doubles.</param>
        /// <returns>Number of pivots raised to SmallestPivot.</returns>
        public static int InsertRowsAndColumns(int n, double* l, int ldl, int p, int r, double* b, int ldb, double* scratch)
        {
            return InsertRowsAndColumns(n, MatrixView<double>.ColumnMajor(l, n + r, n + r, ldl), p, r,
                MatrixView<double>.ColumnMajor(b, n + r, r, ldb), scratch);
        }

        /// <summary>
        /// Required scratch space in doubles to delete r rows and columns from an n by n factor.
        /// </summary>
        /// <param name="first">Index of the first row and column to delete.</param>
        public static int DeleteScratch(int n, int first, int r) => (n - first - r) * r + r + (r + 1) / 2;

        /// <summary>
        /// Update the Cholesky factor in place after deleting r rows and columns from A.
        /// </summary>
        /// <param name="l">Lower triangular factor of the n by n matrix A.
        /// The resulting factor occupies the first n - r rows and columns.</param>
        /// <param name="indices">Indices to delete, in strictly increasing order.</param>
        /// <param name="r">Number of rows and columns to delete.</param>
        /// <param name="scratch">Workspace of at least DeleteScratch(n, indices[0], r) doubles. Unused if r is zero.</param>
        /// <returns>Number of pivots raised to SmallestPivot.</returns>
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

            // Save the deleted columns in the retained rows for the rank update.
            // Subtract c deleted rows and the first unchanged rows to get the scratch row index.
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

                // Zero the rows before this update vector starts.
                for (int i = 0; i < firstRows[k] && i < rows; i++)
                {
                    w[i, k] = 0.0;
                }
            }

            // Close the gaps from the top left to avoid overwriting entries before they move.
            // Leave the leading block before the first deleted index unchanged.
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

        /// <summary>Delete r rows and columns from A, updating its Cholesky factor in column-major order.</summary>
        /// <param name="l">Lower triangular factor of the n by n matrix A.
        /// The resulting factor occupies the first n - r rows and columns.</param>
        /// <param name="ldl">Leading dimension of l, in elements.</param>
        /// <param name="indices">Indices to delete, in strictly increasing order.</param>
        /// <param name="r">Number of rows and columns to delete.</param>
        /// <param name="scratch">Workspace of at least DeleteScratch(n, indices[0], r) doubles. Unused if r is zero.</param>
        /// <returns>Number of pivots raised to SmallestPivot.</returns>
        public static int DeleteRowsAndColumns(int n, double* l, int ldl, int* indices, int r, double* scratch)
        {
            return DeleteRowsAndColumns(MatrixView<double>.ColumnMajor(l, n, n, ldl), indices, r, scratch);
        }
    }
}
