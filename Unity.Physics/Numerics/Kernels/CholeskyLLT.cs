// Ported from faer 0.19.4 (https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md):
//   src/linalg/cholesky/llt/compute.rs
//   src/linalg/cholesky/ldlt_diagonal/compute.rs (new_cholesky, LLT variant)
//   src/linalg/cholesky/llt/solve.rs

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// Cholesky factorization, solves and updates for symmetric, positive definite matrices:
    ///   A = L*L^T
    /// where L is lower triangular.
    /// </summary>
    public static unsafe partial class CholeskyLLT
    {
        /// <summary>
        /// Minimum pivot value used to avoid division by zero when factoring nearly singular matrices.
        /// </summary>
        public const double SmallestPivot = 1e-300;

        /// <summary>
        /// Maximum number of rows for the base case (four double4 vectors).
        /// </summary>
        const int BaseCaseRows = 16;

        /// <summary>
        /// Compute in-place Cholesky factorization A = L*L^T.
        /// Pivots at or below <see cref="SmallestPivot"/> are replaced with that value.
        /// </summary>
        /// <param name="a">Square matrix to factor. The lower triangle is read and overwritten with L.</param>
        /// <param name="keepUpper">Preserve the upper triangle. Set to false to allow faster writes that may overwrite it.</param>
        /// <returns>Number of pivots raised to SmallestPivot.</returns>
        public static int Factor(MatrixView<double> a, bool keepUpper = true)
        {
            int raised = 0;
            Factor(a, false, keepUpper, ref raised);
            return raised;
        }

        /// <summary>
        /// Compute in-place Cholesky factorization A = L*L^T, stopping at the first nonpositive pivot.
        /// </summary>
        /// <param name="a">Square matrix to factor. The lower triangle is read and overwritten with L.
        /// On failure, only columns before the failed pivot contain L; the remaining columns may have been modified.</param>
        /// <param name="keepUpper">Preserve the upper triangle. Set to false to allow faster writes that may overwrite it.</param>
        /// <returns>-1 on success. Otherwise, the index of the first nonpositive pivot.</returns>
        public static int TryFactor(MatrixView<double> a, bool keepUpper = true)
        {
            int raised = 0;
            return Factor(a, true, keepUpper, ref raised);
        }

        /// <summary>Compute in-place Cholesky factorization of an n by n matrix stored in column-major order.</summary>
        /// <param name="a">Matrix to factor. The lower triangle is read and overwritten with L.</param>
        /// <param name="lda">Leading dimension of a, in elements.</param>
        /// <param name="keepUpper">Preserve the upper triangle. Set to false to allow faster writes that may overwrite it.</param>
        /// <returns>Number of pivots raised to SmallestPivot.</returns>
        public static int Factor(int n, double* a, int lda, bool keepUpper = true)
        {
            return Factor(MatrixView<double>.ColumnMajor(a, n, n, lda), keepUpper);
        }

        /// <summary>
        /// Factor recursively. If stop is true, return the first nonpositive pivot index; otherwise raise small pivots.
        /// Return -1 on success and accumulate the number of raised pivots in raised.
        /// </summary>
        static int Factor(MatrixView<double> a, bool stop, bool keepUpper, ref int raised)
        {
            int n = a.Rows;
            if (n <= BaseCaseRows)
            {
                return a.ColumnsAreContiguous
                    ? FactorBaseCase(n, a.Data, a.LeadingDimension, stop, ref raised)
                    : FactorBaseCaseByRows(a, stop, ref raised);
            }

            int bs = n / 2;
            int failed = Factor(a.Block(0, 0, bs, bs), stop, keepUpper, ref raised);
            if (failed >= 0)
            {
                return failed;
            }

            // Solve L_00 * L_10^T = A_10^T for L_10.
            MatrixView<double> a10 = a.Block(bs, 0, n - bs, bs);
            Blas.TrsmLeftLower(a.Block(0, 0, bs, bs), a10.Transpose);
            Blas.SyrkLower(-1.0, a10, a.Block(bs, bs, n - bs, n - bs), keepUpper);
            failed = Factor(a.Block(bs, bs, n - bs, n - bs), stop, keepUpper, ref raised);
            return failed >= 0 ? bs + failed : -1;
        }

        /// <summary>
        /// Factor a small matrix with contiguous columns, using up to four double4 vectors per column.
        /// </summary>
        static int FactorBaseCase(int n, double* a, int lda, bool stop, ref int raised)
        {
            for (int j = 0; j < n; j++)
            {
                double* aj = a + j + (long)j * lda;
                int rows = n - j;
                int vectors = rows >> 2;
                int rest = rows & 3;
                bool window = rest > 0 && n >= 4;
                double4 c0 = vectors > 0 ? *(double4*)aj : 0.0;
                double4 c1 = vectors > 1 ? *(double4*)(aj + 4) : 0.0;
                double4 c2 = vectors > 2 ? *(double4*)(aj + 8) : 0.0;
                double4 c3 = vectors > 3 ? *(double4*)(aj + 12) : 0.0;

                // Load the last four rows to keep the vector within the matrix. Only the remaining lanes are stored.
                // For matrices with fewer than four rows, load the remaining entries individually.
                int first = window ? n - 4 : j + 4 * vectors;
                int lane = window ? 4 - rest : 0;
                double* tail = a + first + (long)j * lda;
                double4 t = 0.0;
                if (window)
                {
                    t = *(double4*)tail;
                }
                else
                {
                    for (int i = 0; i < rest; i++)
                    {
                        t[i] = tail[i];
                    }
                }

                for (int k = 0; k < j; k++)
                {
                    double* ak = a + j + (long)k * lda;
                    double4 f = -ak[0];
                    if (vectors > 0)
                    {
                        c0 = FusedMath.Fma(f, *(double4*)ak, c0);
                    }

                    if (vectors > 1)
                    {
                        c1 = FusedMath.Fma(f, *(double4*)(ak + 4), c1);
                    }

                    if (vectors > 2)
                    {
                        c2 = FusedMath.Fma(f, *(double4*)(ak + 8), c2);
                    }

                    if (vectors > 3)
                    {
                        c3 = FusedMath.Fma(f, *(double4*)(ak + 12), c3);
                    }

                    if (window)
                    {
                        t = FusedMath.Fma(f, *(double4*)(a + first + (long)k * lda), t);
                    }
                    else if (rest > 0)
                    {
                        double* kt = ak + 4 * vectors;
                        double4 v = new double4(kt[0], rest > 1 ? kt[1] : 0.0, rest > 2 ? kt[2] : 0.0, 0.0);
                        t = FusedMath.Fma(f, v, t);
                    }
                }

                double diagonal = vectors > 0 ? c0.x : t[lane];
                if (stop && !(diagonal > 0.0))
                {
                    return j;
                }

                if (!(diagonal > SmallestPivot))
                {
                    diagonal = SmallestPivot;
                    raised++;
                }

                double root = math.sqrt(diagonal);
                double4 inverse = 1.0 / root;
                c0 *= inverse;
                c1 *= inverse;
                c2 *= inverse;
                c3 *= inverse;
                t *= inverse;
                if (vectors > 0)
                {
                    c0.x = root;
                }
                else
                {
                    t[lane] = root;
                }

                if (vectors > 0)
                {
                    *(double4*)aj = c0;
                }

                if (vectors > 1)
                {
                    *(double4*)(aj + 4) = c1;
                }

                if (vectors > 2)
                {
                    *(double4*)(aj + 8) = c2;
                }

                if (vectors > 3)
                {
                    *(double4*)(aj + 12) = c3;
                }

                for (int i = 0; i < rest; i++)
                {
                    tail[lane + i] = t[lane + i];
                }
            }

            return -1;
        }

        /// <summary>
        /// Factor a small matrix whose columns are not contiguous, using dot products of rows.
        /// </summary>
        static int FactorBaseCaseByRows(MatrixView<double> a, bool stop, ref int raised)
        {
            int n = a.Rows;
            int cs = a.ColumnStride;
            for (int j = 0; j < n; j++)
            {
                double* rowJ = a.At(j, 0);
                double diagonal = a[j, j] - Blas.Dot(j, rowJ, cs, rowJ, cs);
                if (stop && !(diagonal > 0.0))
                {
                    return j;
                }

                if (!(diagonal > SmallestPivot))
                {
                    diagonal = SmallestPivot;
                    raised++;
                }

                double root = math.sqrt(diagonal);
                double inverse = 1.0 / root;
                a[j, j] = root;
                for (int i = j + 1; i < n; i++)
                {
                    a[i, j] = (a[i, j] - Blas.Dot(j, a.At(i, 0), cs, rowJ, cs)) * inverse;
                }
            }

            return -1;
        }

        /// <summary>
        /// Solve L*L^T*x = b, overwriting b with x.
        /// </summary>
        /// <param name="l">Lower triangular Cholesky factor.</param>
        /// <param name="x">Right-hand side on entry; solution on return.</param>
        /// <param name="incx">Stride between entries of x, in elements.</param>
        public static void Solve(MatrixView<double> l, double* x, int incx)
        {
            Blas.TrsvLower(l, x, incx);
            Blas.TrsvUpper(l.Transpose, x, incx);
        }

        /// <summary>Solve L*L^T*X = B, overwriting B with X.</summary>
        /// <param name="l">Lower triangular Cholesky factor.</param>
        /// <param name="b">Right-hand sides on entry; solutions on return, one per column.</param>
        public static void Solve(MatrixView<double> l, MatrixView<double> b)
        {
            Blas.TrsmLeftLower(l, b);
            Blas.TrsmLeftUpper(l.Transpose, b);
        }

        /// <summary>Solve L*L^T*x = b, with L stored in column-major order.</summary>
        /// <param name="l">Lower triangular Cholesky factor.</param>
        /// <param name="ldl">Leading dimension of l, in elements.</param>
        /// <param name="x">Right-hand side on entry; solution on return.</param>
        public static void Solve(int n, double* l, int ldl, double* x)
        {
            Solve(MatrixView<double>.ColumnMajor(l, n, n, ldl), x, 1);
        }
    }
}
