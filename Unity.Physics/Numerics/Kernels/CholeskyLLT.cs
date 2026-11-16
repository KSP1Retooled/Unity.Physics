// Ported from faer 0.19.4 (https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md):
//   src/linalg/cholesky/llt/compute.rs          (the recursive factor)
//   src/linalg/cholesky/ldlt_diagonal/compute.rs (new_cholesky, the base case of the factor, LLT variant)
//   src/linalg/cholesky/llt/solve.rs             (the solve: a lower solve, then a solve with L^T)
//   (the updates are in CholeskyLLTUpdate.cs)
// The algorithms are faer's. The code is written for raw double pointers, with double4 in place of faer's SIMD types,
// and calls the BLAS-style routines in Blas. What was adapted is listed in each summary.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// The Cholesky factor A = L L^T of a symmetric positive definite matrix, and changes to it. L is lower triangular.
    /// The factor and the solves take a <see cref="MatrixView{T}"/> in any layout, or a pointer to a matrix that is
    /// stored column by column with a leading dimension: entry (i, j) of L is then l[i + j * ldl]. The changes to the
    /// factor take only the second form. The factor reads and writes only the lower triangle, so the entries above the
    /// diagonal keep their values.
    /// </summary>
    public static unsafe partial class CholeskyLLT
    {
        /// <summary>
        /// The smallest value that a pivot can take. faer stops with an error at a pivot that is not positive. The
        /// physics solver's matrices are positive definite in theory, and a pivot that rounding makes zero or negative
        /// is raised to this value instead, as the solver's own factor did before this port.
        /// </summary>
        public const double SmallestPivot = 1e-300;

        /// <summary>
        /// The largest matrix that the base case factors. faer's base case takes a matrix whose columns fit in four
        /// SIMD vectors, which is 16 rows of doubles with 256-bit vectors.
        /// </summary>
        const int BaseCaseRows = 16;

        /// <summary>
        /// Factors the n by n matrix in the lower triangle of <paramref name="a"/> in place, as faer's LLT factor does:
        /// a matrix of up to 16 rows goes to the base case. A larger one is split in half: the routine factors the
        /// upper left block, solves for the block below it, takes that block's product with itself out of the lower
        /// right block, and factors the lower right block. A pivot that is not larger than
        /// <see cref="SmallestPivot"/> is raised to it, and the routine goes on. Returns how many pivots were raised.
        /// </summary>
        public static int Factor(MatrixView<double> a)
        {
            int raised = 0;
            Factor(a, false, ref raised);
            return raised;
        }

        /// <summary>
        /// Factors the matrix as <see cref="Factor(MatrixView{double})"/> does, but stops at the first pivot that is
        /// not positive, as faer does, and returns its index. Returns -1 if every pivot is positive. When the routine
        /// stops, the columns before that index hold their part of the factor and the rest of the matrix is changed.
        /// </summary>
        public static int TryFactor(MatrixView<double> a)
        {
            int raised = 0;
            return Factor(a, true, ref raised);
        }

        /// <summary>The factor of a matrix that is stored column by column with leading dimension <paramref name="lda"/>.</summary>
        public static int Factor(int n, double* a, int lda)
        {
            return Factor(MatrixView<double>.ColumnMajor(a, n, n, lda));
        }

        /// <summary>
        /// The recursion of <see cref="Factor(MatrixView{double})"/>. If <paramref name="stop"/> is true, it returns
        /// the index of the first pivot that is not positive, or -1. Otherwise it raises small pivots, counts them in
        /// <paramref name="raised"/>, and returns -1.
        /// </summary>
        static int Factor(MatrixView<double> a, bool stop, ref int raised)
        {
            int n = a.Rows;
            if (n <= BaseCaseRows)
            {
                return a.ColumnsAreContiguous
                    ? FactorBaseCase(n, a.Data, a.LeadingDimension, stop, ref raised)
                    : FactorBaseCaseByRows(a, stop, ref raised);
            }

            int bs = n / 2;
            int failed = Factor(a.Block(0, 0, bs, bs), stop, ref raised);
            if (failed >= 0)
            {
                return failed;
            }

            // A_10 := A_10 L_00^-T, which is the lower solve on A_10^T.
            MatrixView<double> a10 = a.Block(bs, 0, n - bs, bs);
            Blas.TrsmLeftLower(a.Block(0, 0, bs, bs), a10.Transpose);
            Blas.SyrkLower(-1.0, a10, a.Block(bs, bs, n - bs, n - bs));
            failed = Factor(a.Block(bs, bs, n - bs, n - bs), stop, ref raised);
            return failed >= 0 ? bs + failed : -1;
        }

        /// <summary>
        /// faer's base case (new_cholesky, LLT variant), for a matrix whose columns are contiguous. It works from the
        /// left, one column at a time. The column's entries from the diagonal down stay in registers while every
        /// earlier column k takes its share out of them, with a multiplier of -L[j, k]. Then the diagonal entry becomes
        /// its square root, and the entries below it are multiplied by the root's reciprocal. faer copies the matrix
        /// into a buffer that is aligned to the vectors and keeps the windows aligned; this port works in place with
        /// unaligned loads, and the entries past the last whole vector are done one at a time.
        /// </summary>
        static int FactorBaseCase(int n, double* a, int lda, bool stop, ref int raised)
        {
            for (int j = 0; j < n; j++)
            {
                double* aj = a + j + (long)j * lda;
                int rows = n - j;
                int vectors = rows >> 2;
                int rest = rows & 3;
                double4 c0 = vectors > 0 ? *(double4*)aj : 0.0;
                double4 c1 = vectors > 1 ? *(double4*)(aj + 4) : 0.0;
                double4 c2 = vectors > 2 ? *(double4*)(aj + 8) : 0.0;
                double4 c3 = vectors > 3 ? *(double4*)(aj + 12) : 0.0;
                double* tail = aj + 4 * vectors;
                double4 t = 0.0;
                for (int i = 0; i < rest; i++)
                {
                    t[i] = tail[i];
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

                    if (rest > 0)
                    {
                        double* kt = ak + 4 * vectors;
                        double4 v = new double4(kt[0], rest > 1 ? kt[1] : 0.0, rest > 2 ? kt[2] : 0.0, 0.0);
                        t = FusedMath.Fma(f, v, t);
                    }
                }

                double diagonal = vectors > 0 ? c0.x : t.x;
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
                    t.x = root;
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
                    tail[i] = t[i];
                }
            }

            return -1;
        }

        /// <summary>
        /// The base case for a matrix whose columns are not contiguous. It works down the rows, one column at a time:
        /// entry (i, j) takes out the dot product of rows i and j to the left of column j. A matrix that is stored row
        /// by row has contiguous rows, so these dot products run in vectors; any other layout runs them in a plain loop.
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
        /// Solves L L^T x = b in place, where x has n entries that are <paramref name="incx"/> apart: x := L^-1 x, then
        /// x := L^-T x.
        /// </summary>
        public static void Solve(MatrixView<double> l, double* x, int incx)
        {
            Blas.TrsvLower(l, x, incx);
            Blas.TrsvUpper(l.Transpose, x, incx);
        }

        /// <summary>Solves L L^T X = B in place for every column of B: B := L^-1 B, then B := L^-T B.</summary>
        public static void Solve(MatrixView<double> l, MatrixView<double> b)
        {
            Blas.TrsmLeftLower(l, b);
            Blas.TrsmLeftUpper(l.Transpose, b);
        }

        /// <summary>Solves L L^T x = b in place, for L stored column by column with leading dimension <paramref name="ldl"/>.</summary>
        public static void Solve(int n, double* l, int ldl, double* x)
        {
            Solve(MatrixView<double>.ColumnMajor(l, n, n, ldl), x, 1);
        }
    }
}
