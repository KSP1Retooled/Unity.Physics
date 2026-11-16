// Ported from faer 0.19.4 (https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md):
//   src/linalg/cholesky/llt/compute.rs          (the recursive factor)
//   src/linalg/cholesky/ldlt_diagonal/compute.rs (new_cholesky, the base case of the factor, LLT variant)
//   src/linalg/cholesky/llt/solve.rs             (the solve: a lower solve, then a solve with L^T)
// The float version, written for vectors of eight floats (float8). faer's base case takes the columns that fit in four
// SIMD vectors, which is 32 rows of floats, so the float base case is twice as tall as the double one.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class CholeskyLLT
    {
        /// <summary>The smallest value that a float pivot can take when <see cref="Factor(MatrixView{float})"/> raises it.</summary>
        public const float SmallestPivotFloat = 1e-30f;

        /// <summary>The largest matrix of floats that the base case factors: four vectors of eight rows.</summary>
        const int FloatBaseCaseRows = 32;

        /// <summary>
        /// Factors the n by n matrix of floats in the lower triangle of <paramref name="a"/> in place, with the
        /// recursion of the double version. A pivot that is not larger than <see cref="SmallestPivotFloat"/> is raised
        /// to it. Returns how many pivots were raised.
        /// </summary>
        public static int Factor(MatrixView<float> a)
        {
            int raised = 0;
            Factor(a, false, ref raised);
            return raised;
        }

        /// <summary>
        /// Factors the matrix of floats, but stops at the first pivot that is not positive and returns its index, as
        /// faer does. Returns -1 if every pivot is positive.
        /// </summary>
        public static int TryFactor(MatrixView<float> a)
        {
            int raised = 0;
            return Factor(a, true, ref raised);
        }

        static int Factor(MatrixView<float> a, bool stop, ref int raised)
        {
            int n = a.Rows;
            if (n <= FloatBaseCaseRows)
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

            MatrixView<float> a10 = a.Block(bs, 0, n - bs, bs);
            Blas.TrsmLeftLower(a.Block(0, 0, bs, bs), a10.Transpose);
            Blas.SyrkLower(-1f, a10, a.Block(bs, bs, n - bs, n - bs));
            failed = Factor(a.Block(bs, bs, n - bs, n - bs), stop, ref raised);
            return failed >= 0 ? bs + failed : -1;
        }

        /// <summary>
        /// faer's base case for floats whose columns are contiguous. Column j's entries from the diagonal down stay in
        /// up to four vectors of eight floats, and the entries past the last whole vector in one more, while every
        /// earlier column takes its share out of them.
        /// </summary>
        static int FactorBaseCase(int n, float* a, int lda, bool stop, ref int raised)
        {
            for (int j = 0; j < n; j++)
            {
                float* aj = a + j + (long)j * lda;
                int rows = n - j;
                int vectors = rows >> 3;
                int rest = rows & 7;
                float8 c0 = vectors > 0 ? *(float8*)aj : 0f;
                float8 c1 = vectors > 1 ? *(float8*)(aj + 8) : 0f;
                float8 c2 = vectors > 2 ? *(float8*)(aj + 16) : 0f;
                float8 c3 = vectors > 3 ? *(float8*)(aj + 24) : 0f;
                float* tail = aj + 8 * vectors;
                float8 t = 0f;
                float* tp = (float*)&t;
                for (int i = 0; i < rest; i++)
                {
                    tp[i] = tail[i];
                }

                for (int k = 0; k < j; k++)
                {
                    float* ak = a + j + (long)k * lda;
                    float8 f = -ak[0];
                    if (vectors > 0)
                    {
                        c0 = FusedMath.Fma(f, *(float8*)ak, c0);
                    }

                    if (vectors > 1)
                    {
                        c1 = FusedMath.Fma(f, *(float8*)(ak + 8), c1);
                    }

                    if (vectors > 2)
                    {
                        c2 = FusedMath.Fma(f, *(float8*)(ak + 16), c2);
                    }

                    if (vectors > 3)
                    {
                        c3 = FusedMath.Fma(f, *(float8*)(ak + 24), c3);
                    }

                    if (rest > 0)
                    {
                        float* kt = ak + 8 * vectors;
                        float8 v = 0f;
                        float* vp = (float*)&v;
                        for (int i = 0; i < rest; i++)
                        {
                            vp[i] = kt[i];
                        }

                        t = FusedMath.Fma(f, v, t);
                    }
                }

                float diagonal = vectors > 0 ? c0.Low.x : t.Low.x;
                if (stop && !(diagonal > 0f))
                {
                    return j;
                }

                if (!(diagonal > SmallestPivotFloat))
                {
                    diagonal = SmallestPivotFloat;
                    raised++;
                }

                float root = math.sqrt(diagonal);
                float8 inverse = 1f / root;
                c0 = c0 * inverse;
                c1 = c1 * inverse;
                c2 = c2 * inverse;
                c3 = c3 * inverse;
                t = t * inverse;
                if (vectors > 0)
                {
                    c0.Low.x = root;
                    *(float8*)aj = c0;
                }
                else
                {
                    tp[0] = root;
                }

                if (vectors > 1)
                {
                    *(float8*)(aj + 8) = c1;
                }

                if (vectors > 2)
                {
                    *(float8*)(aj + 16) = c2;
                }

                if (vectors > 3)
                {
                    *(float8*)(aj + 24) = c3;
                }

                for (int i = 0; i < rest; i++)
                {
                    tail[i] = tp[i];
                }
            }

            return -1;
        }

        /// <summary>
        /// The float base case for a matrix whose columns are not contiguous: entry (i, j) takes out the dot product of
        /// rows i and j to the left of column j, which runs in vectors if the rows are contiguous.
        /// </summary>
        static int FactorBaseCaseByRows(MatrixView<float> a, bool stop, ref int raised)
        {
            int n = a.Rows;
            int cs = a.ColumnStride;
            for (int j = 0; j < n; j++)
            {
                float* rowJ = a.At(j, 0);
                float diagonal = a[j, j] - Blas.Dot(j, rowJ, cs, rowJ, cs);
                if (stop && !(diagonal > 0f))
                {
                    return j;
                }

                if (!(diagonal > SmallestPivotFloat))
                {
                    diagonal = SmallestPivotFloat;
                    raised++;
                }

                float root = math.sqrt(diagonal);
                float inverse = 1f / root;
                a[j, j] = root;
                for (int i = j + 1; i < n; i++)
                {
                    a[i, j] = (a[i, j] - Blas.Dot(j, a.At(i, 0), cs, rowJ, cs)) * inverse;
                }
            }

            return -1;
        }

        /// <summary>Solves L L^T x = b in place for floats, where x has entries <paramref name="incx"/> apart.</summary>
        public static void Solve(MatrixView<float> l, float* x, int incx)
        {
            Blas.TrsvLower(l, x, incx);
            Blas.TrsvUpper(l.Transpose, x, incx);
        }

        /// <summary>Solves L L^T X = B in place for floats, for every column of B.</summary>
        public static void Solve(MatrixView<float> l, MatrixView<float> b)
        {
            Blas.TrsmLeftLower(l, b);
            Blas.TrsmLeftUpper(l.Transpose, b);
        }
    }
}
