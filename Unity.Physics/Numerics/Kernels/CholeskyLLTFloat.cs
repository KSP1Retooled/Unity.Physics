// Ported from faer 0.19.4 (https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md):
//   src/linalg/cholesky/llt/compute.rs
//   src/linalg/cholesky/ldlt_diagonal/compute.rs (new_cholesky, LLT variant)
//   src/linalg/cholesky/llt/solve.rs

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class CholeskyLLT
    {
        /// <summary>Minimum float pivot value used to avoid division by zero when factoring nearly singular matrices.</summary>
        public const float SmallestPivotFloat = 1e-30f;

        /// <summary>Maximum number of rows for the float base case (four float8 vectors).</summary>
        const int FloatBaseCaseRows = 32;

        /// <summary>
        /// Compute in-place Cholesky factorization A = L*L^T.
        /// Pivots at or below <see cref="SmallestPivotFloat"/> are replaced with that value.
        /// </summary>
        /// <param name="a">Square matrix to factor. The lower triangle is read and overwritten with L.</param>
        /// <param name="keepUpper">Preserve the upper triangle. Set to false to allow faster writes that may overwrite it.</param>
        /// <returns>Number of pivots raised to SmallestPivotFloat.</returns>
        public static int Factor(MatrixView<float> a, bool keepUpper = true)
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
        public static int TryFactor(MatrixView<float> a, bool keepUpper = true)
        {
            int raised = 0;
            return Factor(a, true, keepUpper, ref raised);
        }

        static int Factor(MatrixView<float> a, bool stop, bool keepUpper, ref int raised)
        {
            int n = a.Rows;
            if (n <= FloatBaseCaseRows)
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

            MatrixView<float> a10 = a.Block(bs, 0, n - bs, bs);
            Blas.TrsmLeftLower(a.Block(0, 0, bs, bs), a10.Transpose);
            Blas.SyrkLower(-1f, a10, a.Block(bs, bs, n - bs, n - bs), keepUpper);
            failed = Factor(a.Block(bs, bs, n - bs, n - bs), stop, keepUpper, ref raised);
            return failed >= 0 ? bs + failed : -1;
        }

        /// <summary>
        /// Factor a small matrix with contiguous columns, using up to four float8 vectors per column.
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
                // Load the last eight rows to keep the vector within the matrix. Only the remaining lanes are stored.
                // For matrices with fewer than eight rows, load the remaining entries individually.
                bool window = rest > 0 && n >= 8;
                int first = window ? n - 8 : j + 8 * vectors;
                int lane = window ? 8 - rest : 0;
                float* tail = a + first + (long)j * lda;
                float8 t = 0f;
                float* tp = (float*)&t;
                if (window)
                {
                    t = *(float8*)tail;
                }
                else
                {
                    for (int i = 0; i < rest; i++)
                    {
                        tp[i] = tail[i];
                    }
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

                    if (window)
                    {
                        t = FusedMath.Fma(f, *(float8*)(a + first + (long)k * lda), t);
                    }
                    else if (rest > 0)
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

                float diagonal = vectors > 0 ? c0.Low.x : tp[lane];
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
                    tp[lane] = root;
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
                    tail[lane + i] = tp[lane + i];
                }
            }

            return -1;
        }

        /// <summary>
        /// Factor a small matrix whose columns are not contiguous, using dot products of rows.
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

        /// <summary>Solve L*L^T*x = b, overwriting b with x.</summary>
        /// <param name="l">Lower triangular Cholesky factor.</param>
        /// <param name="x">Right-hand side on entry; solution on return.</param>
        /// <param name="incx">Stride between entries of x, in elements.</param>
        public static void Solve(MatrixView<float> l, float* x, int incx)
        {
            Blas.TrsvLower(l, x, incx);
            Blas.TrsvUpper(l.Transpose, x, incx);
        }

        /// <summary>Solve L*L^T*X = B, overwriting B with X.</summary>
        /// <param name="l">Lower triangular Cholesky factor.</param>
        /// <param name="b">Right-hand sides on entry; solutions on return, one per column.</param>
        public static void Solve(MatrixView<float> l, MatrixView<float> b)
        {
            Blas.TrsmLeftLower(l, b);
            Blas.TrsmLeftUpper(l.Transpose, b);
        }
    }
}
