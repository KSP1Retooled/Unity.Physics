// Ported from faer 0.19.4 (https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md):
//   src/linalg/cholesky/llt/compute.rs          (the recursive factor)
//   src/linalg/cholesky/ldlt_diagonal/compute.rs (new_cholesky, the base case of the factor, LLT variant)
//   src/linalg/cholesky/llt/solve.rs             (the solve: a lower solve, then a solve with L^T)
//   (the updates are in CholeskyLLTUpdate.cs)
// The algorithms are faer's. The code is written for raw double pointers in column-major order, with double4 in place
// of faer's SIMD types, and calls the BLAS-style routines in Blas. What was adapted is listed in each summary.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// The Cholesky factor A = L L^T of a symmetric positive definite matrix, and changes to it. L is lower triangular
    /// and is stored in column-major order with a leading dimension: entry (i, j) of L is l[i + j * ldl]. The routines
    /// read and write only the lower triangle, except that the matrix products may also write entries above the
    /// diagonal, so the caller must not keep anything there.
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
        /// right block, and factors the lower right block. Returns how many pivots were raised to
        /// <see cref="SmallestPivot"/>.
        /// </summary>
        public static int Factor(int n, double* a, int lda)
        {
            if (n <= BaseCaseRows)
            {
                return FactorBaseCase(n, a, lda);
            }

            int bs = n / 2;
            int raised = Factor(bs, a, lda);
            double* a10 = a + bs;
            double* a11 = a + bs + (long)bs * lda;
            Blas.TrsmRightLowerTranspose(n - bs, bs, a, lda, a10, lda);
            Blas.SyrkLowerNT(n - bs, bs, -1.0, a10, lda, a11, lda);
            return raised + Factor(n - bs, a11, lda);
        }

        /// <summary>
        /// faer's base case (new_cholesky, LLT variant). It works from the left, one column at a time. The column's
        /// entries from the diagonal down stay in registers while every earlier column k takes its share out of them,
        /// with a multiplier of -L[j, k]. Then the diagonal entry becomes its square root, and the entries below it
        /// are multiplied by the root's reciprocal. faer copies the matrix into a buffer that is aligned to the
        /// vectors and keeps the windows aligned; this port works in place with unaligned loads, and the entries
        /// past the last whole vector are done one at a time.
        /// </summary>
        static int FactorBaseCase(int n, double* a, int lda)
        {
            int raised = 0;
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

            return raised;
        }

        /// <summary>Solves L L^T x = b in place: x := L^-1 x, then x := L^-T x.</summary>
        public static void Solve(int n, double* l, int ldl, double* x)
        {
            Blas.TrsvLower(n, l, ldl, x);
            Blas.TrsvLowerTranspose(n, l, ldl, x);
        }
    }
}
