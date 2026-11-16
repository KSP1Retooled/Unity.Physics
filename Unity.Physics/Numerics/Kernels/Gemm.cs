// Ported from gemm-f64 0.18.0 src/microkernel.rs (https://github.com/sarah-ek/gemm, MIT license): the FMA f64 micro-kernel
//   shape, 8 rows in two vectors of four by 6 columns; faer 0.19.4 src/linalg/matmul/triangular.rs for the
//   lower-triangle product.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers in column-major order, with double4 in place of the SIMD types.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>C += alpha A B, where A is m by k, B is k by n and C is m by n.</summary>
        public static void GemmNN(int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double* c, int ldc)
        {
            Gemm(m, n, k, alpha, a, lda, b, 1, ldb, c, ldc, false);
        }

        /// <summary>C += alpha A B^T, where A is m by k, B is n by k and C is m by n.</summary>
        public static void GemmNT(int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double* c, int ldc)
        {
            Gemm(m, n, k, alpha, a, lda, b, ldb, 1, c, ldc, false);
        }

        /// <summary>
        /// The lower triangle of C += alpha A A^T, where A is n by k and C is n by n. The routine also writes some
        /// entries above the diagonal of C, in the tiles that the diagonal crosses, so the caller must not keep
        /// anything there. faer's triangular matrix product computes a tile that the diagonal crosses into a buffer
        /// and copies its lower part. A factor stored as a lower triangle keeps nothing above its diagonal, so this
        /// port writes the whole tile instead.
        /// </summary>
        public static void SyrkLowerNT(int n, int k, double alpha, double* a, int lda, double* c, int ldc)
        {
            Gemm(n, n, k, alpha, a, lda, a, lda, 1, c, ldc, true);
        }

        /// <summary>
        /// C += alpha op(A) op(B), where entry (p, j) of op(B) is b[p * bRow + j * bColumn]. If
        /// <paramref name="lower"/> is true, only the tiles that hold entries on or below the diagonal of C are done.
        /// <para>
        /// This follows the shape of the gemm crate's FMA micro-kernel for doubles: a tile of C of 8 rows by 6
        /// columns stays in twelve vector registers while the loop runs over the inner dimension. Each step loads two
        /// vectors of four rows of A and multiplies them by six single entries of op(B). The edges use the same
        /// kernel with fewer rows or columns. The gemm crate packs A and B into buffers first; this port reads them
        /// in place, because the solver's matrices are small enough to stay in the caches.
        /// </para>
        /// </summary>
        static void Gemm(int m, int n, int k, double alpha, double* a, int lda, double* b, int bRow, int bColumn,
            double* c, int ldc, bool lower)
        {
            if (m <= 0 || n <= 0 || k <= 0)
            {
                return;
            }

            for (int j = 0; j < n; j += 6)
            {
                int columns = math.min(6, n - j);
                int firstRow = lower ? j & ~7 : 0;
                for (int i = firstRow; i < m; i += 8)
                {
                    int rows = math.min(8, m - i);
                    double* ai = a + i;
                    double* bj = b + (long)j * bColumn;
                    double* cij = c + i + (long)j * ldc;
                    if (rows == 8 && columns == 6)
                    {
                        Kernel8x6(k, alpha, ai, lda, bj, bRow, bColumn, cij, ldc);
                    }
                    else
                    {
                        KernelEdge(rows, columns, k, alpha, ai, lda, bj, bRow, bColumn, cij, ldc);
                    }
                }
            }
        }

        /// <summary>The full tile: 8 rows of C by 6 columns, in twelve vector sums.</summary>
        static void Kernel8x6(int k, double alpha, double* a, int lda, double* b, int bRow, int bColumn, double* c, int ldc)
        {
            double4 c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0, c04 = 0.0, c05 = 0.0;
            double4 c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0, c14 = 0.0, c15 = 0.0;
            for (int p = 0; p < k; p++)
            {
                double* ap = a + (long)p * lda;
                double4 a0 = *(double4*)ap;
                double4 a1 = *(double4*)(ap + 4);
                double* bp = b + (long)p * bRow;
                double4 b0 = bp[0];
                double4 b1 = bp[bColumn];
                double4 b2 = bp[2 * bColumn];
                double4 b3 = bp[3 * bColumn];
                double4 b4 = bp[4 * bColumn];
                double4 b5 = bp[5 * bColumn];
                c00 = FusedMath.Fma(a0, b0, c00);
                c10 = FusedMath.Fma(a1, b0, c10);
                c01 = FusedMath.Fma(a0, b1, c01);
                c11 = FusedMath.Fma(a1, b1, c11);
                c02 = FusedMath.Fma(a0, b2, c02);
                c12 = FusedMath.Fma(a1, b2, c12);
                c03 = FusedMath.Fma(a0, b3, c03);
                c13 = FusedMath.Fma(a1, b3, c13);
                c04 = FusedMath.Fma(a0, b4, c04);
                c14 = FusedMath.Fma(a1, b4, c14);
                c05 = FusedMath.Fma(a0, b5, c05);
                c15 = FusedMath.Fma(a1, b5, c15);
            }

            double4 alpha4 = alpha;
            Store(c, alpha4, c00, c10);
            Store(c + ldc, alpha4, c01, c11);
            Store(c + 2 * (long)ldc, alpha4, c02, c12);
            Store(c + 3 * (long)ldc, alpha4, c03, c13);
            Store(c + 4 * (long)ldc, alpha4, c04, c14);
            Store(c + 5 * (long)ldc, alpha4, c05, c15);
        }

        static void Store(double* c, double4 alpha, double4 top, double4 bottom)
        {
            *(double4*)c = FusedMath.Fma(alpha, top, *(double4*)c);
            *(double4*)(c + 4) = FusedMath.Fma(alpha, bottom, *(double4*)(c + 4));
        }

        /// <summary>
        /// A tile at an edge of C, with up to 8 rows and up to 6 columns. Each column of the tile is one pass over the
        /// inner dimension with two vector sums, and the rows past the last full vector are summed one at a time.
        /// </summary>
        static void KernelEdge(int rows, int columns, int k, double alpha, double* a, int lda, double* b, int bRow, int bColumn,
            double* c, int ldc)
        {
            int vectors = rows >> 2;
            int rest = rows & 3;
            for (int j = 0; j < columns; j++)
            {
                double4 s0 = 0.0, s1 = 0.0, r = 0.0;
                double* bj = b + (long)j * bColumn;
                for (int p = 0; p < k; p++)
                {
                    double* ap = a + (long)p * lda;
                    double4 bp = bj[(long)p * bRow];
                    if (vectors > 0)
                    {
                        s0 = FusedMath.Fma(*(double4*)ap, bp, s0);
                    }

                    if (vectors > 1)
                    {
                        s1 = FusedMath.Fma(*(double4*)(ap + 4), bp, s1);
                    }

                    if (rest > 0)
                    {
                        double* tail = ap + 4 * vectors;
                        double4 t = new double4(tail[0], rest > 1 ? tail[1] : 0.0, rest > 2 ? tail[2] : 0.0, 0.0);
                        r = FusedMath.Fma(t, bp, r);
                    }
                }

                double* cj = c + (long)j * ldc;
                if (vectors > 0)
                {
                    *(double4*)cj = FusedMath.Fma(alpha, s0, *(double4*)cj);
                }

                if (vectors > 1)
                {
                    *(double4*)(cj + 4) = FusedMath.Fma(alpha, s1, *(double4*)(cj + 4));
                }

                double* ct = cj + 4 * vectors;
                for (int i = 0; i < rest; i++)
                {
                    ct[i] = FusedMath.Fma(alpha, r[i], ct[i]);
                }
            }
        }
    }
}
