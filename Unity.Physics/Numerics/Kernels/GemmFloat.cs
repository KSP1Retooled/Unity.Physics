// Ported from gemm-f32 0.18.0 src/microkernel.rs (https://github.com/sarah-ek/gemm, MIT license): the FMA f32 micro-kernel
//   shape, 16 rows in two vectors of eight by 6 columns.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw float pointers in column-major order, with float8 in place of the SIMD types. It is the
// float version of Gemm.cs, written on its own for the wider vectors: a 256-bit register holds 8 floats against 4
// doubles, so the tile has twice the rows.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>C += alpha A B, where A is m by k, B is k by n and C is m by n, all of floats.</summary>
        public static void GemmNN(int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float* c, int ldc)
        {
            Gemm(m, n, k, alpha, a, lda, b, 1, ldb, c, ldc);
        }

        /// <summary>C += alpha A B^T, where A is m by k, B is n by k and C is m by n, all of floats.</summary>
        public static void GemmNT(int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float* c, int ldc)
        {
            Gemm(m, n, k, alpha, a, lda, b, ldb, 1, c, ldc);
        }

        /// <summary>
        /// C += alpha A op(B), where entry (p, j) of op(B) is b[p * bRow + j * bColumn]. A tile of C of 16 rows by 6
        /// columns stays in twelve vector registers of eight floats while the loop runs over the inner dimension. The
        /// edges use one column at a time with vectors of eight and four floats, and the last rows one at a time.
        /// </summary>
        static void Gemm(int m, int n, int k, float alpha, float* a, int lda, float* b, int bRow, int bColumn, float* c, int ldc)
        {
            if (m <= 0 || n <= 0 || k <= 0)
            {
                return;
            }

            for (int j = 0; j < n; j += 6)
            {
                int columns = math.min(6, n - j);
                for (int i = 0; i < m; i += 16)
                {
                    int rows = math.min(16, m - i);
                    float* ai = a + i;
                    float* bj = b + (long)j * bColumn;
                    float* cij = c + i + (long)j * ldc;
                    if (rows == 16 && columns == 6)
                    {
                        Kernel16x6(k, alpha, ai, lda, bj, bRow, bColumn, cij, ldc);
                    }
                    else
                    {
                        KernelEdge(rows, columns, k, alpha, ai, lda, bj, bRow, bColumn, cij, ldc);
                    }
                }
            }
        }

        static void Kernel16x6(int k, float alpha, float* a, int lda, float* b, int bRow, int bColumn, float* c, int ldc)
        {
            float8 c00 = 0f, c01 = 0f, c02 = 0f, c03 = 0f, c04 = 0f, c05 = 0f;
            float8 c10 = 0f, c11 = 0f, c12 = 0f, c13 = 0f, c14 = 0f, c15 = 0f;
            for (int p = 0; p < k; p++)
            {
                float* ap = a + (long)p * lda;
                float8 a0 = *(float8*)ap;
                float8 a1 = *(float8*)(ap + 8);
                float* bp = b + (long)p * bRow;
                float8 b0 = bp[0];
                float8 b1 = bp[bColumn];
                float8 b2 = bp[2 * bColumn];
                float8 b3 = bp[3 * bColumn];
                float8 b4 = bp[4 * bColumn];
                float8 b5 = bp[5 * bColumn];
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

            float8 alpha8 = alpha;
            Store(c, alpha8, c00, c10);
            Store(c + ldc, alpha8, c01, c11);
            Store(c + 2 * (long)ldc, alpha8, c02, c12);
            Store(c + 3 * (long)ldc, alpha8, c03, c13);
            Store(c + 4 * (long)ldc, alpha8, c04, c14);
            Store(c + 5 * (long)ldc, alpha8, c05, c15);
        }

        static void Store(float* c, float8 alpha, float8 top, float8 bottom)
        {
            *(float8*)c = FusedMath.Fma(alpha, top, *(float8*)c);
            *(float8*)(c + 8) = FusedMath.Fma(alpha, bottom, *(float8*)(c + 8));
        }

        /// <summary>
        /// A tile at an edge of C, with up to 16 rows and up to 6 columns. Each column of the tile is one pass over the
        /// inner dimension: a vector of eight rows while eight remain, then one of four, then single rows.
        /// </summary>
        static void KernelEdge(int rows, int columns, int k, float alpha, float* a, int lda, float* b, int bRow, int bColumn,
            float* c, int ldc)
        {
            int eights = rows >> 3;
            bool four = (rows & 4) != 0;
            int start = eights * 8 + (four ? 4 : 0);
            int rest = rows - start;
            for (int j = 0; j < columns; j++)
            {
                float8 s0 = 0f, s1 = 0f;
                float4 s4 = 0f, r = 0f;
                float* bj = b + (long)j * bColumn;
                for (int p = 0; p < k; p++)
                {
                    float* ap = a + (long)p * lda;
                    float bp = bj[(long)p * bRow];
                    if (eights > 0)
                    {
                        s0 = FusedMath.Fma(*(float8*)ap, bp, s0);
                    }

                    if (eights > 1)
                    {
                        s1 = FusedMath.Fma(*(float8*)(ap + 8), bp, s1);
                    }

                    if (four)
                    {
                        s4 = FusedMath.Fma(*(float4*)(ap + eights * 8), bp, s4);
                    }

                    if (rest > 0)
                    {
                        float* tail = ap + start;
                        float4 t = new float4(tail[0], rest > 1 ? tail[1] : 0f, rest > 2 ? tail[2] : 0f, 0f);
                        r = FusedMath.Fma(t, bp, r);
                    }
                }

                float* cj = c + (long)j * ldc;
                if (eights > 0)
                {
                    *(float8*)cj = FusedMath.Fma(alpha, s0, *(float8*)cj);
                }

                if (eights > 1)
                {
                    *(float8*)(cj + 8) = FusedMath.Fma(alpha, s1, *(float8*)(cj + 8));
                }

                if (four)
                {
                    *(float4*)(cj + eights * 8) = FusedMath.Fma(alpha, s4, *(float4*)(cj + eights * 8));
                }

                for (int i = 0; i < rest; i++)
                {
                    cj[start + i] = FusedMath.Fma(alpha, r[i], cj[start + i]);
                }
            }
        }
    }
}
