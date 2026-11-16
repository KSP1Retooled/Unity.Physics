// Ported from gemm-f32 0.18.0 src/microkernel.rs (https://github.com/sarah-ek/gemm, MIT license): the FMA f32 micro-kernel
//   shape, 16 rows in two vectors of eight by 6 columns.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw float pointers, with float8 in place of the SIMD types. It is the
// float version of Gemm.cs, written on its own for the wider vectors: a 256-bit register holds 8 floats against 4
// doubles, so the tile has twice the rows.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>C += alpha A B, where A is m by k, B is k by n and C is m by n, all of floats stored column by column.</summary>
        public static void GemmNN(int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float* c, int ldc)
        {
            Gemm(alpha, MatrixView<float>.ColumnMajor(a, m, k, lda), MatrixView<float>.ColumnMajor(b, k, n, ldb),
                MatrixView<float>.ColumnMajor(c, m, n, ldc));
        }

        /// <summary>C += alpha A B^T, where A is m by k, B is n by k and C is m by n, all of floats stored column by column.</summary>
        public static void GemmNT(int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float* c, int ldc)
        {
            Gemm(alpha, MatrixView<float>.ColumnMajor(a, m, k, lda), MatrixView<float>.ColumnMajor(b, n, k, ldb).Transpose,
                MatrixView<float>.ColumnMajor(c, m, n, ldc));
        }

        /// <summary>C += alpha A B for floats, where A is m by k, B is k by n and C is m by n, in any layout.</summary>
        public static void Gemm(float alpha, MatrixView<float> a, MatrixView<float> b, MatrixView<float> c)
        {
            Gemm(alpha, a, b, c, Triangle.All);
        }

        /// <summary>
        /// The lower triangle of C += alpha A A^T for floats, where A is n by k and C is n by n. As for doubles, the
        /// routine also writes the entries above the diagonal in the tiles that the diagonal crosses.
        /// </summary>
        public static void SyrkLower(float alpha, MatrixView<float> a, MatrixView<float> c)
        {
            Gemm(alpha, a, a.Transpose, c, Triangle.Lower);
        }

        /// <summary>
        /// C += alpha A B for any layout, with the same dispatch as for doubles: a C whose rows are contiguous is
        /// computed as its transpose, an A whose columns are not contiguous is copied into panels of 16 rows, and a C
        /// with neither rows nor columns contiguous takes a plain loop.
        /// </summary>
        static void Gemm(float alpha, MatrixView<float> a, MatrixView<float> b, MatrixView<float> c, Triangle triangle)
        {
            int m = c.Rows;
            int n = c.Columns;
            int k = a.Columns;
            if (m <= 0 || n <= 0 || k <= 0)
            {
                return;
            }

            if (!c.ColumnsAreContiguous)
            {
                if (c.RowsAreContiguous)
                {
                    Gemm(alpha, b.Transpose, a.Transpose, c.Transpose, Flip(triangle));
                }
                else
                {
                    GemmPlain(alpha, a, b, c, triangle);
                }

                return;
            }

            int ldc = c.LeadingDimension;
            if (a.ColumnsAreContiguous)
            {
                GemmTiles(m, n, k, alpha, a.Data, a.LeadingDimension, b.Data, b.RowStride, b.ColumnStride, c.Data, ldc, triangle);
            }
            else
            {
                GemmPacked(alpha, a, b, c.Data, ldc, triangle);
            }
        }

        /// <summary>
        /// Runs the tiles of C over A as it is stored. A tile of C of 16 rows by 6 columns stays in twelve vector
        /// registers of eight floats while the loop runs over the inner dimension. The edges use one column at a time
        /// with vectors of eight and four floats, and the last rows one at a time.
        /// </summary>
        static void GemmTiles(int m, int n, int k, float alpha, float* a, int lda, float* b, int bRow, int bColumn,
            float* c, int ldc, Triangle triangle)
        {
            for (int j = 0; j < n; j += 6)
            {
                int columns = math.min(6, n - j);
                int firstRow = triangle == Triangle.Lower ? j & ~15 : 0;
                int endRow = triangle == Triangle.Upper ? math.min(m, j + columns) : m;
                for (int i = firstRow; i < endRow; i += 16)
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

        /// <summary>
        /// Runs the tiles of C over a copy of A, for an A whose columns are not contiguous. It copies 16 rows of A at
        /// a time, for up to <see cref="PanelDepth"/> steps of the inner dimension (16 KiB of floats), into a buffer
        /// where the panel's columns are contiguous, and runs every tile of those rows of C over it.
        /// </summary>
        static void GemmPacked(float alpha, MatrixView<float> a, MatrixView<float> b, float* c, int ldc, Triangle triangle)
        {
            int m = a.Rows;
            int k = a.Columns;
            int n = b.Columns;
            float* panel = stackalloc float[16 * PanelDepth];
            for (int p0 = 0; p0 < k; p0 += PanelDepth)
            {
                int depth = math.min(PanelDepth, k - p0);
                float* bp = b.At(p0, 0);
                for (int i = 0; i < m; i += 16)
                {
                    int rows = math.min(16, m - i);
                    for (int r = 0; r < rows; r++)
                    {
                        float* source = a.At(i + r, p0);
                        for (int p = 0; p < depth; p++)
                        {
                            panel[p * 16 + r] = source[(long)p * a.ColumnStride];
                        }
                    }

                    int firstColumn = triangle == Triangle.Upper ? i / 6 * 6 : 0;
                    int endColumn = triangle == Triangle.Lower ? math.min(n, i + rows) : n;
                    for (int j = firstColumn; j < endColumn; j += 6)
                    {
                        int columns = math.min(6, n - j);
                        float* bj = bp + (long)j * b.ColumnStride;
                        float* cij = c + i + (long)j * ldc;
                        if (rows == 16 && columns == 6)
                        {
                            Kernel16x6(depth, alpha, panel, 16, bj, b.RowStride, b.ColumnStride, cij, ldc);
                        }
                        else
                        {
                            KernelEdge(rows, columns, depth, alpha, panel, 16, bj, b.RowStride, b.ColumnStride, cij, ldc);
                        }
                    }
                }
            }
        }

        /// <summary>The plain path, for a C whose rows and columns are both not contiguous.</summary>
        static void GemmPlain(float alpha, MatrixView<float> a, MatrixView<float> b, MatrixView<float> c, Triangle triangle)
        {
            int k = a.Columns;
            for (int j = 0; j < c.Columns; j++)
            {
                int firstRow = triangle == Triangle.Lower ? j : 0;
                int endRow = triangle == Triangle.Upper ? math.min(c.Rows, j + 1) : c.Rows;
                for (int i = firstRow; i < endRow; i++)
                {
                    float sum = 0f;
                    for (int p = 0; p < k; p++)
                    {
                        sum = FusedMath.Fma(a[i, p], b[p, j], sum);
                    }

                    c[i, j] = FusedMath.Fma(alpha, sum, c[i, j]);
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
