// Based on gemm-f64 0.18.0 src/microkernel.rs (8 by 6 FMA kernel).
// gemm: https://github.com/sarah-ek/gemm, MIT license.
// Triangular products follow faer 0.19.4 src/linalg/matmul/triangular.rs.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Compute C += alpha*A*B. A is m by k, B is k by n and C is m by n, all in column-major order.</summary>
        public static void GemmNN(int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double* c, int ldc)
        {
            Gemm(alpha, MatrixView<double>.ColumnMajor(a, m, k, lda), MatrixView<double>.ColumnMajor(b, k, n, ldb),
                MatrixView<double>.ColumnMajor(c, m, n, ldc));
        }

        /// <summary>Compute C += alpha*A*B^T. A is m by k, B is n by k and C is m by n, all in column-major order.</summary>
        public static void GemmNT(int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double* c, int ldc)
        {
            Gemm(alpha, MatrixView<double>.ColumnMajor(a, m, k, lda), MatrixView<double>.ColumnMajor(b, n, k, ldb).Transpose,
                MatrixView<double>.ColumnMajor(c, m, n, ldc));
        }

        /// <summary>
        /// Update the lower triangle of C += alpha*A*A^T. A is n by k and C is n by n, both in column-major order.
        /// Preserve the upper triangle of C.
        /// </summary>
        public static void SyrkLowerNT(int n, int k, double alpha, double* a, int lda, double* c, int ldc)
        {
            SyrkLower(alpha, MatrixView<double>.ColumnMajor(a, n, k, lda), MatrixView<double>.ColumnMajor(c, n, n, ldc));
        }

        /// <summary>Compute C += alpha*A*B. A is m by k, B is k by n and C is m by n.</summary>
        public static void Gemm(double alpha, MatrixView<double> a, MatrixView<double> b, MatrixView<double> c)
        {
            Gemm(alpha, a, b, c, Triangle.All, null);
        }

        /// <summary>Update the lower triangle of C += alpha*A*A^T, where A is n by k and C is n by n.</summary>
        /// <param name="keepUpper">Preserve the upper triangle. Set to false to allow faster writes that may overwrite it.</param>
        public static void SyrkLower(double alpha, MatrixView<double> a, MatrixView<double> c, bool keepUpper = true)
        {
            if (keepUpper)
            {
                double* saved = stackalloc double[8 * 6];
                Gemm(alpha, a, a.Transpose, c, Triangle.Lower, saved);
            }
            else
            {
                Gemm(alpha, a, a.Transpose, c, Triangle.Lower, null);
            }
        }

        /// <summary>Compute C += alpha*A*B for the requested region, choosing a kernel for the matrix layout.</summary>
        static void Gemm(double alpha, MatrixView<double> a, MatrixView<double> b, MatrixView<double> c, Triangle triangle, double* saved)
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
                    Gemm(alpha, b.Transpose, a.Transpose, c.Transpose, Flip(triangle), saved);
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
                GemmTiles(m, n, k, alpha, a.Data, a.LeadingDimension, b.Data, b.RowStride, b.ColumnStride, c.Data, ldc, triangle, saved);
            }
            else
            {
                GemmPacked(alpha, a, b, c.Data, ldc, triangle, saved);
            }
        }

        /// <summary>Compute C += alpha*A*B in 8 by 6 tiles. A and C have contiguous columns.</summary>
        /// <param name="bRow">Row stride of b, in elements.</param>
        /// <param name="bColumn">Column stride of b, in elements.</param>
        static void GemmTiles(int m, int n, int k, double alpha, double* a, int lda, double* b, int bRow, int bColumn,
            double* c, int ldc, Triangle triangle, double* saved)
        {
            for (int j = 0; j < n; j += 6)
            {
                int columns = math.min(6, n - j);
                int firstRow = triangle == Triangle.Lower ? j & ~7 : 0;
                int endRow = triangle == Triangle.Upper ? math.min(m, j + columns) : m;
                for (int i = firstRow; i < endRow; i += 8)
                {
                    int rows = math.min(8, m - i);
                    double* ai = a + i;
                    double* bj = b + (long)j * bColumn;
                    double* cij = c + i + (long)j * ldc;
                    Tile(rows, columns, k, alpha, ai, lda, bj, bRow, bColumn, cij, ldc, i - j, triangle, saved);
                }
            }
        }

        /// <summary>Maximum panel depth. An 8 by 256 panel occupies 16 KiB of stack space.</summary>
        const int PanelDepth = 256;

        /// <summary>Compute C += alpha*A*B by copying A into panels of 8 rows with contiguous columns.</summary>
        static void GemmPacked(double alpha, MatrixView<double> a, MatrixView<double> b, double* c, int ldc, Triangle triangle, double* saved)
        {
            int m = a.Rows;
            int k = a.Columns;
            int n = b.Columns;
            double* panel = stackalloc double[8 * math.min(PanelDepth, k)];
            for (int p0 = 0; p0 < k; p0 += PanelDepth)
            {
                int depth = math.min(PanelDepth, k - p0);
                double* bp = b.At(p0, 0);
                for (int i = 0; i < m; i += 8)
                {
                    int rows = math.min(8, m - i);
                    for (int r = 0; r < rows; r++)
                    {
                        double* source = a.At(i + r, p0);
                        for (int p = 0; p < depth; p++)
                        {
                            panel[p * 8 + r] = source[(long)p * a.ColumnStride];
                        }
                    }

                    int firstColumn = triangle == Triangle.Upper ? i / 6 * 6 : 0;
                    int endColumn = triangle == Triangle.Lower ? math.min(n, i + rows) : n;
                    for (int j = firstColumn; j < endColumn; j += 6)
                    {
                        int columns = math.min(6, n - j);
                        double* bj = bp + (long)j * b.ColumnStride;
                        double* cij = c + i + (long)j * ldc;
                        Tile(rows, columns, depth, alpha, panel, 8, bj, b.RowStride, b.ColumnStride, cij, ldc, i - j, triangle, saved);
                    }
                }
            }
        }

        /// <summary>Update a tile of C with at most 8 rows and six columns.</summary>
        /// <param name="offset">Row index minus column index of the tile within C.</param>
        /// <param name="saved">Workspace to preserve entries outside the requested triangle. If null, overwrite the whole tile.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Tile(int rows, int columns, int k, double alpha, double* a, int lda, double* b, int bRow, int bColumn,
            double* c, int ldc, int offset, Triangle triangle, double* saved)
        {
            bool crossed = saved != null &&
                (triangle == Triangle.Lower ? offset < columns - 1 : triangle == Triangle.Upper && offset + rows - 1 > 0);
            if (crossed)
            {
                Outside(rows, columns, offset, triangle, c, ldc, saved, false);
            }

            if (rows == 8 && columns == 6)
            {
                Kernel8x6(k, alpha, a, lda, b, bRow, bColumn, c, ldc);
            }
            else
            {
                KernelEdge(rows, columns, k, alpha, a, lda, b, bRow, bColumn, c, ldc);
            }

            if (crossed)
            {
                Outside(rows, columns, offset, triangle, c, ldc, saved, true);
            }
        }

        /// <summary>Save tile entries outside the requested triangle, or restore them if restore is true.</summary>
        static void Outside(int rows, int columns, int offset, Triangle triangle, double* c, int ldc, double* saved, bool restore)
        {
            int count = 0;
            for (int s = 0; s < columns; s++)
            {
                int from = triangle == Triangle.Lower ? 0 : math.max(0, s - offset + 1);
                int to = triangle == Triangle.Lower ? math.min(rows, math.max(0, s - offset)) : rows;
                double* column = c + (long)s * ldc;
                for (int r = from; r < to; r++)
                {
                    if (restore)
                    {
                        column[r] = saved[count++];
                    }
                    else
                    {
                        saved[count++] = column[r];
                    }
                }
            }
        }

        /// <summary>Compute C += alpha*A*B with scalar loops for arbitrary strides.</summary>
        static void GemmPlain(double alpha, MatrixView<double> a, MatrixView<double> b, MatrixView<double> c, Triangle triangle)
        {
            int k = a.Columns;
            for (int j = 0; j < c.Columns; j++)
            {
                int firstRow = triangle == Triangle.Lower ? j : 0;
                int endRow = triangle == Triangle.Upper ? math.min(c.Rows, j + 1) : c.Rows;
                for (int i = firstRow; i < endRow; i++)
                {
                    double sum = 0.0;
                    for (int p = 0; p < k; p++)
                    {
                        sum = FusedMath.Fma(a[i, p], b[p, j], sum);
                    }

                    c[i, j] = FusedMath.Fma(alpha, sum, c[i, j]);
                }
            }
        }

        /// <summary>Update an 8 by 6 tile of C.</summary>
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

        /// <summary>Update a partial tile of C with at most eight rows and six columns.</summary>
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
