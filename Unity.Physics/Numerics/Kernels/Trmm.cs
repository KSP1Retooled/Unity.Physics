// Based on faer 0.19.4 src/linalg/matmul/triangular.rs.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Compute B = L*B in place.</summary>
        /// <param name="l">Lower triangular n by n matrix.</param>
        /// <param name="b">An n by nrhs matrix. Overwritten with the product.</param>
        /// <param name="unitDiagonal">Treat diagonal entries as 1 without reading them.</param>
        public static void TrmmLeftLower(MatrixView<double> l, MatrixView<double> b, bool unitDiagonal = false)
        {
            int n = l.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= 4)
            {
                TrmmBaseCase(l, b, unitDiagonal, false);
                return;
            }

            int bs = TriangularBlock(n);
            int nrhs = b.Columns;
            TrmmLeftLower(l.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
            Gemm(1.0, l.Block(bs, 0, n - bs, bs), b.Block(0, 0, bs, nrhs), b.Block(bs, 0, n - bs, nrhs));
            TrmmLeftLower(l.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
        }

        /// <summary>Compute B = U*B in place.</summary>
        /// <param name="u">Upper triangular n by n matrix.</param>
        /// <param name="b">An n by nrhs matrix. Overwritten with the product.</param>
        /// <param name="unitDiagonal">Treat diagonal entries as 1 without reading them.</param>
        public static void TrmmLeftUpper(MatrixView<double> u, MatrixView<double> b, bool unitDiagonal = false)
        {
            int n = u.Rows;
            if (n <= 0 || b.Columns <= 0)
            {
                return;
            }

            if (n <= 4)
            {
                TrmmBaseCase(u, b, unitDiagonal, true);
                return;
            }

            int bs = TriangularBlock(n);
            int nrhs = b.Columns;
            TrmmLeftUpper(u.Block(0, 0, bs, bs), b.Block(0, 0, bs, nrhs), unitDiagonal);
            Gemm(1.0, u.Block(0, bs, bs, n - bs), b.Block(bs, 0, n - bs, nrhs), b.Block(0, 0, bs, nrhs));
            TrmmLeftUpper(u.Block(bs, bs, n - bs, n - bs), b.Block(bs, 0, n - bs, nrhs), unitDiagonal);
        }

        /// <summary>
        /// Multiply a triangle with at most four rows by B in place.
        /// Process lower triangles bottom to top and upper triangles top to bottom to preserve input values.
        /// </summary>
        static void TrmmBaseCase(MatrixView<double> t, MatrixView<double> b, bool unitDiagonal, bool upper)
        {
            int n = t.Rows;
            int nrhs = b.Columns;
            if (!b.ColumnsAreContiguous && b.RowsAreContiguous)
            {
                for (int s = 0; s < n; s++)
                {
                    int i = upper ? s : n - 1 - s;
                    double* bi = b.At(i, 0);
                    if (!unitDiagonal)
                    {
                        Scale(nrhs, t[i, i], bi);
                    }

                    int from = upper ? i + 1 : 0;
                    int to = upper ? n : i;
                    for (int p = from; p < to; p++)
                    {
                        Axpy(nrhs, t[i, p], b.At(p, 0), bi);
                    }
                }

                return;
            }

            for (int j = 0; j < nrhs; j++)
            {
                for (int s = 0; s < n; s++)
                {
                    int i = upper ? s : n - 1 - s;
                    int from = upper ? i + 1 : 0;
                    int to = upper ? n : i;
                    double y = unitDiagonal ? b[i, j] : t[i, i] * b[i, j];
                    for (int p = from; p < to; p++)
                    {
                        y = FusedMath.Fma(t[i, p], b[p, j], y);
                    }

                    b[i, j] = y;
                }
            }
        }
    }
}
