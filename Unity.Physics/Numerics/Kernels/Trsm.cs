// Ported from faer 0.19.4 src/linalg/triangular_solve.rs (the recursive triangular solves and their block sizes).
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers in column-major order, with double4 in place of the SIMD types.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// faer's block size for the recursive triangular solve: the lower part of the split is about half of the
        /// rows, rounded up to a multiple of 16, 8 or 4 rows as the matrix gets smaller, so that it fills whole
        /// vectors. Returns the size of the upper part.
        /// </summary>
        static int TriangularBlock(int n)
        {
            int rest = n / 2;
            if (n >= 32)
            {
                rest = (rest + 15) / 16 * 16;
            }
            else if (n >= 16)
            {
                rest = (rest + 7) / 8 * 8;
            }
            else if (n >= 8)
            {
                rest = (rest + 3) / 4 * 4;
            }

            return n - rest;
        }

        /// <summary>
        /// B := L^-1 B, where L is n by n and lower triangular, and B is n by nrhs. This is faer's
        /// solve_lower_triangular_in_place: for more than 4 rows it solves the upper block of rows, takes its share out
        /// of the lower block with a matrix product, and solves the lower block. The base case computes the
        /// reciprocals of the diagonal and the negated multipliers once, and runs the forward substitution down each
        /// column of B. faer also splits B by columns when it has more than 64; this port leaves that out, because the
        /// solver's B has few columns.
        /// </summary>
        public static void TrsmLeftLower(int n, int nrhs, double* l, int ldl, double* b, int ldb)
        {
            if (n <= 0 || nrhs <= 0)
            {
                return;
            }

            if (n <= 4)
            {
                double4 inverse = 0.0;
                for (int i = 0; i < n; i++)
                {
                    inverse[i] = 1.0 / l[i + (long)i * ldl];
                }

                for (int j = 0; j < nrhs; j++)
                {
                    double* bj = b + (long)j * ldb;
                    for (int i = 0; i < n; i++)
                    {
                        double y = bj[i] * inverse[i];
                        for (int p = 0; p < i; p++)
                        {
                            y = FusedMath.Fma(-(l[i + (long)p * ldl] * inverse[i]), bj[p], y);
                        }

                        bj[i] = y;
                    }
                }

                return;
            }

            int bs = TriangularBlock(n);
            TrsmLeftLower(bs, nrhs, l, ldl, b, ldb);
            GemmNN(n - bs, nrhs, bs, -1.0, l + bs, ldl, b, ldb, b + bs, ldb);
            TrsmLeftLower(n - bs, nrhs, l + bs + (long)bs * ldl, ldl, b + bs, ldb);
        }

        /// <summary>
        /// B := B L^-T, where L is n by n and lower triangular, and B is m by n. faer solves this as
        /// L X^T = B^T with its lower solve on a transposed view of B. This port writes the same recursion for B as it
        /// is stored: the left block of columns is solved first, its share is taken out of the right block with
        /// B_1 -= B_0 L_10^T, and the right block is solved. The base case (4 columns or fewer) works down the
        /// columns of B, which are stored in a line.
        /// </summary>
        public static void TrsmRightLowerTranspose(int m, int n, double* l, int ldl, double* b, int ldb)
        {
            if (m <= 0 || n <= 0)
            {
                return;
            }

            if (n <= 4)
            {
                for (int i = 0; i < n; i++)
                {
                    double* bi = b + (long)i * ldb;
                    double inverse = 1.0 / l[i + (long)i * ldl];
                    for (int p = 0; p < i; p++)
                    {
                        Axpy(m, -l[i + (long)p * ldl], b + (long)p * ldb, bi);
                    }

                    Scale(m, inverse, bi);
                }

                return;
            }

            int bs = TriangularBlock(n);
            TrsmRightLowerTranspose(m, bs, l, ldl, b, ldb);
            GemmNT(m, n - bs, bs, -1.0, b, ldb, l + bs, ldl, b + (long)bs * ldb, ldb);
            TrsmRightLowerTranspose(m, n - bs, l + bs + (long)bs * ldl, ldl, b + (long)bs * ldb, ldb);
        }
    }
}
