using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// Holds dense kernels for small blocks: the Cholesky factor of a block, a triangular solve with that factor, the
    /// inverse of a 6 by 6 factor, and products that are subtracted from a target. Every block is a dense matrix of
    /// doubles that is stored row by row, with no gap between the rows. The routines are written for blocks of about
    /// 6 by 6, the size of <see cref="double6x6"/>.
    /// <para>
    /// The sparse constraint solver is the main user. Almost every block in that solver is 6 by 6, one row and column
    /// per body's unknowns. At that size, these routines are faster than the general kernels in this assembly, which
    /// take the blocks as row-major views (SmallBlockKernelBench times both): the factor by about 10%, the products and
    /// the solves by 25% to 3 times. <see cref="AddScaled"/> runs the same loop as the general kernel, so it calls that
    /// kernel.
    /// </para>
    /// </summary>
    public static unsafe class SmallBlockKernels
    {
        /// <summary>
        /// The smallest pivot that <see cref="CholeskyInPlace"/> accepts. The routine raises a smaller pivot to this
        /// value before it takes the square root.
        /// </summary>
        public const double SmallestPivot = 1e-30;

        /// <summary>
        /// Computes the lower Cholesky factor of sign * a in place. The block a is square, has <paramref name="size"/>
        /// rows, and is stored row by row. The routine reads and writes only the lower half of the block. It writes the
        /// reciprocal of each diagonal entry of the factor to <paramref name="inverse"/>. Later columns multiply by
        /// these reciprocals, because a division would delay the sum for the next column.
        /// </summary>
        public static void CholeskyInPlace(double* a, double* inverse, int size, double sign)
        {
            for (int row = 0; row < size; row++)
            {
                double* r = a + row * size;
                for (int column = 0; column < row; column++)
                {
                    double sum = sign * r[column] - FusedMath.ArrayProductSum(r, a + column * size, column);
                    r[column] = sum * inverse[column];
                }

                double pivot = sign * r[row] - FusedMath.ArrayProductSum(r, r, row);
                r[row] = math.sqrt(math.max(pivot, SmallestPivot));
                inverse[row] = 1.0 / r[row];
            }
        }

        /// <summary>
        /// Multiplies each row of <paramref name="block"/> by C^-T in place. The block has <paramref name="rows"/> rows
        /// of <paramref name="size"/> entries. C is a lower triangular factor with <paramref name="size"/> rows, and
        /// <paramref name="inverse"/> holds the reciprocal of each diagonal entry of C. The routine multiplies by these
        /// reciprocals, because a division would delay the sum for the next column.
        /// </summary>
        public static void SolveAgainstTranspose(double* block, int rows, double* c, double* inverse, int size)
        {
            for (int row = 0; row < rows; row++)
            {
                double* x = block + row * size;
                for (int column = 0; column < size; column++)
                {
                    double sum = x[column] - FusedMath.ArrayProductSum(x, c + column * size, column);

                    x[column] = sum * inverse[column];
                }
            }
        }

        /// <summary>
        /// Replaces the lower triangular factor C of a 6x6 block with the full inverse of C. <paramref name="inverse"/>
        /// holds the reciprocal of each diagonal entry of C. A triangular solve has to find each unknown after the one
        /// before it, so every node along a branch of a tree waits on a chain of results. A multiplication by the full
        /// inverse finds all six unknowns at the same time.
        /// </summary>
        public static void InvertTriangle(double* c, double* inverse)
        {
            const int size = double6x6.Size;
            double* result = stackalloc double[size * size];
            for (int column = 0; column < size; column++)
            {
                for (int row = 0; row < column; row++)
                {
                    result[row * size + column] = 0.0;
                }

                for (int row = column; row < size; row++)
                {
                    double sum = row == column ? 1.0 : 0.0;
                    for (int j = column; j < row; j++)
                    {
                        sum = FusedMath.Fma(-c[row * size + j], result[j * size + column], sum);
                    }

                    result[row * size + column] = sum * inverse[row];
                }
            }

            UnsafeUtility.MemCpy(c, result, size * size * sizeof(double));
        }

        /// <summary>
        /// Computes target -= sign * left * right^T. The matrix left has <paramref name="rows"/> rows, the matrix right
        /// has <paramref name="columns"/> rows, and every row of both matrices has <paramref name="inner"/> entries.
        /// Each entry of the result is the dot product of a row of left with a row of right, so the routine reads both
        /// rows in order. It updates four entries of a row of the target at a time. If <paramref name="lowerOnly"/> is
        /// true, the target is square and the routine updates its lower half. In that case, the routine can also write
        /// a few entries just above the diagonal, which the caller doesn't use.
        /// </summary>
        public static void SubtractProductTranspose(double* target, int rows, int columns, double* left, double* right,
            int inner, double sign, bool lowerOnly)
        {
            for (int row = 0; row < rows; row++)
            {
                double* l = left + row * inner;
                double* into = target + row * columns;
                int last = lowerOnly ? row + 1 : columns;
                int column = 0;

                // A group of four entries can reach past the diagonal into the upper half, which the caller doesn't
                // use. This lets the short rows of a diagonal block still use all four vector lanes.
                for (; column < last && column + 4 <= columns; column += 4)
                {
                    double* r = right + column * inner;
                    double4 sum = ProductSums(l, r, r + inner, r + 2 * inner, r + 3 * inner, inner);
                    *(double4*)(into + column) -= sign * sum;
                }

                for (; column < last; column++)
                {
                    into[column] -= sign * FusedMath.ArrayProductSum(l, right + column * inner, inner);
                }
            }
        }

        /// <summary>
        /// Computes target -= sign * block * x. The block is stored row by row and has <paramref name="rows"/> rows of
        /// <paramref name="inner"/> entries. The routine handles four rows at a time.
        /// </summary>
        public static void SubtractProduct(double* target, double* block, double* x, int rows, int inner, double sign)
        {
            int row = 0;
            for (; row + 4 <= rows; row += 4)
            {
                double* b = block + row * inner;
                double4 sum = ProductSums(x, b, b + inner, b + 2 * inner, b + 3 * inner, inner);
                *(double4*)(target + row) -= sign * sum;
            }

            for (; row < rows; row++)
            {
                target[row] -= sign * FusedMath.ArrayProductSum(block + row * inner, x, inner);
            }
        }

        /// <summary>
        /// Computes into -= block * source, where the block has <paramref name="rows"/> rows of <paramref
        /// name="inner"/> entries. If <paramref name="six"/> is true and the block is 6x6, the routine adds the product
        /// to <paramref name="owed"/> instead. The caller then subtracts <paramref name="owed"/> from <paramref
        /// name="into"/> at the end. Otherwise, the routine subtracts the product immediately.
        /// </summary>
        public static void SubtractOrDefer(ref double6 owed, bool six, double* into, double* block, double* source,
            int rows, int inner)
        {
            if (six && inner == double6x6.Size)
            {
                owed += ((double6x6*)block)->Mul(double6.Load(source));
                return;
            }

            SubtractProduct(into, block, source, rows, inner, 1.0);
        }

        /// <summary>
        /// Computes the dot product of <paramref name="shared"/> with each of four other vectors of the same length,
        /// and returns one product in each lane. The routine reads each vector in order, and it uses each load of the
        /// shared vector in all four products.
        /// </summary>
        static double4 ProductSums(double* shared, double* b0, double* b1, double* b2, double* b3, int count)
        {
            double4 s0 = 0.0;
            double4 s1 = 0.0;
            double4 s2 = 0.0;
            double4 s3 = 0.0;
            int k = 0;
            for (; k + 4 <= count; k += 4)
            {
                double4 a = *(double4*)(shared + k);
                s0 = FusedMath.Fma(*(double4*)(b0 + k), a, s0);
                s1 = FusedMath.Fma(*(double4*)(b1 + k), a, s1);
                s2 = FusedMath.Fma(*(double4*)(b2 + k), a, s2);
                s3 = FusedMath.Fma(*(double4*)(b3 + k), a, s3);
            }

            // Use a two-lane vector when two entries remain. This case is common, because a body has six columns.
            if (k + 2 <= count)
            {
                double2 a = *(double2*)(shared + k);
                s0.xy = FusedMath.Fma(*(double2*)(b0 + k), a, s0.xy);
                s1.xy = FusedMath.Fma(*(double2*)(b1 + k), a, s1.xy);
                s2.xy = FusedMath.Fma(*(double2*)(b2 + k), a, s2.xy);
                s3.xy = FusedMath.Fma(*(double2*)(b3 + k), a, s3.xy);
                k += 2;
            }

            // Each accumulator holds the partial sums for one vector. After the transpose, each column holds one
            // partial sum from each vector. Adding the columns gives the total for each vector in its own lane.
            double4x4 partial = math.transpose(new double4x4(s0, s1, s2, s3));
            double4 sum = (partial.c0 + partial.c1) + (partial.c2 + partial.c3);

            if (k < count)
            {
                sum = FusedMath.Fma(new double4(b0[k], b1[k], b2[k], b3[k]), shared[k], sum);
            }

            return sum;
        }

        /// <summary>
        /// Computes into += scale * from for <paramref name="count"/> entries, with the kernel's
        /// <see cref="Blas.Axpy(int, double, double*, double*)"/>, which runs the same loop.
        /// </summary>
        public static void AddScaled(double* into, double* from, double scale, int count)
        {
            Blas.Axpy(count, scale, from, into);
        }

        /// <summary>
        /// Solves C y = own in place, where C is a lower triangular block with <paramref name="size"/> rows that was
        /// not replaced by its inverse. <paramref name="inverse"/> holds the reciprocal of each diagonal entry of C.
        /// </summary>
        public static void SolveLower(double* c, double* own, double* inverse, int size)
        {
            for (int row = 0; row < size; row++)
            {
                double sum = own[row];
                for (int j = 0; j < row; j++)
                {
                    sum = FusedMath.Fma(-c[row * size + j], own[j], sum);
                }

                own[row] = sum * inverse[row];
            }
        }

        /// <summary>
        /// Solves C^T y = own in place, where C is a lower triangular block with <paramref name="size"/> rows that was
        /// not replaced by its inverse. <paramref name="inverse"/> holds the reciprocal of each diagonal entry of C.
        /// </summary>
        public static void SolveUpper(double* c, double* own, double* inverse, int size)
        {
            for (int row = size - 1; row >= 0; row--)
            {
                double answer = own[row] * inverse[row];
                own[row] = answer;
                for (int j = 0; j < row; j++)
                {
                    own[j] = FusedMath.Fma(-c[row * size + j], answer, own[j]);
                }
            }
        }
    }
}
