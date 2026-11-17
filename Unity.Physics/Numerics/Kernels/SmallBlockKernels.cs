using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// Dense operations for small blocks, optimized for 6 by 6 matrices.
    /// All matrices use row-major order with no padding between rows.
    /// </summary>
    public static unsafe class SmallBlockKernels
    {
        /// <summary>Minimum pivot value used by CholeskyInPlace.</summary>
        public const double SmallestPivot = 1e-30;

        /// <summary>Compute the lower Cholesky factor of sign*a in place, clamping small pivots to SmallestPivot.</summary>
        /// <param name="a">A size by size matrix. Only the lower triangle is read and overwritten.</param>
        /// <param name="inverse">Output reciprocal of each diagonal entry of the factor.</param>
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

        /// <summary>Compute block = block*C^-T in place.</summary>
        /// <param name="block">A rows by size matrix. Overwritten with the solution.</param>
        /// <param name="c">Lower triangular size by size factor.</param>
        /// <param name="inverse">Reciprocal of each diagonal entry of c.</param>
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

        /// <summary>Replace the lower triangular 6 by 6 matrix c with its inverse. The upper triangle is set to zero.</summary>
        /// <param name="inverse">Reciprocal of each diagonal entry of c.</param>
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

        /// <summary>Compute target -= sign*left*right^T. The target is rows by columns; left and right have inner columns.</summary>
        /// <param name="lowerOnly">Update only the lower triangle of a square target. Some entries above the diagonal may also be overwritten.</param>
        public static void SubtractProductTranspose(double* target, int rows, int columns, double* left, double* right,
            int inner, double sign, bool lowerOnly)
        {
            for (int row = 0; row < rows; row++)
            {
                double* l = left + row * inner;
                double* into = target + row * columns;
                int last = lowerOnly ? row + 1 : columns;
                int column = 0;

                // Allow vector writes past the diagonal into the unused upper triangle.
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

        /// <summary>Compute target -= sign*block*x, where block is rows by inner.</summary>
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

        /// <summary>Compute into -= block*source for a rows by inner block.</summary>
        /// <param name="six">True if the block has six rows. If it also has six columns, add the product to owed instead of subtracting from into.</param>
        /// <param name="owed">Accumulated products that the caller must subtract from into.</param>
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

        /// <summary>Return the dot products of shared with b0, b1, b2 and b3, one per lane, for count entries.</summary>
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

            // Handle the last two entries of a six-column block with a double2.
            if (k + 2 <= count)
            {
                double2 a = *(double2*)(shared + k);
                s0.xy = FusedMath.Fma(*(double2*)(b0 + k), a, s0.xy);
                s1.xy = FusedMath.Fma(*(double2*)(b1 + k), a, s1.xy);
                s2.xy = FusedMath.Fma(*(double2*)(b2 + k), a, s2.xy);
                s3.xy = FusedMath.Fma(*(double2*)(b3 + k), a, s3.xy);
                k += 2;
            }

            // Transpose the partial sums so that each lane gives one dot product after reduction.
            double4x4 partial = math.transpose(new double4x4(s0, s1, s2, s3));
            double4 sum = (partial.c0 + partial.c1) + (partial.c2 + partial.c3);

            if (k < count)
            {
                sum = FusedMath.Fma(new double4(b0[k], b1[k], b2[k], b3[k]), shared[k], sum);
            }

            return sum;
        }

        /// <summary>Compute into += scale*from for count entries.</summary>
        public static void AddScaled(double* into, double* from, double scale, int count)
        {
            Blas.Axpy(count, scale, from, into);
        }
    }
}
