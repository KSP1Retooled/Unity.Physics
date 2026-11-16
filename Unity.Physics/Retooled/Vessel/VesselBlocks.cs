using System.Runtime.CompilerServices;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics.Numerics.Retooled
{
    /// <summary>
    /// The 6x6 block arithmetic of the section solver. A block is 36 doubles stored row by row, the layout of
    /// <see cref="double6x6"/>. Every product is written as rows of the result built from one row of the right-hand matrix
    /// scaled by one element of the left, which keeps each step a broadcast and a fused multiply-add on whole rows.
    /// </summary>
    public static unsafe class VesselBlocks
    {
        public const int Size = 36;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double6 Fma(double scale, double6 row, double6 sum) => new double6(
            FusedMath.Fma(row.Low, scale, sum.Low),
            FusedMath.Fma(row.High, scale, sum.High));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Zero(double* block)
        {
            for (int i = 0; i < Size; i += 4)
            {
                *(double4*)(block + i) = 0.0;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Copy(double* to, double* from)
        {
            for (int i = 0; i < Size; i += 4)
            {
                *(double4*)(to + i) = *(double4*)(from + i);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(double* to, double* from)
        {
            for (int i = 0; i < Size; i += 4)
            {
                *(double4*)(to + i) += *(double4*)(from + i);
            }
        }

        /// <summary>result = a * b.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mul(double* result, double* a, double* b)
        {
            double6 b0 = double6.Load(b), b1 = double6.Load(b + 6), b2 = double6.Load(b + 12);
            double6 b3 = double6.Load(b + 18), b4 = double6.Load(b + 24), b5 = double6.Load(b + 30);
            for (int r = 0; r < 6; r++)
            {
                double* ar = a + 6 * r;
                double6 first = Fma(ar[1], b1, ar[0] * b0);
                double6 second = Fma(ar[3], b3, ar[2] * b2);
                double6 third = Fma(ar[5], b5, ar[4] * b4);
                ((first + second) + third).Store(result + 6 * r);
            }
        }

        /// <summary>result -= transpose(a) * b.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SubtractTransposeMul(double* result, double* a, double* b)
        {
            double6 b0 = double6.Load(b), b1 = double6.Load(b + 6), b2 = double6.Load(b + 12);
            double6 b3 = double6.Load(b + 18), b4 = double6.Load(b + 24), b5 = double6.Load(b + 30);
            for (int r = 0; r < 6; r++)
            {
                double6 first = Fma(a[6 + r], b1, a[r] * b0);
                double6 second = Fma(a[18 + r], b3, a[12 + r] * b2);
                double6 third = Fma(a[30 + r], b5, a[24 + r] * b4);
                (double6.Load(result + 6 * r) - ((first + second) + third)).Store(result + 6 * r);
            }
        }

        /// <summary>
        /// Computes block += weight * a b^T.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddOuter(double* block, double weight, double6 a, double6 b)
        {
            for (int r = 0; r < 6; r++)
            {
                double scale = weight * a[r];
                (double6.Load(block + 6 * r) + scale * b).Store(block + 6 * r);
            }
        }

        /// <summary>result = transpose(a), for a block that is not symmetric.</summary>
        public static void Transpose(double* result, double* a)
        {
            for (int r = 0; r < 6; r++)
            {
                for (int c = 0; c < 6; c++)
                {
                    result[6 * r + c] = a[6 * c + r];
                }
            }
        }

        /// <summary>
        /// The inverse of a symmetric positive definite block, through its Cholesky factor. A pivot that is not positive,
        /// which only rounding on a degenerate section could produce, is clamped to a tiny positive value and counted.
        /// </summary>
        public static bool InvertSymmetric(double* result, double* a)
        {
            // Lower Cholesky factor, row by row, then the inverse of that triangle, then inv(a) = inv(L)^T inv(L).
            double* l = stackalloc double[36];
            bool ok = true;
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double sum = a[6 * i + j];
                    for (int k = 0; k < j; k++)
                    {
                        sum -= l[6 * i + k] * l[6 * j + k];
                    }

                    if (i == j)
                    {
                        double floor = 1e-300 + 1e-15 * math.abs(a[6 * i + i]);
                        if (!(sum > floor))
                        {
                            sum = floor;
                            ok = false;
                        }

                        l[6 * i + i] = math.sqrt(sum);
                    }
                    else
                    {
                        l[6 * i + j] = sum / l[6 * j + j];
                    }
                }
            }

            double* m = stackalloc double[36];
            for (int i = 0; i < 6; i++)
            {
                m[6 * i + i] = 1.0 / l[6 * i + i];
                for (int j = 0; j < i; j++)
                {
                    double sum = 0.0;
                    for (int k = j; k < i; k++)
                    {
                        sum -= l[6 * i + k] * m[6 * k + j];
                    }

                    m[6 * i + j] = sum * m[6 * i + i];
                }

                for (int j = i + 1; j < 6; j++)
                {
                    m[6 * i + j] = 0.0;
                }
            }

            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double sum = 0.0;
                    for (int k = i; k < 6; k++)
                    {
                        sum += m[6 * k + i] * m[6 * k + j];
                    }

                    result[6 * i + j] = sum;
                    result[6 * j + i] = sum;
                }
            }

            return ok;
        }

        /// <summary>The quarter of a block at the given block row and column (0 or 1).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 Quarter(double* block, int row, int column)
        {
            double* at = block + 18 * row + 3 * column;
            return new double3x3(
                at[0], at[1], at[2],
                at[6], at[7], at[8],
                at[12], at[13], at[14]);
        }

        /// <summary>Writes a 3x3 matrix into the quarter of a block at the given block row and column (0 or 1).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetQuarter(double* block, int row, int column, in double3x3 m)
        {
            double* at = block + 18 * row + 3 * column;
            at[0] = m.c0.x; at[1] = m.c1.x; at[2] = m.c2.x;
            at[6] = m.c0.y; at[7] = m.c1.y; at[8] = m.c2.y;
            at[12] = m.c0.z; at[13] = m.c1.z; at[14] = m.c2.z;
        }

        /// <summary>Adds a 3x3 matrix to the quarter of a block at the given block row and column (0 or 1).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddQuarter(double* block, int row, int column, in double3x3 m)
        {
            double* at = block + 18 * row + 3 * column;
            at[0] += m.c0.x; at[1] += m.c1.x; at[2] += m.c2.x;
            at[6] += m.c0.y; at[7] += m.c1.y; at[8] += m.c2.y;
            at[12] += m.c0.z; at[13] += m.c1.z; at[14] += m.c2.z;
        }
    }
}
