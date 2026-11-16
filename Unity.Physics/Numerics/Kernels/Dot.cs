// Ported from faer 0.19.4 has no routine of its own for this; it is the dot product that the solves use.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers in column-major order, with double4 in place of the SIMD types.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Returns the dot product of the first <paramref name="count"/> entries of two arrays.</summary>
        /// <remarks>
        /// It keeps four vector sums of four entries each. A vector sum has to wait for the last addition to it before
        /// it can take the next one, so four sums that take turns keep the processor busy.
        /// </remarks>
        public static double Dot(int count, double* a, double* b)
        {
            double4 s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            int k = 0;
            for (; k + 16 <= count; k += 16)
            {
                s0 = FusedMath.Fma(*(double4*)(a + k), *(double4*)(b + k), s0);
                s1 = FusedMath.Fma(*(double4*)(a + k + 4), *(double4*)(b + k + 4), s1);
                s2 = FusedMath.Fma(*(double4*)(a + k + 8), *(double4*)(b + k + 8), s2);
                s3 = FusedMath.Fma(*(double4*)(a + k + 12), *(double4*)(b + k + 12), s3);
            }

            for (; k + 4 <= count; k += 4)
            {
                s0 = FusedMath.Fma(*(double4*)(a + k), *(double4*)(b + k), s0);
            }

            double4 sums = (s0 + s1) + (s2 + s3);
            double sum = (sums.x + sums.y) + (sums.z + sums.w);
            for (; k < count; k++)
            {
                sum = FusedMath.Fma(a[k], b[k], sum);
            }

            return sum;
        }

        /// <summary>
        /// Returns the dot product of <paramref name="count"/> entries of two vectors that are <paramref name="incx"/>
        /// and <paramref name="incy"/> entries apart. Two vectors with a stride of 1 take the fast path above; any
        /// other strides take a plain loop.
        /// </summary>
        public static double Dot(int count, double* x, int incx, double* y, int incy)
        {
            if (incx == 1 && incy == 1)
            {
                return Dot(count, x, y);
            }

            double sum = 0.0;
            for (int k = 0; k < count; k++)
            {
                sum = FusedMath.Fma(x[(long)k * incx], y[(long)k * incy], sum);
            }

            return sum;
        }
    }
}
