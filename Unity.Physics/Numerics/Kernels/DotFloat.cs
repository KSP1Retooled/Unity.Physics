// Dot products for the triangular solves, following faer 0.19.4's SIMD style.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Return the dot product of the first count entries of a and b.</summary>
        public static float Dot(int count, float* a, float* b)
        {
            float8 s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
            int k = 0;
            for (; k + 32 <= count; k += 32)
            {
                s0 = FusedMath.Fma(*(float8*)(a + k), *(float8*)(b + k), s0);
                s1 = FusedMath.Fma(*(float8*)(a + k + 8), *(float8*)(b + k + 8), s1);
                s2 = FusedMath.Fma(*(float8*)(a + k + 16), *(float8*)(b + k + 16), s2);
                s3 = FusedMath.Fma(*(float8*)(a + k + 24), *(float8*)(b + k + 24), s3);
            }

            for (; k + 8 <= count; k += 8)
            {
                s0 = FusedMath.Fma(*(float8*)(a + k), *(float8*)(b + k), s0);
            }

            float8 sums = (s0 + s1) + (s2 + s3);
            float4 quarter = sums.Low + sums.High;
            if (k + 4 <= count)
            {
                quarter = FusedMath.Fma(*(float4*)(a + k), *(float4*)(b + k), quarter);
                k += 4;
            }

            float sum = (quarter.x + quarter.y) + (quarter.z + quarter.w);
            for (; k < count; k++)
            {
                sum = FusedMath.Fma(a[k], b[k], sum);
            }

            return sum;
        }

        /// <summary>Return the dot product of count entries of x and y.</summary>
        /// <param name="incx">Stride between entries of x, in elements.</param>
        /// <param name="incy">Stride between entries of y, in elements.</param>
        public static float Dot(int count, float* x, int incx, float* y, int incy)
        {
            if (incx == 1 && incy == 1)
            {
                return Dot(count, x, y);
            }

            float sum = 0f;
            for (int k = 0; k < count; k++)
            {
                sum = FusedMath.Fma(x[(long)k * incx], y[(long)k * incy], sum);
            }

            return sum;
        }
    }
}
