// faer 0.19.4 has no routine of its own for this; it is the dot product that the solves use, written in faer's style.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The float version, written for vectors of eight floats (float8), which fill a 256-bit register.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Returns the dot product of the first <paramref name="count"/> entries of two arrays of floats.</summary>
        /// <remarks>
        /// It keeps four vector sums of eight entries each, so one pass of the loop takes 32 entries. A vector sum has
        /// to wait for the last addition to it before it can take the next one, so four sums that take turns keep the
        /// processor busy. After the whole vectors, a vector of four and then single entries take the rest.
        /// </remarks>
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

        /// <summary>
        /// Returns the dot product of <paramref name="count"/> entries of two vectors of floats that are
        /// <paramref name="incx"/> and <paramref name="incy"/> entries apart. Two vectors with a stride of 1 take the
        /// fast path above; any other strides take a plain loop.
        /// </summary>
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
