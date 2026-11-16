// faer 0.19.4 has no routine of its own for this; it is the column update that the solves use, written in faer's style.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The float version, written for vectors of eight floats (float8), which fill a 256-bit register.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>
        /// y += alpha x, for the first <paramref name="count"/> floats: eight at a time, then four, then one at a time.
        /// </summary>
        public static void Axpy(int count, float alpha, float* x, float* y)
        {
            float8 alpha8 = alpha;
            int k = 0;
            for (; k + 8 <= count; k += 8)
            {
                *(float8*)(y + k) = FusedMath.Fma(alpha8, *(float8*)(x + k), *(float8*)(y + k));
            }

            if (k + 4 <= count)
            {
                *(float4*)(y + k) = FusedMath.Fma(alpha8.Low, *(float4*)(x + k), *(float4*)(y + k));
                k += 4;
            }

            for (; k < count; k++)
            {
                y[k] = FusedMath.Fma(alpha, x[k], y[k]);
            }
        }

        /// <summary>
        /// y += alpha x, for <paramref name="count"/> floats of two vectors that are <paramref name="incx"/> and
        /// <paramref name="incy"/> entries apart. Two vectors with a stride of 1 take the fast path above; any other
        /// strides take a plain loop.
        /// </summary>
        public static void Axpy(int count, float alpha, float* x, int incx, float* y, int incy)
        {
            if (incx == 1 && incy == 1)
            {
                Axpy(count, alpha, x, y);
                return;
            }

            for (int k = 0; k < count; k++)
            {
                float* yk = y + (long)k * incy;
                *yk = FusedMath.Fma(alpha, x[(long)k * incx], *yk);
            }
        }

        /// <summary>x := alpha x, for the first <paramref name="count"/> floats.</summary>
        public static void Scale(int count, float alpha, float* x)
        {
            float8 alpha8 = alpha;
            int k = 0;
            for (; k + 8 <= count; k += 8)
            {
                *(float8*)(x + k) = *(float8*)(x + k) * alpha8;
            }

            if (k + 4 <= count)
            {
                *(float4*)(x + k) *= alpha;
                k += 4;
            }

            for (; k < count; k++)
            {
                x[k] *= alpha;
            }
        }

        /// <summary>x := alpha x, for <paramref name="count"/> floats that are <paramref name="incx"/> entries apart.</summary>
        public static void Scale(int count, float alpha, float* x, int incx)
        {
            if (incx == 1)
            {
                Scale(count, alpha, x);
                return;
            }

            for (int k = 0; k < count; k++)
            {
                x[(long)k * incx] *= alpha;
            }
        }
    }
}
