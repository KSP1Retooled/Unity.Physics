// Vector updates and scaling, following the column updates in faer 0.19.4.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Compute y += alpha*x for count entries.</summary>
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

        /// <summary>Compute y += alpha*x for count entries.</summary>
        /// <param name="incx">Stride between entries of x, in elements.</param>
        /// <param name="incy">Stride between entries of y, in elements.</param>
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

        /// <summary>Scale count entries of x by alpha.</summary>
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

        /// <summary>Scale count entries of x by alpha.</summary>
        /// <param name="incx">Stride between entries of x, in elements.</param>
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
