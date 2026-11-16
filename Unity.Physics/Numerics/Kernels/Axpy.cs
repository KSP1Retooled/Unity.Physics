// faer 0.19.4 has no routine of its own for this; it is the column update that the solves use, written in faer's style.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.
// The code is written for raw double pointers in column-major order, with double4 in place of the SIMD types.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>y += alpha x, for the first <paramref name="count"/> entries.</summary>
        public static void Axpy(int count, double alpha, double* x, double* y)
        {
            double4 alpha4 = alpha;
            int k = 0;
            for (; k + 4 <= count; k += 4)
            {
                *(double4*)(y + k) = FusedMath.Fma(alpha4, *(double4*)(x + k), *(double4*)(y + k));
            }

            for (; k < count; k++)
            {
                y[k] = FusedMath.Fma(alpha, x[k], y[k]);
            }
        }

        /// <summary>
        /// y += alpha x, for <paramref name="count"/> entries of two vectors that are <paramref name="incx"/> and
        /// <paramref name="incy"/> entries apart. Two vectors with a stride of 1 take the fast path above; any other
        /// strides take a plain loop.
        /// </summary>
        public static void Axpy(int count, double alpha, double* x, int incx, double* y, int incy)
        {
            if (incx == 1 && incy == 1)
            {
                Axpy(count, alpha, x, y);
                return;
            }

            for (int k = 0; k < count; k++)
            {
                double* yk = y + (long)k * incy;
                *yk = FusedMath.Fma(alpha, x[(long)k * incx], *yk);
            }
        }

        /// <summary>x := alpha x, for <paramref name="count"/> entries that are <paramref name="incx"/> entries apart.</summary>
        public static void Scale(int count, double alpha, double* x, int incx)
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

        /// <summary>x := alpha x, for the first <paramref name="count"/> entries.</summary>
        public static void Scale(int count, double alpha, double* x)
        {
            double4 alpha4 = alpha;
            int k = 0;
            for (; k + 4 <= count; k += 4)
            {
                *(double4*)(x + k) *= alpha4;
            }

            for (; k < count; k++)
            {
                x[k] *= alpha;
            }
        }
    }
}
