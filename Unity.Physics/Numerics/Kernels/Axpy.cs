// Vector updates and scaling, following the column updates in faer 0.19.4.
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    public static unsafe partial class Blas
    {
        /// <summary>Compute y += alpha*x for count entries.</summary>
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

        /// <summary>Compute y += alpha*x for count entries.</summary>
        /// <param name="incx">Stride between entries of x, in elements.</param>
        /// <param name="incy">Stride between entries of y, in elements.</param>
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

        /// <summary>Scale count entries of x by alpha.</summary>
        /// <param name="incx">Stride between entries of x, in elements.</param>
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

        /// <summary>Scale count entries of x by alpha.</summary>
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
