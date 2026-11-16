// Ported from faer 0.19.4 has no routine of its own for this; it is the column update that the solves use.
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
