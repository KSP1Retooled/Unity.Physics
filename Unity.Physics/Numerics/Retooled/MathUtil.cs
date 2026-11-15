using Unity.Burst.Intrinsics;
using Unity.Mathematics;

namespace Unity.Physics.Numerics.Retooled
{
    /// <summary>
    /// Provides fused multiply-add operations: a * b + c with one rounding. <see cref="math.fma(float, float, float)"/>
    /// fuses only when Burst permits fast math. Fast math also permits reordering sums and assuming that infinities do
    /// not occur wherever the function is inlined. These methods explicitly use x86 FMA instructions when supported.
    /// Otherwise, they use separate multiplication and addition, with two roundings. Results can therefore differ in
    /// their last bits between CPUs and between Burst and Mono.
    /// </summary>
    public static class MathUtil
    {
        public static float Fma(float a, float b, float c)
        {
            if (X86.Fma.IsFmaSupported)
            {
                return X86.Fma.fmadd_ss(new v128(a), new v128(b), new v128(c)).Float0;
            }

            return a * b + c;
        }

        public static float2 Fma(float2 a, float2 b, float2 c)
        {
            if (X86.Fma.IsFmaSupported)
            {
                v128 result = X86.Fma.fmadd_ps(new v128(a.x, a.y, 0f, 0f), new v128(b.x, b.y, 0f, 0f), new v128(c.x, c.y, 0f, 0f));
                return new float2(result.Float0, result.Float1);
            }

            return a * b + c;
        }

        public static float3 Fma(float3 a, float3 b, float3 c)
        {
            if (X86.Fma.IsFmaSupported)
            {
                v128 result = X86.Fma.fmadd_ps(new v128(a.x, a.y, a.z, 0f), new v128(b.x, b.y, b.z, 0f), new v128(c.x, c.y, c.z, 0f));
                return new float3(result.Float0, result.Float1, result.Float2);
            }

            return a * b + c;
        }

        public static float4 Fma(float4 a, float4 b, float4 c)
        {
            if (X86.Fma.IsFmaSupported)
            {
                v128 result = X86.Fma.fmadd_ps(new v128(a.x, a.y, a.z, a.w), new v128(b.x, b.y, b.z, b.w), new v128(c.x, c.y, c.z, c.w));
                return new float4(result.Float0, result.Float1, result.Float2, result.Float3);
            }

            return a * b + c;
        }

        public static double Fma(double a, double b, double c)
        {
            if (X86.Fma.IsFmaSupported)
            {
                return X86.Fma.fmadd_sd(new v128(a), new v128(b), new v128(c)).Double0;
            }

            return a * b + c;
        }

        public static double2 Fma(double2 a, double2 b, double2 c)
        {
            if (X86.Fma.IsFmaSupported)
            {
                v128 result = X86.Fma.fmadd_pd(new v128(a.x, a.y), new v128(b.x, b.y), new v128(c.x, c.y));
                return new double2(result.Double0, result.Double1);
            }

            return a * b + c;
        }

        public static double3 Fma(double3 a, double3 b, double3 c)
        {
            if (X86.Fma.IsFmaSupported)
            {
                v256 result = X86.Fma.mm256_fmadd_pd(new v256(a.x, a.y, a.z, 0.0), new v256(b.x, b.y, b.z, 0.0), new v256(c.x, c.y, c.z, 0.0));
                return new double3(result.Double0, result.Double1, result.Double2);
            }

            return a * b + c;
        }

        public static double4 Fma(double4 a, double4 b, double4 c)
        {
            if (X86.Fma.IsFmaSupported)
            {
                v256 result = X86.Fma.mm256_fmadd_pd(new v256(a.x, a.y, a.z, a.w), new v256(b.x, b.y, b.z, b.w), new v256(c.x, c.y, c.z, c.w));
                return new double4(result.Double0, result.Double1, result.Double2, result.Double3);
            }

            return a * b + c;
        }

        /// <summary>
        /// Computes <c>sum(a[i] * b[i])</c>.
        /// </summary>
        public static unsafe double ArrayProductSum(double* a, double* b, int count)
        {
            double4x4 sum4x4 = 0.0;

            int i = 0;
            for (; i + 16 <= count; i += 16)
            {
                for (int j = 0; j < 4; j++)
                    sum4x4[j] = Fma(*(double4*)&a[i + 4 * j], *(double4*)&b[i + 4 * j], sum4x4[j]);
            }

            double4 sum4 = (sum4x4[0] + sum4x4[1]) + (sum4x4[2] + sum4x4[3]);
            for (; i + 4 <= count; i += 4)
                sum4 = Fma(*(double4*)&a[i], *(double4*)&b[i], sum4);

            double sum = (sum4[0] + sum4[2]) + (sum4[1] + sum4[3]);
            for (; i < count; ++i)
                sum = Fma(a[i], b[i], sum);

            return sum;
        }
    }
}
