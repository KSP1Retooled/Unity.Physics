using Unity.Burst.Intrinsics;
using Unity.Mathematics;

namespace Unity.Numerics
{
    /// <summary>
    /// Fused multiply-add, a * b + c with one rounding. <see cref="math.fma(float, float, float)"/> only fuses when Burst is
    /// allowed fast math, which would also let it reorder sums and assume no infinities everywhere it inlines. These ask for
    /// the instruction itself, so they fuse wherever the target CPU has it and fall back to a multiply and an add elsewhere,
    /// which rounds twice: an answer can differ in its last bits between CPUs, and between Burst and Mono.
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
    }
}
