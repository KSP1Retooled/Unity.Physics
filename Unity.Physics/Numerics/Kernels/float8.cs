using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// Eight floats, which fill one 256-bit vector register. Unity.Mathematics has no type of this width, and the float
    /// kernels need one to hold a tile of eight rows. The layout matches the register's, so that
    /// <see cref="FusedMath.Fma(float8, float8, float8)"/> can pass it to the FMA instruction as it is.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct float8
    {
        public float4 Low;
        public float4 High;

        public float8(float4 low, float4 high)
        {
            Low = low;
            High = high;
        }

        public float8(float value)
        {
            Low = value;
            High = value;
        }

        public static implicit operator float8(float value) => new float8(value);

        public static float8 operator +(float8 a, float8 b) => new float8(a.Low + b.Low, a.High + b.High);

        public static float8 operator *(float8 a, float8 b) => new float8(a.Low * b.Low, a.High * b.High);
    }
}
