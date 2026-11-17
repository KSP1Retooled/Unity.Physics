using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>Eight floats stored as two float4 vectors, with the same layout as a 256-bit vector register.</summary>
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
