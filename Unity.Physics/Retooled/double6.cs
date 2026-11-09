using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace Unity.Numerics
{
    internal unsafe struct double6
    {
        public double4 Low;
        public double2 High;

        public double6(double4 low, double2 high)
        {
            Low = low;
            High = high;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double6 Load(double* from) => new double6(*(double4*)from, *(double2*)(from + 4));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Store(double* to)
        {
            *(double4*)to = Low;
            *(double2*)(to + 4) = High;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double6 operator +(double6 a, double6 b) => new double6(a.Low + b.Low, a.High + b.High);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double6 operator -(double6 a, double6 b) => new double6(a.Low - b.Low, a.High - b.High);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double6 operator *(double scale, double6 a) => new double6(scale * a.Low, scale * a.High);
    }
}
