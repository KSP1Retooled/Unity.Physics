using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>Six doubles stored as a double4 and a double2.</summary>
    public unsafe struct double6
    {
        public double4 Low;
        public double2 High;

        public double6(double4 low, double2 high)
        {
            Low = low;
            High = high;
        }

        public double this[int i] => i < 4 ? Low[i] : High[i - 4];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Dot(double6 a, double6 b) => math.dot(a.Low, b.Low) + math.dot(a.High, b.High);

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
