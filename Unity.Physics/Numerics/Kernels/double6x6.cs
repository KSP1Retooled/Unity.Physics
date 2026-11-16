using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>Represents a 6x6 matrix of doubles, stored row by row.</summary>
    public unsafe struct double6x6
    {
        public const int Size = 6;

        public double6 r0;
        public double6 r1;
        public double6 r2;
        public double6 r3;
        public double6 r4;
        public double6 r5;

        /// <summary>Multiplies this matrix by <paramref name="x"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double6 Mul(double6 x)
        {
            return SumLanes(Products(r0, x), Products(r1, x), Products(r2, x), Products(r3, x), Products(r4, x), Products(r5, x));
        }

        /// <summary>Multiplies the transpose of this matrix by <paramref name="x"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double6 TransposeMul(double6 x)
        {
            // Compute three independent pairs of terms to avoid a long chain of dependent additions.
            double6 first = Fma(x.Low.y, r1, x.Low.x * r0);
            double6 second = Fma(x.Low.w, r3, x.Low.z * r2);
            double6 third = Fma(x.High.y, r5, x.High.x * r4);
            return (first + second) + third;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double4 Products(double6 row, double6 x)
        {
            // Add the high pair as a whole vector, rather than adding into two individual lanes, to prevent the
            // compiler from splitting the sum.
            return FusedMath.Fma(row.Low, x.Low, new double4(row.High * x.High, 0.0, 0.0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double6 Fma(double scale, double6 row, double6 sum) => new double6(
            FusedMath.Fma(row.Low, scale, sum.Low),
            FusedMath.Fma(row.High, scale, sum.High));

        /// <summary>Returns the sum of the four lanes in each input vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double6 SumLanes(double4 p0, double4 p1, double4 p2, double4 p3, double4 p4, double4 p5)
        {
            // Interleave neighbouring lanes from two vectors, then add to compute partial sums and transpose the data
            // together. Each row has two partial sums, one in each half. Add across the halves to finish each row's
            // sum.
            double4 first = UnpackSum(p0, p1);
            double4 second = UnpackSum(p2, p3);
            double4 low = new double4(first.xy, second.zw) + new double4(first.zw, second.xy);

            double4 last = UnpackSum(p4, p5);
            return new double6(low, last.xy + last.zw);
        }

        /// <summary>Returns sums of neighbouring lanes: a.x + a.y, b.x + b.y, a.z + a.w, and b.z + b.w.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double4 UnpackSum(double4 a, double4 b)
        {
            return new double4(a.x, b.x, a.z, b.z) + new double4(a.y, b.y, a.w, b.w);
        }
    }
}
