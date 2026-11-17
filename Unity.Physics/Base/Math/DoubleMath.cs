using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Holds small helpers for 3-vectors, 3x3 matrices and quaternions in double precision. A quaternion is stored in a
    /// double4 as x, y, z, w.
    /// </summary>
    internal static unsafe class DoubleMath
    {
        /// <summary>
        /// Reads three doubles at <paramref name="from"/> as a double3.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3 Load3(double* from) => new double3(from[0], from[1], from[2]);

        /// <summary>
        /// Writes a double3 as three doubles at <paramref name="to"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Store3(double* to, double3 value)
        {
            to[0] = value.x;
            to[1] = value.y;
            to[2] = value.z;
        }

        /// <summary>
        /// Adds a double3 to the three doubles at <paramref name="to"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add3(double* to, double3 value)
        {
            to[0] += value.x;
            to[1] += value.y;
            to[2] += value.z;
        }

        /// <summary>
        /// Returns the trace of a 3x3 matrix, which is the sum of its diagonal entries.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Trace(in double3x3 m) => m.c0.x + m.c1.y + m.c2.z;

        /// <summary>
        /// Returns the outer product a b^T.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 Outer(double3 a, double3 b) => new double3x3(a * b.x, a * b.y, a * b.z);

        /// <summary>
        /// Returns the matrix X for which X v = arm x v, for any vector v.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 Cross(double3 arm) => new double3x3(
            0.0, -arm.z, arm.y,
            arm.z, 0.0, -arm.x,
            -arm.y, arm.x, 0.0);

        /// <summary>
        /// Returns the matrix X for which X v = a x v, for any vector v. This is the single-precision version of <see
        /// cref="Cross(double3)"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float3x3 Cross(float3 a)
        {
            return new float3x3(new float3(0, a[2], -a[1]),
                new float3(-a[2], 0, a[0]),
                new float3(a[1], -a[0], 0));
        }

        /// <summary>
        /// Returns U diag(w) U^T, where the columns of U are the axes and w holds the weights.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 Weighted(in double3x3 axes, double3 weight)
        {
            double3x3 scaled = new double3x3(axes.c0 * weight.x, axes.c1 * weight.y, axes.c2 * weight.z);
            return math.mul(scaled, math.transpose(axes));
        }

        /// <summary>
        /// Return R*diag(inertia)*R^T for the unit quaternion orientation, converting principal inertia to the
        /// rotated frame.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 RotateDiagonal(double4 orientation, double3 inertia) =>
            RotateDiagonal(Rotation(orientation), inertia);

        /// <summary>
        /// Returns R diag(d) R^T, where R is <paramref name="r"/> and d is <paramref name="inertia"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 RotateDiagonal(in double3x3 r, double3 inertia)
        {
            double3x3 scaled = new double3x3(r.c0 * inertia.x, r.c1 * inertia.y, r.c2 * inertia.z);
            return math.mul(scaled, math.transpose(r));
        }

        /// <summary>
        /// Returns the rotation matrix of a unit quaternion that is stored as x, y, z, w.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 Rotation(double4 q)
        {
            double x = q.x, y = q.y, z = q.z, w = q.w;
            return new double3x3(
                1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
        }

        /// <summary>
        /// Returns the quaternion product a b, for quaternions that are stored as x, y, z, w. The rotation of the
        /// product applies b first and then a.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double4 MultiplyQuaternions(double4 a, double4 b) => new double4(
            a.w * b.xyz + b.w * a.xyz + math.cross(a.xyz, b.xyz),
            a.w * b.w - math.dot(a.xyz, b.xyz));

        /// <summary>
        /// Returns the conjugate of a quaternion that is stored as x, y, z, w. For a unit quaternion, the conjugate is
        /// the inverse rotation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double4 ConjugateQuaternion(double4 q) => new double4(-q.xyz, q.w);

        /// <summary>
        /// Convert double to float, flushing magnitudes below 1e-30 to zero to avoid denormal velocities and
        /// impulses.
        /// </summary>
        public static float Flushed(double value) => math.abs(value) < 1e-30 ? 0.0f : (float)value;

        /// <summary>
        /// Converts each lane of a double3 to a float with <see cref="Flushed(double)"/>.
        /// </summary>
        public static float3 Flushed(double3 value) => new float3(Flushed(value.x), Flushed(value.y), Flushed(value.z));
    }
}
