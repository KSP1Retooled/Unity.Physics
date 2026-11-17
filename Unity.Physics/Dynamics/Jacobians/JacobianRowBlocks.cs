using System.Runtime.CompilerServices;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics
{
    /// <summary>
    /// Helpers for twelve-float Jacobian rows acting on two six-double blocks.
    /// </summary>
    internal static unsafe class JacobianRowBlocks
    {
        /// <summary>
        /// Reads six floats at <paramref name="columns"/> as a double6.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double6 Load(float* columns) => new double6(
            new double4(columns[0], columns[1], columns[2], columns[3]), new double2(columns[4], columns[5]));

        /// <summary>
        /// Add scale times a row to its two section blocks. Skip halves with negative section indices.
        /// </summary>
        public static void Add(double* vector, int2 sections, float* columns, double scale)
        {
            if (sections.x >= 0)
            {
                double* to = vector + 6 * sections.x;
                (double6.Load(to) + scale * Load(columns)).Store(to);
            }

            if (sections.y >= 0)
            {
                double* to = vector + 6 * sections.y;
                (double6.Load(to) + scale * Load(columns + 6)).Store(to);
            }
        }

        /// <summary>
        /// Returns the dot product of the row with the block vector. The row uses the same layout as in <see
        /// cref="Add"/>.
        /// </summary>
        public static double Dot(double* vector, int2 sections, float* columns)
        {
            double speed = 0.0;
            if (sections.x >= 0)
            {
                speed += double6.Dot(double6.Load(vector + 6 * sections.x), Load(columns));
            }

            if (sections.y >= 0)
            {
                speed += double6.Dot(double6.Load(vector + 6 * sections.y), Load(columns + 6));
            }

            return speed;
        }
    }
}
