using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Numerics.Memory;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// Contiguous Jacobian rows, twelve floats per row. Preassigned row ranges allow independent construction.
        /// Temporary allocations are released with the job.
        /// </summary>
        unsafe struct JacobianRows
        {
            const int Columns = 12;

            float* start;
            int count;
            int next;

            public JacobianRows(int rows)
                : this((float*)UnsafeUtility.Malloc(rows * Columns * (long)sizeof(float), 16, Allocator.Temp), rows)
            {
            }

            /// <summary>Rows in memory someone else owns, <see cref="Columns"/> floats each.</summary>
            public JacobianRows(float* start, int rows)
            {
                this.start = start;
                count = rows;
                next = 0;
            }

            public float* Start => start;

            public Vector Row(int index)
            {
                return Vector.Create(default, start + index * (long)Columns, 1, Columns);
            }

            /// <summary>The rows from <paramref name="row"/> on, which <see cref="Next"/> hands out in turn.</summary>
            public JacobianRows From(int row)
            {
                JacobianRows rows = this;
                rows.next = row;
                return rows;
            }

            public Vector Next(in MemoryManager heap)
            {
                // A solve counts its rows before building them, so this only runs out if the two disagree.
                SafetyChecks.CheckAreEqualAndThrow(true, next < count);
                return Vector.Create(heap, start + next++ * (long)Columns, 1, Columns);
            }
        }
    }
}
