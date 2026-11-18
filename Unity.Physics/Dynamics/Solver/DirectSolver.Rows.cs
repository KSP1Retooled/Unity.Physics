using Unity.Numerics.Linear;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Numerics.Linear.Dense.Primitives;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// Contiguous Jacobian rows, twelve floats per row. Preassigned row ranges allow independent construction.
        /// Temporary allocations are released with the job.
        /// </summary>
        internal unsafe struct JacobianRows
        {
            const int Columns = 12;

            float* start;
            int count;
            int next;

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

            public Vector Next()
            {
                // A solve counts its rows before building them, so this only runs out if the two disagree.
                SafetyChecks.CheckAreEqualAndThrow(true, next < count);
                return Vector.Create(default, start + next++ * (long)Columns, 1, Columns);
            }
        }

        /// <summary>
        /// Build a Jacobian's rows starting at startRow, initially free and unbounded.
        /// Return whether any bound requires a bounded solve. Parallel Jacobians write disjoint ranges.
        /// </summary>
        internal static unsafe bool BuildHeaderRows(ref JacobianHeader header, int startRow, int jacRows, in IslandRowBuffers buffers,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, in StepInput stepInput,
            RowRegularization* regularization)
        {
            for (int row = startRow; row < startRow + jacRows; row++)
            {
                buffers.Lower[row] = float.MinValue;
                buffers.Upper[row] = float.MaxValue;
                buffers.Flags[row] = LCP.MLCPIndexFlag.Free;
                buffers.Recovery[row] = -1.0f;
            }

            var flags = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<LCP.MLCPIndexFlag>(buffers.Flags, buffers.Rows, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref flags, AtomicSafetyHandle.GetTempMemoryHandle());
#endif
            var context = new RowBuildContext
            {
                MotionVelocities = motionVelocities,
                MotionDatas = motionDatas,
                StepInput = stepInput,
                JacobianRows = new JacobianRows(buffers.Jacobian, buffers.Rows),
                Eps = Vector.Create(default, buffers.Epsilon, 1, buffers.Rows),
                L = Vector.Create(default, buffers.Lower, 1, buffers.Rows),
                U = Vector.Create(default, buffers.Upper, 1, buffers.Rows),
                RightHandSide = Vector.Create(default, buffers.RightHandSide, 1, buffers.Rows),
                Recovery = buffers.Recovery,
                IndexSetArray = flags,
                Regularization = regularization,
            };

            Vector* JBlocks = stackalloc Vector[jacRows];
            BuildJacobianRows(ref header, startRow, jacRows, ref context, JBlocks);
            return context.IsLCP;
        }
    }
}
