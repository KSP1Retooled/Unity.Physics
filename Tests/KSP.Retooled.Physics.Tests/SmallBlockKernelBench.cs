using System.Text;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Times each routine in <see cref="SmallBlockKernels"/> against the kernel in Unity.Numerics.Kernels that does the same
    /// work, at the block sizes that the sparse solver factors in the bench scenes. In those scenes almost every node
    /// has 6 unknowns, one per body, and almost every block below the diagonal is 6 by 6; a few nodes have 7 or 12
    /// (see the histogram that VesselForestBench prints with RETOOLED_BENCH_HISTOGRAM=1). A Burst job runs each routine
    /// on a few hundred blocks that are stored row by row, as the solver stores them, and repeats that many times. The
    /// kernels take the blocks as row-major views. The bench prints nanoseconds per call for the old routine and the
    /// kernel, and the largest difference between their answers. Logs/SmallBlockKernelBench.txt holds the table.
    /// </summary>
    [Explicit("A benchmark, under a minute long")]
    public unsafe class SmallBlockKernelBench
    {
        private const int Blocks = 256;
        private const int Repeats = 400;

        private enum Routine
        {
            Cholesky,
            SolveAgainstTranspose,
            SubtractProductTransposeLower,
            SubtractProductTranspose,
            SubtractProduct,
            AddScaled,
        }

        [BurstCompile(CompileSynchronously = true)]
        private struct RunJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public double* Matrices;
            [NativeDisableUnsafePtrRestriction] public double* Work;
            [NativeDisableUnsafePtrRestriction] public double* Others;
            [NativeDisableUnsafePtrRestriction] public double* Inverse;
            [NativeDisableUnsafePtrRestriction] public long* Ticks;
            public Routine Routine;
            public int Rows;
            public int Size;
            public bool Kernel;
            public int Repeats;
            public int Stride;

            public void Execute()
            {
                int block = Rows * Size;
                long elapsed = 0;
                for (int r = 0; r < Repeats; r++)
                {
                    // Every repeat starts from the same inputs, and the copy is not timed.
                    UnsafeUtility.MemCpy(Work, Matrices, (long)Blocks * Stride * sizeof(double));
                    long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    for (int b = 0; b < Blocks; b++)
                    {
                        double* w = Work + (long)b * Stride;
                        double* o = Others + (long)b * Stride;
                        double* inverse = Inverse + (long)b * Size;
                        switch (Routine)
                        {
                            case Routine.Cholesky:
                                if (Kernel)
                                {
                                    KernelCholesky(w, inverse, Size, -1.0);
                                }
                                else
                                {
                                    SmallBlockKernels.CholeskyInPlace(w, inverse, Size, -1.0);
                                }

                                break;
                            case Routine.SolveAgainstTranspose:
                                if (Kernel)
                                {
                                    Blas.TrsmLeftLower(MatrixView<double>.RowMajor(o, Size, Size, Size),
                                        MatrixView<double>.RowMajor(w, Rows, Size, Size).Transpose);
                                }
                                else
                                {
                                    SmallBlockKernels.SolveAgainstTranspose(w, Rows, o, inverse, Size);
                                }

                                break;
                            case Routine.SubtractProductTransposeLower:
                                if (Kernel)
                                {
                                    Blas.SyrkLower(-1.0, MatrixView<double>.RowMajor(o, Rows, Size, Size), MatrixView<double>.RowMajor(w, Rows, Rows, Rows));
                                }
                                else
                                {
                                    SmallBlockKernels.SubtractProductTranspose(w, Rows, Rows, o, o, Size, 1.0, true);
                                }

                                break;
                            case Routine.SubtractProductTranspose:
                                if (Kernel)
                                {
                                    Blas.Gemm(-1.0, MatrixView<double>.RowMajor(o, Rows, Size, Size), MatrixView<double>.RowMajor(o, Rows, Size, Size).Transpose,
                                        MatrixView<double>.RowMajor(w, Rows, Rows, Rows));
                                }
                                else
                                {
                                    SmallBlockKernels.SubtractProductTranspose(w, Rows, Rows, o, o, Size, 1.0, false);
                                }

                                break;
                            case Routine.SubtractProduct:
                                if (Kernel)
                                {
                                    Blas.Gemv(-1.0, MatrixView<double>.RowMajor(o, Rows, Size, Size), o + block, 1, w, 1);
                                }
                                else
                                {
                                    SmallBlockKernels.SubtractProduct(w, o, o + block, Rows, Size, 1.0);
                                }

                                break;
                            case Routine.AddScaled:
                                if (Kernel)
                                {
                                    Blas.Axpy(Size, -0.5, o, w);
                                }
                                else
                                {
                                    SmallBlockKernels.AddScaled(w, o, -0.5, Size);
                                }

                                break;
                        }
                    }

                    elapsed += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - started;
                }

                Ticks[0] = elapsed;
            }
        }

        /// <summary>
        /// The factor of sign * a for a block stored row by row, through the kernel: the lower triangle takes the sign,
        /// the kernel factors it as a row-major view, and the reciprocals of the diagonal are taken after.
        /// </summary>
        private static void KernelCholesky(double* a, double* inverse, int size, double sign)
        {
            if (sign != 1.0)
            {
                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        a[i * size + j] *= sign;
                    }
                }
            }

            CholeskyLLT.Factor(MatrixView<double>.RowMajor(a, size, size, size));
            for (int i = 0; i < size; i++)
            {
                inverse[i] = 1.0 / a[i * size + i];
            }
        }

        [Test]
        public void Measure()
        {
            var table = new StringBuilder();
            table.AppendLine("[SmallBlockKernelBench] ns per call, old SmallBlockKernels routine / kernel (lower is better), " +
                $"{Blocks} blocks x {Repeats} repeats, Burst job, blocks stored row by row; the largest difference is against the largest entry");
            (Routine routine, int rows, int size)[] cases =
            {
                (Routine.Cholesky, 6, 6), (Routine.Cholesky, 7, 7), (Routine.Cholesky, 12, 12),
                (Routine.SolveAgainstTranspose, 6, 6), (Routine.SolveAgainstTranspose, 7, 6), (Routine.SolveAgainstTranspose, 6, 12),
                (Routine.SubtractProductTransposeLower, 6, 6), (Routine.SubtractProductTransposeLower, 12, 6),
                (Routine.SubtractProductTranspose, 6, 6), (Routine.SubtractProductTranspose, 12, 6),
                (Routine.SubtractProduct, 6, 6), (Routine.SubtractProduct, 6, 12), (Routine.SubtractProduct, 12, 6),
                (Routine.AddScaled, 6, 6), (Routine.AddScaled, 12, 12),
            };

            foreach ((Routine routine, int rows, int size) in cases)
            {
                int stride = math.max(math.max(rows * size, size * size), rows * rows) + 2 * math.max(rows, size);
                var matrices = new NativeArray<double>(Blocks * stride, Allocator.TempJob);
                var others = new NativeArray<double>(Blocks * stride, Allocator.TempJob);
                var oldWork = new NativeArray<double>(Blocks * stride, Allocator.TempJob);
                var newWork = new NativeArray<double>(Blocks * stride, Allocator.TempJob);
                var inverse = new NativeArray<double>(Blocks * size, Allocator.TempJob);
                var ticks = new NativeArray<long>(1, Allocator.TempJob);
                var random = new Unity.Mathematics.Random(17);
                for (int b = 0; b < Blocks; b++)
                {
                    // A symmetric positive definite block (for the factor) whose factor also serves as the triangle of
                    // the solves, and random blocks for the products.
                    double* m = (double*)matrices.GetUnsafePtr() + (long)b * stride;
                    double* o = (double*)others.GetUnsafePtr() + (long)b * stride;
                    int n = routine == Routine.Cholesky ? size : rows;
                    for (int k = 0; k < stride; k++)
                    {
                        m[k] = random.NextDouble(-1.0, 1.0);
                        o[k] = random.NextDouble(-1.0, 1.0);
                    }

                    if (routine == Routine.Cholesky)
                    {
                        // -(J J^T + n I), since the solver factors sign * a with a sign of -1 here.
                        var j = new double[n * n];
                        for (int k = 0; k < j.Length; k++)
                        {
                            j[k] = m[k];
                        }

                        for (int r = 0; r < n; r++)
                        {
                            for (int c = 0; c < n; c++)
                            {
                                double sum = r == c ? n : 0.0;
                                for (int k = 0; k < n; k++)
                                {
                                    sum += j[r * n + k] * j[c * n + k];
                                }

                                m[r * n + c] = -sum;
                            }
                        }
                    }

                    if (routine == Routine.SolveAgainstTranspose)
                    {
                        // A lower triangle with a diagonal between 1 and 2.
                        double* inv = (double*)inverse.GetUnsafePtr() + (long)b * size;
                        for (int r = 0; r < size; r++)
                        {
                            o[r * size + r] = 1.0 + math.abs(o[r * size + r]);
                            inv[r] = 1.0 / o[r * size + r];
                            for (int c = r + 1; c < size; c++)
                            {
                                o[r * size + c] = 0.0;
                            }
                        }
                    }
                }

                double oldNs = Time(routine, rows, size, stride, false, matrices, oldWork, others, inverse, ticks);
                double newNs = Time(routine, rows, size, stride, true, matrices, newWork, others, inverse, ticks);
                double oldAgain = Time(routine, rows, size, stride, false, matrices, oldWork, others, inverse, ticks);
                double newAgain = Time(routine, rows, size, stride, true, matrices, newWork, others, inverse, ticks);

                // Compare the answers in the entries that both routines define: the lower triangle of a factor or of a
                // lower-triangle product, and every entry that the other routines write.
                double largest = 1e-300, difference = 0.0;
                int width = routine == Routine.Cholesky ? size
                    : routine == Routine.SubtractProductTranspose || routine == Routine.SubtractProductTransposeLower ? rows
                    : routine == Routine.SolveAgainstTranspose ? size : 1;
                int height = routine == Routine.Cholesky ? size
                    : routine == Routine.SubtractProductTranspose || routine == Routine.SubtractProductTransposeLower || routine == Routine.SolveAgainstTranspose ? rows
                    : routine == Routine.SubtractProduct ? rows : size;
                bool lowerOnly = routine == Routine.Cholesky || routine == Routine.SubtractProductTransposeLower;
                for (int b = 0; b < Blocks; b++)
                {
                    for (int r = 0; r < height; r++)
                    {
                        for (int c = 0; c < (lowerOnly ? r + 1 : width); c++)
                        {
                            int at = b * stride + r * width + c;
                            largest = math.max(largest, math.abs(oldWork[at]));
                            difference = math.max(difference, math.abs(oldWork[at] - newWork[at]));
                        }
                    }
                }

                table.AppendLine($"{routine,-30} {rows,2} x {size,2}: old {math.min(oldNs, oldAgain),7:F1} / kernel {math.min(newNs, newAgain),7:F1} ns; " +
                    $"largest difference {difference / largest:E1}");
                matrices.Dispose();
                others.Dispose();
                oldWork.Dispose();
                newWork.Dispose();
                inverse.Dispose();
                ticks.Dispose();
            }

            Debug.Log(table.ToString());
            System.IO.File.WriteAllText(System.IO.Path.Combine(Application.dataPath, "..", "Logs", "SmallBlockKernelBench.txt"), table.ToString());
        }

        private static double Time(Routine routine, int rows, int size, int stride, bool kernel, NativeArray<double> matrices, NativeArray<double> work,
            NativeArray<double> others, NativeArray<double> inverse, NativeArray<long> ticks)
        {
            new RunJob
            {
                Matrices = (double*)matrices.GetUnsafePtr(),
                Work = (double*)work.GetUnsafePtr(),
                Others = (double*)others.GetUnsafePtr(),
                Inverse = (double*)inverse.GetUnsafePtr(),
                Ticks = (long*)ticks.GetUnsafePtr(),
                Routine = routine,
                Rows = rows,
                Size = size,
                Kernel = kernel,
                Repeats = Repeats,
                Stride = stride,
            }.Run();
            double ns = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator;
            return ticks[0] * ns / ((double)Blocks * Repeats);
        }
    }
}
