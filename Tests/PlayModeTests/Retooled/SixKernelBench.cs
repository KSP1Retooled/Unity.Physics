using System.Diagnostics;
using NUnit.Framework;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Numerics;
using Unity.Physics;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The sparse solver's 6x6 block kernels against the general ones, and the cost of reading blocks out of order, in
    /// nanoseconds per block. Each test logs its table and checks only that the kernels agree.
    /// </summary>
    [Category("Slow")]
    public unsafe class SixKernelBench
    {
        private const int Blocks = 16384;
        private const int Repeats = 2000;
        private const int BlockBytes = 36 * sizeof(double);

        public delegate void Kernel(double* blocks, double* sources, int count, int repeats, double* target);

        public delegate void OrderedKernel(double* blocks, int* order, double* sources, int count, int repeats, int ahead,
            int toL2, double* target);

        [BurstCompile]
        public static class Kernels
        {
            /// <summary>One target owed by every block in turn, as a top node is.</summary>
            [BurstCompile]
            public static void ForwardSix(double* blocks, double* sources, int count, int repeats, double* target)
            {
                var owed = double6.Load(target);
                for (int r = 0; r < repeats; r++)
                {
                    for (int i = 0; i < count; i++)
                    {
                        owed -= ((double6x6*)(blocks + 36 * i))->Mul(double6.Load(sources + 6 * i));
                    }
                }

                owed.Store(target);
            }

            /// <summary>
            /// The blocks in the given order, first prefetching the one <paramref name="ahead"/> places on, into L1 or, when
            /// <paramref name="toL2"/> is set, only as far as L2.
            /// </summary>
            [BurstCompile]
            public static void ForwardOrdered(double* blocks, int* order, double* sources, int count, int repeats, int ahead,
                int toL2, double* target)
            {
                var owed = double6.Load(target);
                for (int r = 0; r < repeats; r++)
                {
                    for (int i = 0; i < count; i++)
                    {
                        if (ahead > 0 && i + ahead < count)
                        {
                            // A block spans five cache lines, or six when it doesn't start on one.
                            byte* next = (byte*)(blocks + 36 * order[i + ahead]);
                            for (int line = 0; line < BlockBytes; line += 64)
                            {
                                Fetch(next + line, toL2);
                            }

                            Fetch(next + BlockBytes - 1, toL2);
                        }

                        int b = order[i];
                        owed -= ((double6x6*)(blocks + 36 * b))->Mul(double6.Load(sources + 6 * b));
                    }
                }

                owed.Store(target);
            }

            [BurstCompile]
            public static void ForwardGeneral(double* blocks, double* sources, int count, int repeats, double* target)
            {
                for (int r = 0; r < repeats; r++)
                {
                    for (int i = 0; i < count; i++)
                    {
                        SparseConstraintSolver.SubtractProduct(target, blocks + 36 * i, sources + 6 * i, 6, 6, 1.0);
                    }
                }
            }

            /// <summary>One node taking every neighbour's answer in turn, as the backward pass does.</summary>
            [BurstCompile]
            public static void BackwardSix(double* blocks, double* sources, int count, int repeats, double* target)
            {
                var sum = double6.Load(target);
                for (int r = 0; r < repeats; r++)
                {
                    for (int i = 0; i < count; i++)
                    {
                        sum -= ((double6x6*)(blocks + 36 * i))->TransposeMul(double6.Load(sources + 6 * i));
                    }
                }

                sum.Store(target);
            }

            [BurstCompile]
            public static void BackwardGeneral(double* blocks, double* sources, int count, int repeats, double* target)
            {
                for (int r = 0; r < repeats; r++)
                {
                    for (int i = 0; i < count; i++)
                    {
                        for (int row = 0; row < 6; row++)
                        {
                            SparseConstraintSolver.AddScaled(target, blocks + 36 * i + 6 * row, -sources[6 * i + row], 6);
                        }
                    }
                }
            }

            // Prefetch takes its locality as a constant, so the two levels are two calls.
            private static void Fetch(byte* address, int toL2)
            {
                if (toL2 != 0)
                {
                    Common.Prefetch(address, Common.ReadWrite.Read, Common.Locality.LowTemporalLocality);
                }
                else
                {
                    Common.Prefetch(address, Common.ReadWrite.Read);
                }
            }
        }

        [Test]
        public void Measure()
        {
            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            var random = new Unity.Mathematics.Random(7);
            var blocks = new NativeArray<double>(36 * Blocks, Allocator.Persistent);
            var sources = new NativeArray<double>(6 * Blocks, Allocator.Persistent);
            var target = new NativeArray<double>(8, Allocator.Persistent);
            for (int i = 0; i < blocks.Length; i++)
            {
                blocks[i] = random.NextDouble(-1e-3, 1e-3);
            }

            for (int i = 0; i < sources.Length; i++)
            {
                sources[i] = random.NextDouble(-1.0, 1.0);
            }

            var b = (double*)blocks.GetUnsafePtr();
            var s = (double*)sources.GetUnsafePtr();
            var t = (double*)target.GetUnsafePtr();

            // Every kernel does target -= its blocks times its sources, so from zero they all agree.
            foreach ((Kernel kernel, Kernel reference) in new (Kernel, Kernel)[]
            {
                (Kernels.ForwardSix, Kernels.ForwardGeneral),
                (Kernels.BackwardSix, Kernels.BackwardGeneral),
            })
            {
                var expected = new double[6];
                var actual = new double[6];
                fixed (double* e = expected, a = actual)
                {
                    BurstCompiler.CompileFunctionPointer(reference).Invoke(b, s, 64, 1, e);
                    BurstCompiler.CompileFunctionPointer(kernel).Invoke(b, s, 64, 1, a);
                }

                for (int i = 0; i < 6; i++)
                {
                    Assert.AreEqual(expected[i], actual[i], 1e-12, $"{kernel.Method.Name} row {i}");
                }
            }

            var results = new System.Text.StringBuilder("[SixKernelBench]");
            foreach (int count in new[] { 64, 1024, Blocks })
            {
                int repeats = Repeats * 256 / count;
                results.Append($"\n  {count} blocks ({count * BlockBytes / 1024} KB):");
                foreach ((string name, Kernel kernel) in new (string, Kernel)[]
                {
                    ("ForwardSix", Kernels.ForwardSix),
                    ("ForwardGeneral", Kernels.ForwardGeneral),
                    ("BackwardSix", Kernels.BackwardSix),
                    ("BackwardGeneral", Kernels.BackwardGeneral),
                })
                {
                    Kernel compiled = BurstCompiler.CompileFunctionPointer(kernel).Invoke;
                    compiled(b, s, count, 2, t);
                    double best = double.MaxValue;
                    for (int trial = 0; trial < 5; trial++)
                    {
                        var watch = Stopwatch.StartNew();
                        compiled(b, s, count, repeats, t);
                        best = System.Math.Min(best, watch.Elapsed.TotalMilliseconds * 1e6 / ((double)count * repeats));
                    }

                    results.Append($" {name}={best:F2}ns");
                }
            }

            Debug.Log(results.ToString());
            blocks.Dispose();
            sources.Dispose();
            target.Dispose();
        }

        /// <summary>A factor's worth of blocks past L2, read in order and shuffled, prefetching various distances ahead.</summary>
        [Test]
        public void MeasureOrder()
        {
            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            const int count = 4096;
            var random = new Unity.Mathematics.Random(11);
            var blocks = new NativeArray<double>(36 * count, Allocator.Persistent);
            var sources = new NativeArray<double>(6 * count, Allocator.Persistent);
            var sequential = new NativeArray<int>(count, Allocator.Persistent);
            var shuffled = new NativeArray<int>(count, Allocator.Persistent);
            var target = new NativeArray<double>(8, Allocator.Persistent);
            for (int i = 0; i < blocks.Length; i++)
            {
                blocks[i] = random.NextDouble(-1e-3, 1e-3);
            }

            for (int i = 0; i < count; i++)
            {
                sequential[i] = i;
                shuffled[i] = i;
            }

            for (int i = count - 1; i > 0; i--)
            {
                int j = random.NextInt(i + 1);
                (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
            }

            OrderedKernel kernel = BurstCompiler.CompileFunctionPointer<OrderedKernel>(Kernels.ForwardOrdered).Invoke;
            var b = (double*)blocks.GetUnsafePtr();
            var s = (double*)sources.GetUnsafePtr();
            var t = (double*)target.GetUnsafePtr();
            var results = new System.Text.StringBuilder(
                $"[SixKernelBench] {count} blocks ({count * BlockBytes / 1024} KB), ns per block by blocks fetched ahead:");
            foreach ((string name, NativeArray<int> order) in new[] { ("sequential", sequential), ("shuffled", shuffled) })
            {
                var o = (int*)order.GetUnsafePtr();
                foreach (int toL2 in new[] { 0, 1 })
                {
                    results.Append($"\n  {name} into {(toL2 != 0 ? "L2" : "L1")}:");
                    foreach (int ahead in new[] { 0, 2, 4, 8, 16, 32 })
                    {
                        kernel(b, o, s, count, 2, ahead, toL2, t);
                        double best = double.MaxValue;
                        for (int trial = 0; trial < 7; trial++)
                        {
                            var watch = Stopwatch.StartNew();
                            kernel(b, o, s, count, 100, ahead, toL2, t);
                            best = System.Math.Min(best, watch.Elapsed.TotalMilliseconds * 1e6 / (count * 100.0));
                        }

                        results.Append($" {ahead}={best:F2}");
                    }
                }
            }

            Debug.Log(results.ToString());
            blocks.Dispose();
            sources.Dispose();
            sequential.Dispose();
            shuffled.Dispose();
            target.Dispose();
        }
    }
}
