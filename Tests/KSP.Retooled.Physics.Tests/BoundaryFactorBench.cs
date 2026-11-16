using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Numerics.Retooled;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Times the Cholesky factor and the solve that the boundary's dense solve uses, apart from any scene. A Burst job
    /// factors a random symmetric positive definite matrix many times, and solves with the factor many times. The job
    /// does this with the faer port, <see cref="CholeskyLLT.Factor"/> and <see cref="CholeskyLLT.Solve"/>, and with a
    /// copy of the kernel that the solver used before (one entry at a time, each a dot product with one running vector
    /// sum, and a solve with L^T that reads L down its columns). The bench prints microseconds per factor and per
    /// solve for several sizes, and checks that both kernels give the same answer up to rounding.
    /// </summary>
    [Explicit("A benchmark, under a minute long")]
    public unsafe class BoundaryFactorBench
    {
        private const int Stride = 119;
        private const int Repeats = 2000;

        [BurstCompile(CompileSynchronously = true)]
        private struct FactorJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public double* W;
            [NativeDisableUnsafePtrRestriction] public double* Factor;
            [NativeDisableUnsafePtrRestriction] public double* Rhs;
            [NativeDisableUnsafePtrRestriction] public double* X;
            [NativeDisableUnsafePtrRestriction] public int* Order;
            [NativeDisableUnsafePtrRestriction] public long* Ticks;
            public int M;
            public bool Old;
            public int Repeats;

            public void Execute()
            {
                long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                for (int r = 0; r < Repeats; r++)
                {
                    if (Old)
                    {
                        OldFactor(Stride, W, Order, M, Factor);
                    }
                    else
                    {
                        for (int j = 0; j < M; j++)
                        {
                            double* wj = W + (long)Order[j] * Stride;
                            double* lj = Factor + (long)j * Stride;
                            for (int i = j; i < M; i++)
                            {
                                lj[i] = wj[Order[i]];
                            }
                        }

                        CholeskyLLT.Factor(M, Factor, Stride, keepUpper: false);
                    }
                }

                long factored = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                for (int r = 0; r < Repeats; r++)
                {
                    for (int a = 0; a < M; a++)
                    {
                        X[a] = Rhs[a];
                    }

                    if (Old)
                    {
                        OldSolve(Stride, Factor, M, X);
                    }
                    else
                    {
                        CholeskyLLT.Solve(M, Factor, Stride, X);
                    }
                }

                Ticks[0] = factored - started;
                Ticks[1] = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - factored;
            }
        }

        /// <summary>The factor as the solver made it before: L L^T, one entry of L at a time.</summary>
        private static void OldFactor(int n, double* w, int* order, int m, double* l)
        {
            for (int a = 0; a < m; a++)
            {
                double* wi = w + (long)order[a] * n;
                double* ra = l + (long)a * n;
                for (int b = 0; b <= a; b++)
                {
                    ra[b] = wi[order[b]];
                }
            }

            for (int a = 0; a < m; a++)
            {
                double* ra = l + (long)a * n;
                for (int b = 0; b <= a; b++)
                {
                    double* rb = l + (long)b * n;
                    double sum = ra[b] - OldDot(ra, rb, b);
                    if (a == b)
                    {
                        ra[a] = math.sqrt(math.max(sum, 1e-300));
                    }
                    else
                    {
                        ra[b] = sum / rb[b];
                    }
                }
            }
        }

        /// <summary>The solve with the factor as the solver did it before.</summary>
        private static void OldSolve(int n, double* l, int m, double* rhs)
        {
            for (int a = 0; a < m; a++)
            {
                double sum = rhs[a] - OldDot(l + (long)a * n, rhs, a);
                rhs[a] = sum / l[(long)a * n + a];
            }

            for (int a = m - 1; a >= 0; a--)
            {
                double sum = rhs[a];
                for (int k = a + 1; k < m; k++)
                {
                    sum -= l[(long)k * n + a] * rhs[k];
                }

                rhs[a] = sum / l[(long)a * n + a];
            }
        }

        private static double OldDot(double* a, double* b, int count)
        {
            double4 sums = 0.0;
            int k = 0;
            for (; k + 4 <= count; k += 4)
            {
                sums += *(double4*)(a + k) * *(double4*)(b + k);
            }

            double sum = (sums.x + sums.y) + (sums.z + sums.w);
            for (; k < count; k++)
            {
                sum += a[k] * b[k];
            }

            return sum;
        }

        [BurstCompile(CompileSynchronously = true)]
        private struct SolveJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public double* W;
            [NativeDisableUnsafePtrRestriction] public double* Q;
            [NativeDisableUnsafePtrRestriction] public double* Lower;
            [NativeDisableUnsafePtrRestriction] public double* Upper;
            [NativeDisableUnsafePtrRestriction] public byte* Start;
            [NativeDisableUnsafePtrRestriction] public byte* States;
            [NativeDisableUnsafePtrRestriction] public double* Z;
            [NativeDisableUnsafePtrRestriction] public BoundCoupling* Couplings;
            [NativeDisableUnsafePtrRestriction] public BoundarySolveStats* Stats;
            [NativeDisableUnsafePtrRestriction] public long* Ticks;
            public int Rows;
            public int CouplingCount;
            public FactorUpdate Update;
            public int Repeats;

            public void Execute()
            {
                long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                for (int r = 0; r < Repeats; r++)
                {
                    for (int i = 0; i < Rows; i++)
                    {
                        States[i] = Start[i];
                    }

                    BoundarySolver.Solve(Rows, W, Q, Lower, Upper, States, Couplings, CouplingCount, true, Z, 200, true, Update,
                        null, Stats, out _);
                }

                Ticks[0] = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - started;
            }
        }

        /// <summary>
        /// Times whole boundary solves of a problem shaped like a grounded craft's boundary: 24 contacts of 5 rows, two
        /// normal rows that only push and three friction rows coupled to them, 120 rows in all. Each solve starts from
        /// the states of the answer to a nearby problem, as a vessel's solve starts from the last solve's states. The
        /// bench prints microseconds per solve and the solve's counters, for each way of changing the factor.
        /// </summary>
        [Test]
        public void MeasureSolve()
        {
            const int contacts = 24, rows = contacts * 5, dofs = 150;
            var random = new Random(4242);
            var jacobian = new double[rows * dofs];
            for (int k = 0; k < jacobian.Length; k++)
            {
                jacobian[k] = random.NextDouble(-1.0, 1.0);
            }

            var w = new NativeArray<double>(rows * rows, Allocator.Persistent);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    double sum = i == j ? 1e-3 : 0.0;
                    for (int k = 0; k < dofs; k++)
                    {
                        sum += jacobian[i * dofs + k] * jacobian[j * dofs + k];
                    }

                    w[i * rows + j] = sum;
                }
            }

            var q = new NativeArray<double>(rows, Allocator.Persistent);
            var lower = new NativeArray<double>(rows, Allocator.Persistent);
            var upper = new NativeArray<double>(rows, Allocator.Persistent);
            var start = new NativeArray<byte>(rows, Allocator.Persistent);
            var states = new NativeArray<byte>(rows, Allocator.Persistent);
            var z = new NativeArray<double>(rows, Allocator.Persistent);
            var couplings = new NativeArray<BoundCoupling>(contacts, Allocator.Persistent);
            var stats = new NativeArray<BoundarySolveStats>(1, Allocator.Persistent);
            var ticks = new NativeArray<long>(1, Allocator.Persistent);
            for (int c = 0; c < contacts; c++)
            {
                for (int k = 0; k < 5; k++)
                {
                    int row = c * 5 + k;
                    q[row] = k < 2 ? random.NextDouble(-1.0, 3.0) : random.NextDouble(-2.0, 2.0);
                    lower[row] = 0.0;
                    upper[row] = k < 2 ? double.PositiveInfinity : 0.0;
                }

                couplings[c] = new BoundCoupling { FirstRow = c * 5 + 2, RowCount = 3, FirstSourceRow = c * 5, SourceRowCount = 2, Factor = 0.6f };
            }

            var job = new SolveJob
            {
                W = (double*)w.GetUnsafePtr(),
                Q = (double*)q.GetUnsafePtr(),
                Lower = (double*)lower.GetUnsafePtr(),
                Upper = (double*)upper.GetUnsafePtr(),
                Start = (byte*)start.GetUnsafePtr(),
                States = (byte*)states.GetUnsafePtr(),
                Z = (double*)z.GetUnsafePtr(),
                Couplings = (BoundCoupling*)couplings.GetUnsafePtr(),
                Stats = (BoundarySolveStats*)stats.GetUnsafePtr(),
                Ticks = (long*)ticks.GetUnsafePtr(),
                Rows = rows,
                CouplingCount = contacts,
                Update = FactorUpdate.Refactor,
                Repeats = 1,
            };

            // The start: the states of the answer to the problem before q moved a little.
            job.Run();
            for (int i = 0; i < rows; i++)
            {
                start[i] = states[i];
                q[i] += random.NextDouble(-0.05, 0.05);
            }

            string report = "[BoundaryFactorBench] whole boundary solves, 120 rows, warm from a nearby answer";
            var ratio = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
            double toMicroseconds = ratio.Numerator / (double)ratio.Denominator * 1e-3;
            foreach (FactorUpdate update in new[] { FactorUpdate.Refactor, FactorUpdate.KeepPrefix, FactorUpdate.Edit })
            {
                const int repeats = 500;
                job.Update = update;
                job.Repeats = 20;
                job.Run();
                stats[0] = default;
                job.Repeats = repeats;
                job.Run();
                BoundarySolveStats s = stats[0];
                report += $"\n  {update}: {ticks[0] * toMicroseconds / repeats:F2} us a solve, factor {s.FactorTicks * toMicroseconds / repeats:F2} us, " +
                    $"solves {s.SolveTicks * toMicroseconds / repeats:F2} us, checks {s.PivotTicks * toMicroseconds / repeats:F2} us; " +
                    $"rounds {s.Rounds / (double)repeats:F2} ({s.Factorizations / (double)repeats:F2} factored, {s.Updates / (double)repeats:F2} changed, " +
                    $"{s.Reuses / (double)repeats:F2} reused), rows joined / left {s.AddedRows / (double)repeats:F1} / {s.RemovedRows / (double)repeats:F1}, " +
                    $"free rows a round {s.FreeRows / (double)s.Rounds:F1}";
            }

            UnityEngine.Debug.Log(report);
            System.IO.File.AppendAllText(System.IO.Path.Combine(UnityEngine.Application.dataPath, "..", "Logs", "BoundaryFactorBench.txt"), "\n" + report + "\n");
            w.Dispose();
            q.Dispose();
            lower.Dispose();
            upper.Dispose();
            start.Dispose();
            states.Dispose();
            z.Dispose();
            couplings.Dispose();
            stats.Dispose();
            ticks.Dispose();
        }

        [BurstCompile(CompileSynchronously = true)]
        private struct EditJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public double* Saved;
            [NativeDisableUnsafePtrRestriction] public double* Factor;
            [NativeDisableUnsafePtrRestriction] public double* Columns;
            [NativeDisableUnsafePtrRestriction] public double* Scratch;
            [NativeDisableUnsafePtrRestriction] public int* Deleted;
            [NativeDisableUnsafePtrRestriction] public long* Ticks;
            public int M;
            public int Count;
            public int Repeats;

            public void Execute()
            {
                Ticks[0] = Ticks[1] = Ticks[2] = 0;
                for (int r = 0; r < Repeats; r++)
                {
                    long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    UnsafeUtility.MemCpy(Factor, Saved, (long)Stride * Stride * sizeof(double));
                    long copied = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    CholeskyLLT.DeleteRowsAndColumns(M, Factor, Stride, Deleted, Count, Scratch);
                    long deleted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    int kept = M - Count;
                    for (int i = 0; i < (kept + Count) * Count; i++)
                    {
                        Columns[i] = i % (kept + Count + 1) == 0 ? 50.0 : 0.01;
                    }

                    CholeskyLLT.InsertRowsAndColumns(kept, Factor, Stride, kept, Count, Columns, kept + Count, Scratch);
                    long inserted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    Ticks[0] += copied - started;
                    Ticks[1] += deleted - copied;
                    Ticks[2] += inserted - deleted;
                }
            }
        }

        /// <summary>
        /// Times faer's deletion of a few rows and columns from a factor of 68 rows, the free set's average size in the
        /// grounded Tu-160 scene, and the insertion of as many at the end.
        /// </summary>
        [Test]
        public void MeasureEdits()
        {
            const int m = 68;
            var a = new NativeArray<double>(Stride * Stride, Allocator.Persistent);
            var random = new Random(99);
            const int dofs = 90;
            var jacobian = new double[m * dofs];
            for (int k = 0; k < jacobian.Length; k++)
            {
                jacobian[k] = random.NextDouble(-1.0, 1.0);
            }

            for (int j = 0; j < m; j++)
            {
                for (int i = j; i < m; i++)
                {
                    double sum = i == j ? 1e-2 : 0.0;
                    for (int k = 0; k < dofs; k++)
                    {
                        sum += jacobian[i * dofs + k] * jacobian[j * dofs + k];
                    }

                    a[i + j * Stride] = sum;
                }
            }

            CholeskyLLT.Factor(m, (double*)a.GetUnsafePtr(), Stride);
            var factor = new NativeArray<double>(Stride * Stride, Allocator.Persistent);
            var columns = new NativeArray<double>(Stride * Stride, Allocator.Persistent);
            var scratch = new NativeArray<double>(Stride * Stride + 2 * Stride + 8, Allocator.Persistent);
            var deleted = new NativeArray<int>(Stride, Allocator.Persistent);
            var ticks = new NativeArray<long>(3, Allocator.Persistent);
            var ratio = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
            double toMicroseconds = ratio.Numerator / (double)ratio.Denominator * 1e-3;
            string report = "[BoundaryFactorBench] edits of a factor of 68 rows: microseconds to copy / delete / insert at the end";
            foreach (int count in new[] { 1, 3, 6 })
            {
                foreach (int where in new[] { 0, m / 2, m - count })
                {
                    for (int k = 0; k < count; k++)
                    {
                        deleted[k] = where + k;
                    }

                    const int repeats = 2000;
                    var job = new EditJob
                    {
                        Saved = (double*)a.GetUnsafePtr(),
                        Factor = (double*)factor.GetUnsafePtr(),
                        Columns = (double*)columns.GetUnsafePtr(),
                        Scratch = (double*)scratch.GetUnsafePtr(),
                        Deleted = (int*)deleted.GetUnsafePtr(),
                        Ticks = (long*)ticks.GetUnsafePtr(),
                        M = m,
                        Count = count,
                        Repeats = 20,
                    };
                    job.Run();
                    job.Repeats = repeats;
                    job.Run();
                    report += $"\n  {count} rows from place {where}: {ticks[0] * toMicroseconds / repeats:F2} / {ticks[1] * toMicroseconds / repeats:F2} / " +
                        $"{ticks[2] * toMicroseconds / repeats:F2}";
                }
            }

            UnityEngine.Debug.Log(report);
            System.IO.File.AppendAllText(System.IO.Path.Combine(UnityEngine.Application.dataPath, "..", "Logs", "BoundaryFactorBench.txt"), "\n" + report + "\n");
            a.Dispose();
            factor.Dispose();
            columns.Dispose();
            scratch.Dispose();
            deleted.Dispose();
            ticks.Dispose();
        }

        [Test]
        public void Measure()
        {
            // W = J J^T + a small diagonal, with 30 columns in J for each 5 rows, so that W looks like a boundary block
            // of contacts on a few bodies: dense, symmetric, positive definite and not too well conditioned.
            var random = new Random(12345);
            const int dofs = 72;
            var jacobian = new double[Stride * dofs];
            for (int k = 0; k < jacobian.Length; k++)
            {
                jacobian[k] = random.NextDouble(-1.0, 1.0);
            }

            var w = new NativeArray<double>(Stride * Stride, Allocator.Persistent);
            for (int i = 0; i < Stride; i++)
            {
                for (int j = 0; j < Stride; j++)
                {
                    double sum = i == j ? 1e-3 : 0.0;
                    for (int k = 0; k < dofs; k++)
                    {
                        sum += jacobian[i * dofs + k] * jacobian[j * dofs + k];
                    }

                    w[i * Stride + j] = sum;
                }
            }

            var factor = new NativeArray<double>(Stride * Stride, Allocator.Persistent);
            var rhs = new NativeArray<double>(Stride, Allocator.Persistent);
            var oldX = new NativeArray<double>(Stride, Allocator.Persistent);
            var newX = new NativeArray<double>(Stride, Allocator.Persistent);
            var order = new NativeArray<int>(Stride, Allocator.Persistent);
            var ticks = new NativeArray<long>(2, Allocator.Persistent);
            for (int i = 0; i < Stride; i++)
            {
                rhs[i] = random.NextDouble(-1.0, 1.0);
            }

            string report = $"[BoundaryFactorBench] microseconds per factor / per solve, {Repeats} repeats, old kernel against new";
            foreach (int m in new[] { 119, 100, 80, 60, 40 })
            {
                // The free rows are every other row first, then the rest, as a free set is a scattered part of W.
                int next = 0;
                for (int i = 0; i < Stride && next < m; i += 2)
                {
                    order[next++] = i;
                }

                for (int i = 1; i < Stride && next < m; i += 2)
                {
                    order[next++] = i;
                }

                double* times = stackalloc double[4];
                for (int pass = 0; pass < 2; pass++)
                {
                    bool old = pass == 0;
                    var job = new FactorJob
                    {
                        W = (double*)w.GetUnsafePtr(),
                        Factor = (double*)factor.GetUnsafePtr(),
                        Rhs = (double*)rhs.GetUnsafePtr(),
                        X = (double*)(old ? oldX : newX).GetUnsafePtr(),
                        Order = (int*)order.GetUnsafePtr(),
                        Ticks = (long*)ticks.GetUnsafePtr(),
                        M = m,
                        Old = old,
                        Repeats = 20,
                    };

                    // A short run first, so that the timed run doesn't pay for the first touch of the memory.
                    job.Run();
                    job.Repeats = Repeats;
                    job.Run();
                    var ratio = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
                    double scale = ratio.Numerator / (double)ratio.Denominator * 1e-3 / Repeats;
                    times[2 * pass] = ticks[0] * scale;
                    times[2 * pass + 1] = ticks[1] * scale;
                }

                double largest = 0.0, size = 0.0;
                for (int a = 0; a < m; a++)
                {
                    largest = math.max(largest, math.abs(oldX[a] - newX[a]));
                    size = math.max(size, math.abs(oldX[a]));
                }

                report += $"\n  m = {m}: factor {times[0]:F2} -> {times[2]:F2}, solve {times[1]:F2} -> {times[3]:F2}; " +
                    $"largest difference {largest / size:E1} of the largest entry";
                Assert.That(largest, Is.LessThanOrEqualTo(1e-6 * size), $"m = {m}: both kernels give the same answer");
            }

            UnityEngine.Debug.Log(report);
            System.IO.File.WriteAllText(System.IO.Path.Combine(UnityEngine.Application.dataPath, "..", "Logs", "BoundaryFactorBench.txt"), report);
            w.Dispose();
            factor.Dispose();
            rhs.Dispose();
            oldX.Dispose();
            newX.Dispose();
            order.Dispose();
            ticks.Dispose();
        }
    }
}
