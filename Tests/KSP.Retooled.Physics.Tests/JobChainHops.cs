using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Profiling.LowLevel.Unsafe;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Measure median scheduling gaps between serial and deferred parallel jobs.
    /// Write results to the log and Logs/JobChainHops.txt.
    /// </summary>
    [TestFixture]
    [Explicit("profiling")]
    public unsafe class JobChainHops
    {
        const int Stages = 400;
        const int Rounds = 5;

        struct Times
        {
            public long SerialStart;
            public long SerialEnd;
            public long ParallelFirst;
            public long ParallelLastStart;
            public long ParallelLast;
        }

        [BurstCompile]
        struct SerialJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public Times* Times;
            [NativeDisableUnsafePtrRestriction] public int* Counts;
            public int Stage;
            public int Lanes;
            public long WorkTicks;

            public void Execute()
            {
                Times[Stage].SerialStart = ProfilerUnsafeUtility.Timestamp;
                Spin(WorkTicks);
                Counts[Stage] = Lanes;
                Times[Stage].SerialEnd = ProfilerUnsafeUtility.Timestamp;
            }
        }

        [BurstCompile]
        struct ParallelJob : IJobParallelForDefer
        {
            [NativeDisableUnsafePtrRestriction] public Times* Times;
            public int Stage;
            public long WorkTicks;

            public void Execute(int lane)
            {
                long start = ProfilerUnsafeUtility.Timestamp;
                ref long first = ref Times[Stage].ParallelFirst;
                long seen = System.Threading.Volatile.Read(ref first);
                while ((seen == 0 || start < seen) && System.Threading.Interlocked.CompareExchange(ref first, start, seen) != seen)
                {
                    seen = System.Threading.Volatile.Read(ref first);
                }

                ref long lastStart = ref Times[Stage].ParallelLastStart;
                seen = System.Threading.Volatile.Read(ref lastStart);
                while (start > seen && System.Threading.Interlocked.CompareExchange(ref lastStart, start, seen) != seen)
                {
                    seen = System.Threading.Volatile.Read(ref lastStart);
                }

                Spin(WorkTicks);
                long end = ProfilerUnsafeUtility.Timestamp;
                ref long last = ref Times[Stage].ParallelLast;
                seen = System.Threading.Volatile.Read(ref last);
                while (end > seen && System.Threading.Interlocked.CompareExchange(ref last, end, seen) != seen)
                {
                    seen = System.Threading.Volatile.Read(ref last);
                }
            }
        }

        static void Spin(long ticks)
        {
            long until = ProfilerUnsafeUtility.Timestamp + ticks;
            while (ProfilerUnsafeUtility.Timestamp < until)
            {
                Unity.Burst.Intrinsics.Common.Pause();
            }
        }

        [Test]
        public void Measure()
        {
            int workers = JobsUtility.JobWorkerCount;
            int threads = int.TryParse(System.Environment.GetEnvironmentVariable("RETOOLED_BENCH_THREADS"), out int t) && t > 0 ? t : 8;
            JobsUtility.JobWorkerCount = threads - 1;
            ProfilerUnsafeUtility.TimestampConversionRatio ratio = ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
            double usPerTick = ratio.Numerator / (double)ratio.Denominator * 1e-3;
            long TicksOf(double us) => (long)(us / usPerTick);
            var report = new StringBuilder($"[JobChainHops] {threads} threads, {Stages} stages a chain, {Rounds} rounds" + System.Environment.NewLine);
            try
            {
                // Lanes of the parallel jobs: none (a stage that doesn't run), one, and every thread but the serial one.
                foreach (int lanes in new[] { 0, 1, threads - 1 })
                {
                    // The serial jobs' work and each lane's work, in microseconds. A long serial job, like the dense boundary
                    // solve, lets idle workers go to sleep before the parallel job after it.
                    foreach ((double serialUs, double workUs) in new[] { (2.0, 2.0), (20.0, 20.0), (300.0, 20.0) })
                    {
                        var serialToParallel = new List<double>();
                        var parallelToSerial = new List<double>();
                        var lastLane = new List<double>();
                        var perStage = new List<double>();
                        for (int round = 0; round < Rounds; round++)
                        {
                            var times = (Times*)UnsafeUtility.Malloc(sizeof(Times) * Stages, 64, Allocator.Persistent);
                            var counts = (int*)UnsafeUtility.Malloc(sizeof(int) * Stages, 64, Allocator.Persistent);
                            UnsafeUtility.MemClear(times, sizeof(Times) * Stages);
                            UnsafeUtility.MemClear(counts, sizeof(int) * Stages);
                            long started = ProfilerUnsafeUtility.Timestamp;
                            JobHandle handle = default;
                            for (int k = 0; k < Stages; k++)
                            {
                                handle = new SerialJob { Times = times, Counts = counts, Stage = k, Lanes = lanes, WorkTicks = TicksOf(serialUs) }
                                    .Schedule(handle);
                                handle = new ParallelJob { Times = times, Stage = k, WorkTicks = TicksOf(workUs) }.Schedule(counts + k, 1, handle);
                            }

                            JobHandle.ScheduleBatchedJobs();
                            handle.Complete();
                            long ended = ProfilerUnsafeUtility.Timestamp;
                            perStage.Add((ended - started) * usPerTick / Stages);
                            for (int k = 0; k < Stages; k++)
                            {
                                if (lanes > 0)
                                {
                                    serialToParallel.Add((times[k].ParallelFirst - times[k].SerialEnd) * usPerTick);
                                    lastLane.Add((times[k].ParallelLastStart - times[k].SerialEnd) * usPerTick);
                                }

                                if (k + 1 < Stages)
                                {
                                    long previous = lanes > 0 ? times[k].ParallelLast : times[k].SerialEnd;
                                    parallelToSerial.Add((times[k + 1].SerialStart - previous) * usPerTick);
                                }
                            }

                            UnsafeUtility.Free(times, Allocator.Persistent);
                            UnsafeUtility.Free(counts, Allocator.Persistent);
                        }

                        serialToParallel.Sort();
                        parallelToSerial.Sort();
                        lastLane.Sort();
                        perStage.Sort();
                        string Median(List<double> list) => list.Count == 0 ? "-" : list[list.Count / 2].ToString("F1", CultureInfo.InvariantCulture);
                        string Tail(List<double> list) => list.Count == 0 ? "-" : list[list.Count * 9 / 10].ToString("F1", CultureInfo.InvariantCulture);
                        report.Append($"parallel lanes {lanes}, serial work {serialUs:F0} us, lane work {workUs:F0} us: serial end -> first lane {Median(serialToParallel)} us " +
                            $"(90% {Tail(serialToParallel)}), serial end -> last lane's start {Median(lastLane)} us, {(lanes > 0 ? "last lane end" : "serial end")} -> next serial start " +
                            $"{Median(parallelToSerial)} us (90% {Tail(parallelToSerial)}), wall time a stage {Median(perStage)} us" +
                            System.Environment.NewLine);
                    }
                }
            }
            finally
            {
                JobsUtility.JobWorkerCount = workers;
            }

            Debug.Log(report.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "JobChainHops.txt"), report.ToString());
        }
    }
}
