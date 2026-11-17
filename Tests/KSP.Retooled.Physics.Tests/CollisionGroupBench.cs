using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Compare per-pair ignores with collision groups on a cube of overlapping bodies.
    /// Report setup time/memory, pair filtering and whole-step costs.
    /// Configure size, threads and modes with RETOOLED_BENCH_SIDE, RETOOLED_BENCH_THREADS and RETOOLED_BENCH_MODES.
    /// </summary>
    [Explicit("A benchmark, a few minutes long")]
    public class CollisionGroupBench : PhysicsScene
    {
        private const int Repeats = 20;

        [BurstCompile]
        private unsafe struct ShapePairFilterJob : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<IntPtr> First;

            [ReadOnly]
            public NativeArray<IntPtr> Second;

            [NativeDisableUnsafePtrRestriction]
            public StoreGlobals* Globals;

            public NativeArray<byte> Ignored;

            public void Execute(int index)
            {
                Ignored[index] = IgnoreTable.Ignores(Globals, (ShapeData*)First[index], (ShapeData*)Second[index]) ? (byte)1 : (byte)0;
            }
        }

        private static string Setting(string name, string fallback)
        {
            string value = Environment.GetEnvironmentVariable(name);
            return string.IsNullOrEmpty(value) ? fallback : value;
        }

        private List<Rigidbody> Cube(int side)
        {
            var bodies = new List<Rigidbody>(side * side * side);
            for (int x = 0; x < side; x++)
            {
                for (int y = 0; y < side; y++)
                {
                    for (int z = 0; z < side; z++)
                    {
                        GameObject box = Box(new Vector3(x, y + 100.0f, z), new Vector3(1.02f, 1.02f, 1.02f));
                        Rigidbody body = box.AddComponent<Rigidbody>();
                        body.useGravity = false;
                        body.sleepThreshold = 0.0f;
                        bodies.Add(body);
                    }
                }
            }

            return bodies;
        }

        private static List<(int, int)> Neighbours(int side)
        {
            var pairs = new List<(int, int)>();
            int Index(int x, int y, int z) => (x * side + y) * side + z;
            for (int x = 0; x < side; x++)
            {
                for (int y = 0; y < side; y++)
                {
                    for (int z = 0; z < side; z++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            for (int dy = -1; dy <= 1; dy++)
                            {
                                for (int dz = -1; dz <= 1; dz++)
                                {
                                    int nx = x + dx, ny = y + dy, nz = z + dz;
                                    if (nx < 0 || ny < 0 || nz < 0 || nx >= side || ny >= side || nz >= side)
                                    {
                                        continue;
                                    }

                                    int a = Index(x, y, z), b = Index(nx, ny, nz);
                                    if (a < b)
                                    {
                                        pairs.Add((a, b));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return pairs;
        }

        private static string Stats(List<double> samples)
        {
            samples.Sort();
            return string.Format(CultureInfo.InvariantCulture, "min {0:F3} ms, median {1:F3} ms", samples[0], samples[samples.Count / 2]);
        }

        [Test]
        public unsafe void Measure()
        {
            Assume.That(CollisionGroups.Available, "needs --backend retooled");
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");

            int side = int.Parse(Setting("RETOOLED_BENCH_SIDE", "17"), CultureInfo.InvariantCulture);
            int threads = int.Parse(Setting("RETOOLED_BENCH_THREADS", "8"), CultureInfo.InvariantCulture);
            int workers = JobsUtility.JobWorkerCount;
            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
            var report = new StringBuilder();
            report.AppendLine($"CollisionGroupBench: {side * side * side} boxes, {threads} threads, {Environment.ProcessorCount} logical CPUs");
            try
            {
                foreach (string mode in Setting("RETOOLED_BENCH_MODES", "groups,pairs").Split(','))
                {
                    Run(mode, side, report);
                    SetUpScene();
                }
            }
            finally
            {
                JobsUtility.JobWorkerCount = workers;
                Debug.Log(report.ToString());
                TestContext.Out.WriteLine(report.ToString());
            }
        }

        private unsafe void Run(string mode, int side, StringBuilder report)
        {
            StoreGlobals* globals = NativeStore.RetooledPhysics_Globals();
            List<Rigidbody> bodies = Cube(side);
            Collider[] colliders = bodies.Select(b => b.GetComponent<Collider>()).ToArray();

            // No step runs before the set-up, because the overlapping boxes would push each other apart.

            ulong tableBefore = IgnoreTable.Bytes(globals) + GroupTable.Bytes(globals);
            long start = Stopwatch.GetTimestamp();
            long calls = 0;
            if (mode == "groups")
            {
                for (int i = 0; i < colliders.Length; i++)
                {
                    CollisionGroups.Set(colliders[i], 1, (uint)i + 1, false);
                    calls++;
                }
            }
            else
            {
                for (int i = 0; i < colliders.Length; i++)
                {
                    for (int j = i + 1; j < colliders.Length; j++)
                    {
                        UnityEngine.Physics.IgnoreCollision(colliders[i], colliders[j]);
                        calls++;
                    }
                }
            }

            double setupMs = (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
            ulong tableAfter = IgnoreTable.Bytes(globals) + GroupTable.Bytes(globals);
            report.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "{0}: set-up {1} calls in {2:F1} ms; store tables {3:F1} MB -> {4:F1} MB; ignored pairs {5}",
                mode, calls, setupMs, tableBefore / 1048576.0, tableAfter / 1048576.0, IgnoreTable.PairCount(globals)));

            // The pair filter over every overlapping body pair, as DropIgnoredPairsJob runs it when its cache is cold.
            List<(int, int)> neighbours = Neighbours(side);
            var first = new NativeArray<IntPtr>(neighbours.Count, Allocator.TempJob);
            var second = new NativeArray<IntPtr>(neighbours.Count, Allocator.TempJob);
            var ignored = new NativeArray<byte>(neighbours.Count, Allocator.TempJob);
            for (int i = 0; i < neighbours.Count; i++)
            {
                first[i] = (IntPtr)GroupTable.ShapeOf(globals, EntityId.ToULong(colliders[neighbours[i].Item1].GetEntityId()));
                second[i] = (IntPtr)GroupTable.ShapeOf(globals, EntityId.ToULong(colliders[neighbours[i].Item2].GetEntityId()));
                Assert.That(first[i] != IntPtr.Zero && second[i] != IntPtr.Zero, "every collider has a shape");
            }

            var filterSamples = new List<double>();
            for (int r = 0; r < Repeats + 2; r++)
            {
                long filterStart = Stopwatch.GetTimestamp();
                new ShapePairFilterJob { First = first, Second = second, Globals = globals, Ignored = ignored }
                    .Schedule(neighbours.Count, 256).Complete();
                if (r >= 2)
                {
                    filterSamples.Add((Stopwatch.GetTimestamp() - filterStart) * 1000.0 / Stopwatch.Frequency);
                }
            }

            int ignoredCount = ignored.Count(b => b != 0);
            first.Dispose();
            second.Dispose();
            ignored.Dispose();
            report.AppendLine($"{mode}: pair filter over {neighbours.Count} overlapping pairs ({ignoredCount} ignored): {Stats(filterSamples)}");
            Assert.That(ignoredCount, Is.EqualTo(neighbours.Count), "every pair is ignored");

            // Whole steps, first with the filter's cache warm and then with it cold, as after every change to the
            // ignored pairs.
            Simulate(5);
            var warm = new List<double>();
            for (int r = 0; r < Repeats; r++)
            {
                long stepStart = Stopwatch.GetTimestamp();
                UnityEngine.Physics.Simulate(Step);
                warm.Add((Stopwatch.GetTimestamp() - stepStart) * 1000.0 / Stopwatch.Frequency);
            }

            var cold = new List<double>();
            for (int r = 0; r < Repeats; r++)
            {
                globals->ignoreVersion++;
                long stepStart = Stopwatch.GetTimestamp();
                UnityEngine.Physics.Simulate(Step);
                cold.Add((Stopwatch.GetTimestamp() - stepStart) * 1000.0 / Stopwatch.Frequency);
            }

            report.AppendLine($"{mode}: whole step, cache warm: {Stats(warm)}; cache cold: {Stats(cold)}");
            AssertVector(new Vector3(0.0f, 100.0f, 0.0f), bodies[0].position, 0.001f, "the boxes stay where they are");

            long teardownStart = Stopwatch.GetTimestamp();
            TearDownScene();

            double teardownMs = (Stopwatch.GetTimestamp() - teardownStart) * 1000.0 / Stopwatch.Frequency;
            report.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0}: destroying the boxes took {1:F1} ms; ignored pairs left {2}",
                mode, teardownMs, IgnoreTable.PairCount(globals)));
        }
    }
}
