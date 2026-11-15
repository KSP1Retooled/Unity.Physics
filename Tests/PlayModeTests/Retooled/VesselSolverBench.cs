using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Physics;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The section solver spec's falsifier measurement: whole steps of <see cref="VesselSolver"/> on the reference craft
    /// and the synthetic 4,500-body rocket under each lock rule, on one thread and on eight (seven job workers and the
    /// waiting main thread), back to back ("warm") and with 2 ms of unrelated memory traffic between steps ("cold"). Each
    /// session starts only once the machine's total CPU load is under 15%, waiting up to ten minutes, and reports the
    /// load it started at. Results go to the log and to VesselSolverBench.txt in the project's Logs folder.
    /// </summary>
    [Category("Slow")]
    public unsafe class VesselSolverBench
    {
        const int WarmSteps = 400;
        const int ColdSteps = 200;
        const double GapMilliseconds = 2.0;
        const int Toggled = 36;

        delegate void StreamKernel(long* data, long count);

        [BurstCompile(CompileSynchronously = true)]
        static class Traffic
        {
            /// <summary>Reads and writes every element: what the rest of a frame does to the caches between two steps.</summary>
            [BurstCompile(CompileSynchronously = true)]
            public static void Stream(long* data, long count)
            {
                long sum = 0;
                for (long i = 0; i < count; i++)
                {
                    sum += data[i];
                    data[i] = sum;
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct ProbeJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public bool* Managed;

            public void Execute() => Mark();

            [BurstDiscard]
            void Mark() => *Managed = true;
        }

        /// <summary>Total CPU load in percent over three seconds, from the Windows performance counter.</summary>
        static double SampleLoad()
        {
            var start = new ProcessStartInfo("powershell.exe",
                "-NoProfile -Command \"(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 3)." +
                "CounterSamples | Measure-Object -Property CookedValue -Average | ForEach-Object { $_.Average }\"")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };
            using Process process = Process.Start(start);
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            return double.TryParse(output.Trim(), NumberStyles.Float, CultureInfo.CurrentCulture, out double load) ? load
                : double.TryParse(output.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out load) ? load : -1.0;
        }

        /// <summary>Waits for a quiet machine, up to ten minutes, and returns the load it started at.</summary>
        static double WaitForQuiet()
        {
            var watch = Stopwatch.StartNew();
            double load = SampleLoad();
            while (load > 15.0 && watch.Elapsed.TotalMinutes < 10.0)
            {
                System.Threading.Thread.Sleep(20000);
                load = SampleLoad();
            }

            return load;
        }

        static (double min, double median) Summary(List<double> samples)
        {
            var sorted = samples.OrderBy(x => x).ToList();
            return (sorted[0], sorted[sorted.Count / 2]);
        }

        enum Case
        {
            LiveJointsOnly,
            StackLocked,
            AllYielding,
            Switching,
        }

        [Test]
        [Timeout(7200000)]
        public void Measure()
        {
            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            bool managed = false;
            new ProbeJob { Managed = &managed }.Run();
            Assert.IsFalse(managed, "Jobs are not running in Burst");
            NativeLeakDetection.Mode = NativeLeakDetectionMode.Disabled;
            bool jobDebugger = JobsUtility.JobDebuggerEnabled;
            JobsUtility.JobDebuggerEnabled = false;
            int workers = JobsUtility.JobWorkerCount;

            var traffic = new NativeArray<long>(8 << 20, Allocator.Persistent);
            StreamKernel stream = BurstCompiler.CompileFunctionPointer<StreamKernel>(Traffic.Stream).Invoke;
            var report = new StringBuilder("[VesselSolverBench] whole step, microseconds min / median; load is total CPU % at " +
                "the start of each session\n");
            report.Append("craft | case | sections | live links | chunks | top | threads | warm | cold | load\n");
            var stages = new StringBuilder("[VesselSolverBench] one thread, warm, median microseconds by stage: sections | " +
                "factor chunks | top | substitute chunks | loads and integration\n");
            try
            {
                foreach (string craft in new[] { "tu160", "epsilon", "synthetic4500", "behemoth", "pegasus" })
                {
                    VesselGraph graph = VesselGraph.Load(craft);
                    foreach (Case kind in Enum.GetValues(typeof(Case)))
                    {
                        foreach (int threads in new[] { 1, 8 })
                        {
                            Session(graph, kind, threads, workers, traffic, stream, report, stages,
                                VesselSolverSettings.Default.ChunkTarget);
                        }
                    }
                }

                // How the cut into chunks moves the pass/fail case: fewer chunks leave less to the serial top.
                VesselGraph synthetic = VesselGraph.Load("synthetic4500");
                foreach (int chunkTarget in new[] { 16, 32, 48, 192 })
                {
                    Session(synthetic, Case.AllYielding, 8, workers, traffic, stream, report, stages, chunkTarget);
                }
            }
            finally
            {
                JobsUtility.JobWorkerCount = workers;
                JobsUtility.JobDebuggerEnabled = jobDebugger;
                traffic.Dispose();
            }

            Debug.Log(report.ToString());
            Debug.Log(stages.ToString());
            string path = Path.GetFullPath(Path.Combine(UnityEngine.Application.dataPath, "..", "Logs", "VesselSolverBench.txt"));
            File.WriteAllText(path, report + "\n" + stages);
        }

        static void Session(VesselGraph graph, Case kind, int threads, int workers, NativeArray<long> traffic,
            StreamKernel stream, StringBuilder report, StringBuilder stages, int chunkTarget)
        {
            VesselGraph.LockRule rule = kind switch
            {
                Case.StackLocked => VesselGraph.LockRule.StackLocked,
                Case.AllYielding => VesselGraph.LockRule.AllYielding,
                _ => VesselGraph.LockRule.LiveJointsOnly,
            };
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.ChunkTarget = chunkTarget;
            graph.Build(rule, settings.TimeStep, 1.0, Allocator.Persistent, out var parts, out var edges, out var closures);
            var solver = new VesselSolver(parts, edges, closures, settings);

            // The edges that switch in the switching case: a few dozen locked ones spread through the vessel.
            var toggled = new List<int>();
            if (kind == Case.Switching)
            {
                var locked = Enumerable.Range(1, graph.Count).Where(i => i < graph.Count && edges[i].State == VesselEdgeState.Locked).ToList();
                for (int k = 0; k < Toggled && k < locked.Count; k++)
                {
                    toggled.Add(locked[(int)((long)k * locked.Count / Toggled)]);
                }
            }

            bool unlocked = false;
            void Toggle()
            {
                if (toggled.Count == 0)
                {
                    return;
                }

                unlocked = !unlocked;
                foreach (int i in toggled)
                {
                    VesselEdge edge = edges[i];
                    edge.State = unlocked ? VesselEdgeState.Yielding : VesselEdgeState.Locked;
                    edges[i] = edge;
                }

                solver.MarkStructureChanged();
            }

            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : workers;
            bool useJobs = threads > 1;
            for (int step = 0; step < 20; step++)
            {
                Toggle();
                solver.Step(useJobs);
            }

            double load = WaitForQuiet();
            var warm = new List<double>(WarmSteps);
            var ticks = new long[5];
            var stageSamples = new List<double>[5];
            for (int k = 0; k < 5; k++)
            {
                stageSamples[k] = new List<double>(WarmSteps);
            }

            double tickMicroseconds = 1e6 / Stopwatch.Frequency;
            for (int step = 0; step < WarmSteps; step++)
            {
                Toggle();
                long start = Stopwatch.GetTimestamp();
                solver.Step(useJobs);
                warm.Add((Stopwatch.GetTimestamp() - start) * tickMicroseconds);
            }

            var cold = new List<double>(ColdSteps);
            for (int step = 0; step < ColdSteps; step++)
            {
                long gapStart = Stopwatch.GetTimestamp();
                var data = (long*)traffic.GetUnsafePtr();
                while ((Stopwatch.GetTimestamp() - gapStart) * tickMicroseconds < GapMilliseconds * 1000.0)
                {
                    stream(data + (step * 131072L) % (traffic.Length - 131072), 131072);
                }

                Toggle();
                long start = Stopwatch.GetTimestamp();
                solver.Step(useJobs);
                cold.Add((Stopwatch.GetTimestamp() - start) * tickMicroseconds);
            }

            if (threads == 1 && kind != Case.Switching)
            {
                for (int step = 0; step < 100; step++)
                {
                    solver.RunStages(ticks);
                    for (int k = 0; k < 5; k++)
                    {
                        stageSamples[k].Add(ticks[k] * tickMicroseconds);
                    }
                }

                stages.Append($"{graph.Name} | {kind} | {string.Join(" | ", stageSamples.Select(s => Summary(s).median.ToString("F1")))}\n");
            }

            bool finite = true;
            for (int i = 0; i < parts.Length; i++)
            {
                finite &= Unity.Mathematics.math.all(Unity.Mathematics.math.isfinite(parts[i].Position));
            }

            (double warmMin, double warmMedian) = Summary(warm);
            (double coldMin, double coldMedian) = Summary(cold);
            string label = chunkTarget == VesselSolverSettings.Default.ChunkTarget ? kind.ToString() : $"{kind}, chunk target {chunkTarget}";
            report.Append($"{graph.Name} | {label} | {solver.SectionCount} | {solver.LinkCount} | {solver.ChunkCount} | " +
                $"{solver.TopCount} | {threads} | {warmMin:F0} / {warmMedian:F0} | {coldMin:F0} / {coldMedian:F0} | " +
                $"{load:F0}%{(finite ? "" : " (NOT FINITE)")}\n");
            Debug.Log($"[VesselSolverBench] {graph.Name} {kind} {threads}T: warm {warmMin:F0} / {warmMedian:F0}, cold " +
                $"{coldMin:F0} / {coldMedian:F0} us at {load:F0}% load");
            solver.Dispose();
            parts.Dispose();
            edges.Dispose();
            closures.Dispose();
        }
    }
}
