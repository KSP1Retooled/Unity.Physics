using KSP.Retooled.Physics.Tests.Benchmarks;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Diagnostics;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Unity.Jobs.LowLevel.Unsafe;
using UnityEditor.Profiling;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.TestTools;
using Debug = UnityEngine.Debug;
using Unity.Physics.Configuration;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Tests.Tools
{
    using Graph = VesselForestBench.Graph;

    // Profile craft steps and report median marker start/end times and CPU totals.
    // M rows cover the main thread, A all threads, and O work within 10 ms around Simulate.
    // Use PROFILE_ONLY, _PATHS, _SCENE and _TAG to select configurations and output.
    // PROFILE_SAVE=1 also saves Profiler data files in Logs.
    [Explicit("profiling")]
    [Category("Slow")]
    public class Timeline : PhysicsScene
    {
        private const int PartLayer = 30;
        private const int Profiled = 30;

        private static ConfigurableJoint Joint(Rigidbody host, Rigidbody target, Vector3 anchor, float spring, bool locked)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(anchor);
            joint.connectedAnchor = target.transform.InverseTransformPoint(anchor);
            if (locked)
            {
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
                return joint;
            }

            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
            joint.angularXMotion = ConfigurableJointMotion.Locked;
            joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 1.0f };
            joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
            joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
            joint.xDrive = joint.yDrive = joint.zDrive = new JointDrive { positionSpring = 1e20f, maximumForce = 1e20f };
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularYZDrive = new JointDrive { positionSpring = spring, positionDamper = 0.02f * spring, maximumForce = 1e20f };
            return joint;
        }

        private void Build(Graph graph, string scene)
        {
            bool ground = scene != "floating";
            var bodies = new List<Rigidbody>(graph.Parent.Length);
            float lowest = float.MaxValue;
            for (int i = 0; i < graph.Parent.Length; i++)
            {
                float mass = Mathf.Max(graph.Mass[i], 1e-3f);
                float inertia = Mathf.Max((graph.Inertia[i].x + graph.Inertia[i].y + graph.Inertia[i].z) / 3.0f, 1e-6f);
                float side = Mathf.Clamp(Mathf.Sqrt(6.0f * inertia / mass), 0.05f, 5.0f);
                GameObject part = Box(graph.Position[i], Vector3.one * side);
                part.layer = PartLayer;
                Rigidbody body = part.AddComponent<Rigidbody>();
                body.mass = mass;
                body.sleepThreshold = 0.0f;
                bodies.Add(body);
                lowest = Mathf.Min(lowest, graph.Position[i].y - 0.5f * side);
            }

            for (int i = 0; i < graph.Parent.Length; i++)
            {
                if (graph.Parent[i] >= 0)
                {
                    Joint(bodies[i], bodies[graph.Parent[i]], graph.Joint[i], graph.Swing[i] > 0.0f ? graph.Swing[i] : 1e5f, false);
                }
            }

            for (int c = 0; c < graph.ClosureA.Length; c++)
            {
                int a = graph.ClosureA[c], b = graph.ClosureB[c];
                Joint(bodies[b], bodies[a], 0.5f * (graph.Position[a] + graph.Position[b]), 0.0f, true);
            }

            if (ground)
            {
                // A fleet of copies that reaches past the 400 m ground gets a ground that covers it, as in the bench.
                float left = graph.Position.Min(p => p.x), right = graph.Position.Max(p => p.x);
                bool fits = left > -190.0f && right < 190.0f;
                Box(new Vector3(fits ? 0.0f : 0.5f * (left + right), lowest - 0.5f, 0.0f),
                    new Vector3(fits ? 400.0f : right - left + 100.0f, 1.0f, 400.0f));
            }

            if (scene == VesselForestBench.StaticsScene)
            {
                VesselForestBench.StaticField(graph, lowest, Track);
            }
        }


        private sealed class Stat
        {
            public readonly List<double> First = new List<double>();
            public readonly List<double> Last = new List<double>();
            public readonly List<double> Cpu = new List<double>();
        }

        private static double Median(List<double> values)
        {
            if (values.Count == 0)
            {
                return 0.0;
            }

            var sorted = values.OrderBy(v => v).ToList();
            return sorted[sorted.Count / 2];
        }

        [UnityTest]
        [Timeout(7200000)]
        public IEnumerator Profile()
        {
            string tag = Environment.GetEnvironmentVariable("PROFILE_TAG") ?? "run";
            string only = Environment.GetEnvironmentVariable("PROFILE_ONLY");
            var report = new StringBuilder();
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(PartLayer, PartLayer);
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, true);
            JobsUtility.JobDebuggerEnabled = false;
            Unity.Collections.NativeLeakDetection.Mode = Unity.Collections.NativeLeakDetectionMode.Disabled;
            int workers = JobsUtility.JobWorkerCount;
            bool wasForest = ForestSwitches.Enabled;
            bool wasLocking = ForestSwitches.Locking;
            bool wasVessels = VesselSwitches.Enabled;
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            ForestSwitches.Initialize();
            report.Append($"CPU load at start {SampleLoad():F1}%\n");
            UnityEngine.Profiling.Profiler.maxUsedMemory = 1 << 30;
            // PROFILE_CRAFT lists the crafts, separated by commas, in the names that VesselForestBench reads: "tu160x2"
            // for two copies of the Tu-160, or "tu160:128x16" for sixteen copies of its first 128 parts.
            string crafts = Environment.GetEnvironmentVariable("PROFILE_CRAFT");
            foreach (string craft in string.IsNullOrEmpty(crafts) ? new[] { "tu160", "synthetic4500" } : crafts.Split(','))
            {
                Graph graph = VesselForestBench.Load(craft);
                foreach (string scene in (Environment.GetEnvironmentVariable("PROFILE_SCENE") ?? "floating,ground").Split(','))
                {
                    bool ground = scene != "floating";
                    foreach (int threads in new[] { 8, 1 })
                    {
                        foreach (string path in (Environment.GetEnvironmentVariable("PROFILE_PATHS") ?? "compiled").Split(','))
                        {
                            string name = $"{craft} {scene} {threads}T {path}";
                            if (!string.IsNullOrEmpty(only) && !name.Contains(only))
                            {
                                continue;
                            }

                            TearDownScene();
                            SetUpScene();
                            if (threads == 1 && ground)
                            {
                                continue;
                            }

                            // "compiled:nocontacts" keeps the islands with contacts or joints out of the fused vessel job,
                            // and "compiled:parts=64" sends vessel islands of 64 parts or more to it.
                            string[] options = path.Split(':');
                            Unity.Physics.Solver.FusedContacts = !options.Contains("nocontacts");
                            Unity.Physics.Solver.FusedVesselParts = Unity.Physics.Solver.DefaultFusedVesselParts;
                            foreach (string option in options.Where(o => o.StartsWith("parts=")))
                            {
                                Unity.Physics.Solver.FusedVesselParts = int.Parse(option.Substring("parts=".Length), CultureInfo.InvariantCulture);
                            }

                            ForestSwitches.Enabled = path != "sparse";
                            VesselSwitches.Enabled = options[0] == "compiled";
                            ForestDiagnostics.ForgetIslands();
                            ForestSwitches.Locking = path == "locked";
                            ForestSwitches.BoundaryRowCap = ForestSwitches.DefaultBoundaryRowCap;
                            ForestSwitches.MinimumHoldSteps = -1;
                            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
                            Build(graph, scene);
                            for (int i = 0; i < 30; i++)
                            {
                                UnityEngine.Physics.Simulate(Step);
                            }

                            ForestDiagnostics.GetCounts(out long solves0, out _, out _);
                            ForestDiagnostics.GetSectionCounts(out long sections0, out long locked0, out long builds0, out _, out _, out _, out _, out _);
                            var unprofiled = new List<double>();
                            for (int i = 0; i < 20; i++)
                            {
                                long start = Stopwatch.GetTimestamp();
                                UnityEngine.Physics.Simulate(Step);
                                unprofiled.Add((Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
                            }

                            // Yield an editor frame between these steps so that pending background work can
                            // finish before the next step.
                            var framed = new List<double>();
                            for (int i = 0; i < 20; i++)
                            {
                                yield return null;
                                long start = Stopwatch.GetTimestamp();
                                UnityEngine.Physics.Simulate(Step);
                                framed.Add((Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
                            }

                            ForestDiagnostics.GetCounts(out long solves1, out _, out _);
                            ForestDiagnostics.GetSectionCounts(out long sections1, out long locked1, out long builds1, out _, out _, out _, out _, out _);
                            long solved = Math.Max(1, solves1 - solves0);
                            report.Append($"# {craft} {scene} {threads}T: {solves1 - solves0} vessel solves in 40 steps, " +
                                $"{(sections1 - sections0) / (double)solved:F0} sections and {(locked1 - locked0) / (double)solved:F0} locked edges per solve, " +
                                $"{builds1 - builds0} structure builds\n");
                            report.Append($"# unprofiled step median: {Median(unprofiled):F2} ms back to back, {Median(framed):F2} ms with an editor frame between steps\n");
                            ProfilerDriver.ClearAllFrames();
                            ProfilerDriver.profileEditor = true;
                            ProfilerDriver.enabled = true;
                            UnityEngine.Profiling.Profiler.enabled = true;
                            yield return null;
                            yield return null;

                            // On a busy machine the profiler can take many frames to start recording, and the first
                            // configuration then had no profiled steps. Wait until it has recorded two frames.
                            int started = ProfilerDriver.lastFrameIndex;
                            for (int wait = 0; wait < 600 && ProfilerDriver.lastFrameIndex < started + 2; wait++)
                            {
                                yield return null;
                            }

                            var timedWhileProfiled = new List<double>();
                            for (int s = 0; s < Profiled; s++)
                            {
                                long start = Stopwatch.GetTimestamp();
                                UnityEngine.Physics.Simulate(Step);
                                timedWhileProfiled.Add((Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
                                yield return null;
                            }

                            report.Append($"# profiled steps by the stopwatch: median {Median(timedWhileProfiled):F2} ms\n");
                            yield return null;
                            UnityEngine.Profiling.Profiler.enabled = false;
                            ProfilerDriver.enabled = false;
                            JobsUtility.JobWorkerCount = workers;
                            report.Append(Analyse(name, Median(unprofiled)));
                            if (Environment.GetEnvironmentVariable("PROFILE_SAVE") == "1")
                            {
                                // The saved capture opens in the Profiler window (Load). Its name has the tag and the
                                // configuration, for example Logs/Timeline-run-tu160-ground-8T-compiled.data.
                                string capture = $"Timeline-{tag}-{name.Replace(' ', '-').Replace(':', '_')}.data";
                                ProfilerDriver.SaveProfile(Path.Combine(Application.dataPath, "..", "Logs", capture));
                            }

                            ProfilerDriver.ClearAllFrames();
                        }
                    }
                }
            }

            ForestSwitches.Enabled = wasForest;
            ForestSwitches.Locking = wasLocking;
            VesselSwitches.Enabled = wasVessels;
            Unity.Physics.Solver.FusedContacts = true;
            Unity.Physics.Solver.FusedVesselParts = Unity.Physics.Solver.DefaultFusedVesselParts;
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, !collided);
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", $"Timeline-{tag}.txt"), report.ToString());
            Debug.Log("[Timeline] done");
        }

        private static double SampleLoad()
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

        private static string Analyse(string name, double unprofiled)
        {
            var stats = new Dictionary<string, Stat>();
            var steps = new List<double>();
            int first = ProfilerDriver.firstFrameIndex;
            int last = ProfilerDriver.lastFrameIndex;
            for (int frame = first; frame <= last; frame++)
            {
                ulong windowStart = 0;
                ulong windowEnd = 0;
                using (RawFrameDataView main = ProfilerDriver.GetRawFrameDataView(frame, 0))
                {
                    if (!main.valid)
                    {
                        continue;
                    }

                    for (int i = 0; i < main.sampleCount; i++)
                    {
                        if (main.GetSampleName(i) == "Physics.Simulate")
                        {
                            windowStart = main.GetSampleStartTimeNs(i);
                            windowEnd = windowStart + (ulong)(main.GetSampleTimeMs(i) * 1e6);
                            break;
                        }
                    }
                }

                if (windowStart == 0)
                {
                    continue;
                }

                steps.Add((windowEnd - windowStart) / 1e6);

                // The main thread's work within 10 ms before and after the step's marker. Physics.Simulate does work
                // of its own around the backend's step, which the stopwatch counts and the marker doesn't.
                using (RawFrameDataView main = ProfilerDriver.GetRawFrameDataView(frame, 0))
                {
                    var around = new Dictionary<string, (double from, double ms)>();
                    for (int i = 0; i < main.sampleCount; i++)
                    {
                        ulong start = main.GetSampleStartTimeNs(i);
                        double ms = main.GetSampleTimeMs(i);
                        ulong end = start + (ulong)(ms * 1e6);
                        bool before = end <= windowStart && start + 10_000_000 >= windowStart;
                        bool after = start >= windowEnd && start <= windowEnd + 10_000_000;
                        if (!before && !after)
                        {
                            continue;
                        }

                        double from = before ? -((windowStart - start) / 1e6) : (start - windowStart) / 1e6;
                        string key = "O " + main.GetSampleName(i);
                        around[key] = around.TryGetValue(key, out var have) ? (Math.Min(have.from, from), have.ms + ms) : (from, ms);
                    }

                    foreach (var pair in around)
                    {
                        if (!stats.TryGetValue(pair.Key, out Stat stat))
                        {
                            stat = new Stat();
                            stats[pair.Key] = stat;
                        }

                        stat.First.Add(pair.Value.from);
                        stat.Last.Add(pair.Value.from + pair.Value.ms);
                        stat.Cpu.Add(pair.Value.ms);
                    }
                }

                var seen = new Dictionary<string, (double first, double last, double cpu)>();
                for (int thread = 0; ; thread++)
                {
                    using RawFrameDataView view = ProfilerDriver.GetRawFrameDataView(frame, thread);
                    if (!view.valid)
                    {
                        break;
                    }

                    string prefix = thread == 0 ? "M " : "  ";
                    for (int i = 0; i < view.sampleCount; i++)
                    {
                        ulong start = view.GetSampleStartTimeNs(i);
                        if (start < windowStart || start > windowEnd)
                        {
                            continue;
                        }

                        double ms = view.GetSampleTimeMs(i);
                        double from = (start - windowStart) / 1e6;
                        string key = prefix + view.GetSampleName(i);
                        foreach (string k in new[] { key, "A " + view.GetSampleName(i) })
                        {
                            if (seen.TryGetValue(k, out var have))
                            {
                                seen[k] = (Math.Min(have.first, from), Math.Max(have.last, from + ms), have.cpu + ms);
                            }
                            else
                            {
                                seen[k] = (from, from + ms, ms);
                            }
                        }
                    }
                }

                foreach (var pair in seen)
                {
                    if (!stats.TryGetValue(pair.Key, out Stat stat))
                    {
                        stat = new Stat();
                        stats[pair.Key] = stat;
                    }

                    stat.First.Add(pair.Value.first);
                    stat.Last.Add(pair.Value.last);
                    stat.Cpu.Add(pair.Value.cpu);
                }
            }

            var text = new StringBuilder();
            text.Append(WhileParked(first, last));
            text.Append(SolveTail(first, last));
            text.Append($"## {name}: profiled step median {Median(steps):F2} ms over {steps.Count}, unprofiled median {unprofiled:F2}\n");
            text.Append("   first     last     span      cpu  marker\n");
            foreach (var pair in stats.OrderBy(p => Median(p.Value.First)))
            {
                double cpu = pair.Value.Cpu.Sum() / Math.Max(1, steps.Count);
                double span = Median(pair.Value.Last) - Median(pair.Value.First);
                // Keep the short jobs that find and index the islands, so that their cost shows even when it is small.
                bool islandStage = pair.Key.Contains("Island") || pair.Key.Contains("DispatchPair");
                if (cpu < 0.02 && span < 0.05 && !islandStage)
                {
                    continue;
                }

                text.Append($"{Median(pair.Value.First),8:F3} {Median(pair.Value.Last),8:F3} {span,8:F3} {cpu,8:F3}  {pair.Key}\n");
            }

            return text.ToString();
        }

        /// <summary>
        /// The jobs that the substeps of the direct islands run, and the jobs that free their buffers. SolveTail looks
        /// for them after the fused vessel job.
        /// </summary>
        private static readonly string[] SubstepJobs =
        {
            "Solver:UpdateIslandJacobiansJob", "Solver:CountIslandRowsJob", "Solver:BuildIslandRowsJob",
            "Solver:SolveDirectIslandJob", "Solver:SolveSharedDirectIslandsJob", "Solver:IntegrateIslandBodiesJob",
            "Solver:SolveUnfusedSubstepsJob",
        };

        /// <summary>
        /// Report solve completion and write-back timing, with the intervening jobs for the first three frames.
        /// </summary>
        private static string SolveTail(int first, int last)
        {
            const string fusedJob = "Solver:SolveFusedVesselIslandsJob (Burst)";
            const string writeBackJob = "StoreBodies.WriteBackJob (Burst)";
            var chainEnds = new List<double>();
            var writeBacks = new List<double>();
            var writeBackEnds = new List<double>();
            var stepEnds = new List<double>();
            var substepJobCounts = new List<double>();
            var listing = new StringBuilder();
            int listed = 0;
            for (int frame = first; frame <= last; frame++)
            {
                var samples = new List<(int thread, ulong start, ulong end, string name)>();
                for (int thread = 0; ; thread++)
                {
                    using RawFrameDataView view = ProfilerDriver.GetRawFrameDataView(frame, thread);
                    if (!view.valid)
                    {
                        break;
                    }

                    for (int i = 0; i < view.sampleCount; i++)
                    {
                        string name = view.GetSampleName(i);
                        bool stepMarker = thread == 0 && name != null && name.StartsWith("Retooled.Physics.Step");
                        if (name == null || !(name.EndsWith("(Burst)") || name.EndsWith("(Cleanup)") || stepMarker))
                        {
                            continue;
                        }

                        ulong start = view.GetSampleStartTimeNs(i);
                        samples.Add((thread, start, start + (ulong)(view.GetSampleTimeMs(i) * 1e6), name));
                    }
                }

                ulong fusedEnd = samples.Where(s => s.name == fusedJob).Select(s => s.end).DefaultIfEmpty(0UL).Max();
                ulong writeBack = samples.Where(s => s.name == writeBackJob).Select(s => s.start).DefaultIfEmpty(0UL).Min();
                if (fusedEnd == 0 || writeBack == 0 || writeBack < fusedEnd - 5_000_000)
                {
                    continue;
                }

                var chain = samples.Where(s => SubstepJobs.Any(j => s.name.StartsWith(j)) && s.start >= fusedEnd - 5_000_000 && s.start < writeBack).ToList();
                ulong chainEnd = chain.Select(s => s.end).DefaultIfEmpty(fusedEnd).Max();
                chainEnds.Add(((long)chainEnd - (long)fusedEnd) / 1e6);
                writeBacks.Add(((long)writeBack - (long)fusedEnd) / 1e6);
                writeBackEnds.Add(((long)samples.Where(s => s.name == writeBackJob).Max(s => s.end) - (long)fusedEnd) / 1e6);
                ulong stepEnd = samples.Where(s => s.name == "Retooled.Physics.Step").Select(s => s.end).DefaultIfEmpty(0UL).Max();
                stepEnds.Add(((long)stepEnd - (long)fusedEnd) / 1e6);
                substepJobCounts.Add(chain.Count(s => s.name.EndsWith("(Burst)")));
                if (listed < 3)
                {
                    listed++;
                    listing.Append($"#   frame {frame}, times in ms from the end of the fused job:\n");
                    foreach (var s in samples.Where(s => s.end + 50_000 >= fusedEnd && s.start <= writeBack).OrderBy(s => s.start))
                    {
                        listing.Append($"#     {((long)s.start - (long)fusedEnd) / 1e6,8:F3} {(s.end - s.start) / 1e6,7:F3}  thread {s.thread,2}  {s.name}\n");
                    }
                }
            }

            if (writeBacks.Count == 0)
            {
                return "";
            }

            return $"# solve tail over {writeBacks.Count} frames, medians in ms from the end of the fused job: last substep job ends " +
                $"{Median(chainEnds):F3}, StoreBodies.WriteBackJob starts {Median(writeBacks):F3} and ends {Median(writeBackEnds):F3}, the step " +
                $"(Retooled.Physics.Step) ends {Median(stepEnds):F3}; substep jobs that start within 5 ms before the " +
                $"fused job ends or after it: {Median(substepJobCounts):F0}\n" + listing;
        }

        /// <summary>
        /// Report exclusive marker time while at least two solver lanes wait for their leader, in milliseconds per
        /// frame.
        /// </summary>
        private static string WhileParked(int first, int last)
        {
            const string park = "Retooled.Physics.Island.Park";
            var total = new Dictionary<string, double>();
            double parkedWindow = 0.0;
            int frames = 0;
            for (int frame = first; frame <= last; frame++)
            {
                var events = new List<(ulong time, int change)>();
                var views = new List<RawFrameDataView>();
                for (int thread = 0; ; thread++)
                {
                    RawFrameDataView view = ProfilerDriver.GetRawFrameDataView(frame, thread);
                    if (!view.valid)
                    {
                        view.Dispose();
                        break;
                    }

                    views.Add(view);
                    for (int i = 0; i < view.sampleCount; i++)
                    {
                        if (view.GetSampleName(i) == park)
                        {
                            ulong start = view.GetSampleStartTimeNs(i);
                            events.Add((start, 1));
                            events.Add((start + (ulong)(view.GetSampleTimeMs(i) * 1e6), -1));
                        }
                    }
                }

                // The windows in which two or more lanes are parked.
                var windows = new List<(ulong from, ulong to)>();
                events.Sort((a, b) => a.time != b.time ? a.time.CompareTo(b.time) : a.change.CompareTo(b.change));
                int parked = 0;
                ulong opened = 0;
                foreach (var e in events)
                {
                    int before = parked;
                    parked += e.change;
                    if (before < 2 && parked >= 2)
                    {
                        opened = e.time;
                    }
                    else if (before >= 2 && parked < 2)
                    {
                        windows.Add((opened, e.time));
                    }
                }

                if (windows.Count > 0)
                {
                    frames++;
                    foreach (var w in windows)
                    {
                        parkedWindow += (w.to - w.from) / 1e6;
                    }

                    foreach (RawFrameDataView view in views)
                    {
                        int index = 0;
                        while (index < view.sampleCount)
                        {
                            index = AddSelfOverlap(view, index, windows, total);
                        }
                    }
                }

                foreach (RawFrameDataView view in views)
                {
                    view.Dispose();
                }
            }

            var text = new StringBuilder();
            if (frames == 0)
            {
                return "";
            }

            text.Append($"# while two or more lanes are parked: {parkedWindow / frames:F3} ms per frame; own time of each marker in those windows, all threads:\n");
            // Each thread's root sample has no name and covers the whole frame, so its own time is mostly the thread's
            // idle time; leave it out.
            foreach (var pair in total.Where(p => p.Key != park && p.Key != "(a sample without a name)").OrderByDescending(p => p.Value).Take(25))
            {
                text.Append($"#   {pair.Value / frames,8:F3}  {pair.Key}\n");
            }

            return text.ToString();
        }

        /// <summary>
        /// Accumulate exclusive sample time within profiling windows, recurse into children and return the next
        /// subtree index.
        /// </summary>
        private static int AddSelfOverlap(RawFrameDataView view, int index, List<(ulong from, ulong to)> windows, Dictionary<string, double> total)
        {
            ulong start = view.GetSampleStartTimeNs(index);
            ulong end = start + (ulong)(view.GetSampleTimeMs(index) * 1e6);
            double own = Overlap(start, end, windows);
            int children = view.GetSampleChildrenCount(index);
            int next = index + 1;
            for (int c = 0; c < children && next < view.sampleCount; c++)
            {
                ulong childStart = view.GetSampleStartTimeNs(next);
                ulong childEnd = childStart + (ulong)(view.GetSampleTimeMs(next) * 1e6);
                own -= Overlap(childStart, childEnd, windows);
                next = AddSelfOverlap(view, next, windows, total);
            }

            string name = view.GetSampleName(index);
            name = string.IsNullOrEmpty(name) ? "(a sample without a name)" : name;
            total[name] = (total.TryGetValue(name, out double have) ? have : 0.0) + Math.Max(0.0, own);
            return next;
        }

        private static double Overlap(ulong start, ulong end, List<(ulong from, ulong to)> windows)
        {
            double ms = 0.0;
            foreach (var w in windows)
            {
                ulong from = Math.Max(start, w.from), to = Math.Min(end, w.to);
                if (to > from)
                {
                    ms += (to - from) / 1e6;
                }
            }

            return ms;
        }
    }
}
