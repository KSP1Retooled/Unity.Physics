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

namespace KSP.Retooled.Physics.Tests
{
    // A profiling harness. It builds the VesselForestBench crafts, steps them with the editor profiler on, and reads the
    // profiler's raw frames for every thread. For each marker it reports the first start, the last end and the summed
    // CPU inside each Physics.Simulate call, as medians per step. Rows that start with "M" are the main thread, rows that
    // start with "A" merge all threads. PROFILE_ONLY picks configurations by name, PROFILE_PATHS picks the paths
    // (compiled by default), and PROFILE_TAG names the output file Logs/Timeline-<tag>.txt. PROFILE_SAVE=1 also saves
    // each configuration's profiler frames as Logs/Timeline-<tag>-<configuration>.data, which the Profiler window opens.
    [Explicit("profiling")]
    [Category("Slow")]
    public class ZzTimeline : PhysicsScene
    {
        private const int PartLayer = 30;
        private const int Profiled = 30;

        private struct Graph
        {
            public int[] Parent;
            public float[] Mass;
            public Vector3[] Inertia;
            public Vector3[] Position;
            public Vector3[] Joint;
            public float[] Swing;
            public int[] ClosureA;
            public int[] ClosureB;
        }

        private static Graph Load(string name)
        {
            string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "io.github.ksp1retooled.physics",
                "Tests", "VesselGraphs~"));
            using var file = File.OpenRead(Path.Combine(folder, name + ".txt.gz"));
            using var unzip = new GZipStream(file, CompressionMode.Decompress);
            using var reader = new StreamReader(unzip);
            reader.ReadLine();
            int count = int.Parse(reader.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
            var graph = new Graph
            {
                Parent = new int[count], Mass = new float[count], Inertia = new Vector3[count], Position = new Vector3[count],
                Joint = new Vector3[count], Swing = new float[count],
            };
            for (int i = 0; i < count; i++)
            {
                string[] f = reader.ReadLine().Split(' ');
                float F(int k) => float.Parse(f[k], CultureInfo.InvariantCulture);
                graph.Parent[i] = int.Parse(f[0], CultureInfo.InvariantCulture);
                graph.Mass[i] = F(2);
                graph.Inertia[i] = new Vector3(F(3), F(4), F(5));
                graph.Position[i] = new Vector3(F(6), F(7), F(8));
                graph.Joint[i] = new Vector3(F(9), F(10), F(11));
                graph.Swing[i] = F(15);
            }

            int closures = int.Parse(reader.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
            graph.ClosureA = new int[closures];
            graph.ClosureB = new int[closures];
            for (int c = 0; c < closures; c++)
            {
                string[] f = reader.ReadLine().Split(' ');
                graph.ClosureA[c] = int.Parse(f[0], CultureInfo.InvariantCulture);
                graph.ClosureB[c] = int.Parse(f[1], CultureInfo.InvariantCulture);
            }

            return graph;
        }

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

        private void Build(Graph graph, bool ground)
        {
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
                Box(new Vector3(0.0f, lowest - 0.5f, 0.0f), new Vector3(400.0f, 1.0f, 400.0f));
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
            bool wasForest = VesselForest.Enabled;
            bool wasLocking = VesselForest.Locking;
            bool wasVessels = VesselPipeline.Enabled;
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            VesselForest.Initialize();
            report.Append($"CPU load at start {SampleLoad():F1}%\n");
            UnityEngine.Profiling.Profiler.maxUsedMemory = 1 << 30;
            foreach (string craft in new[] { "tu160", "synthetic4500" })
            {
                Graph graph = Load(craft);
                foreach (bool ground in new[] { false, true })
                {
                    foreach (int threads in new[] { 8, 1 })
                    {
                        foreach (string path in (Environment.GetEnvironmentVariable("PROFILE_PATHS") ?? "compiled").Split(','))
                        {
                            string name = $"{craft} {(ground ? "ground" : "floating")} {threads}T {path}";
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

                            VesselForest.Enabled = path != "sparse";
                            VesselPipeline.Enabled = options[0] == "compiled";
                            VesselForest.ForgetIslands();
                            VesselForest.Locking = path == "locked";
                            VesselForest.BoundaryRowCap = VesselForest.DefaultBoundaryRowCap;
                            VesselForest.MinimumHoldSteps = -1;
                            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
                            Build(graph, ground);
                            for (int i = 0; i < 30; i++)
                            {
                                UnityEngine.Physics.Simulate(Step);
                            }

                            VesselForest.GetCounts(out long solves0, out _, out _);
                            VesselForest.GetSectionCounts(out long sections0, out long locked0, out long builds0, out _, out _, out _, out _, out _);
                            var unprofiled = new List<double>();
                            for (int i = 0; i < 20; i++)
                            {
                                long start = Stopwatch.GetTimestamp();
                                UnityEngine.Physics.Simulate(Step);
                                unprofiled.Add((Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
                            }

                            VesselForest.GetCounts(out long solves1, out _, out _);
                            VesselForest.GetSectionCounts(out long sections1, out long locked1, out long builds1, out _, out _, out _, out _, out _);
                            long solved = Math.Max(1, solves1 - solves0);
                            report.Append($"# {craft} {(ground ? "ground" : "floating")} {threads}T: {solves1 - solves0} vessel solves in 20 steps, " +
                                $"{(sections1 - sections0) / (double)solved:F0} sections and {(locked1 - locked0) / (double)solved:F0} locked edges per solve, " +
                                $"{builds1 - builds0} structure builds\n");
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

                            for (int s = 0; s < Profiled; s++)
                            {
                                UnityEngine.Physics.Simulate(Step);
                                yield return null;
                            }

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

            VesselForest.Enabled = wasForest;
            VesselForest.Locking = wasLocking;
            VesselPipeline.Enabled = wasVessels;
            Unity.Physics.Solver.FusedContacts = true;
            Unity.Physics.Solver.FusedVesselParts = Unity.Physics.Solver.DefaultFusedVesselParts;
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, !collided);
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", $"Timeline-{tag}.txt"), report.ToString());
            Debug.Log("[ZzTimeline] done");
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
    }
}
