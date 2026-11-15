using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Unity.Burst;
using Unity.Jobs.LowLevel.Unsafe;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Whole physics steps of a large craft built from a vessel graph (Tests/VesselGraphs~) as Rigidbodies and stock-style
    /// ConfigurableJoints, on the sparse path, the forest path and the forest path with locking, on one thread and on eight,
    /// warm (RETOOLED_BENCH_CRAFT, _SCENE, _THREADS and _PATHS pick part of the table). The craft floats
    /// (every row a forest row) or rests on the ground (contacts make a boundary). Parts never collide with each other.
    /// Reports the machine's CPU load when it starts. Results go to the log and to VesselForestBench.txt in Logs.
    /// </summary>
    [Explicit("A benchmark, minutes long")]
    [Category("Slow")]
    public class VesselForestBench : PhysicsScene
    {
        private const int PartLayer = 30;
        private const int WarmUp = 20;
        private const int Measured = 60;

        private struct Graph
        {
            public string Name;
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
            var graph = new Graph { Name = reader.ReadLine().Substring("vessel ".Length) };
            int count = int.Parse(reader.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
            graph.Parent = new int[count];
            graph.Mass = new float[count];
            graph.Inertia = new Vector3[count];
            graph.Position = new Vector3[count];
            graph.Joint = new Vector3[count];
            graph.Swing = new float[count];
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

        /// <summary>Total CPU load in percent over three seconds, from the Windows performance counter.</summary>
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

        private List<Rigidbody> Build(Graph graph, bool ground)
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
                int parent = graph.Parent[i];
                if (parent >= 0)
                {
                    Joint(bodies[i], bodies[parent], graph.Joint[i], graph.Swing[i] > 0.0f ? graph.Swing[i] : 1e5f, false);
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

            return bodies;
        }

        private string lastTimes;

        private (double min, double median, long solves, long fallbacks) Session(Graph graph, bool ground, string path, int threads,
            int workers)
        {
            TearDownScene();
            SetUpScene();
            VesselForest.Enabled = path != "sparse";
            VesselForest.Locking = path == "locked";
            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
            Build(graph, ground);
            Simulate(WarmUp);
            VesselForest.GetCounts(out long solvesBefore, out long fallbacksBefore, out _);
            VesselForest.GetTimes(out double s0, out double a0, out double b0, out double c0, out double f0, out double w0, out double d0);
            VesselForest.GetSectionCounts(out long sections0, out long locked0, out long builds0, out long changes0, out double rebuild0,
                out _, out _, out _);
            double settle0 = VesselForest.SettleMilliseconds;
            var samples = new List<double>(Measured);
            for (int i = 0; i < Measured; i++)
            {
                long start = Stopwatch.GetTimestamp();
                UnityEngine.Physics.Simulate(Step);
                samples.Add((Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
            }

            VesselForest.GetCounts(out long solves, out long fallbacks, out _);
            VesselForest.GetTimes(out double s1, out double a1, out double b1, out double c1, out double f1, out double w1, out double d1);
            VesselForest.GetSectionCounts(out long sections1, out long locked1, out long builds1, out long changes1, out double rebuild1,
                out double residual, out _, out _);
            long forestSolves = Math.Max(1, solves - solvesBefore);
            lastTimes = path == "sparse" ? $"sparse {(s1 - s0) / Measured:F2}"
                : $"start {(a1 - a0) / Measured:F2} (factor {(f1 - f0) / Measured:F2}, boundary {(w1 - w0) / Measured:F2}), " +
                $"solve {(b1 - b0) / Measured:F2}, finish {(c1 - c0) / Measured:F2} (settle {(VesselForest.SettleMilliseconds - settle0) / Measured:F2}, " +
                $"drift aim {(d1 - d0) / Measured:F2}), " +
                $"fallback sparse {(s1 - s0) / Measured:F2}";
            if (path == "locked")
            {
                lastTimes += $"; sections {(sections1 - sections0) / (double)forestSolves:F0}, locked pairs {(locked1 - locked0) / (double)forestSolves:F0} " +
                    $"per solve, structure rebuilds {builds1 - builds0} ({(rebuild1 - rebuild0) / Math.Max(1, builds1 - builds0):F2} ms each), " +
                    $"lock changes {changes1 - changes0}, largest residual {residual:E1} N s";
            }
            JobsUtility.JobWorkerCount = workers;
            samples.Sort();
            return (samples[0], samples[samples.Count / 2], solves - solvesBefore, fallbacks - fallbacksBefore);
        }

        /// <summary>A comma-separated list from the environment, to bench a part of the table, or the default.</summary>
        private static string[] Choice(string variable, string fallback)
        {
            string value = System.Environment.GetEnvironmentVariable(variable);
            return (string.IsNullOrEmpty(value) ? fallback : value).Split(',');
        }

        [Test]
        [Timeout(7200000)]
        public void Measure()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The forest is part of the Unity Physics backend.");
            }

            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            bool wasEnabled = VesselForest.Enabled;
            bool wasLocking = VesselForest.Locking;
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(PartLayer, PartLayer);
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, true);
            bool jobDebugger = JobsUtility.JobDebuggerEnabled;
            JobsUtility.JobDebuggerEnabled = false;
            Unity.Collections.NativeLeakDetection.Mode = Unity.Collections.NativeLeakDetectionMode.Disabled;
            int workers = JobsUtility.JobWorkerCount;
            double load = SampleLoad();
            var report = new StringBuilder($"[VesselForestBench] whole Physics.Simulate step, ms min / median over {Measured} warm steps; " +
                $"CPU load at start {load:F1}%, {SystemInfo.processorCount} logical processors, {workers} job workers by default\n");
            string[] paths = Choice("RETOOLED_BENCH_PATHS", "sparse,forest,locked");
            report.Append($"craft | bodies | scene | threads | {string.Join(" | ", paths)} (ms min / median) | " +
                "island solve ms per step and counts, by path\n");
            try
            {
                foreach (string craft in Choice("RETOOLED_BENCH_CRAFT", "tu160,synthetic4500"))
                {
                    Graph graph = Load(craft);
                    foreach (bool ground in Choice("RETOOLED_BENCH_SCENE", "floating,ground").Select(s => s == "ground"))
                    {
                        foreach (int threads in Choice("RETOOLED_BENCH_THREADS", "1,8").Select(int.Parse))
                        {
                            var cells = new List<string>();
                            var notes = new List<string>();
                            foreach (string path in paths)
                            {
                                var result = Session(graph, ground, path, threads, workers);
                                cells.Add($"{result.min:F2} / {result.median:F2}");
                                notes.Add(path == "sparse" ? $"{path}: {lastTimes}"
                                    : $"{path}: {result.solves} forest solves / {result.fallbacks} fallbacks, {lastTimes}");
                            }

                            report.Append($"{craft} | {graph.Parent.Length} | {(ground ? "ground" : "floating")} | {threads} | " +
                                $"{string.Join(" | ", cells)} | {string.Join("; ", notes)}\n");
                        }
                    }
                }
            }
            finally
            {
                JobsUtility.JobWorkerCount = workers;
                JobsUtility.JobDebuggerEnabled = jobDebugger;
                VesselForest.Enabled = wasEnabled;
                VesselForest.Locking = wasLocking;
                UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, !collided);
            }

            Debug.Log(report.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "VesselForestBench.txt"), report.ToString());
        }
    }
}
