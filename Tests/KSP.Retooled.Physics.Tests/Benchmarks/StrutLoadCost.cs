using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using Unity.Burst;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Vessels.Sections;
using KSP.Retooled.Physics.Diagnostics;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Benchmarks
{
    /// <summary>
    /// What the strut loads cost: the factor of each section's equilibrium system when the section forms, and each section's
    /// solve in every step, from the counters around them. STRUT_COST_CRAFT lists the crafts (tu160 and pegasus by
    /// default), STRUT_COST_SCENE the scenes (ground by default). Output in Logs/StrutLoadCost.txt.
    /// </summary>
    [Explicit("A measurement")]
    public class StrutLoadCost : PhysicsScene
    {
        [Test]
        public void Measure()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            EnvironmentSwitches.Read();
            double toMs = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6;
            var text = new System.Text.StringBuilder("[StrutLoadCost] craft | scene | steps | systems factored | unknowns each | ms per factor | " +
                "section solves per step | ms per solve | ms of solves per step | blocks off the diagonal | " +
                "ms per solve: subtree loads, substitution, link loads | chunks, top columns, ms of the top per solve | matches | ms per factor: ordering and structure, numbers\n");
            string crafts = System.Environment.GetEnvironmentVariable("STRUT_COST_CRAFT") ?? "tu160,pegasus";
            string scenes = System.Environment.GetEnvironmentVariable("STRUT_COST_SCENE") ?? "ground";
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(30, 30);
            UnityEngine.Physics.IgnoreLayerCollision(30, 30, true);
            MethodInfo build = typeof(VesselForestBench).GetMethod("Build", BindingFlags.NonPublic | BindingFlags.Instance);
            try
            {
                foreach (string craft in crafts.Split(','))
                {
                    foreach (string scene in scenes.Split(','))
                    {
                        var bench = new VesselForestBench();
                        bench.SetUpScene();
                        try
                        {
                            build.Invoke(bench, new object[] { VesselForestBench.Load(craft), scene == "ground", true });
                            StrutLoadsCounters before = StrutLoads.Counts.Data;
                            const int steps = 200;
                            Simulate(steps);
                            StrutLoadsCounters after = StrutLoads.Counts.Data;
                            long factored = after.Factored - before.Factored;
                            long unknowns = after.FactoredUnknowns - before.FactoredUnknowns;
                            long solves = after.Solves - before.Solves;
                            double factorMs = (after.FactorTicks - before.FactorTicks) * toMs;
                            double solveMs = (after.SolveTicks - before.SolveTicks) * toMs;
                            text.Append($"{craft} | {scene} | {steps} | {factored} | {unknowns / (double)System.Math.Max(1, factored):F0} | " +
                                $"{factorMs / System.Math.Max(1, factored):F3} | {solves / (double)steps:F1} | " +
                                $"{solveMs / System.Math.Max(1, solves):F4} | {solveMs / steps:F4} | " +
                                $"{(after.FactoredBlocks - before.FactoredBlocks) / (double)System.Math.Max(1, factored):F0} | " +
                                $"{(after.LoadTicks - before.LoadTicks) * toMs / System.Math.Max(1, solves):F4}, " +
                                $"{(after.SubstituteTicks - before.SubstituteTicks) * toMs / System.Math.Max(1, solves):F4}, " +
                                $"{(after.LinkTicks - before.LinkTicks) * toMs / System.Math.Max(1, solves):F4} | " +
                                $"{(after.FactoredChunks - before.FactoredChunks) / (double)System.Math.Max(1, factored):F0}, " +
                                $"{(after.FactoredTop - before.FactoredTop) / (double)System.Math.Max(1, factored):F0}, " +
                                $"{(after.TopTicks - before.TopTicks) * toMs / System.Math.Max(1, solves):F4} | {after.Prepares - before.Prepares} | " +
                                $"{(after.OrderTicks - before.OrderTicks) * toMs / System.Math.Max(1, factored):F3}, " +
                                $"{(after.NumericTicks - before.NumericTicks) * toMs / System.Math.Max(1, factored):F3}\n");
                        }
                        finally
                        {
                            bench.TearDownScene();
                        }
                    }
                }
            }
            finally
            {
                UnityEngine.Physics.IgnoreLayerCollision(30, 30, !collided);
            }

            Debug.Log(text.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "StrutLoadCost.txt"), text.ToString());
        }

        /// <summary>
        /// How often the static systems are factored over 400 steps: the bench's crafts on the ground and floating, whose
        /// joints never break, and the same crafts dropped 2 m with every joint's break force at 20 kN, as JointLoadTrace
        /// drops the Tu-160, so that edges yield and lock again. Each line lists the steps that factored a system, and the
        /// structure builds and lock changes over the run. Output in Logs/StrutLoadFactors.txt.
        /// </summary>
        [Test]
        public void FactorReruns()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            EnvironmentSwitches.Read();
            double toMs = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6;
            string crafts = System.Environment.GetEnvironmentVariable("STRUT_COST_CRAFT") ?? "tu160,pegasus";
            var text = new System.Text.StringBuilder("[StrutLoadFactors] craft | scene | systems factored (kept ones used again) | ms of factors | " +
                "steps that factored | structure builds | lock changes\n");
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(30, 30);
            UnityEngine.Physics.IgnoreLayerCollision(30, 30, true);
            MethodInfo build = typeof(VesselForestBench).GetMethod("Build", BindingFlags.NonPublic | BindingFlags.Instance);
            bool wasForest = ForestSwitches.Enabled, wasLocking = ForestSwitches.Locking;
            int wasHold = ForestSwitches.MinimumHoldSteps;
            try
            {
                foreach (string craft in crafts.Split(','))
                {
                    foreach (string scene in new[] { "ground", "floating", "dropped, 20 kN" })
                    {
                        ForestSwitches.Enabled = true;
                        ForestSwitches.Locking = false;
                        ForestSwitches.MinimumHoldSteps = -1;
                        ForestDiagnostics.ForgetIslands();
                        var bench = new VesselForestBench();
                        bench.SetUpScene();
                        try
                        {
                            bool dropped = scene.StartsWith("dropped");
                            var bodies = (List<Rigidbody>)build.Invoke(bench, new object[] { VesselForestBench.Load(craft), scene != "floating", true });
                            if (dropped)
                            {
                                foreach (Rigidbody body in bodies)
                                {
                                    body.position += new Vector3(0.0f, 2.0f, 0.0f);
                                    body.transform.position = body.position;
                                    foreach (ConfigurableJoint joint in body.GetComponents<ConfigurableJoint>())
                                    {
                                        joint.breakForce = 2e4f;
                                    }
                                }
                            }

                            ForestDiagnostics.GetSectionCounts(out _, out _, out long builds0, out long changes0, out _, out _, out _, out _);
                            var factors = new List<string>();
                            StrutLoadsCounters start = StrutLoads.Counts.Data;
                            for (int i = 0; i < 400; i++)
                            {
                                StrutLoadsCounters before = StrutLoads.Counts.Data;
                                Simulate();
                                StrutLoadsCounters after = StrutLoads.Counts.Data;
                                if (after.Factored != before.Factored)
                                {
                                    long n = after.Factored - before.Factored;
                                    factors.Add($"{i} ({n} x {(after.FactoredUnknowns - before.FactoredUnknowns) / n} unknowns, " +
                                        $"{(after.FactorTicks - before.FactorTicks) * toMs:F1} ms)");
                                }
                            }

                            StrutLoadsCounters end = StrutLoads.Counts.Data;
                            ForestDiagnostics.GetSectionCounts(out _, out _, out long builds, out long changes, out _, out _, out _, out _);
                            text.Append($"{craft} | {scene} | {end.Factored - start.Factored} ({end.Reused - start.Reused}) | " +
                                $"{(end.FactorTicks - start.FactorTicks) * toMs:F1} | {(factors.Count == 0 ? "none" : string.Join(", ", factors))} | " +
                                $"{builds - builds0} | {changes - changes0}\n");
                        }
                        finally
                        {
                            bench.TearDownScene();
                        }
                    }
                }
            }
            finally
            {
                ForestSwitches.Enabled = wasForest;
                ForestSwitches.Locking = wasLocking;
                ForestSwitches.MinimumHoldSteps = wasHold;
                UnityEngine.Physics.IgnoreLayerCollision(30, 30, !collided);
            }

            Debug.Log(text.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "StrutLoadFactors.txt"), text.ToString());
        }

        /// <summary>
        /// The whole step with the strut loads off and on, in turn, on the same craft and scene, with the job workers that
        /// STRUT_COST_THREADS gives (8 by default). For each run: the median step over the last 300 of 400 steps, the strut
        /// solves' own time per step, the step less that time, every step at which a system was factored, and the sections,
        /// locked edges and structure builds at the end. Output in Logs/StrutLoadSteps.txt.
        /// </summary>
        [Test]
        public void StepTimes()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            EnvironmentSwitches.Read();
            double toMs = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6;
            string crafts = System.Environment.GetEnvironmentVariable("STRUT_COST_CRAFT") ?? "tu160,pegasus";
            string scenes = System.Environment.GetEnvironmentVariable("STRUT_COST_SCENE") ?? "ground,floating";
            int threads = int.TryParse(System.Environment.GetEnvironmentVariable("STRUT_COST_THREADS"), out int t) ? t : 8;
            int rounds = int.TryParse(System.Environment.GetEnvironmentVariable("STRUT_COST_ROUNDS"), out int r) ? r : 2;
            var text = new System.Text.StringBuilder($"[StrutLoadSteps] {threads} threads, {rounds} rounds\n" +
                "craft | scene | strut loads | median step ms | strut ms per step | median step less strut ms | factors at steps | " +
                "sections, locked edges, structure builds\n");
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(30, 30);
            UnityEngine.Physics.IgnoreLayerCollision(30, 30, true);
            MethodInfo build = typeof(VesselForestBench).GetMethod("Build", BindingFlags.NonPublic | BindingFlags.Instance);
            int workers = Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount;
            bool wasForest = ForestSwitches.Enabled, wasLocking = ForestSwitches.Locking, wasStrut = VesselSwitches.StrutLoadsEnabled;
            int wasHold = ForestSwitches.MinimumHoldSteps;
            try
            {
                Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
                foreach (string craft in crafts.Split(','))
                {
                    foreach (string scene in scenes.Split(','))
                    {
                        for (int run = 0; run < 2 * rounds; run++)
                        {
                            bool on = run % 2 == 1;
                            VesselSwitches.StrutLoadsEnabled = on;
                            ForestSwitches.Enabled = true;
                            ForestSwitches.Locking = false;
                            ForestSwitches.MinimumHoldSteps = -1;
                            ForestDiagnostics.ForgetIslands();
                            var bench = new VesselForestBench();
                            bench.SetUpScene();
                            try
                            {
                                build.Invoke(bench, new object[] { VesselForestBench.Load(craft), scene == "ground", true });
                                const int steps = 400, skip = 100;
                                var times = new List<double>();
                                var factors = new List<int>();
                                double strutMs = 0.0;
                                long builds0 = 0;
                                for (int i = 0; i < steps; i++)
                                {
                                    if (i == skip)
                                    {
                                        ForestDiagnostics.GetSectionCounts(out _, out _, out builds0, out _, out _, out _, out _, out _);
                                    }

                                    StrutLoadsCounters before = StrutLoads.Counts.Data;
                                    long start = System.Diagnostics.Stopwatch.GetTimestamp();
                                    UnityEngine.Physics.Simulate(Step);
                                    double ms = (System.Diagnostics.Stopwatch.GetTimestamp() - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                                    StrutLoadsCounters after = StrutLoads.Counts.Data;
                                    if (after.Factored != before.Factored)
                                    {
                                        factors.Add(i);
                                    }

                                    if (i >= skip)
                                    {
                                        times.Add(ms);
                                        strutMs += (after.SolveTicks - before.SolveTicks + after.AccumulateTicks - before.AccumulateTicks) * toMs;
                                    }
                                }

                                ForestDiagnostics.GetSectionCounts(out _, out _, out long builds, out _, out _, out _, out long sections, out long locked);
                                times.Sort();
                                double median = times[times.Count / 2];
                                double strut = strutMs / times.Count;
                                text.Append($"{craft} | {scene} | {(on ? "on" : "off")} | {median:F3} | {strut:F3} | {median - strut:F3} | " +
                                    $"{(factors.Count == 0 ? "none" : string.Join(" ", factors))} | {sections}, {locked}, {builds - builds0} in the last {steps - skip}\n");
                            }
                            finally
                            {
                                bench.TearDownScene();
                            }
                        }
                    }
                }
            }
            finally
            {
                Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount = workers;
                ForestSwitches.Enabled = wasForest;
                ForestSwitches.Locking = wasLocking;
                ForestSwitches.MinimumHoldSteps = wasHold;
                VesselSwitches.StrutLoadsEnabled = wasStrut;
                UnityEngine.Physics.IgnoreLayerCollision(30, 30, !collided);
            }

            Debug.Log(text.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "StrutLoadSteps.txt"), text.ToString());
        }
    }
}
