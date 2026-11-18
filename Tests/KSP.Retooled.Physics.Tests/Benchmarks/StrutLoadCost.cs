using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using Unity.Burst;
using Unity.Physics;
using Unity.Physics.Dynamics.Vessels.Sections;
using UnityEngine;
using Unity.Physics.Configuration;

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
                "ms per solve: subtree loads, substitution, link loads\n");
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
                            StrutLoads.Totals before = StrutLoads.Counts.Data;
                            const int steps = 200;
                            Simulate(steps);
                            StrutLoads.Totals after = StrutLoads.Counts.Data;
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
                                $"{(after.LinkTicks - before.LinkTicks) * toMs / System.Math.Max(1, solves):F4}\n");
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
    }
}
