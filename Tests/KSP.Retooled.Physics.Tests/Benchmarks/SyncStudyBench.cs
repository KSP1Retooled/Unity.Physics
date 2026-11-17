using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
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
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Jobs;

namespace KSP.Retooled.Physics.Tests.Benchmarks
{
    /// <summary>
    /// Measure Unity step stages, pose read-back commands and direct pose writes for independent transform
    /// hierarchies.
    /// Write results to Logs/SyncStudyBench.txt.
    /// </summary>
    [Explicit("A benchmark")]
    [Category("Slow")]
    public class SyncStudyBench : PhysicsScene
    {
        private const int WarmUp = 20;
        private const int Measured = 60;

        [BurstCompile]
        private struct WritePoses : IJobParallelForTransform
        {
            [ReadOnly] public NativeArray<Vector3> Positions;
            [ReadOnly] public NativeArray<Quaternion> Rotations;

            public void Execute(int index, TransformAccess transform)
            {
                transform.SetPositionAndRotation(Positions[index], Rotations[index]);
            }
        }

        private readonly StringBuilder report = new StringBuilder();

        private static string Stat(List<double> samples)
        {
            samples.Sort();
            return $"{samples[0]:F3} / {samples[samples.Count / 2]:F3}";
        }

        private static double Ms(long start) => (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;

        private List<Rigidbody> Build(int count, int children)
        {
            var bodies = new List<Rigidbody>(count);
            int side = (int)Math.Ceiling(Math.Sqrt(count));
            var random = new System.Random(7);
            for (int i = 0; i < count; i++)
            {
                Rigidbody body = WeightlessBox(new Vector3(3.0f * (i % side), 10.0f, 3.0f * (i / side)));
                body.sleepThreshold = 0.0f;
                body.linearVelocity = new Vector3((float)random.NextDouble() - 0.5f, 0.0f, (float)random.NextDouble() - 0.5f) * 0.1f;
                body.angularVelocity = new Vector3((float)random.NextDouble(), (float)random.NextDouble(), (float)random.NextDouble());
                if (children > 0)
                {
                    // A model node with the rest of the children under it, as a part's model is.
                    Transform model = new GameObject("model").transform;
                    model.SetParent(body.transform, false);
                    for (int c = 1; c < children; c++)
                    {
                        new GameObject("mesh").transform.SetParent(model, false);
                    }
                }

                bodies.Add(body);
            }

            return bodies;
        }

        private void Session(int count, int children)
        {
            TearDownScene();
            SetUpScene();
            UnityEngine.Physics.autoSyncTransforms = false;
            List<Rigidbody> bodies = Build(count, children);
            Transform[] transforms = bodies.Select(b => b.transform).ToArray();
            for (int i = 0; i < WarmUp; i++)
            {
                UnityEngine.Physics.Simulate(Step);
            }

            var prepare = new List<double>();
            var run = new List<double>();
            var publish = new List<double>();
            var sleeping = new List<double>();
            var position = new List<double>();
            var mainWrite = new List<double>();
            var jobWrite = new List<double>();
            var prepareAfterJob = new List<double>();
            var publishAsleep = new List<double>();
            var positions = new NativeArray<Vector3>(count, Allocator.Persistent);
            var rotations = new NativeArray<Quaternion>(count, Allocator.Persistent);
            var access = new TransformAccessArray(transforms);
            UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
            try
            {
                // Unity's own step, one stage at a time.
                for (int i = 0; i < Measured; i++)
                {
                    long start = Stopwatch.GetTimestamp();
                    scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation, SimulationOption.All);
                    prepare.Add(Ms(start));
                    start = Stopwatch.GetTimestamp();
                    scene.RunSimulationStages(Step, SimulationStage.RunSimulation, SimulationOption.All);
                    run.Add(Ms(start));
                    start = Stopwatch.GetTimestamp();
                    scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
                    publish.Add(Ms(start));
                }

                // The commands that the read-back sends per body, from script: GetIsSleeping alone, and GetPose behind
                // the position getter. Each also pays for the call from managed code into Unity.
                for (int i = 0; i < Measured; i++)
                {
                    long start = Stopwatch.GetTimestamp();
                    int asleep = 0;
                    foreach (Rigidbody body in bodies)
                    {
                        asleep += body.IsSleeping() ? 1 : 0;
                    }

                    sleeping.Add(Ms(start));
                    start = Stopwatch.GetTimestamp();
                    for (int b = 0; b < count; b++)
                    {
                        positions[b] = bodies[b].position;
                    }

                    position.Add(Ms(start));
                    Assert.That(asleep, Is.EqualTo(0));
                }

                for (int b = 0; b < count; b++)
                {
                    positions[b] = bodies[b].position;
                    rotations[b] = bodies[b].rotation;
                }

                // Writing the same poses on the main thread, as the read-back does, but with every system told.
                for (int i = 0; i < Measured; i++)
                {
                    float shift = (i & 1) == 0 ? 0.001f : -0.001f;
                    long start = Stopwatch.GetTimestamp();
                    for (int b = 0; b < count; b++)
                    {
                        transforms[b].SetPositionAndRotation(positions[b] + new Vector3(shift, 0.0f, 0.0f), rotations[b]);
                    }

                    mainWrite.Add(Ms(start));
                }

                // The step without Unity's read-back: a parallel transform job writes the poses, and the next Prepare
                // stage then finds every body's transform changed.
                for (int i = 0; i < Measured; i++)
                {
                    long start = Stopwatch.GetTimestamp();
                    scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation, SimulationOption.All);
                    prepareAfterJob.Add(Ms(start));
                    scene.RunSimulationStages(Step, SimulationStage.RunSimulation, SimulationOption.All);
                    for (int b = 0; b < count; b++)
                    {
                        positions[b] = bodies[b].position;
                        rotations[b] = bodies[b].rotation;
                    }

                    start = Stopwatch.GetTimestamp();
                    new WritePoses { Positions = positions, Rotations = rotations }.Schedule(access).Complete();
                    jobWrite.Add(Ms(start));
                }

                // The read-back when every body sleeps.
                UnityEngine.Physics.Simulate(Step);
                foreach (Rigidbody body in bodies)
                {
                    body.linearVelocity = Vector3.zero;
                    body.angularVelocity = Vector3.zero;
                    body.Sleep();
                }

                int stillAwake = 0;
                for (int i = 0; i < Measured; i++)
                {
                    scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation | SimulationStage.RunSimulation, SimulationOption.All);
                    long start = Stopwatch.GetTimestamp();
                    scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
                    publishAsleep.Add(Ms(start));
                }

                foreach (Rigidbody body in bodies)
                {
                    stillAwake += body.IsSleeping() ? 0 : 1;
                }

                string line = $"[SyncStudyBench] {count} bodies, {children} child transforms each, workers {Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount}; " +
                    $"ms min / median: prepare {Stat(prepare)}, run {Stat(run)}, publish {Stat(publish)}; " +
                    $"IsSleeping loop {Stat(sleeping)}, position loop {Stat(position)}; " +
                    $"main-thread SetPositionAndRotation {Stat(mainWrite)}, transform job {Stat(jobWrite)}, prepare after the job {Stat(prepareAfterJob)}; " +
                    $"publish with all asleep {Stat(publishAsleep)} ({stillAwake} awake at the end)";
                UnityEngine.Debug.Log(line);
                report.AppendLine(line);
            }
            finally
            {
                access.Dispose();
                positions.Dispose();
                rotations.Dispose();
            }
        }

        [Test]
        public void Measure()
        {
            report.Clear();
            string sizes = Environment.GetEnvironmentVariable("RETOOLED_SYNC_SIZES");
            foreach (string size in string.IsNullOrEmpty(sizes) ? new[] { "2700x0", "2700x16" } : sizes.Split(','))
            {
                string[] parts = size.Split('x');
                Session(int.Parse(parts[0], CultureInfo.InvariantCulture), int.Parse(parts[1], CultureInfo.InvariantCulture));
            }

            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "SyncStudyBench.txt"), report.ToString());
        }
    }
}
