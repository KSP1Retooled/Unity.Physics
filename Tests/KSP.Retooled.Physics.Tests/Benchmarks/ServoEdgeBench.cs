using System.Collections.Generic;
using System.Diagnostics;
using KSP.Retooled.Physics.Stepping;
using NUnit.Framework;
using Unity.Burst;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Forest;
using UnityEngine;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using PistonEdgeTests = KSP.Retooled.Physics.Tests.Conformance.Vessels.PistonEdgeTests;
using ServoEdgeTests = KSP.Retooled.Physics.Tests.Conformance.Vessels.ServoEdgeTests;

namespace KSP.Retooled.Physics.Tests.Benchmarks
{
    [Explicit("Measures floating servo vessels with Burst enabled.")]
    [Category("Slow")]
    public unsafe class ServoEdgeBench : PhysicsScene
    {
        private const int BlocksPerMode = 3;
        private const int WarmupSteps = 20;
        private const int StepsPerBlock = 20;

        private Rigidbody Part(Vector3 position)
            => ServoEdgeTests.Body(Track(new GameObject("benchmark part")), position, 1.0f, 0.2f);

        private const float PistonStroke = 2.4f;

        private (ConfigurableJoint[] pistons, ConfigurableJoint[] hinges) Build(int parts, int pistons, int hinges)
        {
            int branches = Mathf.Max(pistons, hinges);
            int branchParts = parts / (branches + 1);
            int spineParts = parts - branches * branchParts;
            var spine = new Rigidbody[spineParts];
            for (int i = 0; i < spineParts; i++)
            {
                spine[i] = Part(0.33f * i * Vector3.right);
                if (i > 0)
                {
                    ServoEdgeTests.Link(spine[i], spine[i - 1], spine[i - 1].position, Vector3.right);
                }
            }

            var sliders = new ConfigurableJoint[pistons];
            var rotary = new ConfigurableJoint[hinges];
            for (int i = 0; i < branches; i++)
            {
                Rigidbody parent = spine[i % spineParts];
                Vector3 pivot = parent.position;
                bool hasPiston = i < pistons;
                float pistonOffset = hasPiston ? 0.5f * PistonStroke - 0.33f : 0.0f;
                for (int part = 0; part < branchParts; part++)
                {
                    Rigidbody body = Part(pivot + (0.33f * (part + 1) + pistonOffset) * Vector3.right);
                    if (hasPiston && part == 0)
                    {
                        sliders[i] = PistonEdgeTests.Piston(body, parent, Vector3.right, PistonStroke);
                    }
                    else
                    {
                        bool hinge = i < hinges && part == (hasPiston ? 1 : 0);
                        ConfigurableJoint joint = ServoEdgeTests.Link(body, parent, parent.position, Vector3.right, hinge);
                        if (hinge)
                        {
                            rotary[i] = joint;
                            ServoEdgeTests.Drive(joint, 40.0f, 12.0f, 80.0f);
                        }
                    }

                    parent = body;
                }
            }

            return (sliders, rotary);
        }

        [Test]
        public void Measure()
            => MeasureCases(new[] { (500, 0, 0), (500, 0, 5), (500, 0, 20), (500, 0, 60), (60, 0, 10) });

        [Test]
        public void MeasurePistons()
            => MeasureCases(new[] { (500, 20, 0), (500, 20, 20) });

        private void MeasureCases((int parts, int pistons, int hinges)[] cases)
        {
            Assert.That(BackendInfo.IsRetooled, Is.True);
            Assert.That(BurstCompiler.IsEnabled, Is.True);
            EnvironmentSwitches.Read();
            var settings = RecordedPhysicsSettings.Read();
            int workers = JobsUtility.JobWorkerCount;
            try
            {
                VesselSwitches.Enabled = true;
                VesselSwitches.MinimumParts = 2;
                foreach (int count in new[] { 8, 1 })
                {
                    JobsUtility.JobWorkerCount = count;
                    Assert.That(JobsUtility.JobWorkerCount, Is.EqualTo(count));
                    foreach (var (parts, pistons, hinges) in cases)
                    {
                        MeasureCraft(count, parts, pistons, hinges);
                    }
                }
            }
            finally
            {
                settings.Apply();
                JobsUtility.JobWorkerCount = workers;
            }
        }

        private void MeasureCraft(int workers, int parts, int pistons, int hinges)
        {
            var (sliders, rotary) = Build(parts, pistons, hinges);
            var times = new[] { new List<double>(), new List<double>() };
            var rows = new long[2];
            var samples = new long[2];
            int step = 0;
            for (int block = 0; block < BlocksPerMode * 2; block++)
            {
                int mode = block % 2;
                VesselSwitches.ServoEdges = mode == 1;
                for (int i = 0; i < WarmupSteps + StepsPerBlock; i++)
                {
                    for (int drive = 0; drive < sliders.Length; drive++)
                    {
                        sliders[drive].targetPosition = 0.2f * Mathf.Sin(0.05f * step + 0.27f * drive) * Vector3.right;
                    }
                    for (int drive = 0; drive < rotary.Length; drive++)
                    {
                        float angle = 22.0f * Mathf.Sin(0.05f * step + 0.27f * drive);
                        rotary[drive].targetRotation = Quaternion.AngleAxis(angle, Vector3.right);
                    }
                    step++;
                    BoundaryCounters before = BoundaryCounters.Current;
                    long start = Stopwatch.GetTimestamp();
                    UnityEngine.Physics.Simulate(Step);
                    StepEntryPoints.Current->FinishPendingRefit();
                    long elapsed = Stopwatch.GetTimestamp() - start;
                    if (i >= WarmupSteps)
                    {
                        times[mode].Add(elapsed * 1000.0 / Stopwatch.Frequency);
                        BoundaryCounters after = BoundaryCounters.Current;
                        rows[mode] += after.OtherRows - before.OtherRows;
                        samples[mode] += after.RowKindSamples - before.RowKindSamples;
                    }
                }
            }

            for (int mode = 0; mode < times.Length; mode++)
            {
                times[mode].Sort();
                int middle = times[mode].Count / 2;
                double median = 0.5 * (times[mode][middle - 1] + times[mode][middle]);
                double rowCount = samples[mode] > 0 ? rows[mode] / (double)samples[mode] : 0.0;
                TestContext.WriteLine($"{workers} workers, {parts} parts, {pistons} pistons, {hinges} hinges, edges {(mode == 1 ? "on" : "off")}: "
                    + $"median {median:F3} ms, minimum {times[mode][0]:F3} ms, {rowCount:F2} boundary rows.");
                Assert.That(rowCount, Is.EqualTo((pistons + hinges) * (mode == 1 ? 1 : 6)));
            }
            DestroyAll();
        }
    }
}
