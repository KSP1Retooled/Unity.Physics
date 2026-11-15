using System.Collections;
using System.Diagnostics;
using System.Threading;
using NUnit.Framework;
using Unity.Burst;
using Unity.Jobs.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.TestTools;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>The step runs jobs from inside the engine's Simulate call, whichever state Burst is in.</summary>
    public class StepThreadTests : PhysicsScene
    {
        private const double PhaseSeconds = 10.0;
        private const double KernelCompileSeconds = 120.0;

        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
        }

        /// <summary>
        /// Runs a command from each kernel and waits for Burst compilation. If Burst cannot compile a kernel, that
        /// kernel keeps running in Mono without an obvious failure. The same compilation error would fail a player
        /// build.
        /// </summary>
        [UnityTest]
        public IEnumerator EveryKernelLeavesMonoOnceItsCompileLands()
        {
            Assume.That(BurstCompiler.IsEnabled, "this run has Burst off");

            Rigidbody body = WeightlessBox(Vector3.zero);
            Collider collider = body.GetComponent<Collider>();
            ConfigurableJoint joint = body.gameObject.AddComponent<ConfigurableJoint>();
            Stopwatch wait = Stopwatch.StartNew();
            while (true)
            {
                long before = BackendHost.ManagedCommands;
                body.linearVelocity = Vector3.right;
                collider.isTrigger = !collider.isTrigger;
                joint.breakForce = joint.breakForce > 1.0f ? 1.0f : 2.0f;
                Object.DestroyImmediate(Box(new Vector3(5.0f, 0.0f, 0.0f), Vector3.one));
                if (BackendHost.ManagedCommands == before)
                {
                    Debug.Log($"Kernels in Burst after {wait.Elapsed.TotalSeconds:F1} s here, {before} commands ran in Mono before that");
                    yield break;
                }

                Assert.That(wait.Elapsed.TotalSeconds, Is.LessThan(KernelCompileSeconds), "the kernels are still in Mono");
                yield return null;
            }
        }

        [Test]
        public void TheStepRunsOnTheMainThread()
        {
            WeightlessBox(Vector3.zero);

            Simulate(1);

            Assert.That(BackendHost.LastStepThreadId, Is.EqualTo(Thread.CurrentThread.ManagedThreadId));
            Assert.That(BackendHost.LastStepThreadId, Is.EqualTo(BackendHost.MainThreadId));
        }

        /// <summary>
        /// Steps a pile of boxes in three phases: the initial compilation state after a domain reload, forced Mono jobs
        /// as while asynchronous compilation is in progress, and compiled jobs enabled again while the scene is
        /// running. Yields between steps so that the editor can update and background compilation can finish. After a
        /// cold start without Library/BurstCache, the log shows when the first phase stopped using Mono.
        /// <para>
        /// The first phase ends as soon as one whole step runs without Mono jobs, because from then on the step is in
        /// the same state as the last phase. The middle phase ends after its first step that ran Mono jobs, because
        /// that is all that the test checks about it. Each step in the middle phase takes several seconds in Mono. Both
        /// phases still stop after PhaseSeconds if that point never comes. The last phase always runs for PhaseSeconds.
        /// </para>
        /// </summary>
        [UnityTest]
        [Category("Slow")]
        public IEnumerator StepsKeepWorkingAcrossMonoAndBurstJobs()
        {
            Box(new Vector3(0.0f, -0.5f, 0.0f), new Vector3(100.0f, 1.0f, 100.0f));
            const int bodies = 500;
            for (int i = 0; i < bodies; i++)
            {
                DynamicBox(new Vector3((i % 25) * 1.5f - 18.0f, 1.0f + (i / 25) * 1.5f, 0.0f));
            }

            bool compilerWasEnabled = JobsUtility.JobCompilerEnabled;
            long[] monoRuns = new long[3];
            string[] phases = { "as found", "Mono", "compiled again" };
            try
            {
                for (int phase = 0; phase < phases.Length; phase++)
                {
                    JobsUtility.JobCompilerEnabled = phase != 1 && compilerWasEnabled;

                    long monoBefore = BackendHost.ManagedJobRuns;
                    Stopwatch watch = Stopwatch.StartNew();
                    double slowest = 0.0;
                    int steps = 0;
                    while (watch.Elapsed.TotalSeconds < PhaseSeconds)
                    {
                        long monoBeforeStep = BackendHost.ManagedJobRuns;
                        Stopwatch step = Stopwatch.StartNew();
                        UnityEngine.Physics.Simulate(Step);
                        slowest = System.Math.Max(slowest, step.Elapsed.TotalMilliseconds);
                        steps++;
                        bool stepRanMono = BackendHost.ManagedJobRuns != monoBeforeStep;
                        if (steps % 100 == 1)
                        {
                            Debug.Log($"[RetooledPhysics] {phases[phase]}: step {steps} at {watch.Elapsed.TotalSeconds:F1} s, " +
                                $"{step.Elapsed.TotalMilliseconds:F2} ms, Mono job runs {BackendHost.ManagedJobRuns - monoBefore}");
                        }

                        if ((phase == 0 && !stepRanMono) || (phase == 1 && stepRanMono))
                        {
                            break;
                        }

                        yield return null;
                    }

                    monoRuns[phase] = BackendHost.ManagedJobRuns - monoBefore;
                    Debug.Log($"[RetooledPhysics] {phases[phase]}: {steps} steps, slowest {slowest:F2} ms, " +
                        $"Mono job runs {monoRuns[phase]}, Burst enabled {BurstCompiler.IsEnabled}");
                }
            }
            finally
            {
                JobsUtility.JobCompilerEnabled = compilerWasEnabled;
            }

            Assume.That(compilerWasEnabled && BurstCompiler.IsEnabled, "this run has Burst off, so it never left Mono");
            Assert.That(monoRuns[1], Is.GreaterThan(0), "the middle phase ran Mono jobs");
            Assert.That(monoRuns[2], Is.EqualTo(0), "the last phase ran compiled jobs");
        }
    }
}
