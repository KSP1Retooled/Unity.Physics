using System.Collections;
using System.Diagnostics;
using System.Threading;
using NUnit.Framework;
using Unity.Burst;
using UnityEngine;
using UnityEngine.TestTools;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>The step runs jobs from inside the engine's Simulate call, whichever state Burst is in.</summary>
    public class StepThreadTests : PhysicsScene
    {
        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
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
        /// Keeps stepping a scene large enough to show Mono job code in its timing for as long as an async Burst compile
        /// takes after a cold start, between editor updates so that the compile can finish. In a run without
        /// Library/BurstCache the log shows the step time fall when the compiled jobs come in.
        /// </summary>
        [UnityTest]
        [Category("Slow")]
        public IEnumerator StepsKeepWorkingWhileBurstCompiles()
        {
            Ground();
            const int bodies = 500;
            Rigidbody[] boxes = new Rigidbody[bodies];
            for (int i = 0; i < bodies; i++)
            {
                boxes[i] = DynamicBox(new Vector3((i % 25) * 1.5f - 18.0f, 1.0f + (i / 25) * 1.5f, 0.0f));
            }

            Stopwatch total = Stopwatch.StartNew();
            Stopwatch step = new Stopwatch();
            double firstMilliseconds = 0.0;
            double bestMilliseconds = double.MaxValue;
            int steps = 0;
            while (total.Elapsed.TotalSeconds < 90.0)
            {
                step.Restart();
                Physics.Simulate(Step);
                step.Stop();

                double milliseconds = step.Elapsed.TotalMilliseconds;
                firstMilliseconds = steps == 0 ? milliseconds : firstMilliseconds;
                bestMilliseconds = System.Math.Min(bestMilliseconds, milliseconds);
                if (steps++ % 25 == 0)
                {
                    Debug.Log($"[RetooledPhysics] step {steps} at {total.Elapsed.TotalSeconds:F1} s: {milliseconds:F2} ms");
                }

                yield return null;
            }

            Debug.Log($"[RetooledPhysics] {steps} steps, first {firstMilliseconds:F2} ms, best {bestMilliseconds:F2} ms, " +
                $"Burst enabled {BurstCompiler.IsEnabled}");
            Assert.That(boxes[0].position.y, Is.LessThan(1.1f).And.GreaterThan(0.0f));
        }
    }
}
