using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using System.Collections;
using System.Diagnostics;
using NUnit.Framework;
using Unity.Burst;
using UnityEngine;
using UnityEngine.TestTools;

namespace KSP.Retooled.Physics.Tests.Host
{
    /// <summary>
    /// Verify stepping uses compiled Burst code rather than silently falling back to Mono.
    /// </summary>
    public class StepBurstTests : PhysicsScene
    {
        private const double CompileSeconds = 240.0;

        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
            Assume.That(BurstCompiler.IsEnabled, "this run has Burst off");
        }

        [UnityTest]
        public IEnumerator StepsAndQueryBuildsRunCompiled()
        {
            Rigidbody body = DynamicBox(Vector3.zero);
            long stepsBefore = BackendHost.StepCount;

            Stopwatch wait = Stopwatch.StartNew();
            while (true)
            {
                long before = BackendHost.ManagedSteps;
                UnityEngine.Physics.Simulate(0.02f);
                if (BackendHost.ManagedSteps == before)
                {
                    break;
                }

                Assert.That(wait.Elapsed.TotalSeconds, Is.LessThan(CompileSeconds), "the step is still in Mono");
                yield return null;
            }

            long ranInMono = BackendHost.ManagedSteps;
            body.transform.position = Vector3.up;
            UnityEngine.Physics.SyncTransforms();
            UnityEngine.Physics.Raycast(Vector3.zero, Vector3.up, 2.0f);
            UnityEngine.Physics.Simulate(0.02f);
            Assert.That(BackendHost.StepCount, Is.GreaterThan(stepsBefore), "nothing stepped");
            Assert.That(BackendHost.ManagedSteps, Is.EqualTo(ranInMono), "a step or query build went back to Mono");
        }
    }
}
