using System.Collections;
using System.Diagnostics;
using NUnit.Framework;
using Unity.Burst;
using UnityEngine;
using UnityEngine.TestTools;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The step once Burst has it. A step that doesn't compile keeps working in Mono and only gets slower, so nothing else
    /// would notice it, and most fixtures finish before the compile does.
    /// </summary>
    public class StepBurstTests
    {
        private const double CompileSeconds = 240.0;

        private GameObject body;

        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
            Assume.That(BurstCompiler.IsEnabled, "this run has Burst off");
        }

        [TearDown]
        public void DestroyTheBody()
        {
            Object.DestroyImmediate(body);
        }

        [UnityTest]
        public IEnumerator StepsAndQueryBuildsRunCompiled()
        {
            body = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            body.AddComponent<Rigidbody>();

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
            Assert.That(BackendHost.ManagedSteps, Is.EqualTo(ranInMono), "a step or query build went back to Mono");
        }
    }
}
