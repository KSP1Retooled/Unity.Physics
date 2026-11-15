using System.Collections;
using System.Diagnostics;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using UnityEngine;
using UnityEngine.TestTools;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Tests the query function after Burst compilation. Most conformance fixtures finish while it still runs in Mono.
    /// A query that fails only in compiled code, such as a collector called through a function pointer, could therefore
    /// crash a player without failing those fixtures.
    /// </summary>
    public class QueryBurstTests : QueryScene
    {
        private const double CompileSeconds = 240.0;
        private const float Tolerance = 3e-3f;

        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
        }

        [UnityTest]
        public IEnumerator EveryKindOfQueryRunsCompiled()
        {
            Assume.That(BurstCompiler.IsEnabled, "this run has Burst off");

            Stopwatch wait = Stopwatch.StartNew();
            while (true)
            {
                long before = BackendHost.ManagedQueries;
                UnityEngine.Physics.Raycast(Vector3.zero, Vector3.forward, 1.0f);
                if (BackendHost.ManagedQueries == before)
                {
                    UnityEngine.Debug.Log($"Queries in Burst after {wait.Elapsed.TotalSeconds:F1} s here");
                    break;
                }

                Assert.That(wait.Elapsed.TotalSeconds, Is.LessThan(CompileSeconds), "the query function is still in Mono");
                yield return null;
            }

            long ranInMono = BackendHost.ManagedQueries;
            RunEveryKind();
            Assert.That(BackendHost.ManagedQueries, Is.EqualTo(ranInMono), "a query went back to Mono");
        }

        private void RunEveryKind()
        {
            Collider box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one).GetComponent<Collider>();
            Collider convex = ConvexCube(new Vector3(3.0f, 0.0f, 5.0f)).GetComponent<Collider>();
            Collider quad = TriangleMeshQuad(new Vector3(-6.0f, 0.0f, 5.0f)).GetComponent<Collider>();
            Rigidbody body = ConvexCube(new Vector3(3.0f, 0.0f, 0.0f)).AddComponent<Rigidbody>();
            body.useGravity = false;
            UnityEngine.Physics.SyncTransforms();

            Assert.That(UnityEngine.Physics.RaycastAll(Vector3.zero, Vector3.forward, 100.0f).Length, Is.EqualTo(1));
            Assert.That(box.Raycast(new Ray(Vector3.zero, Vector3.forward), out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.5f).Within(Tolerance));
            Assert.That(quad.Raycast(new Ray(new Vector3(-6.0f, 0.0f, 0.0f), Vector3.forward), out hit, 100.0f), Is.True);

            Assert.That(UnityEngine.Physics.CheckSphere(new Vector3(0.0f, 0.0f, 4.0f), 1.0f), Is.True);
            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(0.0f, 0.0f, 4.0f), 1.0f).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.OverlapBox(new Vector3(0.0f, 0.0f, 4.0f), Vector3.one).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.OverlapCapsule(new Vector3(0.0f, -1.0f, 4.0f), new Vector3(0.0f, 1.0f, 4.0f), 1.0f).Length, Is.EqualTo(1));

            Assert.That(UnityEngine.Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
            Assert.That(UnityEngine.Physics.CapsuleCast(Vector3.zero, Vector3.up, 0.25f, Vector3.forward, out hit, 100.0f), Is.True);
            Assert.That(UnityEngine.Physics.BoxCast(Vector3.zero, Vector3.one * 0.25f, Vector3.forward, out hit, Quaternion.identity, 100.0f), Is.True);
            Assert.That(UnityEngine.Physics.SphereCastAll(Vector3.zero, 0.25f, Vector3.forward, 100.0f).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.SphereCastAll(new Vector3(-6.0f, 0.0f, 0.0f), 0.25f, Vector3.forward, 100.0f).Length, Is.EqualTo(1));

            Assert.That(body.SweepTest(Vector3.forward, out hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.0f).Within(Tolerance));
            Assert.That(body.SweepTestAll(Vector3.forward, 100.0f).Length, Is.EqualTo(1));

            AssertVector(new Vector3(0.0f, 0.0f, 4.5f), box.ClosestPoint(Vector3.zero), Tolerance);
            AssertVector(new Vector3(3.0f, 0.0f, 4.5f), convex.ClosestPoint(new Vector3(3.0f, 0.0f, 0.0f)), Tolerance);

            bool overlapping = UnityEngine.Physics.ComputePenetration(convex, Vector3.zero, Quaternion.identity, box, Vector3.right * 0.75f, Quaternion.identity, out Vector3 direction, out float distance);
            Assert.That(overlapping, Is.True);
            Assert.That(distance, Is.EqualTo(0.25f).Within(Tolerance));
            Assert.That(UnityEngine.Physics.ComputePenetration(quad, Vector3.zero, Quaternion.identity, box, Vector3.back * 0.25f, Quaternion.identity, out direction, out distance), Is.True);

            using NativeArray<OverlapSphereCommand> commands = new NativeArray<OverlapSphereCommand>(1, Allocator.TempJob);
            using NativeArray<ColliderHit> hits = new NativeArray<ColliderHit>(2, Allocator.TempJob);
            NativeArray<OverlapSphereCommand> command = commands;
            BackendHost.SyncQueryWorlds();
            command[0] = new OverlapSphereCommand(new Vector3(0.0f, 0.0f, 4.0f), 1.0f, QueryParameters.Default);
            OverlapSphereCommand.ScheduleBatch(commands, hits, 1, 2).Complete();
            Assert.That(hits[0].collider, Is.SameAs(box));

            Box(new Vector3(0.0f, -20.0f, 0.0f), new Vector3(20.0f, 1.0f, 20.0f));
            ParticleSystem particles = Track(new GameObject("particles")).AddComponent<ParticleSystem>();
            Assert.That(ParticleAfterASecond(particles, new Vector3(0.0f, -18.0f, 0.0f)).position.y, Is.EqualTo(-19.5f).Within(0.1f));
        }
    }
}
