using KSP.Retooled.Physics.Stepping;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;

namespace KSP.Retooled.Physics.Tests.Conformance.Contacts
{
    public class WireContactTests : PhysicsScene
    {
        private Mesh wireMesh;
        private Mesh groundMesh;
        private int recoveriesBeforeTest;

        [SetUp]
        public unsafe void SetUpWire()
        {
            var state = StepEntryPoints.Current;
            recoveriesBeforeTest = state == null ? 0 : state->Counters.NonFiniteBodies;
            wireMesh = new Mesh
            {
                vertices = new[]
                {
                new Vector3(-0.00624993257f, -8.82602544E-05f, -0.342075616f),
                new Vector3(-0.003141746f, 0.005402772f, -1.03861169E-06f),
                new Vector3(-0.00314166956f, 0.005334103f, -0.3420762f),
                new Vector3(-0.0062500136f, -1.95909342E-05f, -4.2826008E-07f),
                new Vector3(-0.00310822413f, -0.005491092f, -0.342073768f),
                new Vector3(-0.00310830516f, -0.00542242732f, 1.40279451E-06f),
                new Vector3(0.0031417422f, -0.005471866f, -0.3420741f),
                new Vector3(0.00314166117f, -0.00540320156f, 1.09761879E-06f),
                new Vector3(0.006250005f, -4.95029271E-05f, -0.342075616f),
                new Vector3(0.0062499335f, 1.916639E-05f, -1.23084291E-07f),
                new Vector3(0.00310829212f, 0.00535332924f, -0.3420762f),
                new Vector3(0.00310821575f, 0.00542199844f, -1.03861169E-06f),
                new Vector3(-0.00314166956f, 0.005334103f, -0.3420762f),
                new Vector3(-0.003141746f, 0.005402772f, -1.03861169E-06f),
                new Vector3(-0.00624993257f, -8.82602544E-05f, -0.342075616f),
                new Vector3(-0.00314166956f, 0.005334103f, -0.3420762f),
                new Vector3(-0.00314169819f, 0.00530877337f, -0.469518244f),
                new Vector3(-0.006249961f, -0.000113589842f, -0.469517618f),
                new Vector3(-0.00310822413f, -0.005491092f, -0.342073768f),
                new Vector3(-0.00310825277f, -0.005516419f, -0.4695158f),
                new Vector3(0.0031417422f, -0.005471866f, -0.3420741f),
                new Vector3(0.00314171356f, -0.00549719343f, -0.4695161f),
                new Vector3(0.006250005f, -4.95029271E-05f, -0.342075616f),
                new Vector3(0.00624998147f, -7.483252E-05f, -0.469517618f),
                new Vector3(0.00310829212f, 0.00535332924f, -0.3420762f),
                new Vector3(0.00310826348f, 0.00532799959f, -0.469518244f),
                new Vector3(-0.00314166956f, 0.005334103f, -0.3420762f),
                new Vector3(-0.00314169819f, 0.00530877337f, -0.469518244f),
                new Vector3(-0.003141746f, 0.005402772f, -1.03861169E-06f),
                new Vector3(-0.0062500136f, -1.95909342E-05f, -4.2826008E-07f),
                new Vector3(-0.00310830516f, -0.00542242732f, 1.40279451E-06f),
                new Vector3(0.00310821575f, 0.00542199844f, -1.03861169E-06f),
                new Vector3(0.00314166117f, -0.00540320156f, 1.09761879E-06f),
                new Vector3(0.0062499335f, 1.916639E-05f, -1.23084291E-07f),
                new Vector3(-0.00310825277f, -0.005516419f, -0.4695158f),
                new Vector3(-0.006249961f, -0.000113589842f, -0.469517618f),
                new Vector3(-0.00314169819f, 0.00530877337f, -0.469518244f),
                new Vector3(0.00310826348f, 0.00532799959f, -0.469518244f),
                new Vector3(0.00314171356f, -0.00549719343f, -0.4695161f),
                new Vector3(0.00624998147f, -7.483252E-05f, -0.469517618f),
                },
                triangles = new[] { 0, 1, 2, 0, 3, 1, 4, 3, 0, 4, 5, 3, 6, 5, 4, 6, 7, 5, 8, 7, 6, 8, 9, 7, 10, 9, 8, 10, 11, 9, 12, 11, 10, 12, 13, 11, 14, 15, 16, 14, 16, 17, 18, 14, 17, 18, 17, 19, 20, 18, 19, 20, 19, 21, 22, 20, 21, 22, 21, 23, 24, 22, 23, 24, 23, 25, 26, 24, 25, 26, 25, 27, 28, 29, 30, 30, 31, 28, 30, 32, 31, 32, 33, 31, 34, 35, 36, 36, 37, 34, 37, 38, 34, 37, 39, 38 },
            };
            UnityEngine.Physics.gravity = Vector3.zero;
        }

        [TearDown]
        public void DestroyMeshes()
        {
            DestroyAll();
            Object.DestroyImmediate(wireMesh);
            Object.DestroyImmediate(groundMesh);
        }

        private Rigidbody Wire(Vector3 position, Quaternion rotation, float angularCap)
        {
            var wire = Track(new GameObject("magnetometer wire"));
            wire.transform.SetPositionAndRotation(position, rotation);
            wire.transform.localScale = new Vector3(1.00000262f, 1.000003f, 0.579111636f);
            var collider = wire.AddComponent<MeshCollider>();
            collider.convex = true;
            collider.sharedMesh = wireMesh;
            var body = wire.AddComponent<Rigidbody>();
            body.useGravity = false;
            body.mass = 0.01f;
            body.inertiaTensor = new Vector3(6.2e-5f, 6.2e-5f, 1.6e-7f);
            body.maxAngularVelocity = angularCap;
            body.maxDepenetrationVelocity = 10f;
            return body;
        }

        private void MeshGround(float extent)
        {
            groundMesh = new Mesh
            {
                vertices = new[]
                {
                    new Vector3(-extent, 0, -extent), new Vector3(-extent, 0, extent),
                    new Vector3(extent, 0, extent), new Vector3(extent, 0, -extent),
                },
                triangles = new[] { 0, 1, 2, 0, 2, 3 },
            };
            Track(new GameObject("mesh pad")).AddComponent<MeshCollider>().sharedMesh = groundMesh;
        }

        [TestCase(10f)]
        [TestCase(150f)]
        public unsafe void WireCrossingMesh(float speed)
        {
            MeshGround(10f);
            var wire = Wire(Vector3.zero, Quaternion.identity, 7f);
            var random = new Unity.Mathematics.Random(1234);
            int contacts = 0;
            for (int i = 0; i < 100; i++)
            {
                wire.rotation = random.NextQuaternionRotation();
                wire.position = new Vector3(random.NextFloat(-1f, 1f), random.NextFloat(-0.19f, 0f), random.NextFloat(-1f, 1f));
                wire.angularVelocity = (Vector3)random.NextFloat3Direction() * 3f;
                wire.linearVelocity = Vector3.up * speed;
                Simulate();
                var state = StepEntryPoints.Current;
                if (state != null)
                {
                    contacts += state->Manifolds.Length;
                    Assert.That(state->Counters.NonFiniteBodies, Is.EqualTo(recoveriesBeforeTest), $"orientation {i}");
                }

                AssertFinite(wire, $"orientation {i}");
            }

            if (BackendInfo.IsRetooled)
                Assert.That(contacts, Is.GreaterThan(0));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void WireApproachingMeshWithMixedReachableContacts(bool fromBelow)
        {
            MeshGround(10f);
            // Replay the first non-finite body's incoming pose and spin from the mesh pile.
            Quaternion turn = fromBelow ? new Quaternion(1f, 0f, 0f, 0f) : Quaternion.identity;
            var wire = Wire(turn * new Vector3(-0.105569400f, 0.305987000f, 0.053452600f),
                turn * new Quaternion(-0.859293300f, 0.222993700f, 0.246101800f, 0.389002200f), 7f);
            wire.linearVelocity = turn * new Vector3(-0.733339400f, -4.977560000f, 0.048843180f);
            wire.angularVelocity = turn * new Vector3(0.820048800f, -4.229281000f, 5.508426000f);

            Simulate();

            AssertFinite(wire, "near the mesh");
            AssertNoRecovery();
        }

        [TestCase(false, 7f)]
        [TestCase(true, 7f)]
        [TestCase(false, 50f)]
        [TestCase(true, 50f)]
        public void WirePile(bool meshPad, float angularCap)
        {
            if (meshPad)
                MeshGround(10f);
            else
                Ground();

            var wires = new Rigidbody[12];
            var previous = new Vector3[wires.Length];
            var random = new Unity.Mathematics.Random(1);
            for (int i = 0; i < wires.Length; i++)
                wires[i] = Wire(Vector3.zero, Quaternion.identity, angularCap);

            for (int i = 0; i < wires.Length; i++)
            {
                wires[i].position = new Vector3(random.NextFloat(-0.05f, 0.05f), 0.27f + i * 0.15f, random.NextFloat(-0.05f, 0.05f));
                wires[i].rotation = random.NextQuaternionRotation();
                wires[i].linearVelocity = Vector3.down * 1.5f;
                wires[i].angularVelocity = (Vector3)random.NextFloat3Direction() * random.NextFloat(0f, 3f);
            }

            float peak = 0f;
            float peakChange = 0f;
            for (int step = 0; step < 80; step++)
            {
                for (int i = 0; i < wires.Length; i++)
                {
                    previous[i] = wires[i].linearVelocity;
                    wires[i].AddForce(Vector3.up * Gravity, ForceMode.Acceleration);
                }

                Simulate();
                AssertNoRecovery();
                for (int i = 0; i < wires.Length; i++)
                {
                    var wire = wires[i];
                    AssertFinite(wire, $"wire {i} at step {step}");
                    float change = (wire.linearVelocity - previous[i]).magnitude;
                    peakChange = Mathf.Max(peakChange, change);
                    peak = Mathf.Max(peak, wire.linearVelocity.magnitude);
                    Assert.That(change, Is.LessThan(15f), $"wire {i} at step {step}");
                }
            }

            TestContext.Out.WriteLine($"Peak speed {peak:R} m/s and peak change {peakChange:R} m/s");
            Assert.That(System.Array.Exists(wires, wire => Mathf.Abs(wire.position.y) < 0.2f && wire.linearVelocity.magnitude < 0.3f),
                "At least one wire should have settled on the pad.");
        }

        private static void AssertFinite(Rigidbody body, string where)
        {
            Quaternion rotation = body.rotation;
            Assert.That(math.all(math.isfinite((float3)body.linearVelocity)), where);
            Assert.That(math.all(math.isfinite((float3)body.angularVelocity)), where);
            Assert.That(math.all(math.isfinite((float3)body.position)), where);
            Assert.That(math.all(math.isfinite(new float4(rotation.x, rotation.y, rotation.z, rotation.w))), where);
        }

        private unsafe void AssertNoRecovery()
        {
            var state = StepEntryPoints.Current;
            if (state != null)
            {
                Assert.That(state->Counters.NonFiniteBodies, Is.EqualTo(recoveriesBeforeTest));
            }
        }
    }
}
