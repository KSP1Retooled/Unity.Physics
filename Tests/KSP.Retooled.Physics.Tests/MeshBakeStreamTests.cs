using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.TestTools;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>
    /// The bytes a content build stores for a pre-baked mesh and hands back at load. Only a content build drives these
    /// commands through Unity, so the tests call the plugin's own path directly.
    /// </summary>
    public unsafe class MeshBakeStreamTests
    {
        private static readonly float3[] Tetrahedron =
        {
            new float3(0.0f, 0.0f, 0.0f), new float3(1.0f, 0.0f, 0.0f), new float3(0.0f, 1.0f, 0.0f), new float3(0.0f, 0.0f, 1.0f),
        };

        private static readonly int[] TetrahedronTriangles = { 0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3 };

        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
        }

        private static byte[] Cook(float3[] vertices, int[] triangles, bool convex)
        {
            fixed (float3* v = vertices)
            fixed (int* t = triangles)
            {
                int size = NativeStore.RetooledPhysics_CookStream(v, vertices.Length, t, triangles.Length, convex ? 1 : 0, null, 0);
                Assert.That(size, Is.GreaterThan(0));

                byte[] stream = new byte[size];
                fixed (byte* output = stream)
                {
                    NativeStore.RetooledPhysics_CookStream(v, vertices.Length, t, triangles.Length, convex ? 1 : 0, output, size);
                }

                return stream;
            }
        }

        private static bool Refused(byte[] stream, bool convex)
        {
            LogAssert.ignoreFailingMessages = true;
            fixed (byte* s = stream)
            {
                NativeStore.MeshData* mesh = NativeStore.RetooledPhysics_LoadStream(s, stream.Length, convex ? 1 : 0);
                if (mesh == null)
                {
                    return true;
                }

                NativeStore.RetooledPhysics_ReleaseMesh(mesh);
                return false;
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void CookingTheSameMeshGivesTheSameBytes(bool convex)
        {
            Assert.That(Cook(Tetrahedron, TetrahedronTriangles, convex), Is.EqualTo(Cook(Tetrahedron, TetrahedronTriangles, convex)));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AStreamLoadsAsTheMeshItCameFrom(bool convex)
        {
            byte[] stream = Cook(Tetrahedron, TetrahedronTriangles, convex);

            fixed (byte* s = stream)
            {
                NativeStore.MeshData* mesh = NativeStore.RetooledPhysics_LoadStream(s, stream.Length, convex ? 1 : 0);
                Assert.That(mesh != null, Is.True);
                try
                {
                    Assert.That(mesh->vertexCount, Is.EqualTo(4));
                    Assert.That(mesh->triangleCount, Is.EqualTo(4));
                    Assert.That(mesh->convex != 0, Is.EqualTo(convex));
                    Assert.That(mesh->upper, Is.EqualTo(new float3(1.0f, 1.0f, 1.0f)));
                    Assert.That(mesh->lower, Is.EqualTo(float3.zero));
                }
                finally
                {
                    NativeStore.RetooledPhysics_ReleaseMesh(mesh);
                }
            }
        }

        [Test]
        public void ConvexCookingWeldsAndDropsInteriorPoints()
        {
            float3[] vertices =
            {
                new float3(0.0f, 0.0f, 0.0f), new float3(1.0f, 0.0f, 0.0f), new float3(0.0f, 1.0f, 0.0f), new float3(0.0f, 0.0f, 1.0f),
                new float3(0.1f, 0.1f, 0.1f), new float3(1.0f, 0.0f, 0.0f),
            };

            byte[] stream = Cook(vertices, new[] { 0, 1, 2 }, true);

            fixed (byte* s = stream)
            {
                NativeStore.MeshData* mesh = NativeStore.RetooledPhysics_LoadStream(s, stream.Length, 1);
                Assert.That(mesh->vertexCount, Is.EqualTo(4));
                Assert.That(mesh->triangleCount, Is.EqualTo(4));
                NativeStore.RetooledPhysics_ReleaseMesh(mesh);
            }
        }

        [Test]
        public void AStreamOfTheOtherKindIsRefused()
        {
            Assert.That(Refused(Cook(Tetrahedron, TetrahedronTriangles, true), false), Is.True);
        }

        [Test]
        public void ATruncatedStreamIsRefused()
        {
            byte[] stream = Cook(Tetrahedron, TetrahedronTriangles, false);
            System.Array.Resize(ref stream, stream.Length - 4);

            Assert.That(Refused(stream, false), Is.True);
        }

        [Test]
        public void ForeignBytesAreRefused()
        {
            // A PhysX stream opens with "NXS".
            byte[] stream = new byte[512];
            stream[0] = (byte)'N';
            stream[1] = (byte)'X';
            stream[2] = (byte)'S';
            stream[3] = 1;

            Assert.That(Refused(stream, true), Is.True);
        }

        [Test]
        public void AnIndexPastTheVerticesIsRefused()
        {
            byte[] stream = Cook(Tetrahedron, TetrahedronTriangles, false);
            stream[stream.Length - 4] = 200;

            Assert.That(Refused(stream, false), Is.True);
        }

        [Test]
        public void VerticesThatArentFiniteFailTheCook()
        {
            float3[] vertices = { new float3(float.NaN, 0.0f, 0.0f), new float3(1.0f, 0.0f, 0.0f), new float3(0.0f, 1.0f, 0.0f) };

            fixed (float3* v = vertices)
            fixed (int* t = new[] { 0, 1, 2 })
            {
                Assert.That(NativeStore.RetooledPhysics_CookStream(v, 3, t, 3, 0, null, 0), Is.EqualTo(-1));
            }
        }

        [Test]
        public void AMeshWithNoUsableTrianglesCooksEmpty()
        {
            byte[] stream = Cook(Tetrahedron, new[] { 0, 0, 1 }, false);

            fixed (byte* s = stream)
            {
                NativeStore.MeshData* mesh = NativeStore.RetooledPhysics_LoadStream(s, stream.Length, 0);
                Assert.That(mesh != null, Is.True);
                Assert.That(mesh->triangleCount, Is.EqualTo(0));
                NativeStore.RetooledPhysics_ReleaseMesh(mesh);
            }
        }
    }
}
