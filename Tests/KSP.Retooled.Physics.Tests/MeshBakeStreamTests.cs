using NUnit.Framework;
using Unity.Mathematics;
using Unity.Physics.Store.Shapes;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Tests the bytes stored for a pre-baked mesh by a content build and their validation when loaded.
    /// </summary>
    public unsafe class MeshBakeStreamTests
    {
        private static readonly float3[] Tetrahedron =
        {
            new float3(0.0f, 0.0f, 0.0f), new float3(1.0f, 0.0f, 0.0f), new float3(0.0f, 1.0f, 0.0f), new float3(0.0f, 0.0f, 1.0f),
        };

        private static readonly int[] TetrahedronTriangles = { 0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3 };

        private static int CookStatus(float3[] vertices, int[] triangles, bool convex, out byte[] stream)
        {
            fixed (float3* v = vertices)
            fixed (int* t = triangles)
            {
                CookInput input = new CookInput
                {
                    vertices = v,
                    indices = t,
                    vertexCount = (uint)vertices.Length,
                    indexCount = (uint)triangles.Length,
                    vertexStride = (uint)sizeof(float3),
                    indexStride = sizeof(int),
                    convex = convex ? 1u : 0u,
                };

                CookOutput output = default;
                MeshCook.Cook(&input, &output);

                byte[] buffer = new byte[output.size];
                fixed (byte* b = buffer)
                {
                    output.bytes = b;
                    MeshCook.Cook(&input, &output);
                }

                System.Array.Resize(ref buffer, (int)output.size);
                stream = buffer;
                return output.status;
            }
        }

        private static byte[] Cook(float3[] vertices, int[] triangles, bool convex)
        {
            CookStatus(vertices, triangles, convex, out byte[] stream);
            return stream;
        }

        private static bool Accepted(byte[] stream, bool convex)
        {
            fixed (byte* s = stream)
            {
                return MeshCook.Validate(s, stream.Length, convex ? 1 : 0) != 0;
            }
        }

        private static MeshData Header(byte[] stream)
        {
            fixed (byte* s = stream)
            {
                return *(MeshData*)s;
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
        public void AStreamIsAcceptedAndHoldsTheMesh(bool convex)
        {
            byte[] stream = Cook(Tetrahedron, TetrahedronTriangles, convex);

            Assert.That(Accepted(stream, convex), Is.True);
            MeshData mesh = Header(stream);
            Assert.That(mesh.vertexCount, Is.EqualTo(4));
            Assert.That(mesh.triangleCount, Is.EqualTo(4));
            Assert.That(mesh.lower, Is.EqualTo(float3.zero));
            Assert.That(mesh.upper, Is.EqualTo(new float3(1.0f, 1.0f, 1.0f)));
        }

        [Test]
        public void ATriangleMeshsBoundsCoverOnlyItsTriangles()
        {
            float3[] vertices =
            {
                new float3(0.0f, 0.0f, 0.0f), new float3(1.0f, 0.0f, 0.0f), new float3(0.0f, 1.0f, 0.0f), new float3(50.0f, 0.0f, 0.0f),
            };

            MeshData mesh = Header(Cook(vertices, new[] { 0, 2, 1 }, false));

            Assert.That(mesh.upper, Is.EqualTo(new float3(1.0f, 1.0f, 0.0f)));
        }

        [Test]
        public void AConvexMeshsBoundsCoverEveryVertex()
        {
            float3[] vertices =
            {
                new float3(0.0f, 0.0f, 0.0f), new float3(1.0f, 0.0f, 0.0f), new float3(0.0f, 1.0f, 0.0f), new float3(0.0f, 0.0f, 1.0f),
                new float3(50.0f, 0.0f, 0.0f),
            };

            MeshData mesh = Header(Cook(vertices, new[] { 0, 1, 2 }, true));

            Assert.That(mesh.upper, Is.EqualTo(new float3(50.0f, 1.0f, 1.0f)));
        }

        [Test]
        public void AConvexMeshNeedsNoIndices()
        {
            fixed (float3* v = Tetrahedron)
            {
                CookInput input = new CookInput
                {
                    vertices = v,
                    vertexCount = (uint)Tetrahedron.Length,
                    vertexStride = (uint)sizeof(float3),
                    convex = 1u,
                };

                CookOutput output = default;
                MeshCook.Cook(&input, &output);
                byte[] buffer = new byte[output.size];
                fixed (byte* b = buffer)
                {
                    output.bytes = b;
                    MeshCook.Cook(&input, &output);
                }

                Assert.That(output.status, Is.EqualTo((int)MeshCookStatus.Ok));
                MeshData mesh = Header(buffer);
                Assert.That(mesh.vertexCount, Is.EqualTo(4));
                Assert.That(mesh.triangleCount, Is.EqualTo(0));
            }
        }

        [Test]
        public void AStreamOfTheOtherKindIsRefused()
        {
            Assert.That(Accepted(Cook(Tetrahedron, TetrahedronTriangles, true), false), Is.False);
        }

        [Test]
        public void ATruncatedStreamIsRefused()
        {
            byte[] stream = Cook(Tetrahedron, TetrahedronTriangles, false);
            System.Array.Resize(ref stream, stream.Length - 4);

            Assert.That(Accepted(stream, false), Is.False);
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

            Assert.That(Accepted(stream, true), Is.False);
        }

        [Test]
        public void AnIndexPastTheVerticesIsRefused()
        {
            byte[] stream = Cook(Tetrahedron, TetrahedronTriangles, false);
            stream[stream.Length - 4] = 200;

            Assert.That(Accepted(stream, false), Is.False);
        }

        [Test]
        public void VerticesThatArentFiniteFailTheCook()
        {
            float3[] vertices = { new float3(float.NaN, 0.0f, 0.0f), new float3(1.0f, 0.0f, 0.0f), new float3(0.0f, 1.0f, 0.0f) };

            Assert.That(CookStatus(vertices, new[] { 0, 1, 2 }, false, out _), Is.EqualTo((int)MeshCookStatus.BadVertices));
        }

        [Test]
        public void AnIndexPastTheSourceVerticesFailsTheCook()
        {
            Assert.That(CookStatus(Tetrahedron, new[] { 0, 1, 9 }, false, out _), Is.EqualTo((int)MeshCookStatus.BadIndices));
        }

        [Test]
        public void AMeshWithNoUsableTrianglesCooksEmpty()
        {
            Assert.That(CookStatus(Tetrahedron, new[] { 0, 0, 1 }, false, out byte[] stream), Is.EqualTo((int)MeshCookStatus.Empty));
            Assert.That(Header(stream).triangleCount, Is.EqualTo(0));
            Assert.That(Accepted(stream, false), Is.True);
        }

        [Test]
        public void AFlatConvexMeshCooksASlab()
        {
            float3[] vertices =
            {
                new float3(-2.0f, 0.0f, -2.0f), new float3(-2.0f, 0.0f, 2.0f), new float3(2.0f, 0.0f, 2.0f), new float3(2.0f, 0.0f, -2.0f),
            };

            MeshData mesh = Header(Cook(vertices, new[] { 0, 1, 2, 0, 2, 3 }, true));

            Assert.That(mesh.vertexCount, Is.EqualTo(8));
            Assert.That(mesh.upper.y, Is.EqualTo(0.2f).Within(1e-5f));
        }

        [Test]
        public void ACloudFlatToWithinRoundingCooksASlab()
        {
            var random = new Unity.Mathematics.Random(1234);
            float3[] vertices = new float3[2000];
            for (int i = 0; i < vertices.Length; i++)
            {
                float2 flat = random.NextFloat2(-1.0f, 1.0f);
                vertices[i] = new float3(flat.x, random.NextFloat(-1e-6f, 1e-6f) + flat.y * 0.25f, flat.y);
            }

            CookStatus(vertices, new[] { 0, 1, 2 }, true, out byte[] stream);

            Assert.That(Accepted(stream, true), Is.True);
            Assert.That(Header(stream).vertexCount, Is.EqualTo(vertices.Length * 2));
        }

        [Test]
        public void ALineHasNoConvexCollider()
        {
            float3[] vertices = { new float3(0.0f), new float3(1.0f, 0.0f, 0.0f), new float3(2.0f, 0.0f, 0.0f) };

            Assert.That(CookStatus(vertices, new[] { 0, 1, 2 }, true, out _), Is.EqualTo((int)MeshCookStatus.Empty));
        }
    }
}
