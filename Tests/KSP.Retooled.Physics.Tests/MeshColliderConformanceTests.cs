using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>What a MeshCollider does once its mesh is cooked: rays, mass, bounds, contacts and mesh lifetime.</summary>
    public class MeshColliderConformanceTests : PhysicsScene
    {
        private readonly List<Object> assets = new List<Object>();

        [TearDown]
        public void DestroyAssets()
        {
            foreach (Object asset in assets)
            {
                Object.DestroyImmediate(asset);
            }

            assets.Clear();
        }

        private T TrackAsset<T>(T asset) where T : Object
        {
            assets.Add(asset);
            return asset;
        }

        /// <summary>A closed box with outward winding.</summary>
        private Mesh BoxMesh(Vector3 center, Vector3 size)
        {
            Vector3 h = size * 0.5f;
            Vector3[] corners =
            {
                center + new Vector3(-h.x, -h.y, -h.z),
                center + new Vector3(h.x, -h.y, -h.z),
                center + new Vector3(h.x, h.y, -h.z),
                center + new Vector3(-h.x, h.y, -h.z),
                center + new Vector3(-h.x, -h.y, h.z),
                center + new Vector3(h.x, -h.y, h.z),
                center + new Vector3(h.x, h.y, h.z),
                center + new Vector3(-h.x, h.y, h.z),
            };

            int[] triangles =
            {
                0, 2, 1, 0, 3, 2, // -z
                4, 5, 6, 4, 6, 7, // +z
                0, 1, 5, 0, 5, 4, // -y
                3, 7, 6, 3, 6, 2, // +y
                0, 4, 7, 0, 7, 3, // -x
                1, 2, 6, 1, 6, 5, // +x
            };

            return MeshOf(corners, triangles);
        }

        private Mesh MeshOf(Vector3[] vertices, int[] triangles)
        {
            Mesh mesh = TrackAsset(new Mesh());
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>The tetrahedron with unit legs along the axes, whose moments have a closed form.</summary>
        private Mesh RightTetrahedron()
        {
            return MeshOf(
                new[] { Vector3.zero, Vector3.right, Vector3.up, Vector3.forward },
                new[] { 0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3 });
        }

        /// <summary>A flat square facing +y, one-sided, and open.</summary>
        private Mesh Floor(float size)
        {
            float h = size * 0.5f;
            return MeshOf(
                new[] { new Vector3(-h, 0, -h), new Vector3(-h, 0, h), new Vector3(h, 0, h), new Vector3(h, 0, -h) },
                new[] { 0, 1, 2, 0, 2, 3 });
        }

        private MeshCollider Collider(Mesh mesh, bool convex, Vector3 position = default)
        {
            GameObject gameObject = Track(new GameObject("mesh"));
            gameObject.transform.position = position;
            MeshCollider collider = gameObject.AddComponent<MeshCollider>();
            collider.convex = convex;
            collider.sharedMesh = mesh;
            return collider;
        }

        private static float RayDistance(Vector3 origin, Vector3 direction)
        {
            Physics.SyncTransforms();
            return Physics.Raycast(origin, direction, out RaycastHit hit, 100.0f) ? hit.distance : -1.0f;
        }

        [Test]
        public void ARayHitsATriangleMeshAtItsSurface()
        {
            MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one * 2.0f), false);

            Physics.SyncTransforms();
            bool hit = Physics.Raycast(new Vector3(0.25f, 5.0f, 0.25f), Vector3.down, out RaycastHit result, 100.0f);

            Assert.That(hit, Is.True);
            Assert.That(result.collider, Is.SameAs(collider));
            Assert.That(result.distance, Is.EqualTo(4.0f).Within(1e-3f));
            AssertVector(Vector3.up, result.normal, 1e-3f, "normal");
        }

        [Test]
        public void ARayHitsAConvexMeshAtItsHull()
        {
            // A ray along the diagonal face of the tetrahedron, whose plane is x + y + z = 1.
            MeshCollider collider = Collider(RightTetrahedron(), true);

            Physics.SyncTransforms();
            Vector3 towards = new Vector3(-1.0f, -1.0f, -1.0f).normalized;
            bool hit = Physics.Raycast(Vector3.one, towards, out RaycastHit result, 100.0f);

            Assert.That(hit, Is.True);
            Assert.That(result.collider, Is.SameAs(collider));
            Assert.That(result.distance, Is.EqualTo(Vector3.one.magnitude - 1.0f / Mathf.Sqrt(3.0f)).Within(1e-3f));
            AssertVector(-towards, result.normal, 1e-3f, "normal");
        }

        [Test]
        public void AConvexMeshIsItsHullEvenWhereTheMeshIsConcave()
        {
            // A dent in the top of a block: the convex collider covers it and the triangle mesh doesn't.
            Vector3[] vertices =
            {
                new Vector3(-1, 0, -1), new Vector3(1, 0, -1), new Vector3(1, 0, 1), new Vector3(-1, 0, 1),
                new Vector3(-1, 1, -1), new Vector3(1, 1, -1), new Vector3(1, 1, 1), new Vector3(-1, 1, 1),
                new Vector3(0, 0.25f, 0),
            };

            int[] triangles =
            {
                0, 1, 2, 0, 2, 3,
                0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0,
                4, 7, 8, 7, 6, 8, 6, 5, 8, 5, 4, 8,
            };

            Mesh mesh = MeshOf(vertices, triangles);
            Collider(mesh, true, new Vector3(-5.0f, 0.0f, 0.0f));
            Collider(mesh, false, new Vector3(5.0f, 0.0f, 0.0f));

            Assert.That(RayDistance(new Vector3(-5.0f, 3.0f, 0.0f), Vector3.down), Is.EqualTo(2.0f).Within(1e-3f));
            Assert.That(RayDistance(new Vector3(5.0f, 3.0f, 0.0f), Vector3.down), Is.EqualTo(2.75f).Within(1e-3f));
        }

        [Test]
        public void ARayFromInsideATriangleMeshPassesThroughItsBackfaces()
        {
            Collider(BoxMesh(Vector3.zero, Vector3.one * 2.0f), false);

            Assert.That(RayDistance(Vector3.zero, Vector3.up), Is.EqualTo(-1.0f));
        }

        [Test]
        public void ARayFromBehindAFloorPassesThrough()
        {
            Collider(Floor(10.0f), false);

            Assert.That(RayDistance(new Vector3(0.5f, -1.0f, 0.25f), Vector3.up), Is.EqualTo(-1.0f));
            Assert.That(RayDistance(new Vector3(0.5f, 1.0f, 0.25f), Vector3.down), Is.EqualTo(1.0f).Within(1e-3f));
        }

        [Test]
        public void BackfacesCountWhenTheProjectAsksForThem()
        {
            bool before = Physics.queriesHitBackfaces;
            Collider(BoxMesh(Vector3.zero, Vector3.one * 2.0f), false);
            try
            {
                Physics.queriesHitBackfaces = true;

                Assert.That(RayDistance(Vector3.zero, Vector3.up), Is.EqualTo(1.0f).Within(1e-3f));
            }
            finally
            {
                Physics.queriesHitBackfaces = before;
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void TheTransformsScaleReachesTheCollider(bool convex)
        {
            MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one), convex);
            collider.transform.localScale = new Vector3(2.0f, 4.0f, 8.0f);

            Assert.That(RayDistance(new Vector3(10.0f, 0.1f, 0.1f), Vector3.left), Is.EqualTo(9.0f).Within(1e-3f));
            Assert.That(RayDistance(new Vector3(0.1f, 10.0f, 0.1f), Vector3.down), Is.EqualTo(8.0f).Within(1e-3f));
            Assert.That(RayDistance(new Vector3(0.1f, 0.1f, 10.0f), Vector3.back), Is.EqualTo(6.0f).Within(1e-3f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AMirroringScaleKeepsTheOutsideOutside(bool convex)
        {
            MeshCollider collider = Collider(BoxMesh(new Vector3(1.0f, 0.0f, 0.0f), Vector3.one), convex);
            collider.transform.localScale = new Vector3(-1.0f, 1.0f, 1.0f);

            Assert.That(RayDistance(new Vector3(-5.0f, 0.1f, 0.1f), Vector3.right), Is.EqualTo(3.5f).Within(1e-3f));
            Assert.That(RayDistance(new Vector3(5.0f, 0.1f, 0.1f), Vector3.left), Is.EqualTo(5.5f).Within(1e-3f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void TheMeshFollowsItsBody(bool convex)
        {
            MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one), convex, new Vector3(3.0f, 0.0f, 0.0f));
            Rigidbody body = collider.gameObject.AddComponent<Rigidbody>();
            body.isKinematic = true;

            Assert.That(RayDistance(new Vector3(3.0f, 5.0f, 0.0f), Vector3.down), Is.EqualTo(4.5f).Within(1e-3f));

            body.transform.position = new Vector3(3.0f, 2.0f, 0.0f);

            Assert.That(RayDistance(new Vector3(3.0f, 5.0f, 0.0f), Vector3.down), Is.EqualTo(2.5f).Within(1e-3f));
        }

        [Test]
        public void AMeshBesideAnotherColliderOnOneBodyIsHitAsItself()
        {
            GameObject root = Track(new GameObject("root"));
            root.AddComponent<Rigidbody>().isKinematic = true;
            BoxCollider box = root.AddComponent<BoxCollider>();
            GameObject child = new GameObject("child");
            child.transform.SetParent(root.transform);
            child.transform.localPosition = new Vector3(4.0f, 0.0f, 0.0f);
            MeshCollider mesh = child.AddComponent<MeshCollider>();
            mesh.convex = false;
            mesh.sharedMesh = BoxMesh(Vector3.zero, Vector3.one);

            Physics.SyncTransforms();

            Assert.That(Physics.Raycast(new Vector3(4.1f, 5.0f, 0.1f), Vector3.down, out RaycastHit onMesh, 100.0f), Is.True);
            Assert.That(onMesh.collider, Is.SameAs(mesh));
            Assert.That(Physics.Raycast(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down, out RaycastHit onBox, 100.0f), Is.True);
            Assert.That(onBox.collider, Is.SameAs(box));
        }

        [Test]
        public void ChangingTheMeshChangesTheCollider()
        {
            MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one), false);
            Assert.That(RayDistance(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down), Is.EqualTo(4.5f).Within(1e-3f));

            collider.sharedMesh = BoxMesh(Vector3.zero, Vector3.one * 4.0f);

            Assert.That(RayDistance(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down), Is.EqualTo(3.0f).Within(1e-3f));
        }

        [Test]
        public void RemovingTheMeshRemovesTheCollider()
        {
            MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one), false);
            Assert.That(RayDistance(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down), Is.GreaterThan(0.0f));

            collider.sharedMesh = null;

            Assert.That(RayDistance(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down), Is.EqualTo(-1.0f));
        }

        [Test]
        public void DestroyingAMeshInUseLeavesTheWorldSteppable()
        {
            Mesh mesh = BoxMesh(Vector3.zero, Vector3.one);
            Collider(mesh, false);
            Collider(mesh, false, new Vector3(5.0f, 0.0f, 0.0f));
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            Simulate();

            assets.Remove(mesh);
            Object.DestroyImmediate(mesh);
            Simulate(5);
            Physics.SyncTransforms();
            Physics.Raycast(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down, 100.0f);

            Assert.That(body.position.y, Is.LessThan(5.0f));
        }

        [Test]
        public void TheMeshIsSharedByItsColliders()
        {
            Mesh mesh = BoxMesh(Vector3.zero, Vector3.one);
            Collider(mesh, false, new Vector3(-5.0f, 0.0f, 0.0f));
            MeshCollider second = Collider(mesh, false, new Vector3(5.0f, 0.0f, 0.0f));

            Object.DestroyImmediate(second.gameObject);

            Assert.That(RayDistance(new Vector3(-5.1f, 5.0f, 0.1f), Vector3.down), Is.EqualTo(4.5f).Within(1e-3f));
        }

        [Test]
        public void AConvexMeshsCentreOfMassIsItsCentroid()
        {
            MeshCollider collider = Collider(BoxMesh(new Vector3(1.0f, 2.0f, 3.0f), new Vector3(1.0f, 2.0f, 3.0f)), true);
            Rigidbody body = collider.gameObject.AddComponent<Rigidbody>();

            body.mass = 12.0f;

            AssertVector(new Vector3(1.0f, 2.0f, 3.0f), body.centerOfMass, 1e-3f, "centre of mass");
            AssertVector(new Vector3(13.0f, 10.0f, 5.0f), body.inertiaTensor, 0.02f, "inertia");
        }

        [Test]
        public void AConvexMeshsInertiaIsOfItsSolid()
        {
            MeshCollider collider = Collider(RightTetrahedron(), true);
            Rigidbody body = collider.gameObject.AddComponent<Rigidbody>();

            body.mass = 80.0f;

            AssertVector(new Vector3(0.25f, 0.25f, 0.25f), body.centerOfMass, 1e-3f, "centre of mass");
            float[] moments = { body.inertiaTensor.x, body.inertiaTensor.y, body.inertiaTensor.z };
            System.Array.Sort(moments);
            Assert.That(moments[0], Is.EqualTo(5.0f).Within(0.01f));
            Assert.That(moments[1], Is.EqualTo(5.0f).Within(0.01f));
            Assert.That(moments[2], Is.EqualTo(8.0f).Within(0.01f));
        }

        [Test]
        public void AScaledConvexMeshsMassPropertiesScale()
        {
            MeshCollider collider = Collider(BoxMesh(new Vector3(1.0f, 1.0f, 1.0f), Vector3.one), true);
            collider.transform.localScale = new Vector3(1.0f, 2.0f, 3.0f);
            Rigidbody body = collider.gameObject.AddComponent<Rigidbody>();

            body.mass = 12.0f;

            AssertVector(new Vector3(1.0f, 2.0f, 3.0f), body.centerOfMass, 1e-3f, "centre of mass");
            AssertVector(new Vector3(13.0f, 10.0f, 5.0f), body.inertiaTensor, 0.02f, "inertia");
        }

        [Test]
        public void AConvexMeshSharesTheBodysMassByVolume()
        {
            MeshCollider collider = Collider(BoxMesh(new Vector3(2.0f, 0.0f, 0.0f), Vector3.one), true);
            BoxCollider box = collider.gameObject.AddComponent<BoxCollider>();
            box.size = Vector3.one * 2.0f;
            Rigidbody body = collider.gameObject.AddComponent<Rigidbody>();

            body.mass = 9.0f;

            // Volumes 1 and 8, so the centroid sits 2/9 of the way to the mesh.
            AssertVector(new Vector3(2.0f / 9.0f, 0.0f, 0.0f), body.centerOfMass, 1e-3f, "centre of mass");
        }

        [Test]
        public void ABoxShapedMeshHasTheBoundsOfTheBox()
        {
            MeshCollider collider = Collider(BoxMesh(new Vector3(1.0f, 0.0f, 0.0f), new Vector3(1.0f, 2.0f, 3.0f)), true);
            collider.transform.position = new Vector3(0.0f, 5.0f, 0.0f);
            collider.transform.localScale = new Vector3(2.0f, 1.0f, 1.0f);

            Physics.SyncTransforms();

            Bounds bounds = collider.bounds;
            AssertVector(new Vector3(2.0f, 5.0f, 0.0f), bounds.center, 0.05f, "center");
            AssertVector(new Vector3(1.0f, 1.0f, 1.5f), bounds.extents, 0.05f, "extents");
        }

        [Test]
        public void ARotatedMeshsBoundsContainIt()
        {
            MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one), false);
            collider.transform.rotation = Quaternion.Euler(0.0f, 45.0f, 0.0f);

            Physics.SyncTransforms();

            Assert.That(collider.bounds.extents.x, Is.EqualTo(Mathf.Sqrt(0.5f)).Within(0.05f));
            Assert.That(collider.bounds.extents.y, Is.EqualTo(0.5f).Within(0.05f));
        }

        [Test]
        public void ABoxRestsOnATriangleMeshFloor()
        {
            Collider(Floor(20.0f), false);
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));

            Simulate(150);

            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.05f));
        }

        [Test]
        public void AConvexMeshRestsOnGround()
        {
            Ground();
            MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one), true, new Vector3(0.0f, 1.0f, 0.0f));
            Rigidbody body = collider.gameObject.AddComponent<Rigidbody>();

            Simulate(150);

            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.05f));
        }

        [Test]
        public void AConvexMeshRestsOnATriangleMeshFloor()
        {
            Collider(Floor(20.0f), false);
            MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one), true, new Vector3(0.0f, 1.0f, 0.0f));
            Rigidbody body = collider.gameObject.AddComponent<Rigidbody>();

            Simulate(150);

            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.05f));
        }

        [Test]
        public void ABoxRestsOnAKinematicTriangleMesh()
        {
            MeshCollider floor = Collider(Floor(20.0f), false);
            floor.gameObject.AddComponent<Rigidbody>().isKinematic = true;
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));

            Simulate(150);

            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.05f));
        }

        [Test]
        public void ABoxFallsThroughATriggerMesh()
        {
            MeshCollider floor = Collider(BoxMesh(new Vector3(0.0f, -0.5f, 0.0f), new Vector3(20.0f, 1.0f, 20.0f)), true);
            floor.isTrigger = true;
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));

            Simulate(100);

            Assert.That(body.position.y, Is.LessThan(-1.0f));
        }

        [Test]
        public void AFlatConvexMeshGainsThickness()
        {
            Collider(Floor(4.0f), true);

            Physics.SyncTransforms();
            bool hit = Physics.Raycast(new Vector3(0.5f, 2.0f, 0.5f), Vector3.down, out RaycastHit result, 100.0f);

            Assert.That(hit, Is.True);
            Assert.That(result.distance, Is.EqualTo(1.8f).Within(1e-2f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void BakingFirstGivesTheSameCollider(bool convex)
        {
            Mesh mesh = BoxMesh(Vector3.zero, Vector3.one * 2.0f);

            Physics.BakeMesh(mesh.GetEntityId(), convex);
            Collider(mesh, convex);

            Assert.That(RayDistance(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down), Is.EqualTo(4.0f).Within(1e-3f));
        }

        private struct BakeJob : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<EntityId> Meshes;

            public bool Convex;

            public void Execute(int index)
            {
                Physics.BakeMesh(Meshes[index], Convex);
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void MeshesBakeInParallelFromJobs(bool convex)
        {
            Mesh[] meshes = Enumerable.Range(1, 32).Select(i => BoxMesh(Vector3.zero, Vector3.one * i)).ToArray();
            using (var ids = new NativeArray<EntityId>(meshes.Select(m => m.GetEntityId()).ToArray(), Allocator.TempJob))
            {
                new BakeJob { Meshes = ids, Convex = convex }.Schedule(ids.Length, 1).Complete();
            }

            for (int i = 0; i < meshes.Length; i++)
            {
                Collider(meshes[i], convex, new Vector3(1000.0f * i, 0.0f, 0.0f));
            }

            float expected = 50.0f - 0.5f * meshes.Length;
            Assert.That(RayDistance(new Vector3(1000.0f * (meshes.Length - 1), 50.0f, 0.1f), Vector3.down), Is.EqualTo(expected).Within(1e-2f));
        }

        [Test]
        public void DestroyingAMaterialInUseLeavesTheShapeWorking()
        {
            PhysicsMaterial material = new PhysicsMaterial { dynamicFriction = 0.2f };
            GameObject ground = Ground();
            BoxCollider collider = ground.GetComponent<BoxCollider>();
            collider.sharedMaterial = material;
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));
            Simulate();

            Object.DestroyImmediate(material);
            ground.transform.position += Vector3.up * 0.01f;
            Simulate(100);

            Assert.That(body.position.y, Is.EqualTo(0.51f).Within(0.05f));
        }

        [Test]
        public void ANonConvexMeshOnADynamicBodyIsLeftOutOfTheSimulation()
        {
            LogAssert.ignoreFailingMessages = true;
            try
            {
                Ground();
                MeshCollider collider = Collider(BoxMesh(Vector3.zero, Vector3.one), false, new Vector3(0.0f, 3.0f, 0.0f));
                Rigidbody body = collider.gameObject.AddComponent<Rigidbody>();

                Simulate(100);
                // Unity keeps such a shape away from the body: it neither lands nor takes rays, and the ray reaches the ground.
                Assert.That(body.position.y, Is.LessThan(0.0f));
                Assert.That(RayDistance(new Vector3(0.1f, 20.0f, 0.1f), Vector3.down), Is.EqualTo(20.0f).Within(1e-3f));
            }
            finally
            {
                LogAssert.ignoreFailingMessages = false;
            }
        }
    }
}
