using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>What asks about one collider or one body: Physics.ComputePenetration, Collider.Raycast and Rigidbody.SweepTest.</summary>
    public class ShapeQueryConformanceTests : QueryScene
    {
        private const float Tolerance = 3e-3f;

        private static readonly Quaternion Upright = Quaternion.identity;

        private static bool Penetration(Collider a, Vector3 positionA, Collider b, Vector3 positionB, out Vector3 direction, out float distance)
        {
            return UnityEngine.Physics.ComputePenetration(a, positionA, Upright, b, positionB, Upright, out direction, out distance);
        }

        [Test]
        public void PenetrationPushesTheFirstColliderOut()
        {
            Collider a = Box(Vector3.zero, Vector3.one).GetComponent<Collider>();
            Collider b = Box(Vector3.right, Vector3.one).GetComponent<Collider>();

            Assert.That(Penetration(a, Vector3.zero, b, new Vector3(0.75f, 0.0f, 0.0f), out Vector3 direction, out float distance), Is.True);

            AssertVector(Vector3.left, direction, Tolerance);
            Assert.That(distance, Is.EqualTo(0.25f).Within(Tolerance));
        }

        [Test]
        public void PenetrationTakesTheShortestWayOut()
        {
            Collider a = Box(Vector3.zero, Vector3.one).GetComponent<Collider>();
            Collider b = Box(Vector3.right, Vector3.one).GetComponent<Collider>();

            Penetration(a, Vector3.zero, b, new Vector3(0.1f, 0.0f, 0.0f), out Vector3 direction, out float distance);

            AssertVector(Vector3.left, direction, Tolerance);
            Assert.That(distance, Is.EqualTo(0.9f).Within(Tolerance));
        }

        [Test]
        public void CollidersApartDoNotPenetrate()
        {
            Collider a = Box(Vector3.zero, Vector3.one).GetComponent<Collider>();
            Collider b = Box(Vector3.right, Vector3.one).GetComponent<Collider>();

            Assert.That(Penetration(a, Vector3.zero, b, new Vector3(1.5f, 0.0f, 0.0f), out Vector3 _, out float _), Is.False);
        }

        [Test]
        public void CollidersTouchingPenetrateByNothing()
        {
            Collider a = Box(Vector3.zero, Vector3.one).GetComponent<Collider>();
            Collider b = Box(Vector3.right, Vector3.one).GetComponent<Collider>();

            Assert.That(Penetration(a, Vector3.zero, b, Vector3.right, out Vector3 _, out float distance), Is.True);
            Assert.That(distance, Is.EqualTo(0.0f).Within(Tolerance));
        }

        [Test]
        public void PenetrationGoesByTheCentresGiven()
        {
            Collider a = Box(Vector3.zero, Vector3.one).GetComponent<Collider>();
            BoxCollider b = Box(Vector3.right, Vector3.one).GetComponent<BoxCollider>();
            b.center = Vector3.right;

            Assert.That(Penetration(a, Vector3.zero, b, new Vector3(0.75f, 0.0f, 0.0f), out Vector3 _, out float _), Is.False);
            Assert.That(Penetration(a, Vector3.zero, b, new Vector3(-0.25f, 0.0f, 0.0f), out Vector3 direction, out float distance), Is.True);
            AssertVector(Vector3.left, direction, Tolerance);
            Assert.That(distance, Is.EqualTo(0.25f).Within(Tolerance));
        }

        [Test]
        public void ADisabledColliderDoesNotPenetrate()
        {
            Collider a = Box(Vector3.zero, Vector3.one).GetComponent<Collider>();
            Collider b = Box(Vector3.right, Vector3.one).GetComponent<Collider>();
            b.enabled = false;

            Assert.That(Penetration(a, Vector3.zero, b, new Vector3(0.75f, 0.0f, 0.0f), out Vector3 _, out float _), Is.False);
        }

        [Test]
        public void ConvexMeshesAndSpheresPenetrate()
        {
            Collider convex = ConvexCube(new Vector3(0.0f, 30.0f, 0.0f)).GetComponent<Collider>();
            Collider sphere = Sphere(new Vector3(0.0f, 60.0f, 0.0f), 0.5f);

            Assert.That(Penetration(convex, new Vector3(0.0f, 30.0f, 0.0f), sphere, new Vector3(0.75f, 30.0f, 0.0f), out Vector3 direction, out float distance), Is.True);
            AssertVector(Vector3.left, direction, Tolerance);
            Assert.That(distance, Is.EqualTo(0.25f).Within(Tolerance));

            Assert.That(Penetration(convex, Vector3.zero, convex, Vector3.up * 0.5f, out direction, out distance), Is.True);
            AssertVector(Vector3.down, direction, Tolerance);
            Assert.That(distance, Is.EqualTo(0.5f).Within(Tolerance));
        }

        [Test]
        public void PenetrationDoesNotGoByLayers()
        {
            Collider a = ConvexCube(Vector3.zero).GetComponent<Collider>();
            GameObject other = ConvexCube(new Vector3(0.0f, 0.0f, 5.0f));
            bool ignored = UnityEngine.Physics.GetIgnoreLayerCollision(28, 29);
            UnityEngine.Physics.IgnoreLayerCollision(28, 29, true);
            try
            {
                a.gameObject.layer = 28;
                other.layer = 29;
                UnityEngine.Physics.SyncTransforms();

                Assert.That(Penetration(a, Vector3.zero, other.GetComponent<Collider>(), Vector3.right * 0.75f, out Vector3 _, out float _), Is.True);
            }
            finally
            {
                UnityEngine.Physics.IgnoreLayerCollision(28, 29, ignored);
            }
        }

        [Test]
        public void ATriangleMeshPushesOutOfItsFrontOnly()
        {
            GameObject quad = TriangleMeshQuad(Vector3.zero);
            Collider mesh = quad.GetComponent<Collider>();
            Collider box = Box(new Vector3(0.0f, 0.0f, 20.0f), Vector3.one).GetComponent<Collider>();

            Assert.That(Penetration(mesh, Vector3.zero, box, new Vector3(0.0f, 0.0f, -0.25f), out Vector3 direction, out float distance), Is.True);
            AssertVector(Vector3.forward, direction, Tolerance);
            Assert.That(distance, Is.EqualTo(0.25f).Within(Tolerance));

            Assert.That(Penetration(box, new Vector3(0.0f, 0.0f, -0.25f), mesh, Vector3.zero, out direction, out distance), Is.True);
            AssertVector(Vector3.back, direction, Tolerance);
            Assert.That(distance, Is.EqualTo(0.25f).Within(Tolerance));

            Assert.That(Penetration(mesh, Vector3.zero, box, new Vector3(0.0f, 0.0f, 0.25f), out Vector3 _, out float _), Is.False);
            Assert.That(Penetration(box, new Vector3(0.0f, 0.0f, 0.25f), mesh, Vector3.zero, out Vector3 _, out float _), Is.False);
        }

        [Test]
        public void TwoTriangleMeshesDoNotPenetrate()
        {
            Collider quad = TriangleMeshQuad(Vector3.zero).GetComponent<Collider>();
            Collider cube = TriangleMeshCube(new Vector3(0.0f, 0.0f, 9.0f)).GetComponent<Collider>();

            Assert.That(Penetration(quad, Vector3.zero, cube, Vector3.zero, out Vector3 _, out float _), Is.False);
        }

        [Test]
        public void ColliderRaycastSeesOnlyThatCollider()
        {
            Box(new Vector3(0.0f, 0.0f, 2.0f), Vector3.one);
            Collider box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one).GetComponent<Collider>();

            Assert.That(box.Raycast(new Ray(Vector3.zero, Vector3.forward), out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.collider, Is.SameAs(box));
            Assert.That(hit.distance, Is.EqualTo(4.5f).Within(Tolerance));
            AssertVector(new Vector3(0.0f, 0.0f, 4.5f), hit.point, Tolerance);
            AssertVector(Vector3.back, hit.normal, Tolerance);
            Assert.That(hit.triangleIndex, Is.EqualTo(-1));
        }

        [Test]
        public void ColliderRaycastHonoursItsDistance()
        {
            Collider box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one).GetComponent<Collider>();

            Assert.That(box.Raycast(new Ray(Vector3.zero, Vector3.forward), out RaycastHit _, 1.0f), Is.False);
        }

        [Test]
        public void ColliderRaycastFromInsideMisses()
        {
            Collider box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one).GetComponent<Collider>();

            Assert.That(box.Raycast(new Ray(new Vector3(0.0f, 0.0f, 5.0f), Vector3.forward), out RaycastHit _, 100.0f), Is.False);
        }

        [Test]
        public void ColliderRaycastSkipsADisabledCollider()
        {
            Collider box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one).GetComponent<Collider>();
            box.enabled = false;

            Assert.That(box.Raycast(new Ray(Vector3.zero, Vector3.forward), out RaycastHit _, 100.0f), Is.False);
        }

        [Test]
        public void ColliderRaycastHitsAMeshFromTheFrontOnly()
        {
            Collider mesh = TriangleMeshQuad(Vector3.zero).GetComponent<Collider>();

            Assert.That(mesh.Raycast(new Ray(new Vector3(0.0f, 0.0f, 3.0f), Vector3.back), out RaycastHit _, 100.0f), Is.False);
            Assert.That(mesh.Raycast(new Ray(new Vector3(0.0f, 0.0f, -3.0f), Vector3.forward), out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(3.0f).Within(Tolerance));
        }

        [Test]
        public void ColliderRaycastHitsAMeshOfACompoundBodyAtItsChildPose()
        {
            GameObject root = Box(Vector3.zero, Vector3.one);
            GameObject child = MeshObject(CubeMesh(), Vector3.zero, true);
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, 0.0f, 5.0f);
            root.AddComponent<Rigidbody>().useGravity = false;
            UnityEngine.Physics.SyncTransforms();

            Assert.That(child.GetComponent<Collider>().Raycast(new Ray(Vector3.zero, Vector3.forward), out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.5f).Within(Tolerance));
        }

        [Test]
        public void SweepTestFindsWhatABodyWouldHit()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            GameObject wall = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.collider, Is.SameAs(wall.GetComponent<Collider>()));
            Assert.That(hit.distance, Is.EqualTo(4.0f).Within(Tolerance));
            AssertVector(Vector3.back, hit.normal, Tolerance);
        }

        [Test]
        public void SweepTestIgnoresWhatABodyStartsIn()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            Box(new Vector3(0.0f, 0.0f, 0.25f), Vector3.one);
            GameObject wall = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.collider, Is.SameAs(wall.GetComponent<Collider>()));

            RaycastHit[] hits = body.SweepTestAll(Vector3.forward, 100.0f);
            Assert.That(hits.Length, Is.EqualTo(1));
            Assert.That(hits[0].collider, Is.SameAs(wall.GetComponent<Collider>()));
        }

        [Test]
        public void SweepTestDoesNotHitItsOwnBody()
        {
            GameObject root = Box(Vector3.zero, Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, 0.0f, 2.0f);
            child.AddComponent<BoxCollider>();
            Rigidbody body = root.AddComponent<Rigidbody>();
            body.useGravity = false;

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit _, 100.0f), Is.False);
            Assert.That(body.SweepTest(Vector3.back, out RaycastHit _, 100.0f), Is.False);
        }

        [Test]
        public void EveryColliderOfABodySweeps()
        {
            GameObject root = Box(Vector3.zero, Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, 0.0f, 2.0f);
            child.AddComponent<BoxCollider>();
            GameObject trigger = Track(new GameObject("trigger"));
            trigger.transform.SetParent(root.transform, false);
            trigger.transform.localPosition = new Vector3(0.0f, 0.0f, 3.0f);
            trigger.AddComponent<BoxCollider>().isTrigger = true;
            Rigidbody body = root.AddComponent<Rigidbody>();
            body.useGravity = false;
            Box(new Vector3(0.0f, 0.0f, 8.0f), new Vector3(3.0f, 3.0f, 1.0f));

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.0f).Within(Tolerance));

            Assert.That(body.SweepTest(Vector3.forward, out hit, 100.0f, QueryTriggerInteraction.Ignore), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.0f).Within(Tolerance));

            Assert.That(Distances(body.SweepTestAll(Vector3.forward, 100.0f)), Is.EqualTo(new[] { 4.0f, 5.0f, 7.0f }).Within(Tolerance));
        }

        [Test]
        public void AConvexMeshSweeps()
        {
            Rigidbody body = ConvexCube(Vector3.zero).AddComponent<Rigidbody>();
            body.useGravity = false;
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.0f).Within(Tolerance));
        }

        [Test]
        public void ATriangleMeshDoesNotSweep()
        {
            Rigidbody body = TriangleMeshCube(Vector3.zero).AddComponent<Rigidbody>();
            body.isKinematic = true;
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            LogAssert.ignoreFailingMessages = true;

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit _, 100.0f), Is.False);
        }

        [Test]
        public void SweepTestFollowsTheFilters()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            GameObject wall = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            wall.GetComponent<BoxCollider>().isTrigger = true;

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit _, 100.0f, QueryTriggerInteraction.Ignore), Is.False);
            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit _, 100.0f, QueryTriggerInteraction.Collide), Is.True);
        }

        [Test]
        public void SweepTestAtAnAngleMeetsTheFirstEdge()
        {
            GameObject gameObject = Box(Vector3.zero, Vector3.one);
            gameObject.transform.rotation = Quaternion.Euler(0.0f, 45.0f, 0.0f);
            Rigidbody body = gameObject.AddComponent<Rigidbody>();
            body.useGravity = false;
            Box(new Vector3(0.0f, 0.0f, 5.0f), new Vector3(3.0f, 3.0f, 1.0f));

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.5f - 0.5f * Mathf.Sqrt(2.0f)).Within(Tolerance));
        }

        [Test]
        public void SweepTestWorksFarFromTheOrigin()
        {
            Vector3 origin = new Vector3(100000.0f, -50000.0f, 75000.0f);
            Rigidbody body = WeightlessBox(origin);
            Box(origin + new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(body.SweepTest(Vector3.forward, out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.0f).Within(0.02f));
        }
    }
}
