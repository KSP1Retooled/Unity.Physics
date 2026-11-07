using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>Collider.ClosestPoint and Physics.ClosestPoint.</summary>
    public class ClosestPointConformanceTests : QueryScene
    {
        private const float Tolerance = 1e-3f;

        [Test]
        public void APointOutsideLandsOnTheSurface()
        {
            Collider box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one).GetComponent<Collider>();

            AssertVector(new Vector3(0.0f, 0.0f, 4.5f), box.ClosestPoint(Vector3.zero), Tolerance);
            AssertVector(new Vector3(0.5f, 0.5f, 4.5f), box.ClosestPoint(new Vector3(3.0f, 3.0f, 0.0f)), Tolerance);
        }

        [Test]
        public void APointInsideIsItsOwnAnswer()
        {
            Collider box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one).GetComponent<Collider>();
            Vector3 inside = new Vector3(0.1f, 0.0f, 5.0f);

            Assert.That(box.ClosestPoint(inside), Is.EqualTo(inside));
        }

        [Test]
        public void ASphereAndACapsuleAnswerTooWithTheirCentreOffset()
        {
            SphereCollider sphere = Sphere(new Vector3(10.0f, 0.0f, 0.0f), 1.0f);
            sphere.center = new Vector3(0.0f, 1.0f, 0.0f);
            AssertVector(new Vector3(10.0f, 2.0f, 0.0f), sphere.ClosestPoint(new Vector3(10.0f, 5.0f, 0.0f)), Tolerance);

            GameObject gameObject = Track(new GameObject("capsule"));
            CapsuleCollider capsule = gameObject.AddComponent<CapsuleCollider>();
            capsule.height = 4.0f;
            capsule.radius = 0.5f;
            capsule.direction = 2;
            AssertVector(new Vector3(0.0f, 0.0f, 2.0f), capsule.ClosestPoint(new Vector3(0.0f, 0.0f, 9.0f)), Tolerance);
            AssertVector(new Vector3(0.5f, 0.0f, 1.0f), capsule.ClosestPoint(new Vector3(3.0f, 0.0f, 1.0f)), Tolerance);
        }

        [Test]
        public void TheCentreRotatesWithThePoseGiven()
        {
            BoxCollider box = Box(new Vector3(10.0f, 0.0f, 0.0f), Vector3.one).GetComponent<BoxCollider>();
            box.center = new Vector3(0.0f, 3.0f, 0.0f);

            AssertVector(new Vector3(10.0f, 3.5f, 0.0f), box.ClosestPoint(new Vector3(10.0f, 10.0f, 0.0f)), Tolerance);

            Vector3 rotated = UnityEngine.Physics.ClosestPoint(Vector3.zero, box, new Vector3(10.0f, 0.0f, 0.0f), Quaternion.Euler(0.0f, 0.0f, 90.0f));
            AssertVector(new Vector3(6.5f, 0.0f, 0.0f), rotated, Tolerance);
        }

        [Test]
        public void PhysicsClosestPointGoesByThePoseNotTheTransform()
        {
            Collider box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one).GetComponent<Collider>();

            AssertVector(new Vector3(0.0f, 0.0f, 9.5f), UnityEngine.Physics.ClosestPoint(Vector3.zero, box, new Vector3(0.0f, 0.0f, 10.0f), Quaternion.identity), Tolerance);
        }

        [Test]
        public void AConvexMeshAnswers()
        {
            Collider convex = ConvexCube(new Vector3(0.0f, 30.0f, 0.0f)).GetComponent<Collider>();

            AssertVector(new Vector3(0.0f, 30.0f, -0.5f), convex.ClosestPoint(new Vector3(0.0f, 30.0f, -3.0f)), Tolerance);

            Vector3 inside = new Vector3(0.0f, 30.1f, 0.0f);
            Assert.That(convex.ClosestPoint(inside), Is.EqualTo(inside));
        }

        [Test]
        public void AConvexMeshOnAChildObjectGoesByItsOwnPose()
        {
            GameObject root = Track(new GameObject("root"));
            root.transform.position = new Vector3(0.0f, 5.0f, 0.0f);
            root.AddComponent<Rigidbody>().useGravity = false;
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(2.0f, 0.0f, 0.0f);
            MeshCollider single = child.AddComponent<MeshCollider>();
            single.convex = true;
            single.sharedMesh = CubeMesh();

            AssertVector(new Vector3(2.5f, 5.0f, 0.0f), single.ClosestPoint(new Vector3(9.0f, 5.0f, 0.0f)), Tolerance);
        }

        [Test]
        public void EachConvexMeshOfACompoundBodyAnswersForItself()
        {
            GameObject root = Track(new GameObject("root"));
            root.AddComponent<Rigidbody>().useGravity = false;
            MeshCollider left = null;
            MeshCollider right = null;
            foreach (float x in new[] { -2.0f, 2.0f })
            {
                GameObject child = Track(new GameObject("child"));
                child.transform.SetParent(root.transform, false);
                child.transform.localPosition = new Vector3(x, 0.0f, 0.0f);
                MeshCollider collider = child.AddComponent<MeshCollider>();
                collider.convex = true;
                collider.sharedMesh = CubeMesh();
                left = x < 0.0f ? collider : left;
                right = x > 0.0f ? collider : right;
            }

            AssertVector(new Vector3(-1.5f, 0.0f, 0.0f), left.ClosestPoint(new Vector3(9.0f, 0.0f, 0.0f)), Tolerance);
            AssertVector(new Vector3(2.5f, 0.0f, 0.0f), right.ClosestPoint(new Vector3(9.0f, 0.0f, 0.0f)), Tolerance);
        }

        [Test]
        public void ATriangleMeshGivesThePointBack()
        {
            Collider mesh = TriangleMeshQuad(Vector3.zero).GetComponent<Collider>();
            Vector3 point = new Vector3(0.0f, 0.0f, -3.0f);
            LogAssert.ignoreFailingMessages = true;

            Assert.That(mesh.ClosestPoint(point), Is.EqualTo(point));
        }

        [Test]
        public void ADisabledColliderGivesThePointBack()
        {
            Collider box = Box(Vector3.zero, Vector3.one).GetComponent<Collider>();
            box.enabled = false;
            Vector3 point = new Vector3(0.0f, 0.0f, 3.0f);

            Assert.That(box.ClosestPoint(point), Is.EqualTo(point));
            Assert.That(UnityEngine.Physics.ClosestPoint(point, box, Vector3.zero, Quaternion.identity), Is.EqualTo(point));
        }

        [Test]
        public void ClosestPointWorksFarFromTheOrigin()
        {
            Vector3 origin = new Vector3(100000.0f, -50000.0f, 75000.0f);
            Collider box = Box(origin, Vector3.one).GetComponent<Collider>();

            AssertVector(origin + new Vector3(0.0f, 0.0f, 0.5f), box.ClosestPoint(origin + new Vector3(0.0f, 0.0f, 3.0f)), 0.02f);
        }
    }
}
