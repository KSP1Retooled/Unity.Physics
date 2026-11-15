using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Tests synchronization between transforms and physics bodies with autosync disabled, as in Principia and KSPCF.
    /// </summary>
    public class TransformSyncConformanceTests : PhysicsScene
    {
        [SetUp]
        public void TurnAutoSyncOff()
        {
            UnityEngine.Physics.autoSyncTransforms = false;
        }

        [Test]
        public void AMovedTransformReachesTheBodyOnSync()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            UnityEngine.Physics.SyncTransforms();
            body.transform.position = new Vector3(3.0f, 0.0f, 0.0f);

            Assert.That(body.position.x, Is.EqualTo(0.0f), "not before the sync");

            UnityEngine.Physics.SyncTransforms();
            Assert.That(body.position.x, Is.EqualTo(3.0f));
        }

        [Test]
        public void TheStepPicksUpAMovedTransformWithoutASync()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            UnityEngine.Physics.SyncTransforms();
            body.transform.position = new Vector3(3.0f, 0.0f, 0.0f);

            Simulate(1);

            Assert.That(body.position.x, Is.EqualTo(3.0f).Within(1e-4f));
        }

        [Test]
        public void AStepWritesThePoseToTheTransform()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.linearVelocity = new Vector3(5.0f, 0.0f, 0.0f);

            Simulate(10);

            Assert.That(body.transform.position.x, Is.EqualTo(1.0f).Within(1e-3f));
        }

        [Test]
        public void ARayMissesAMovedColliderUntilTheSync()
        {
            GameObject box = Box(Vector3.zero, Vector3.one);
            UnityEngine.Physics.SyncTransforms();

            box.transform.position = new Vector3(10.0f, 0.0f, 0.0f);
            Vector3 origin = new Vector3(10.0f, 5.0f, 0.0f);
            Assert.That(UnityEngine.Physics.Raycast(origin, Vector3.down, 10.0f), Is.False, "the query sees the last synced pose");

            UnityEngine.Physics.SyncTransforms();
            Assert.That(UnityEngine.Physics.Raycast(origin, Vector3.down, 10.0f), Is.True);
        }

        [Test]
        public void AChildColliderMovedWithinItsBodyIsFoundWhereItWent()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(body.transform, false);
            child.transform.localPosition = new Vector3(3.0f, 0.0f, 0.0f);
            SphereCollider sphere = child.AddComponent<SphereCollider>();
            Simulate(1);

            child.transform.localPosition = new Vector3(6.0f, 0.0f, 0.0f);
            Simulate(1);

            Assert.That(UnityEngine.Physics.Raycast(new Vector3(3.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.False, "left its old place");
            Assert.That(UnityEngine.Physics.Raycast(new Vector3(6.0f, 5.0f, 0.0f), Vector3.down, out RaycastHit hit, 10.0f), Is.True);
            Assert.That(hit.collider, Is.SameAs(sphere));
            Assert.That(UnityEngine.Physics.Raycast(new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, out hit, 10.0f), Is.True, "the box stays");
            Assert.That(hit.collider, Is.SameAs(body.GetComponent<Collider>()));
        }

        // A floating origin shift moves every vessel through its transform. The child's pose relative to the body stays
        // unchanged, so Unity does not write a separate pose for the child. Its collider must still follow the body.
        [Test]
        public void AChildColliderFollowsABodyMovedThroughItsTransform()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(body.transform, false);
            child.transform.localPosition = new Vector3(3.0f, 0.0f, 0.0f);
            SphereCollider sphere = child.AddComponent<SphereCollider>();
            Simulate(1);

            body.transform.SetPositionAndRotation(new Vector3(10.0f, 0.0f, 0.0f), Quaternion.Euler(0.0f, 90.0f, 0.0f));
            Simulate(1);

            Assert.That(UnityEngine.Physics.Raycast(new Vector3(13.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.False, "turned with the body");
            Assert.That(UnityEngine.Physics.Raycast(new Vector3(10.0f, 5.0f, -3.0f), Vector3.down, out RaycastHit hit, 10.0f), Is.True);
            Assert.That(hit.collider, Is.SameAs(sphere));
        }

        [Test]
        public void AMeshAloneOnAMovedChildIsFoundWhereItWent()
        {
            GameObject primitive = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Mesh mesh = Object.Instantiate(primitive.GetComponent<MeshFilter>().sharedMesh);
            Object.DestroyImmediate(primitive);
            try
            {
                Rigidbody body = WeightlessBox(Vector3.zero);
                Object.DestroyImmediate(body.GetComponent<Collider>());
                GameObject child = Track(new GameObject("child"));
                child.transform.SetParent(body.transform, false);
                child.transform.localPosition = new Vector3(3.0f, 0.0f, 0.0f);
                MeshCollider hull = child.AddComponent<MeshCollider>();
                hull.convex = true;
                hull.sharedMesh = mesh;
                Simulate(1);

                for (int x = 4; x <= 6; x++)
                {
                    child.transform.localPosition = new Vector3(x, 0.0f, 0.0f);
                    Simulate(1);

                    Assert.That(UnityEngine.Physics.Raycast(new Vector3(x - 1.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.False, $"left {x - 1}");
                    Assert.That(UnityEngine.Physics.Raycast(new Vector3(x, 5.0f, 0.0f), Vector3.down, out RaycastHit hit, 10.0f), Is.True, $"at {x}");
                    Assert.That(hit.collider, Is.SameAs(hull));
                }
            }
            finally
            {
                Object.DestroyImmediate(mesh);
            }
        }

        [Test]
        public void MovingAParentCarriesAChildBody()
        {
            GameObject parent = Track(new GameObject("parent"));
            Rigidbody body = WeightlessBox(new Vector3(1.0f, 0.0f, 0.0f));
            body.transform.SetParent(parent.transform, true);

            parent.transform.position = new Vector3(0.0f, 4.0f, 0.0f);
            UnityEngine.Physics.SyncTransforms();

            AssertVector(new Vector3(1.0f, 4.0f, 0.0f), body.position, 1e-4f);
        }

        [Test]
        public void SettingTheBodysPositionMovesTheTransformInTheStep()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.position = new Vector3(2.0f, 0.0f, 0.0f);

            Simulate(1);

            Assert.That(body.transform.position.x, Is.EqualTo(2.0f).Within(1e-4f));
        }

        [Test]
        public void AKinematicMoveArrivesInTheStep()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.isKinematic = true;

            body.MovePosition(new Vector3(1.0f, 0.0f, 0.0f));
            Simulate(1);

            Assert.That(body.position.x, Is.EqualTo(1.0f).Within(1e-4f));
            Assert.That(body.transform.position.x, Is.EqualTo(1.0f).Within(1e-4f));
        }
    }
}
