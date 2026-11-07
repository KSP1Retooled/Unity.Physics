using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>Transforms and bodies reaching each other with autosync off, which is how Principia and KSPCF run.</summary>
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
