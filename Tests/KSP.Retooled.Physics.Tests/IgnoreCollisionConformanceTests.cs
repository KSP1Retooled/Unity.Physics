using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Tests Physics.IgnoreCollision, which operates per collider and is used by KSP's CollisionManager.
    /// </summary>
    public class IgnoreCollisionConformanceTests : PhysicsScene
    {
        private const float RestingHeight = 0.5f;

        [Test]
        public void AnIgnoredPairPassesThrough()
        {
            Collider ground = Ground().GetComponent<Collider>();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));

            UnityEngine.Physics.IgnoreCollision(body.GetComponent<Collider>(), ground);
            Simulate(60);

            Assert.That(body.position.y, Is.LessThan(-1.0f));
        }

        [Test]
        public void GetIgnoreCollisionAnswersPerPairInEitherOrder()
        {
            Collider first = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f)).GetComponent<Collider>();
            Collider second = DynamicBox(new Vector3(3.0f, 5.0f, 0.0f)).GetComponent<Collider>();
            Collider third = DynamicBox(new Vector3(6.0f, 5.0f, 0.0f)).GetComponent<Collider>();

            Assert.That(UnityEngine.Physics.GetIgnoreCollision(first, second), Is.False, "nothing ignored yet");

            UnityEngine.Physics.IgnoreCollision(first, second);
            Assert.That(UnityEngine.Physics.GetIgnoreCollision(first, second), Is.True);
            Assert.That(UnityEngine.Physics.GetIgnoreCollision(second, first), Is.True);
            Assert.That(UnityEngine.Physics.GetIgnoreCollision(first, third), Is.False, "another pair with one collider in common");

            UnityEngine.Physics.IgnoreCollision(second, first, false);
            Assert.That(UnityEngine.Physics.GetIgnoreCollision(first, second), Is.False);
        }

        [Test]
        public void RestoringAPairCollidesAgain()
        {
            Collider ground = Ground().GetComponent<Collider>();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 2.0f, 0.0f));
            Collider collider = body.GetComponent<Collider>();

            UnityEngine.Physics.IgnoreCollision(collider, ground);
            Simulate(5);
            UnityEngine.Physics.IgnoreCollision(collider, ground, false);
            Simulate(150);

            Assert.That(body.position.y, Is.EqualTo(RestingHeight).Within(0.02f));
        }

        [Test]
        public void IgnoringOneChildOfACompoundLeavesTheOthersColliding()
        {
            Collider ground = Ground().GetComponent<Collider>();
            GameObject root = Box(new Vector3(0.0f, 2.0f, 0.0f), Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, -1.0f, 0.0f);
            Collider lowest = child.AddComponent<BoxCollider>();
            Rigidbody body = root.AddComponent<Rigidbody>();
            body.constraints = RigidbodyConstraints.FreezeRotation;

            UnityEngine.Physics.IgnoreCollision(lowest, ground);
            Simulate(200);

            Assert.That(body.position.y, Is.EqualTo(RestingHeight).Within(0.02f), "the root collider lands, with the child sunk in");
        }

        [Test]
        public void IgnoredPairsOfTwoBodiesLetThemOverlap()
        {
            Rigidbody first = WeightlessBox(new Vector3(-3.0f, 0.0f, 0.0f));
            Rigidbody second = WeightlessBox(new Vector3(3.0f, 0.0f, 0.0f));
            first.linearVelocity = new Vector3(5.0f, 0.0f, 0.0f);

            UnityEngine.Physics.IgnoreCollision(first.GetComponent<Collider>(), second.GetComponent<Collider>());
            Simulate(100);

            AssertVector(new Vector3(7.0f, 0.0f, 0.0f), first.position, 0.05f, "moved through");
            AssertVector(new Vector3(3.0f, 0.0f, 0.0f), second.position, 0.01f, "left alone");
        }

        [Test]
        public void AColliderAddedToAnIgnoredBodyCollides()
        {
            Rigidbody first = WeightlessBox(new Vector3(0.0f, 0.0f, 0.0f));
            Rigidbody second = WeightlessBox(new Vector3(0.2f, 0.0f, 0.0f));
            UnityEngine.Physics.IgnoreCollision(first.GetComponent<Collider>(), second.GetComponent<Collider>());

            // The boxes overlap from the start, so their bodies are already known to ignore each other when the new
            // collider arrives.
            Simulate(5);
            AssertVector(new Vector3(0.2f, 0.0f, 0.0f), second.position, 0.01f, "ignored, so left overlapping");

            first.gameObject.AddComponent<BoxCollider>();
            Simulate(100);

            Assert.That(second.position.x - first.position.x, Is.GreaterThan(0.5f), "the new collider isn't ignored, so it pushes them apart");
        }

        [Test]
        public void AnIgnoredPairStaysIgnoredThroughDisablingACollider()
        {
            Collider ground = Ground().GetComponent<Collider>();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));
            Collider collider = body.GetComponent<Collider>();

            UnityEngine.Physics.IgnoreCollision(collider, ground);
            collider.enabled = false;
            Simulate(1);
            collider.enabled = true;
            Simulate(60);

            Assert.That(UnityEngine.Physics.GetIgnoreCollision(collider, ground), Is.True);
            Assert.That(body.position.y, Is.LessThan(-1.0f));
        }

        [Test]
        public void AColliderDestroyedWhileIgnoredGivesUpItsPairs()
        {
            Collider ground = Ground().GetComponent<Collider>();
            Rigidbody first = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));
            UnityEngine.Physics.IgnoreCollision(first.GetComponent<Collider>(), ground);
            Object.DestroyImmediate(first.gameObject);

            Rigidbody second = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));
            Simulate(100);

            Assert.That(UnityEngine.Physics.GetIgnoreCollision(second.GetComponent<Collider>(), ground), Is.False);
            Assert.That(second.position.y, Is.EqualTo(RestingHeight).Within(0.02f));
        }
    }
}
