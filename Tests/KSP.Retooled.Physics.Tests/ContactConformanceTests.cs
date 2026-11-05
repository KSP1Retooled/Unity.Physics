using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>How bodies meet each other and the ground, and how layers keep them apart.</summary>
    public class ContactConformanceTests : PhysicsScene
    {
        // Layers no stock code uses, with nothing tied to them in the project's collision matrix.
        private const int LayerA = 28;
        private const int LayerB = 29;

        [Test]
        public void ABoxLandsOnAStaticBoxAndRests()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 2.0f, 0.0f));

            Simulate(200);

            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.02f));
            Assert.That(body.linearVelocity.magnitude, Is.LessThan(0.05f));
            Assert.That(body.angularVelocity.magnitude, Is.LessThan(0.05f));
        }

        [Test]
        public void ASphereLandsOnAStaticBox()
        {
            Ground();
            GameObject gameObject = Track(new GameObject("sphere"));
            gameObject.transform.position = new Vector3(0.0f, 2.0f, 0.0f);
            gameObject.AddComponent<SphereCollider>().radius = 0.5f;
            Rigidbody body = gameObject.AddComponent<Rigidbody>();

            Simulate(200);

            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.02f));
        }

        [Test]
        public void ACapsuleOnItsEndStaysAsTallAsItIs()
        {
            Ground();
            GameObject gameObject = Track(new GameObject("capsule"));
            gameObject.transform.position = new Vector3(0.0f, 2.0f, 0.0f);
            CapsuleCollider capsule = gameObject.AddComponent<CapsuleCollider>();
            capsule.radius = 0.5f;
            capsule.height = 3.0f;
            capsule.direction = 1;
            Rigidbody body = gameObject.AddComponent<Rigidbody>();
            body.constraints = RigidbodyConstraints.FreezeRotation;

            Simulate(100);

            Assert.That(body.position.y, Is.EqualTo(1.5f).Within(0.02f));
        }

        [Test]
        public void ABoxLandsOnATwoColliderBody()
        {
            Ground();
            GameObject root = Box(new Vector3(0.0f, 2.0f, 0.0f), Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, -1.0f, 0.0f);
            child.AddComponent<BoxCollider>();
            Rigidbody body = root.AddComponent<Rigidbody>();
            body.constraints = RigidbodyConstraints.FreezeRotation;

            Simulate(200);

            Assert.That(body.position.y, Is.EqualTo(1.5f).Within(0.02f));
        }

        [Test]
        public void TwoDynamicBoxesStack()
        {
            Ground();
            Rigidbody lower = DynamicBox(new Vector3(0.0f, 0.6f, 0.0f));
            Rigidbody upper = DynamicBox(new Vector3(0.0f, 2.0f, 0.0f));

            Simulate(250);

            Assert.That(lower.position.y, Is.EqualTo(0.5f).Within(0.03f));
            Assert.That(upper.position.y, Is.EqualTo(1.5f).Within(0.05f));
        }

        [Test]
        public void FrictionStopsASlidingBox()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            body.linearVelocity = new Vector3(2.0f, 0.0f, 0.0f);

            Simulate(100);

            Assert.That(body.linearVelocity.magnitude, Is.LessThan(0.05f));
            Assert.That(body.position.x, Is.GreaterThan(0.1f));
            Assert.That(body.position.x, Is.LessThan(1.0f));
        }

        [Test]
        public void IgnoredLayersDoNotCollide()
        {
            bool previous = Physics.GetIgnoreLayerCollision(LayerA, LayerB);
            try
            {
                Physics.IgnoreLayerCollision(LayerA, LayerB, true);
                GameObject ground = Ground();
                ground.layer = LayerB;
                Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));
                body.gameObject.layer = LayerA;

                Simulate(100);

                Assert.That(body.position.y, Is.LessThan(-1.0f));
            }
            finally
            {
                Physics.IgnoreLayerCollision(LayerA, LayerB, previous);
            }
        }

        [Test]
        public void IgnoringLayersAfterTheyMetLetsTheBodyFall()
        {
            bool previous = Physics.GetIgnoreLayerCollision(LayerA, LayerB);
            try
            {
                Physics.IgnoreLayerCollision(LayerA, LayerB, false);
                GameObject ground = Ground();
                ground.layer = LayerB;
                Rigidbody body = DynamicBox(new Vector3(0.0f, 0.6f, 0.0f));
                body.gameObject.layer = LayerA;

                Simulate(50);
                Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.02f), "the layers collide at first");

                Physics.IgnoreLayerCollision(LayerA, LayerB, true);
                body.WakeUp();
                Simulate(50);

                Assert.That(body.position.y, Is.LessThan(0.0f));
            }
            finally
            {
                Physics.IgnoreLayerCollision(LayerA, LayerB, previous);
            }
        }

        [Test]
        public void DetectCollisionsFalseLetsTheBodyFallThrough()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));
            body.detectCollisions = false;

            Simulate(100);

            Assert.That(body.position.y, Is.LessThan(-1.0f));
        }

        [Test]
        public void ATriggerDoesNotStopABody()
        {
            GameObject ground = Ground();
            ground.GetComponent<BoxCollider>().isTrigger = true;
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));

            Simulate(100);

            Assert.That(body.position.y, Is.LessThan(-1.0f));
        }
    }
}
