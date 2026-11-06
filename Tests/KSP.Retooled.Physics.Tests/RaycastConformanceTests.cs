using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>Physics.Raycast against the scene.</summary>
    public class RaycastConformanceTests : PhysicsScene
    {
        private const float Tolerance = 1e-3f;

        [Test]
        public void ARayHitsTheColliderInItsWay()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.collider, Is.SameAs(box.GetComponent<Collider>()));
            Assert.That(hit.distance, Is.EqualTo(4.5f).Within(Tolerance));
            AssertVector(new Vector3(0.0f, 0.0f, 4.5f), hit.point, Tolerance);
            AssertVector(Vector3.back, hit.normal, Tolerance);
        }

        [Test]
        public void ARayStopsAtItsMaximumDistance()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 4.0f), Is.False);
        }

        [Test]
        public void ARayWithoutADistanceReachesFar()
        {
            Box(new Vector3(0.0f, 0.0f, 5000.0f), Vector3.one);

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward), Is.True);
        }

        [Test]
        public void TheClosestColliderWins()
        {
            Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);
            GameObject near = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Physics.Raycast(Vector3.zero, Vector3.forward, out RaycastHit hit, 100.0f);

            Assert.That(hit.collider, Is.SameAs(near.GetComponent<Collider>()));
        }

        [Test]
        public void TheLayerMaskFiltersColliders()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            box.layer = 28;

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f, 1 << 29), Is.False);
            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f, 1 << 28), Is.True);
        }

        [Test]
        public void TriggersFollowTheQueryInteraction()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            box.GetComponent<BoxCollider>().isTrigger = true;

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f, ~0, QueryTriggerInteraction.Ignore), Is.False);
            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f, ~0, QueryTriggerInteraction.Collide), Is.True);
        }

        [Test]
        public void ARayFindsABodyWhereTheStepLeftIt()
        {
            Rigidbody body = WeightlessBox(new Vector3(0.0f, 0.0f, 5.0f));
            body.linearVelocity = new Vector3(0.0f, 0.0f, 10.0f);
            Simulate(10);

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.distance, Is.EqualTo(4.5f + 10.0f * Step * 10).Within(0.01f));
            Assert.That(hit.rigidbody, Is.SameAs(body));
        }

        [Test]
        public void ARayFindsATransformMovedByHand()
        {
            GameObject box = Box(Vector3.zero, Vector3.one);
            box.transform.position = new Vector3(0.0f, 0.0f, 7.0f);

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.distance, Is.EqualTo(6.5f).Within(Tolerance));
        }

        [Test]
        public void ARaySeesEachColliderOfACompoundBody()
        {
            GameObject root = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, 3.0f, 0.0f);
            BoxCollider childCollider = child.AddComponent<BoxCollider>();
            root.AddComponent<Rigidbody>();

            Physics.Raycast(new Vector3(0.0f, 3.0f, 0.0f), Vector3.forward, out RaycastHit hit, 100.0f);

            Assert.That(hit.collider, Is.SameAs(childCollider));
        }

        [TestCase(0, 1.0f)]
        [TestCase(1, 0.5f)]
        [TestCase(2, 0.5f)]
        public void ACapsuleLiesAlongItsDirection(int direction, float halfWidthAlongX)
        {
            GameObject gameObject = Track(new GameObject("capsule"));
            gameObject.transform.position = new Vector3(5.0f, 0.0f, 0.0f);
            CapsuleCollider capsule = gameObject.AddComponent<CapsuleCollider>();
            capsule.radius = 0.5f;
            capsule.height = 2.0f;
            capsule.direction = direction;

            Physics.Raycast(Vector3.zero, Vector3.right, out RaycastHit hit, 100.0f);

            Assert.That(hit.distance, Is.EqualTo(5.0f - halfWidthAlongX).Within(Tolerance));
        }

        [Test]
        public void ARayStartingInsideAColliderMissesIt()
        {
            Box(Vector3.zero, Vector3.one * 4.0f);

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f), Is.False);
        }

        [Test]
        public void ARayFromAJobFindsABodyAStepMoved()
        {
            Rigidbody body = WeightlessBox(new Vector3(0.0f, 0.0f, 5.0f));
            body.linearVelocity = new Vector3(0.0f, 0.0f, 10.0f);
            Simulate(10);

            RaycastHit hit = RayFromAJob(Vector3.zero, Vector3.forward, 100.0f);

            Assert.That(hit.collider, Is.SameAs(body.GetComponent<Collider>()));
            Assert.That(hit.distance, Is.EqualTo(4.5f + 10.0f * Step * 10).Within(0.01f));
        }

        [Test]
        public void ARayFromAJobFindsATransformMovedByHand()
        {
            GameObject box = Box(Vector3.zero, Vector3.one);
            box.transform.position = new Vector3(0.0f, 0.0f, 7.0f);
            Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            RaycastHit hit = RayFromAJob(Vector3.zero, Vector3.forward, 100.0f);

            Assert.That(hit.distance, Is.EqualTo(6.5f).Within(Tolerance));
        }

        [Test]
        public void ARayFromAJobSeesACompoundBodyCreatedJustBefore()
        {
            GameObject root = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, 3.0f, 0.0f);
            BoxCollider childCollider = child.AddComponent<BoxCollider>();
            root.AddComponent<Rigidbody>();
            Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            RaycastHit hit = RayFromAJob(new Vector3(0.0f, 3.0f, 0.0f), Vector3.forward, 100.0f);

            Assert.That(hit.collider, Is.SameAs(childCollider));
        }

        [Test]
        public void ARayFromAJobMissesABodyDestroyedJustBefore()
        {
            GameObject near = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            GameObject far = Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);
            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f), Is.True);

            Object.DestroyImmediate(near);
            BackendHost.SyncQueryWorlds();

            RaycastHit hit = RayFromAJob(Vector3.zero, Vector3.forward, 100.0f);
            Assert.That(hit.collider, Is.SameAs(far.GetComponent<Collider>()));
        }

        [Test]
        public void ARayFromAJobSurvivesABodyDestroyedWithoutASync()
        {
            GameObject near = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);
            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f), Is.True);

            Object.DestroyImmediate(near);
            RaycastHit hit = RayFromAJob(Vector3.zero, Vector3.forward, 100.0f);

            // PhysX finds the far box and the backend may still report the old near one, which reads back as no collider.
            Assert.That(hit.collider == null || hit.distance > 8.0f, Is.True, $"hit {hit.collider} at {hit.distance}");
        }

        [Test]
        public void AColliderThatCollidesWithNothingStillStopsRays()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            box.GetComponent<BoxCollider>().excludeLayers = ~0;
            Physics.SyncTransforms();

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f), Is.True);
        }

        [Test]
        public void AColliderThatCollidesWithNothingStaysOutOfContacts()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));
            body.GetComponent<BoxCollider>().excludeLayers = ~0;

            Simulate(100);

            Assert.That(body.position.y, Is.LessThan(-1.0f), "fell through the ground");
        }

        private static RaycastHit RayFromAJob(Vector3 origin, Vector3 direction, float distance)
        {
            using NativeArray<RaycastCommand> commands = new NativeArray<RaycastCommand>(32, Allocator.TempJob);
            using NativeArray<RaycastHit> hits = new NativeArray<RaycastHit>(32, Allocator.TempJob);
            NativeArray<RaycastCommand> rays = commands;
            for (int i = 0; i < rays.Length; i++)
            {
                rays[i] = new RaycastCommand(origin, direction, QueryParameters.Default, distance);
            }

            RaycastCommand.ScheduleBatch(commands, hits, 1, 1).Complete();
            for (int i = 1; i < hits.Length; i++)
            {
                Assert.That(hits[i].collider, Is.SameAs(hits[0].collider), "rays of one batch agree");
            }

            return hits[0];
        }

        [Test]
        public void AColliderRemovedFromTheSceneIsGone()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f), Is.True);

            Object.DestroyImmediate(box.GetComponent<BoxCollider>());

            Assert.That(Physics.Raycast(Vector3.zero, Vector3.forward, 100.0f), Is.False);
        }
    }
}
