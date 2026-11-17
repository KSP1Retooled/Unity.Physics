using KSP.Retooled.Physics.Tests.Scenes;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Queries
{
    /// <summary>Physics.OverlapSphere, OverlapBox, OverlapCapsule and the Check family.</summary>
    public class OverlapConformanceTests : QueryScene
    {
        [Test]
        public void ASphereFindsWhatItReaches()
        {
            GameObject near = Box(new Vector3(0.0f, 0.0f, 1.0f), Vector3.one);
            Box(new Vector3(0.0f, 0.0f, 3.0f), Vector3.one);

            Collider[] found = UnityEngine.Physics.OverlapSphere(Vector3.zero, 1.0f);

            Assert.That(found, Is.EqualTo(new[] { near.GetComponent<Collider>() }));
        }

        [Test]
        public void ASphereTouchingAFaceCounts()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(0.0f, 0.0f, 4.0f), 0.5f).Length, Is.EqualTo(1));
        }

        [Test]
        public void ASphereInsideACollidersVolumeCounts()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one * 4.0f);

            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(0.0f, 0.0f, 5.0f), 0.1f).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(0.0f, 0.0f, 5.0f), 0.0f).Length, Is.EqualTo(1));
        }

        [Test]
        public void ASphereMissingABoxByItsCornerFindsNothing()
        {
            Box(Vector3.zero, Vector3.one);

            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(1.0f, 1.0f, 1.0f), 0.8f).Length, Is.EqualTo(0));
            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(1.0f, 1.0f, 1.0f), 0.9f).Length, Is.EqualTo(1));
        }

        [Test]
        public void EachColliderOfACompoundBodyCountsOnce()
        {
            GameObject root = Box(Vector3.zero, Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(1.0f, 0.0f, 0.0f);
            BoxCollider childCollider = child.AddComponent<BoxCollider>();
            root.AddComponent<Rigidbody>();

            Assert.That(UnityEngine.Physics.OverlapSphere(Vector3.zero, 3.0f).Length, Is.EqualTo(2));
            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(1.5f, 0.0f, 0.0f), 0.25f), Is.EqualTo(new Collider[] { childCollider }));
        }

        [Test]
        public void AMeshCountsOnceHoweverManyTrianglesItReaches()
        {
            TriangleMeshCube(Vector3.zero);

            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(0.5f, 0.5f, 0.5f), 0.4f).Length, Is.EqualTo(1));

            Collider[] results = new Collider[4];
            Assert.That(UnityEngine.Physics.OverlapSphereNonAlloc(new Vector3(0.5f, 0.5f, 0.5f), 0.4f, results), Is.EqualTo(1));
        }

        [Test]
        public void ATriangleMeshIsHollow()
        {
            TriangleMeshCube(Vector3.zero);

            Assert.That(UnityEngine.Physics.OverlapSphere(Vector3.zero, 0.1f).Length, Is.EqualTo(0));
        }

        [Test]
        public void AConvexMeshIsSolid()
        {
            ConvexCube(Vector3.zero);

            Assert.That(UnityEngine.Physics.OverlapSphere(Vector3.zero, 0.1f).Length, Is.EqualTo(1));
        }

        [Test]
        public void ATriangleHasNoBackForAnOverlap()
        {
            TriangleMeshQuad(Vector3.zero);

            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(0.0f, 0.0f, 0.1f), 0.2f).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(0.0f, 0.0f, -0.1f), 0.2f).Length, Is.EqualTo(1));
        }

        [Test]
        public void TheLayerMaskFiltersOverlaps()
        {
            GameObject box = Box(Vector3.zero, Vector3.one);
            box.layer = 28;

            Assert.That(UnityEngine.Physics.OverlapSphere(Vector3.zero, 1.0f, 1 << 29).Length, Is.EqualTo(0));
            Assert.That(UnityEngine.Physics.OverlapSphere(Vector3.zero, 1.0f, 1 << 28).Length, Is.EqualTo(1));
        }

        [Test]
        public void TriggersFollowTheQueryInteraction()
        {
            GameObject box = Box(Vector3.zero, Vector3.one);
            box.GetComponent<BoxCollider>().isTrigger = true;

            Assert.That(UnityEngine.Physics.OverlapSphere(Vector3.zero, 1.0f, ~0, QueryTriggerInteraction.Ignore).Length, Is.EqualTo(0));
            Assert.That(UnityEngine.Physics.OverlapSphere(Vector3.zero, 1.0f, ~0, QueryTriggerInteraction.Collide).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.CheckSphere(Vector3.zero, 1.0f, ~0, QueryTriggerInteraction.Ignore), Is.False);
            Assert.That(UnityEngine.Physics.CheckSphere(Vector3.zero, 1.0f, ~0, QueryTriggerInteraction.Collide), Is.True);
        }

        [Test]
        public void NonAllocStopsWhenTheArrayIsFull()
        {
            Box(new Vector3(0.0f, 0.0f, 1.0f), Vector3.one);
            Box(new Vector3(0.0f, 0.0f, -1.0f), Vector3.one);
            Collider[] results = new Collider[1];

            Assert.That(UnityEngine.Physics.OverlapSphereNonAlloc(Vector3.zero, 1.0f, results), Is.EqualTo(1));
            Assert.That(results[0], Is.Not.Null);
        }

        [Test]
        public void ABoxOverlapFollowsItsOrientation()
        {
            Box(new Vector3(0.0f, 0.0f, 1.3f), Vector3.one);
            Vector3 thin = new Vector3(1.0f, 0.1f, 0.1f);

            Assert.That(UnityEngine.Physics.OverlapBox(Vector3.zero, thin).Length, Is.EqualTo(0));
            Assert.That(UnityEngine.Physics.OverlapBox(Vector3.zero, thin, Quaternion.Euler(0.0f, 90.0f, 0.0f)).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.CheckBox(Vector3.zero, thin), Is.False);
            Assert.That(UnityEngine.Physics.CheckBox(Vector3.zero, thin, Quaternion.Euler(0.0f, 90.0f, 0.0f)), Is.True);

            Collider[] results = new Collider[2];
            Assert.That(UnityEngine.Physics.OverlapBoxNonAlloc(Vector3.zero, thin, results, Quaternion.Euler(0.0f, 90.0f, 0.0f)), Is.EqualTo(1));
        }

        [Test]
        public void ACapsuleOverlapReachesAlongItsLength()
        {
            Box(new Vector3(0.0f, 3.0f, 0.0f), Vector3.one);
            Vector3 foot = Vector3.zero;

            Assert.That(UnityEngine.Physics.OverlapCapsule(foot, new Vector3(0.0f, 2.0f, 0.0f), 0.25f).Length, Is.EqualTo(0));
            Assert.That(UnityEngine.Physics.OverlapCapsule(foot, new Vector3(0.0f, 2.0f, 0.0f), 0.75f).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.CheckCapsule(foot, new Vector3(0.0f, 2.0f, 0.0f), 0.25f), Is.False);
            Assert.That(UnityEngine.Physics.CheckCapsule(foot, new Vector3(0.0f, 2.0f, 0.0f), 0.75f), Is.True);

            Collider[] results = new Collider[2];
            Assert.That(UnityEngine.Physics.OverlapCapsuleNonAlloc(foot, new Vector3(0.0f, 2.0f, 0.0f), 0.75f, results), Is.EqualTo(1));
        }

        [Test]
        public void CheckSphereSaysWhetherAnythingIsThere()
        {
            Box(Vector3.zero, Vector3.one);

            Assert.That(UnityEngine.Physics.CheckSphere(new Vector3(0.0f, 0.0f, 0.9f), 0.5f), Is.True);
            Assert.That(UnityEngine.Physics.CheckSphere(new Vector3(0.0f, 0.0f, 1.1f), 0.5f), Is.False);
        }

        [Test]
        public void AnOverlapFindsABodyWhereTheStepLeftIt()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.linearVelocity = new Vector3(0.0f, 0.0f, 10.0f);
            Simulate(50);

            Assert.That(UnityEngine.Physics.OverlapSphere(Vector3.zero, 1.0f).Length, Is.EqualTo(0));
            Assert.That(UnityEngine.Physics.OverlapSphere(new Vector3(0.0f, 0.0f, 10.0f), 1.0f).Length, Is.EqualTo(1));
        }

        [Test]
        public void AnOverlapWorksFarFromTheOrigin()
        {
            Vector3 origin = new Vector3(100000.0f, -50000.0f, 75000.0f);
            Box(origin, Vector3.one);

            Assert.That(UnityEngine.Physics.OverlapSphere(origin + new Vector3(0.0f, 0.0f, 1.0f), 0.75f).Length, Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.OverlapSphere(origin + new Vector3(0.0f, 0.0f, 1.0f), 0.25f).Length, Is.EqualTo(0));
        }

        [Test]
        public void TheDefaultSceneOverlapsLikePhysics()
        {
            Box(Vector3.zero, Vector3.one);
            UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
            Collider[] results = new Collider[2];

            Assert.That(scene.OverlapSphere(Vector3.zero, 1.0f, results, ~0, QueryTriggerInteraction.UseGlobal), Is.EqualTo(1));
            Assert.That(scene.OverlapBox(Vector3.zero, Vector3.one, results), Is.EqualTo(1));
            Assert.That(scene.OverlapCapsule(Vector3.zero, Vector3.up, 1.0f, results), Is.EqualTo(1));
        }
    }
}
