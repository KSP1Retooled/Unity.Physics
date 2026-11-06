using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>The managed core going away and coming back, which is what a domain reload does to it.</summary>
    public class BackendAttachTests : PhysicsScene
    {
        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
        }

        [Test]
        public void DetachedTheWorldStandsStillAndRaysHitNothing()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 2.0f, 0.0f));

            BackendHost.Detach();
            try
            {
                Simulate(10);

                Assert.That(body.position.y, Is.EqualTo(2.0f));
                Assert.That(Physics.Raycast(new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.False);
            }
            finally
            {
                BackendHost.Attach();
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AMeshCookedWhileDetachedHasItsColliderAfterAttaching(bool convex)
        {
            Mesh mesh = Resources.GetBuiltinResource<Mesh>("Cube.fbx");
            GameObject gameObject = Track(new GameObject("mesh"));
            MeshCollider collider = gameObject.AddComponent<MeshCollider>();
            collider.convex = convex;

            BackendHost.Detach();
            try
            {
                collider.sharedMesh = mesh;
            }
            finally
            {
                BackendHost.Attach();
            }

            Physics.SyncTransforms();

            Assert.That(Physics.Raycast(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down, out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.collider, Is.SameAs(collider));
            Assert.That(hit.distance, Is.EqualTo(4.5f).Within(1e-3f));
        }

        [Test]
        public void AttachingAgainPicksUpTheStoreWhereItWas()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 2.0f, 0.0f));
            Simulate(10);
            float height = body.position.y;
            Vector3 velocity = body.linearVelocity;

            BackendHost.Detach();
            BackendHost.Attach();

            AssertVector(velocity, body.linearVelocity, 1e-4f);
            Simulate(1);
            Assert.That(body.position.y, Is.LessThan(height));

            Simulate(200);
            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.02f), "the colliders came back");
            Assert.That(Physics.Raycast(new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.True);
        }
    }
}
