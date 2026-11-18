using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Interop;
using KSP.Retooled.Physics.Diagnostics;
using KSP.Retooled.Physics.Stepping;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Tests.Host
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
        public unsafe void DetachClearsTheLastSteppedWorld()
        {
            DynamicBox(new Vector3(0.0f, 2.0f, 0.0f));
            Simulate();
            Assert.That((ulong)StepEntryPoints.Current, Is.Not.Zero);

            BackendHost.Detach();
            try
            {
                Assert.That((ulong)StepEntryPoints.Current, Is.Zero);
                ForestDiagnostics.GetCounts(out long solves, out long fallbacks, out long builds);
                Assert.That((solves, fallbacks, builds), Is.EqualTo((0L, 0L, 0L)));
            }
            finally
            {
                BackendHost.Attach();
            }

            Simulate();
            Assert.That((ulong)StepEntryPoints.Current, Is.Not.Zero);
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

                Assert.That(UnityEngine.Physics.Raycast(new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.False);
            }
            finally
            {
                // The ten steps (2.9) and the ray (5.3) did nothing, which the plugin says when the core attaches again.
                LogAssert.Expect(LogType.Warning, new Regex(@"lost because the managed core was detached.* 2\.9x10.* 5\.3x1"));
                BackendHost.Attach();
            }

            Assert.That(body.position.y, Is.EqualTo(2.0f));
        }

        [Test]
        public unsafe void JointsOfAnotherLayoutAreRefused()
        {
            StoreGlobals* globals = NativeLibrary.RetooledPhysics_Globals();
            uint size = globals->jointDataSize;
            Assert.That(LayoutChecks.FirstSizeMismatch(), Is.Null);

            globals->jointDataSize = size + 8;
            try
            {
                Assert.That(LayoutChecks.FirstSizeMismatch(), Does.Contain(nameof(JointData)));
            }
            finally
            {
                globals->jointDataSize = size;
            }
        }

        [Test]
        public void DetachedCommandsAreSafeNoOps()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 2.0f, 0.0f));

            BackendHost.Detach();
            try
            {
                body.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
                body.AddForce(new Vector3(1.0f, 0.0f, 0.0f));
                Assert.That(body.position, Is.EqualTo(Vector3.zero), "a detached read answers zero");
                Assert.That(body.rotation, Is.EqualTo(Quaternion.identity), "a detached rotation is valid");

                // Actor creation requires the managed core. Expect Unity to log an error for each failed creation:
                // first the static actor for the collider, then the dynamic actor for the Rigidbody.
                LogAssert.Expect(LogType.Error, new Regex("Could not create Rigidbody actor"));
                LogAssert.Expect(LogType.Error, new Regex("Could not create Rigidbody actor"));
                GameObject created = Track(new GameObject("detached"));
                created.AddComponent<BoxCollider>();
                created.AddComponent<Rigidbody>();
                UnityEngine.Physics.SyncTransforms();
            }
            finally
            {
                // The plugin names the commands that it couldn't run in a warning when the core attaches again: here
                // CreateBody (2.14), AddForce (3.16) and SetLinearVelocity (3.20), among others.
                LogAssert.Expect(LogType.Warning, new Regex(@"lost because the managed core was detached.* 2\.14x.* 3\.16x1.* 3\.20x1"));
                BackendHost.Attach();
            }

            Assert.That(body.linearVelocity, Is.EqualTo(Vector3.zero), "the velocity set while detached was dropped");
            Assert.That(body.position.y, Is.EqualTo(2.0f), "the store is as it was");
            Simulate(1);
            Assert.That(body.position.y, Is.EqualTo(2.0f).Within(0.01f));
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

            // Mesh cooking was queued while the core was detached, but no shape could be created. Re-enable the
            // collider now that the core is attached so that its shape can be built from the mesh.
            collider.enabled = false;
            collider.enabled = true;
            UnityEngine.Physics.SyncTransforms();

            Assert.That(UnityEngine.Physics.Raycast(new Vector3(0.1f, 5.0f, 0.1f), Vector3.down, out RaycastHit hit, 100.0f), Is.True);
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
            Assert.That(UnityEngine.Physics.Raycast(new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.True);
        }
    }
}
