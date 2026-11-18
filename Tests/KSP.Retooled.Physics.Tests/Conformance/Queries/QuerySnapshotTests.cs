using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Stepping;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Queries
{
    /// <summary>
    /// Verify query snapshot reuse after pose-only changes, tree refits, shape changes and body row changes.
    /// </summary>
    public unsafe class QuerySnapshotTests : PhysicsScene
    {
        private static long Kept => BackendInfo.IsRetooled ? StepEntryPoints.Current->Queries.KeptSnapshots : 0;

        private static Collider HitDown(Vector3 from)
        {
            return UnityEngine.Physics.Raycast(from, Vector3.down, out RaycastHit hit, 100.0f) ? hit.collider : null;
        }

        [Test]
        public void AFallingBoxIsHitAtItsNewPlace()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 10.0f, 0.0f));
            Simulate(2);
            long kept = Kept;

            Simulate(40);

            if (BackendInfo.IsRetooled)
            {
                Assert.That(Kept - kept, Is.GreaterThanOrEqualTo(39), "the snapshot holds no poses, so moving steps keep it");
            }

            Assert.That(body.position.y, Is.LessThan(9.0f));
            Assert.That(UnityEngine.Physics.Raycast(new Vector3(0.0f, 20.0f, 0.0f), Vector3.down, out RaycastHit hit, 100.0f), Is.True);
            Assert.That(hit.collider, Is.SameAs(body.GetComponent<Collider>()));
            Assert.That(hit.point.y, Is.EqualTo(body.position.y + 0.5f).Within(1e-3f));
        }

        /// <summary>
        /// The boxes start in a row and move at different speeds, so that they pass each other. The tree that each
        /// step builds groups the boxes by where they were at the start of the step, so its groups end up far apart.
        /// </summary>
        [Test]
        public void BoxesThatPassEachOtherAreAllFound()
        {
            const int count = 24;
            var bodies = new Rigidbody[count];
            for (int i = 0; i < count; i++)
            {
                // Each box has its own lane in z, so the boxes pass each other without touching.
                bodies[i] = WeightlessBox(new Vector3(2.0f * i, 0.0f, i));
                bodies[i].linearVelocity = new Vector3(((i * 7) % count - count / 2) * 1.5f, 0.0f, 0.0f);
                bodies[i].GetComponent<BoxCollider>().size = new Vector3(0.5f, 0.5f, 0.5f);
            }

            for (int step = 0; step < 30; step++)
            {
                Simulate();
                for (int i = 0; i < count; i++)
                {
                    Vector3 above = bodies[i].position + new Vector3(0.0f, 10.0f, 0.0f);
                    Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, out RaycastHit hit, 100.0f), Is.True, $"box {i} after step {step}");
                    Assert.That(hit.collider.attachedRigidbody, Is.SameAs(bodies[i]), $"box {i} after step {step}");
                }
            }

            Collider[] all = UnityEngine.Physics.OverlapBox(new Vector3(0.0f, 0.0f, count / 2), new Vector3(1000.0f, 1.0f, count));
            Assert.That(all.Length, Is.EqualTo(count));
        }

        [Test]
        public void AColliderAddedToAMovingBodyIsFound()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            Simulate(5);

            GameObject child = new GameObject("child");
            child.transform.SetParent(body.transform, false);
            child.transform.localPosition = new Vector3(0.0f, 0.0f, 3.0f);
            SphereCollider sphere = child.AddComponent<SphereCollider>();
            sphere.radius = 0.5f;
            Simulate(2);

            Vector3 above = sphere.transform.position + new Vector3(0.0f, 10.0f, 0.0f);
            Assert.That(HitDown(above), Is.SameAs(sphere));
            Assert.That(UnityEngine.Physics.OverlapSphere(sphere.transform.position, 0.1f), Does.Contain(sphere));
        }

        [Test]
        public void ARemovedColliderIsNotFound()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            GameObject child = new GameObject("child");
            child.transform.SetParent(body.transform, false);
            child.transform.localPosition = new Vector3(0.0f, 0.0f, 3.0f);
            SphereCollider sphere = child.AddComponent<SphereCollider>();
            Simulate(5);
            Vector3 above = sphere.transform.position + new Vector3(0.0f, 10.0f, 0.0f);
            Assert.That(HitDown(above), Is.SameAs(sphere));

            Object.DestroyImmediate(sphere);
            Simulate(2);

            Assert.That(HitDown(above), Is.Null);
            Assert.That(HitDown(body.position + new Vector3(0.0f, 10.0f, 0.0f)), Is.SameAs(body.GetComponent<Collider>()));
        }

        [Test]
        public void ACollidersTriggerFlagReachesQueries()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            Simulate(5);
            Collider box = body.GetComponent<Collider>();
            Assert.That(HitDown(body.position + new Vector3(0.0f, 10.0f, 0.0f)), Is.SameAs(box));

            box.isTrigger = true;
            Simulate(2);

            Vector3 above = body.position + new Vector3(0.0f, 10.0f, 0.0f);
            Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, 100.0f, ~0, QueryTriggerInteraction.Ignore), Is.False);
            Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, 100.0f, ~0, QueryTriggerInteraction.Collide), Is.True);
        }

        /// <summary>
        /// Destroying a body moves the last body into its row. A body created next takes the row at the end. The built
        /// order has the same length as before, but the rows hold other bodies.
        /// </summary>
        [Test]
        public void ABodyInAReusedRowIsFoundAsItself()
        {
            Rigidbody first = WeightlessBox(new Vector3(0.0f, 0.0f, 0.0f));
            Rigidbody second = WeightlessBox(new Vector3(5.0f, 0.0f, 0.0f));
            Simulate(3);

            Object.DestroyImmediate(first.gameObject);
            Rigidbody third = WeightlessBox(new Vector3(10.0f, 0.0f, 0.0f));
            Simulate(2);

            Assert.That(HitDown(new Vector3(0.0f, 10.0f, 0.0f)), Is.Null);
            Assert.That(HitDown(new Vector3(5.0f, 10.0f, 0.0f)), Is.SameAs(second.GetComponent<Collider>()));
            Assert.That(HitDown(new Vector3(10.0f, 10.0f, 0.0f)), Is.SameAs(third.GetComponent<Collider>()));
        }

        [Test]
        public void ABodyThatStopsBeingSimulatedLeavesQueries()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            Rigidbody other = WeightlessBox(new Vector3(5.0f, 0.0f, 0.0f));
            Simulate(3);

            body.gameObject.SetActive(false);
            Simulate(2);

            Assert.That(HitDown(new Vector3(0.0f, 10.0f, 0.0f)), Is.Null);
            Assert.That(HitDown(new Vector3(5.0f, 10.0f, 0.0f)), Is.SameAs(other.GetComponent<Collider>()));

            body.gameObject.SetActive(true);
            Simulate(2);

            Assert.That(HitDown(new Vector3(0.0f, 10.0f, 0.0f)), Is.SameAs(body.GetComponent<Collider>()));
        }
    }
}
