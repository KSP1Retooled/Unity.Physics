using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>Physics.SphereCast, CapsuleCast and BoxCast against the scene.</summary>
    public class ShapeCastConformanceTests : QueryScene
    {
        // Unity Physics stops a cast 1e-4 short of the surface, and a PhysX box sweep 0.0025.
        private const float Tolerance = 3e-3f;

        [Test]
        public void ASphereStopsWhereItsSurfaceMeetsTheBox()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.collider, Is.SameAs(box.GetComponent<Collider>()));
            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
            AssertVector(new Vector3(0.0f, 0.0f, 4.5f), hit.point, Tolerance);
            AssertVector(Vector3.back, hit.normal, Tolerance);
            Assert.That(hit.triangleIndex, Is.EqualTo(-1));
        }

        [Test]
        public void ASphereGrazingAnEdgeGetsTheEdgeNormal()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.SphereCast(new Vector3(0.6f, 0.0f, 0.0f), 0.25f, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.distance, Is.EqualTo(4.2708f).Within(Tolerance));
            AssertVector(new Vector3(0.5f, 0.0f, 4.5f), hit.point, Tolerance);
            AssertVector(new Vector3(0.4f, 0.0f, -0.9165f), hit.normal, Tolerance);
        }

        [Test]
        public void ACastPassingBesideTheBoxMisses()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.SphereCast(new Vector3(0.8f, 0.0f, 0.0f), 0.25f, Vector3.forward, out RaycastHit _, 100.0f), Is.False);
        }

        [Test]
        public void ACastStopsAtItsMaximumDistance()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit _, 4.0f), Is.False);
        }

        [Test]
        public void ACastWithoutADistanceReachesFar()
        {
            Box(new Vector3(0.0f, 0.0f, 5000.0f), Vector3.one);

            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit hit), Is.True);

            Assert.That(hit.distance, Is.EqualTo(4999.25f).Within(0.01f));
        }

        [Test]
        public void TheDirectionsLengthDoesNotScaleTheCast()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward * 7.0f, out RaycastHit hit, 100.0f);

            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
        }

        [Test]
        public void ADegenerateCastHitsNothing()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.zero, out RaycastHit _, 100.0f), Is.False);
            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit _, 0.0f), Is.False);
        }

        [Test]
        public void ASphereWithoutARadiusActsLikeARay()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.SphereCast(Vector3.zero, 0.0f, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.distance, Is.EqualTo(4.5f).Within(Tolerance));
        }

        [TestCase(0.0f, 0.0f, 5.0f)]
        [TestCase(0.0f, 0.0f, 4.4f)]
        [TestCase(0.0f, 0.0f, 4.25f)]
        public void ACastStartingInAColliderIgnoresIt(float x, float y, float z)
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.SphereCast(new Vector3(x, y, z), 0.25f, Vector3.forward, out RaycastHit _, 100.0f), Is.False);
        }

        [Test]
        public void ACastGoesOnPastAColliderItStartsIn()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            GameObject far = Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);

            Assert.That(Physics.SphereCast(new Vector3(0.0f, 0.0f, 5.0f), 0.25f, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.collider, Is.SameAs(far.GetComponent<Collider>()));
        }

        [Test]
        public void CastAllReportsAStartOverlapByConvention()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            RaycastHit[] hits = Physics.SphereCastAll(new Vector3(0.0f, 0.0f, 5.0f), 0.25f, Vector3.forward, 100.0f);

            Assert.That(hits.Length, Is.EqualTo(1));
            Assert.That(hits[0].collider, Is.SameAs(box.GetComponent<Collider>()));
            Assert.That(hits[0].distance, Is.EqualTo(0.0f));
            Assert.That(hits[0].point, Is.EqualTo(Vector3.zero));
            AssertVector(Vector3.back, hits[0].normal, Tolerance);
        }

        [Test]
        public void CastAllFindsEveryColliderOnTheWay()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);

            RaycastHit[] hits = Physics.SphereCastAll(Vector3.zero, 0.25f, Vector3.forward, 100.0f);

            Assert.That(Distances(hits), Is.EqualTo(new[] { 4.25f, 8.25f }).Within(Tolerance));
        }

        [Test]
        public void NonAllocFillsTheArrayAndStops()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);
            RaycastHit[] hits = new RaycastHit[1];

            Assert.That(Physics.SphereCastNonAlloc(Vector3.zero, 0.25f, Vector3.forward, hits, 100.0f), Is.EqualTo(1));

            RaycastHit[] room = new RaycastHit[4];
            Assert.That(Physics.SphereCastNonAlloc(Vector3.zero, 0.25f, Vector3.forward, room, 100.0f), Is.EqualTo(2));
        }

        [Test]
        public void TheLayerMaskFiltersCasts()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            box.layer = 28;

            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit _, 100.0f, 1 << 29), Is.False);
            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit _, 100.0f, 1 << 28), Is.True);
        }

        [Test]
        public void TriggersFollowTheQueryInteraction()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            box.GetComponent<BoxCollider>().isTrigger = true;

            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit _, 100.0f, ~0, QueryTriggerInteraction.Ignore), Is.False);
            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit _, 100.0f, ~0, QueryTriggerInteraction.Collide), Is.True);
        }

        [Test]
        public void ACastFindsABodyWhereTheStepLeftIt()
        {
            Rigidbody body = WeightlessBox(new Vector3(0.0f, 0.0f, 5.0f));
            body.linearVelocity = new Vector3(0.0f, 0.0f, 10.0f);
            Simulate(10);

            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.distance, Is.EqualTo(4.25f + 10.0f * Step * 10).Within(0.01f));
            Assert.That(hit.rigidbody, Is.SameAs(body));
        }

        [Test]
        public void ACastNamesTheChildOfACompoundBody()
        {
            GameObject root = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, 3.0f, 0.0f);
            BoxCollider childCollider = child.AddComponent<BoxCollider>();
            root.AddComponent<Rigidbody>();

            Physics.SphereCast(new Vector3(0.0f, 3.0f, 0.0f), 0.25f, Vector3.forward, out RaycastHit hit, 100.0f);

            Assert.That(hit.collider, Is.SameAs(childCollider));
        }

        [Test]
        public void ACapsuleSweepsItsWholeLength()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            bool cast = Physics.CapsuleCast(new Vector3(0.0f, -3.0f, 0.0f), new Vector3(0.0f, 3.0f, 0.0f), 0.25f, Vector3.forward, out RaycastHit hit, 100.0f);

            Assert.That(cast, Is.True);
            Assert.That(hit.collider, Is.SameAs(box.GetComponent<Collider>()));
            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
            AssertVector(Vector3.back, hit.normal, Tolerance);
            Assert.That(hit.point.z, Is.EqualTo(4.5f).Within(Tolerance));
            Assert.That(Mathf.Abs(hit.point.y), Is.LessThanOrEqualTo(0.5f + Tolerance));
        }

        [Test]
        public void ACapsuleCapMeetsAnEdge()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            bool cast = Physics.CapsuleCast(new Vector3(0.0f, 0.6f, 0.0f), new Vector3(0.0f, 3.0f, 0.0f), 0.25f, Vector3.forward, out RaycastHit hit, 100.0f);

            Assert.That(cast, Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.2708f).Within(Tolerance));
            AssertVector(new Vector3(0.0f, 0.5f, 4.5f), hit.point, Tolerance);
        }

        [Test]
        public void ACapsuleWithBothEndsTogetherIsASphere()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Physics.CapsuleCast(Vector3.zero, Vector3.zero, 0.25f, Vector3.forward, out RaycastHit hit, 100.0f);

            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
        }

        [Test]
        public void CapsuleCastAllStartingInAColliderReportsIt()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            RaycastHit[] hits = Physics.CapsuleCastAll(new Vector3(0.0f, -1.0f, 5.0f), new Vector3(0.0f, 1.0f, 5.0f), 0.25f, Vector3.right, 100.0f);

            Assert.That(hits.Length, Is.EqualTo(1));
            Assert.That(hits[0].distance, Is.EqualTo(0.0f));
            AssertVector(Vector3.left, hits[0].normal, Tolerance);
        }

        [Test]
        public void ABoxSweepsItsFace()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.BoxCast(Vector3.zero, new Vector3(1.0f, 1.0f, 0.5f), Vector3.forward, out RaycastHit hit, Quaternion.identity, 100.0f), Is.True);

            Assert.That(hit.distance, Is.EqualTo(4.0f).Within(Tolerance));
            AssertVector(Vector3.back, hit.normal, Tolerance);
            Assert.That(hit.point.z, Is.EqualTo(4.5f).Within(Tolerance));
        }

        [Test]
        public void ARotatedBoxLeadsWithItsEdge()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Physics.BoxCast(Vector3.zero, Vector3.one * 0.25f, Vector3.forward, out RaycastHit hit, Quaternion.Euler(0.0f, 45.0f, 0.0f), 100.0f);

            Assert.That(hit.distance, Is.EqualTo(4.5f - 0.25f * Mathf.Sqrt(2.0f)).Within(Tolerance));
        }

        [Test]
        public void BoxCastAllStartingInAColliderReportsIt()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            RaycastHit[] hits = Physics.BoxCastAll(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one * 0.1f, Vector3.right, Quaternion.identity, 100.0f);

            Assert.That(hits.Length, Is.EqualTo(1));
            Assert.That(hits[0].distance, Is.EqualTo(0.0f));
        }

        [Test]
        public void ACastHitsAConvexMesh()
        {
            ConvexCube(new Vector3(0.0f, 0.0f, 5.0f));

            Assert.That(Physics.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
            Assert.That(hit.triangleIndex, Is.EqualTo(-1));
        }

        [Test]
        public void ACastHitsTheFrontOfATriangleMesh()
        {
            GameObject mesh = TriangleMeshQuad(Vector3.zero);

            Assert.That(Physics.SphereCast(new Vector3(0.0f, 0.0f, -3.0f), 0.25f, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.collider, Is.SameAs(mesh.GetComponent<Collider>()));
            Assert.That(hit.distance, Is.EqualTo(2.75f).Within(Tolerance));
            AssertVector(Vector3.back, hit.normal, Tolerance);
        }

        [Test]
        public void ACastPassesThroughTheBackOfATriangleMesh()
        {
            TriangleMeshQuad(Vector3.zero);

            Assert.That(Physics.SphereCast(new Vector3(0.0f, 0.0f, 3.0f), 0.25f, Vector3.back, out RaycastHit _, 100.0f), Is.False);
        }

        [Test]
        public void HittingBackfacesGivesTheTrianglesOwnNormal()
        {
            TriangleMeshQuad(Vector3.zero);
            bool backfaces = Physics.queriesHitBackfaces;
            Physics.queriesHitBackfaces = true;
            try
            {
                Assert.That(Physics.SphereCast(new Vector3(0.0f, 0.0f, 3.0f), 0.25f, Vector3.back, out RaycastHit hit, 100.0f), Is.True);

                Assert.That(hit.distance, Is.EqualTo(2.75f).Within(Tolerance));
                AssertVector(Vector3.back, hit.normal, Tolerance);
            }
            finally
            {
                Physics.queriesHitBackfaces = backfaces;
            }
        }

        [Test]
        public void ACastIsOneHitPerMesh()
        {
            TriangleMeshCube(Vector3.zero);

            Assert.That(Physics.SphereCastAll(new Vector3(0.0f, 0.0f, -3.0f), 0.25f, Vector3.forward, 100.0f).Length, Is.EqualTo(1));
        }

        [Test]
        public void ACastStraddlingAMeshFromTheFrontOverlapsIt()
        {
            TriangleMeshQuad(Vector3.zero);
            Vector3 start = new Vector3(0.0f, 0.0f, -0.1f);

            Assert.That(Physics.SphereCast(start, 0.25f, Vector3.forward, out RaycastHit _, 10.0f), Is.False);

            RaycastHit[] hits = Physics.SphereCastAll(start, 0.25f, Vector3.forward, 10.0f);
            Assert.That(hits.Length, Is.EqualTo(1));
            Assert.That(hits[0].distance, Is.EqualTo(0.0f));
            Assert.That(hits[0].point, Is.EqualTo(Vector3.zero));
        }

        [Test]
        public void ACastStraddlingAMeshFromBehindNeverTouchesIt()
        {
            TriangleMeshQuad(Vector3.zero);
            Vector3 start = new Vector3(0.0f, 0.0f, 0.1f);

            Assert.That(Physics.SphereCast(start, 0.25f, Vector3.back, out RaycastHit _, 10.0f), Is.False);
            Assert.That(Physics.SphereCastAll(start, 0.25f, Vector3.back, 10.0f).Length, Is.EqualTo(0));
        }

        [Test]
        public void AMeshOverlappedAtTheStartIsOutForTheWholeCast()
        {
            TriangleMeshCube(Vector3.zero);
            Vector3 start = new Vector3(0.0f, 0.0f, -0.55f);

            Assert.That(Physics.SphereCast(start, 0.25f, Vector3.forward, out RaycastHit _, 10.0f), Is.False);

            RaycastHit[] hits = Physics.SphereCastAll(start, 0.25f, Vector3.forward, 10.0f);
            Assert.That(hits.Length, Is.EqualTo(1));
            Assert.That(hits[0].distance, Is.EqualTo(0.0f));
        }

        [Test]
        public void ACastInsideAClosedMeshTouchesNothing()
        {
            TriangleMeshCube(Vector3.zero);

            Assert.That(Physics.SphereCast(Vector3.zero, 0.1f, Vector3.forward, out RaycastHit _, 10.0f), Is.False);
            Assert.That(Physics.SphereCastAll(Vector3.zero, 0.1f, Vector3.forward, 10.0f).Length, Is.EqualTo(0));
        }

        [Test]
        public void ACastWorksFarFromTheOrigin()
        {
            Vector3 origin = new Vector3(100000.0f, -50000.0f, 75000.0f);
            Box(origin + new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);

            Assert.That(Physics.SphereCast(origin, 0.25f, Vector3.forward, out RaycastHit hit, 100.0f), Is.True);

            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(0.02f));
        }

        [Test]
        public void TheDefaultSceneCastsLikePhysics()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            UnityEngine.PhysicsScene scene = Physics.defaultPhysicsScene;

            Assert.That(scene.SphereCast(Vector3.zero, 0.25f, Vector3.forward, out RaycastHit hit), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
            Assert.That(scene.BoxCast(Vector3.zero, Vector3.one * 0.25f, Vector3.forward, out hit), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
            Assert.That(scene.CapsuleCast(Vector3.zero, Vector3.up, 0.25f, Vector3.forward, out hit), Is.True);
            Assert.That(hit.distance, Is.EqualTo(4.25f).Within(Tolerance));
        }
    }
}
