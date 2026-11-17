using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Contacts
{
    /// <summary>
    /// Test static and sliding friction on static and dynamic bodies.
    /// Expected values include PhysX's two friction anchors per patch.
    /// </summary>
    public class FrictionConformanceTests : PhysicsScene
    {
        private const float Slip = 0.01f;

        private static PhysicsMaterial Material(float friction, PhysicsMaterialCombine combine = PhysicsMaterialCombine.Average)
        {
            return new PhysicsMaterial { dynamicFriction = friction, staticFriction = friction, frictionCombine = combine };
        }

        private static void Wear(GameObject gameObject, PhysicsMaterial material)
        {
            gameObject.GetComponent<Collider>().sharedMaterial = material;
        }

        /// <summary>A box lying on a static slope, sized to stay on it for the length of a test.</summary>
        private Rigidbody BoxOnSlope(float degrees, PhysicsMaterial material)
        {
            GameObject slope = Box(new Vector3(0.0f, -0.5f, 0.0f), new Vector3(40.0f, 1.0f, 40.0f));
            slope.transform.rotation = Quaternion.Euler(0.0f, 0.0f, degrees);
            Rigidbody box = DynamicBox(slope.transform.position + slope.transform.up);
            box.transform.rotation = slope.transform.rotation;
            Wear(slope, material);
            Wear(box.gameObject, material);
            return box;
        }

        /// <summary>The deceleration of a body set sliding across the ground, in units of g.</summary>
        private static float FrictionOfSlide(Rigidbody body)
        {
            Simulate(10);
            body.linearVelocity = new Vector3(8.0f, 0.0f, 0.0f);
            Simulate(2);
            float before = body.linearVelocity.x;
            Simulate(10);
            return (before - body.linearVelocity.x) / (10.0f * Step) / -Gravity;
        }

        private float FrictionBetween(PhysicsMaterial bottom, PhysicsMaterial top)
        {
            GameObject ground = Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Wear(ground, bottom);
            Wear(body.gameObject, top);
            body.freezeRotation = true;
            return FrictionOfSlide(body);
        }

        [Test]
        public void ABoxStaysOnASlopeItsFrictionHolds()
        {
            Rigidbody box = BoxOnSlope(20.0f, Material(0.25f));

            Simulate(25);
            Vector3 settled = box.position;
            Simulate(100);

            AssertVector(settled, box.position, 0.002f, "a coefficient of 0.25 holds to about 26 degrees");
        }

        [Test]
        public void ABoxSlidesOffASlopeBeyondItsFriction()
        {
            Rigidbody box = BoxOnSlope(35.0f, Material(0.25f));

            Simulate(25);
            Vector3 settled = box.position;
            Simulate(50);

            Assert.That((box.position - settled).magnitude, Is.GreaterThan(1.0f));
        }

        [Test]
        public void TheDefaultMaterialHoldsLikeAQuarter()
        {
            Rigidbody box = BoxOnSlope(20.0f, null);

            Simulate(25);
            Vector3 settled = box.position;
            Simulate(100);

            AssertVector(settled, box.position, 0.002f);
        }

        [TestCase(0.1f, 0.2f)]
        [TestCase(0.25f, 0.5f)]
        [TestCase(0.5f, 1.0f)]
        public void ASlidingBoxSpendsTwiceItsCoefficient(float coefficient, float expected)
        {
            float friction = FrictionBetween(Material(coefficient), Material(coefficient));

            Assert.That(friction, Is.EqualTo(expected).Within(0.03f));
        }

        // Use explicit materials so that friction expectations do not depend on the project's default material.
        [Test]
        public void ASlidingSphereSpendsItsCoefficient()
        {
            Wear(Ground(), Material(0.25f));
            GameObject sphere = Track(new GameObject("sphere"));
            sphere.transform.position = new Vector3(0.0f, 0.5f, 0.0f);
            sphere.AddComponent<SphereCollider>();
            Wear(sphere, Material(0.25f));
            Rigidbody body = sphere.AddComponent<Rigidbody>();
            body.freezeRotation = true;

            Assert.That(FrictionOfSlide(body), Is.EqualTo(0.25f).Within(0.03f), "one point is one anchor");
        }

        [Test]
        public void MaximumOutranksAverage()
        {
            float friction = FrictionBetween(Material(0.1f), Material(0.5f, PhysicsMaterialCombine.Maximum));

            Assert.That(friction, Is.EqualTo(1.0f).Within(0.1f));
        }

        [Test]
        public void MinimumOutranksAverage()
        {
            float friction = FrictionBetween(Material(0.5f), Material(0.1f, PhysicsMaterialCombine.Minimum));

            Assert.That(friction, Is.EqualTo(0.2f).Within(0.03f));
        }

        [Test]
        public void MultiplyOutranksMinimum()
        {
            float friction = FrictionBetween(Material(0.5f, PhysicsMaterialCombine.Multiply), Material(0.2f, PhysicsMaterialCombine.Minimum));

            Assert.That(friction, Is.EqualTo(0.2f).Within(0.03f), "0.5 times 0.2, twice over");
        }

        [Test]
        public void StaticFrictionHoldsWhatDynamicFrictionWouldNot()
        {
            PhysicsMaterial material = new PhysicsMaterial { dynamicFriction = 0.02f, staticFriction = 0.5f, frictionCombine = PhysicsMaterialCombine.Average };
            Rigidbody box = BoxOnSlope(20.0f, material);

            Simulate(25);
            Vector3 settled = box.position;
            Simulate(100);

            AssertVector(settled, box.position, 0.005f);
        }

        [Test]
        public void DynamicFrictionActsOnceABoxSlides()
        {
            PhysicsMaterial material = new PhysicsMaterial { dynamicFriction = 0.1f, staticFriction = 1.0f, frictionCombine = PhysicsMaterialCombine.Average };

            Assert.That(FrictionBetween(material, material), Is.EqualTo(0.2f).Within(0.03f));
        }

        private void AssertSettles(params Rigidbody[] bodies)
        {
            Vector3[] starts = new Vector3[bodies.Length];
            for (int i = 0; i < bodies.Length; i++)
            {
                starts[i] = bodies[i].position;
            }

            Simulate(150);

            for (int i = 0; i < bodies.Length; i++)
            {
                Vector3 moved = bodies[i].position - starts[i];
                moved.y = 0.0f;
                Assert.That(moved.magnitude, Is.LessThan(Slip), $"body {i} of the pile stays where it was put");
                Assert.That(bodies[i].IsSleeping(), Is.True, $"body {i} of the pile comes to rest");
            }
        }

        [Test]
        public void ABoxRestsOnAnEqualBox()
        {
            Ground();

            AssertSettles(DynamicBox(new Vector3(0.0f, 0.5f, 0.0f)), DynamicBox(new Vector3(0.0f, 1.5f, 0.0f)));
        }

        [Test]
        public void AHeavyBoxRestsOnALightBox()
        {
            Ground();

            AssertSettles(DynamicBox(new Vector3(0.0f, 0.5f, 0.0f)), DynamicBox(new Vector3(0.0f, 1.5f, 0.0f), 10.0f));
        }

        [Test]
        public void ALightBoxRestsOnAHeavyBox()
        {
            Ground();

            AssertSettles(DynamicBox(new Vector3(0.0f, 0.5f, 0.0f), 10.0f), DynamicBox(new Vector3(0.0f, 1.5f, 0.0f)));
        }

        [Test]
        public void ThreeBoxesRestInAColumn()
        {
            Ground();
            Rigidbody bottom = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Rigidbody middle = DynamicBox(new Vector3(0.0f, 1.5f, 0.0f));
            Rigidbody top = DynamicBox(new Vector3(0.0f, 2.5f, 0.0f));

            AssertSettles(bottom, middle, top);
        }

        private Rigidbody Plank()
        {
            Rigidbody plank = Box(new Vector3(0.0f, 0.1f, 0.0f), new Vector3(10.0f, 0.2f, 10.0f)).AddComponent<Rigidbody>();
            plank.mass = 50.0f;
            return plank;
        }

        [Test]
        public void ABoxRestsOnADynamicPlank()
        {
            Ground();

            AssertSettles(Plank(), DynamicBox(new Vector3(0.0f, 0.7f, 0.0f)));
        }

        // Use explicit materials for all bodies to keep the measured sliding distance independent of project
        // defaults.
        [Test]
        public void ABoxRidesAMovingPlank()
        {
            PhysicsMaterial material = Material(0.25f);
            Wear(Ground(), material);
            Rigidbody plank = Plank();
            Rigidbody box = DynamicBox(new Vector3(0.0f, 0.7f, 0.0f));
            Wear(plank.gameObject, material);
            Wear(box.gameObject, material);
            Simulate(50);

            plank.linearVelocity = new Vector3(0.5f, 0.0f, 0.0f);
            box.linearVelocity = plank.linearVelocity;
            Simulate(50);

            Assert.That(plank.position.x, Is.GreaterThan(0.01f), "the ground stops the plank");
            Assert.That(box.position.x - plank.position.x, Is.InRange(-Slip, Slip), "and the box stays on the spot it rode");
        }

        [Test]
        public void ABoxStaysOnABoxOnASlope()
        {
            GameObject slope = Box(new Vector3(0.0f, -0.5f, 0.0f), new Vector3(40.0f, 1.0f, 40.0f));
            slope.transform.rotation = Quaternion.Euler(0.0f, 0.0f, 15.0f);
            Rigidbody lower = DynamicBox(slope.transform.position + slope.transform.up, 20.0f);
            Rigidbody upper = DynamicBox(slope.transform.position + slope.transform.up * 2.0f);
            lower.transform.rotation = slope.transform.rotation;
            upper.transform.rotation = slope.transform.rotation;

            Simulate(25);
            Vector3 settled = upper.position;
            Simulate(125);

            AssertVector(settled, upper.position, 0.005f);
            Assert.That(upper.IsSleeping(), Is.True);
        }
    }
}
