using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Bodies
{
    /// <summary>How a Rigidbody moves when nothing touches it.</summary>
    public class RigidbodyConformanceTests : PhysicsScene
    {
        private const float Tolerance = 1e-3f;

        [Test]
        public void GravityAcceleratesADynamicBoxWithSemiImplicitEuler()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 10.0f, 0.0f));

            const int steps = 50;
            Simulate(steps);

            float expectedDrop = Gravity * Step * Step * steps * (steps + 1) / 2.0f;
            Assert.That(body.position.y, Is.EqualTo(10.0f + expectedDrop).Within(0.002f));
            Assert.That(body.linearVelocity.y, Is.EqualTo(Gravity * Step * steps).Within(0.002f));
        }

        [Test]
        public void UseGravityFalseLeavesTheBodyInPlace()
        {
            Rigidbody body = WeightlessBox(new Vector3(0.0f, 10.0f, 0.0f));

            Simulate(25);

            AssertVector(new Vector3(0.0f, 10.0f, 0.0f), body.position, Tolerance);
            AssertVector(Vector3.zero, body.linearVelocity, Tolerance);
        }

        [Test]
        public void TransformFollowsTheBody()
        {
            Rigidbody body = DynamicBox(new Vector3(1.0f, 10.0f, 2.0f));

            Simulate(10);

            AssertVector(body.position, body.transform.position, Tolerance);
            Assert.That(body.transform.position.y, Is.LessThan(10.0f));
        }

        [TestCase(ForceMode.Force, 10.0f / 2.0f * Step)]
        [TestCase(ForceMode.Acceleration, 10.0f * Step)]
        [TestCase(ForceMode.Impulse, 10.0f / 2.0f)]
        [TestCase(ForceMode.VelocityChange, 10.0f)]
        public void AddForceChangesVelocityByMode(ForceMode mode, float expected)
        {
            Rigidbody body = WeightlessBox(Vector3.zero, mass: 2.0f);

            body.AddForce(new Vector3(10.0f, 0.0f, 0.0f), mode);
            Simulate();

            Assert.That(body.linearVelocity.x, Is.EqualTo(expected).Within(Tolerance));
            Assert.That(body.position.x, Is.EqualTo(expected * Step).Within(Tolerance));

            Simulate();

            Assert.That(body.linearVelocity.x, Is.EqualTo(expected).Within(Tolerance), "a force lasts one step");
        }

        // A unit cube of mass 2 has a moment of inertia of 1/3 about each axis.
        [TestCase(ForceMode.Force, 3.0f * 3.0f * Step)]
        [TestCase(ForceMode.Acceleration, 3.0f * Step)]
        [TestCase(ForceMode.Impulse, 3.0f * 3.0f)]
        [TestCase(ForceMode.VelocityChange, 3.0f)]
        public void AddTorqueChangesAngularVelocityByMode(ForceMode mode, float expected)
        {
            Rigidbody body = WeightlessBox(Vector3.zero, mass: 2.0f);
            body.maxAngularVelocity = 1000.0f;

            body.AddTorque(new Vector3(0.0f, 3.0f, 0.0f), mode);
            Simulate();

            Assert.That(body.angularVelocity.y, Is.EqualTo(expected).Within(0.01f));
            Assert.That(body.rotation.eulerAngles.y, Is.EqualTo(expected * Step * Mathf.Rad2Deg).Within(0.05f));
        }

        [Test]
        public void AddForceAtPositionAddsTorque()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.maxAngularVelocity = 1000.0f;

            body.AddForceAtPosition(new Vector3(0.0f, 0.0f, 1.0f), new Vector3(1.0f, 0.0f, 0.0f), ForceMode.Impulse);
            Simulate();

            Assert.That(body.linearVelocity.z, Is.EqualTo(1.0f).Within(Tolerance));
            Assert.That(body.angularVelocity.y, Is.EqualTo(-6.0f).Within(0.01f));
        }

        [Test]
        public void LinearDampingScalesVelocityPerStep()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.linearDamping = 2.0f;
            body.linearVelocity = new Vector3(10.0f, 0.0f, 0.0f);

            Simulate(10);

            float expected = 10.0f * Mathf.Pow(1.0f - 2.0f * Step, 10);
            Assert.That(body.linearVelocity.x, Is.EqualTo(expected).Within(0.005f));
        }

        [Test]
        public void AngularDampingScalesAngularVelocityPerStep()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.angularDamping = 2.0f;
            body.angularVelocity = new Vector3(0.0f, 5.0f, 0.0f);

            Simulate(10);

            float expected = 5.0f * Mathf.Pow(1.0f - 2.0f * Step, 10);
            Assert.That(body.angularVelocity.y, Is.EqualTo(expected).Within(0.005f));
        }

        [Test]
        public void MaxAngularVelocityClampsSpin()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.maxAngularVelocity = 2.0f;

            body.angularVelocity = new Vector3(0.0f, 50.0f, 0.0f);
            Simulate();

            Assert.That(body.angularVelocity.magnitude, Is.EqualTo(2.0f).Within(Tolerance));
        }

        [Test]
        public void MaxLinearVelocityClampsSpeed()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.maxLinearVelocity = 3.0f;

            body.linearVelocity = new Vector3(50.0f, 0.0f, 0.0f);
            Simulate();

            Assert.That(body.linearVelocity.magnitude, Is.EqualTo(3.0f).Within(Tolerance));
        }

        [Test]
        public void KinematicBodyIgnoresForcesAndGravity()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            body.isKinematic = true;

            body.AddForce(Vector3.right, ForceMode.VelocityChange);
            Simulate(10);

            AssertVector(new Vector3(0.0f, 5.0f, 0.0f), body.position, Tolerance);
        }

        /// <summary>
        /// Unity sends the MovePosition target to the physics engine during the next step. Sleeping the body before
        /// that step must not prevent it from reaching the target.
        /// </summary>
        [Test]
        public void MovePositionBeforeSleepingStillReachesItsTarget()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.isKinematic = true;
            Simulate();

            Vector3 target = new Vector3(1.0f, 0.0f, 0.0f);
            body.MovePosition(target);
            body.Sleep();
            Simulate(2);

            AssertVector(target, body.position, 1e-5f, "position");
        }

        [Test]
        public void APoseSetAfterAKinematicTargetWins()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.isKinematic = true;
            Simulate();

            body.MovePosition(new Vector3(1.0f, 0.0f, 0.0f));
            body.position = new Vector3(0.0f, 1.0f, 0.0f);
            Simulate(2);

            AssertVector(new Vector3(0.0f, 1.0f, 0.0f), body.position, 1e-5f, "position");
        }

        [Test]
        public void MovePositionTakesAKinematicBodyToItsTarget()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            body.isKinematic = true;

            Vector3 target = new Vector3(1.0f, 6.0f, -2.0f);
            body.MovePosition(target);
            Simulate();

            AssertVector(target, body.position, Tolerance);
            AssertVector(target, body.transform.position, Tolerance);
        }

        [Test]
        public void MoveRotationTakesAKinematicBodyToItsTarget()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            body.isKinematic = true;

            Quaternion target = Quaternion.Euler(0.0f, 30.0f, 0.0f);
            body.MoveRotation(target);
            Simulate();

            Assert.That(Quaternion.Angle(target, body.rotation), Is.LessThan(0.1f));
        }

        [Test]
        public void PushingADynamicBodyWithAKinematicOne()
        {
            Rigidbody pusher = DynamicBox(new Vector3(-2.0f, 0.0f, 0.0f));
            pusher.isKinematic = true;
            Rigidbody pushed = WeightlessBox(Vector3.zero);

            for (int i = 1; i <= 60; i++)
            {
                pusher.MovePosition(new Vector3(-2.0f + i * 0.05f, 0.0f, 0.0f));
                Simulate();
            }

            Assert.That(pushed.position.x, Is.GreaterThan(0.5f));
        }

        [Test]
        public void ASleepingBodyStaysPutUntilWoken()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));

            body.Sleep();
            Assert.That(body.IsSleeping(), Is.True);
            Simulate(5);
            AssertVector(new Vector3(0.0f, 5.0f, 0.0f), body.position, Tolerance);

            body.WakeUp();
            Assert.That(body.IsSleeping(), Is.False);
            Simulate(5);
            Assert.That(body.position.y, Is.LessThan(4.99f));
        }

        [Test]
        public void GetAccumulatedForceSeesPendingForces()
        {
            Rigidbody body = WeightlessBox(Vector3.zero, mass: 2.0f);

            body.AddForce(new Vector3(3.0f, 0.0f, 0.0f));
            body.AddForce(new Vector3(0.0f, 1.0f, 0.0f), ForceMode.Acceleration);

            AssertVector(new Vector3(3.0f, 2.0f, 0.0f), body.GetAccumulatedForce(Step), Tolerance);

            Simulate();

            AssertVector(Vector3.zero, body.GetAccumulatedForce(Step), Tolerance);
        }

        [Test]
        public void VelocityAtAPointIncludesSpin()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            body.angularVelocity = new Vector3(0.0f, 0.0f, 2.0f);

            AssertVector(new Vector3(1.0f, 2.0f, 0.0f), body.GetPointVelocity(Vector3.right), Tolerance);
        }
    }
}
