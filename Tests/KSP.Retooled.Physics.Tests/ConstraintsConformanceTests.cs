using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Test Rigidbody.constraints, applied by Unity during read-back rather than through backend commands.
    /// </summary>
    public class ConstraintsConformanceTests : PhysicsScene
    {
        private const float Tolerance = 1e-3f;

        [Test]
        public void FreezeRotationStopsTorque()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.constraints = RigidbodyConstraints.FreezeRotation;

            body.AddTorque(new Vector3(1.0f, 2.0f, 3.0f), ForceMode.Impulse);
            Simulate(5);

            AssertVector(Vector3.zero, body.angularVelocity, Tolerance);
            Assert.That(Quaternion.Angle(Quaternion.identity, body.rotation), Is.LessThan(0.01f));
        }

        [Test]
        public void FreezeRotationOnOneAxisLeavesTheOthers()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.maxAngularVelocity = 1000.0f;
            body.constraints = RigidbodyConstraints.FreezeRotationY;

            body.AddTorque(new Vector3(1.0f, 1.0f, 0.0f), ForceMode.VelocityChange);
            Simulate();

            Assert.That(body.angularVelocity.x, Is.GreaterThan(0.1f));
            Assert.That(body.angularVelocity.y, Is.EqualTo(0.0f).Within(Tolerance));
        }

        [Test]
        public void FreezeRotationKeepsABoxUprightOnAnEdgeOfTheGround()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(9.8f, 1.0f, 0.0f));
            body.constraints = RigidbodyConstraints.FreezeRotation;

            Simulate(100);

            Assert.That(Quaternion.Angle(Quaternion.identity, body.rotation), Is.LessThan(0.01f));
        }

        [Test]
        public void FreezePositionStopsMotionOnThatAxis()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.constraints = RigidbodyConstraints.FreezePositionX;

            body.AddForce(new Vector3(5.0f, 5.0f, 0.0f), ForceMode.VelocityChange);
            Simulate(5);

            Assert.That(body.position.x, Is.EqualTo(0.0f).Within(Tolerance));
            Assert.That(body.position.y, Is.GreaterThan(0.1f));
        }

        [Test]
        public void FreezePositionHoldsAgainstGravity()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            body.constraints = RigidbodyConstraints.FreezePosition;

            Simulate(25);

            AssertVector(new Vector3(0.0f, 5.0f, 0.0f), body.position, Tolerance);
        }

        [Test]
        public void ClearingConstraintsReleasesTheBody()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            body.constraints = RigidbodyConstraints.FreezeAll;
            Simulate(5);

            body.constraints = RigidbodyConstraints.None;
            body.AddTorque(new Vector3(0.0f, 1.0f, 0.0f), ForceMode.VelocityChange);
            Simulate(5);

            Assert.That(body.position.y, Is.LessThan(4.95f));
            Assert.That(body.angularVelocity.y, Is.GreaterThan(0.5f));
        }
    }
}
