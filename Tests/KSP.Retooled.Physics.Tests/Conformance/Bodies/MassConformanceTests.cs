using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Bodies
{
    /// <summary>
    /// Tests mass, centre of mass and inertia readback, both when set explicitly and derived from colliders.
    /// </summary>
    public class MassConformanceTests : PhysicsScene
    {
        private const float Tolerance = 1e-3f;

        [Test]
        public void MassRoundTrips()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);

            body.mass = 12.5f;

            Assert.That(body.mass, Is.EqualTo(12.5f));
        }

        [Test]
        public void InertiaComesFromTheBoxAndScalesWithMass()
        {
            Rigidbody body = Box(Vector3.zero, new Vector3(1.0f, 2.0f, 3.0f)).AddComponent<Rigidbody>();

            body.mass = 12.0f;

            AssertVector(new Vector3(13.0f, 10.0f, 5.0f), body.inertiaTensor, 0.01f);
        }

        [Test]
        public void SphereInertia()
        {
            GameObject gameObject = Track(new GameObject("sphere"));
            gameObject.AddComponent<SphereCollider>().radius = 2.0f;
            Rigidbody body = gameObject.AddComponent<Rigidbody>();

            body.mass = 5.0f;

            AssertVector(Vector3.one * 8.0f, body.inertiaTensor, 0.01f);
        }

        [Test]
        public void CapsuleInertiaFollowsItsAxis()
        {
            GameObject gameObject = Track(new GameObject("capsule"));
            CapsuleCollider capsule = gameObject.AddComponent<CapsuleCollider>();
            capsule.radius = 0.5f;
            capsule.height = 3.0f;
            capsule.direction = 1;
            Rigidbody body = gameObject.AddComponent<Rigidbody>();

            body.mass = 1.0f;

            Vector3 inertia = body.inertiaTensor;
            Assert.That(inertia.x, Is.EqualTo(inertia.z).Within(0.001f));
            Assert.That(inertia.y, Is.LessThan(inertia.x));
            Assert.That(body.inertiaTensorRotation.w * body.inertiaTensorRotation.w, Is.GreaterThan(0.2f));
        }

        [Test]
        public void CentreOfMassFollowsTheVolumeOfTheColliders()
        {
            GameObject root = Box(Vector3.zero, Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(2.0f, 0.0f, 0.0f);
            child.AddComponent<BoxCollider>().size = new Vector3(1.0f, 1.0f, 3.0f);
            Rigidbody body = root.AddComponent<Rigidbody>();

            AssertVector(new Vector3(1.5f, 0.0f, 0.0f), body.centerOfMass, Tolerance);
            AssertVector(new Vector3(1.5f, 0.0f, 0.0f), body.worldCenterOfMass, Tolerance);
        }

        [Test]
        public void CentreOfMassRoundTripsAndResets()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);

            body.centerOfMass = new Vector3(0.1f, 0.2f, 0.3f);
            AssertVector(new Vector3(0.1f, 0.2f, 0.3f), body.centerOfMass, Tolerance);

            body.ResetCenterOfMass();
            AssertVector(Vector3.zero, body.centerOfMass, Tolerance);
        }

        [Test]
        public void InertiaTensorRoundTripsAndResets()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            Vector3 derived = body.inertiaTensor;

            body.inertiaTensor = new Vector3(2.0f, 3.0f, 4.0f);
            AssertVector(new Vector3(2.0f, 3.0f, 4.0f), body.inertiaTensor, Tolerance);

            body.ResetInertiaTensor();
            AssertVector(derived, body.inertiaTensor, Tolerance);
        }

        [Test]
        public void InertiaTensorRotationRoundTrips()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            Quaternion rotation = Quaternion.Euler(10.0f, 20.0f, 30.0f);

            body.inertiaTensorRotation = rotation;

            Assert.That(Quaternion.Angle(rotation, body.inertiaTensorRotation), Is.LessThan(0.1f));
        }

        [Test]
        public void ForceThroughTheCentreOfMassMakesNoSpin()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.centerOfMass = new Vector3(0.0f, 0.25f, 0.0f);

            body.AddForceAtPosition(Vector3.right, body.worldCenterOfMass, ForceMode.Impulse);
            Simulate();

            AssertVector(new Vector3(1.0f, 0.0f, 0.0f), body.linearVelocity, Tolerance);
            AssertVector(Vector3.zero, body.angularVelocity, Tolerance);
        }

        [Test]
        public void ABodySpinsAboutItsCentreOfMass()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.centerOfMass = new Vector3(0.5f, 0.0f, 0.0f);
            body.maxAngularVelocity = 1000.0f;
            body.angularVelocity = new Vector3(0.0f, Mathf.PI / 2.0f / (50.0f * Step), 0.0f);

            Simulate(50);

            AssertVector(new Vector3(0.5f, 0.0f, 0.0f), body.worldCenterOfMass, 0.01f, "the centre of mass stays put");
            AssertVector(new Vector3(0.5f, 0.0f, 0.5f), body.position, 0.03f, "a quarter turn around it");
        }

        [Test]
        public void HeavierInertiaAxisTurnsLess()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.maxAngularVelocity = 1000.0f;
            body.inertiaTensor = new Vector3(1.0f, 4.0f, 1.0f);

            body.AddTorque(new Vector3(0.0f, 2.0f, 0.0f), ForceMode.Impulse);
            Simulate();

            Assert.That(body.angularVelocity.y, Is.EqualTo(0.5f).Within(Tolerance));
        }
    }
}
