using NUnit.Framework;
using Unity.Physics.Configuration;
using UnityEngine;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// KSP's servos read their angle and speed from the moving part's local rotation. The moving part is a child
    /// transform of the servo part with a Rigidbody of its own, jointed to the servo part's body. Its transform must
    /// follow its body after every step.
    /// </summary>
    public class ServoReadoutTests : PhysicsScene
    {
        private int wasMinimumParts;

        [SetUp]
        public void SaveSwitches()
        {
            EnvironmentSwitches.Read();
            wasMinimumParts = VesselSwitches.MinimumParts;
        }

        [TearDown]
        public void RestoreSwitches()
        {
            VesselSwitches.MinimumParts = wasMinimumParts;
        }

        private Rigidbody Body(GameObject gameObject, float mass)
        {
            Rigidbody body = gameObject.AddComponent<Rigidbody>();
            body.mass = mass;
            body.inertiaTensor = 0.1f * Vector3.one;
            body.useGravity = false;
            body.sleepThreshold = 0.0f;
            body.angularDamping = body.linearDamping = 0.0f;
            return body;
        }

        private static ConfigurableJoint Lock(Rigidbody owner, Rigidbody connected, Vector3 axis)
        {
            ConfigurableJoint joint = owner.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = connected;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = Vector3.zero;
            joint.connectedAnchor = connected.transform.InverseTransformPoint(owner.position);
            joint.axis = axis;
            joint.secondaryAxis = Vector3.forward;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            return joint;
        }

        /// <summary>
        /// A servo part on a heavy base, with its moving part as a child body that turns about the servo part's up axis,
        /// and a part attached to the moving part. Returns the servo part's body and the moving part's body.
        /// </summary>
        private (Rigidbody servo, Rigidbody moving) Servo(bool movingFirst)
        {
            GameObject servoObject = Track(new GameObject("servo part"));
            servoObject.transform.position = Vector3.zero;
            GameObject movingObject = new GameObject("moving part");
            movingObject.transform.SetParent(servoObject.transform, false);
            movingObject.transform.localPosition = 0.2f * Vector3.up;

            Rigidbody moving = movingFirst ? Body(movingObject, 0.1f) : null;
            Rigidbody servo = Body(servoObject, 1.0f);
            moving ??= Body(movingObject, 0.1f);

            Rigidbody baseBody = Body(Track(new GameObject("base")), 50.0f);
            baseBody.position = baseBody.transform.position = -0.5f * Vector3.up;
            Lock(servo, baseBody, Vector3.up);

            Rigidbody child = Body(Track(new GameObject("attached part")), 0.5f);
            child.position = child.transform.position = 0.6f * Vector3.up;
            Lock(child, moving, Vector3.up);

            ConfigurableJoint joint = Lock(moving, servo, Vector3.up);
            joint.angularXMotion = ConfigurableJointMotion.Free;
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularXDrive = new JointDrive { positionSpring = 0.0f, positionDamper = 1000.0f, maximumForce = 1000.0f };
            joint.targetAngularVelocity = 2.0f * Vector3.right;
            return (servo, moving);
        }

        /// <summary>The moving part's turn about the servo's up axis, in degrees, from -180 to 180.</summary>
        private static float Turn(Quaternion local)
        {
            local.ToAngleAxis(out float angle, out Vector3 axis);
            angle *= Mathf.Sign(Vector3.Dot(axis, Vector3.up));
            return Mathf.DeltaAngle(0.0f, angle);
        }

        [TestCase(false, false)]
        [TestCase(false, true)]
        [TestCase(true, false)]
        [TestCase(true, true)]
        public void TheMovingPartsTransformFollowsItsBody(bool vessel, bool movingFirst)
        {
            VesselSwitches.MinimumParts = vessel ? 2 : int.MaxValue;
            (Rigidbody servo, Rigidbody moving) = Servo(movingFirst);

            Simulate(25);

            float bodyTurn = Turn(Quaternion.Inverse(servo.rotation) * moving.rotation);
            float transformTurn = Turn(moving.transform.localRotation);
            TestContext.WriteLine($"Body turn {bodyTurn:F3} deg, transform turn {transformTurn:F3} deg, "
                + $"transform to body {Quaternion.Angle(moving.transform.rotation, moving.rotation):F4} deg.");
            Assert.That(bodyTurn, Is.EqualTo(2.0f * 25 * Step * Mathf.Rad2Deg).Within(5.0f), "the moving body's turn");
            Assert.That(Quaternion.Angle(moving.transform.rotation, moving.rotation), Is.LessThan(0.01f), "the moving part's transform");
            Assert.That(transformTurn, Is.EqualTo(bodyTurn).Within(0.01f), "the moving part's local turn");
        }
    }
}
