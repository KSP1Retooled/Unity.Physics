using NUnit.Framework;
using Unity.Physics.Configuration;
using UnityEngine;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// Spin KSP rotors carrying blades inside vessels. The rotor joint is a ConfigurableJoint with every axis locked
    /// except angular X, driven like KSP's rotors: a spring of 1e-10, a damper and a target angular velocity.
    /// </summary>
    public class RotorTests : PhysicsScene
    {

        private int wasMinimumParts;

        [SetUp]
        public void TurnVesselsOn()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Rotor vessels require the Retooled backend.");
            }

            EnvironmentSwitches.Read();

            // These crafts are small, so let them compile into vessels.
            wasMinimumParts = VesselSwitches.MinimumParts;
            VesselSwitches.MinimumParts = 2;
        }

        [TearDown]
        public void RestoreSwitches()
        {
            if (BackendInfo.IsRetooled)
            {
                VesselSwitches.MinimumParts = wasMinimumParts;
            }
        }

        private static Rigidbody Body(GameObject part, Vector3 position, float mass, float inertia, bool gravity)
        {
            part.transform.position = position;
            Rigidbody body = part.AddComponent<Rigidbody>();
            body.mass = mass;
            body.inertiaTensor = Vector3.one * inertia;
            body.useGravity = gravity;
            body.sleepThreshold = 0.0f;
            body.angularDamping = body.linearDamping = 0.0f;
            body.maxAngularVelocity = 100.0f;
            return body;
        }

        private Rigidbody Part(Vector3 position, float mass = 1.0f, float inertia = 0.2f, bool gravity = false)
            => Body(Track(new GameObject("rotor part")), position, mass, inertia, gravity);

        private static ConfigurableJoint Link(Rigidbody owner, Rigidbody connected, Vector3 point, Vector3 axis, bool rotor = false)
        {
            ConfigurableJoint joint = owner.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = connected;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = owner.transform.InverseTransformPoint(point);
            joint.connectedAnchor = connected.transform.InverseTransformPoint(point);
            joint.axis = axis;
            joint.secondaryAxis = Vector3.up;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = rotor ? ConfigurableJointMotion.Free : ConfigurableJointMotion.Locked;
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            return joint;
        }

        /// <summary>KSP's rotors give their velocity drive a spring of 1e-10 rather than zero.</summary>
        private static void Drive(ConfigurableJoint joint, float damper, float maxForce)
            => joint.angularXDrive = new JointDrive { positionSpring = 1e-10f, positionDamper = damper, maximumForce = maxForce };

        private static void Limits(ConfigurableJoint joint, float low, float high)
        {
            joint.lowAngularXLimit = new SoftJointLimit { limit = low };
            joint.highAngularXLimit = new SoftJointLimit { limit = high };
            joint.angularXMotion = ConfigurableJointMotion.Limited;
        }

        private static float AnchorGap(ConfigurableJoint joint) => Vector3.Distance(
            joint.transform.TransformPoint(joint.anchor), joint.connectedBody.transform.TransformPoint(joint.connectedAnchor));

        private static float AxisGap(ConfigurableJoint joint)
        {
            Vector3 a = joint.transform.TransformDirection(joint.axis);
            Vector3 b = joint.connectedBody.transform.TransformDirection(joint.axis);
            return Mathf.Atan2(Vector3.Cross(a, b).magnitude, Vector3.Dot(a, b)) * Mathf.Rad2Deg;
        }

        private static float RotorSpeed(Rigidbody rotor, Rigidbody mount, Vector3 axis)
            => Vector3.Dot(rotor.angularVelocity - mount.angularVelocity, mount.rotation * axis);

        [TestCase(false)]
        [TestCase(true)]
        public void ABladeRotorSpinsAtFortyRadiansPerSecond(bool limited)
        {
            var bodies = new[] { Part(-Vector3.right), Part(Vector3.zero), Part(Vector3.right), Part(2.0f * Vector3.right) };
            Link(bodies[1], bodies[0], bodies[1].position, Vector3.right);
            Link(bodies[3], bodies[2], bodies[2].position, Vector3.right);
            bodies[0].mass = 50.0f;
            bodies[0].inertiaTensor = 10.0f * Vector3.one;
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            if (limited)
            {
                Limits(joint, -177.0f, 177.0f);
                joint.angularXLimitSpring = new SoftJointLimitSpring { spring = 1e-8f };
            }

            var blades = new ConfigurableJoint[4];
            for (int i = 0; i < blades.Length; i++)
            {
                Vector3 offset = (i % 2 == 0 ? Vector3.up : Vector3.forward) * (i < 2 ? 0.35f : -0.35f);
                Rigidbody blade = Part(bodies[2].position + offset, 0.25f, 0.04f);
                blades[i] = Link(blade, bodies[2], blade.position, Vector3.right);
            }

            Drive(joint, 80.0f, 1000.0f);
            joint.targetAngularVelocity = 40.0f * Vector3.right;
            float anchor = 0.0f, axis = 0.0f, bladeGap = 0.0f, peakSpeed = 0.0f, speed = 0.0f;
            for (int i = 0; i < 75; i++)
            {
                Simulate();
                anchor = Mathf.Max(anchor, AnchorGap(joint));
                axis = Mathf.Max(axis, AxisGap(joint));
                foreach (ConfigurableJoint blade in blades)
                {
                    bladeGap = Mathf.Max(bladeGap, AnchorGap(blade));
                }

                speed = RotorSpeed(bodies[2], bodies[1], Vector3.right);
                peakSpeed = Mathf.Max(peakSpeed, speed);
            }

            TestContext.WriteLine($"Rotor limited {limited}, speed {speed:F4} rad/s, peak speed {peakSpeed:F4} rad/s, "
                + $"peak anchor gap {anchor:E4} m, peak axis deflection {axis:E4} deg, peak blade anchor gap {bladeGap:E4} m.");
            Assert.That(float.IsFinite(speed + anchor + axis + bladeGap), Is.True);
            Assert.That(peakSpeed, Is.GreaterThan(39.0f).And.LessThanOrEqualTo(40.5f));
            Assert.That(speed, Is.EqualTo(40.0f).Within(0.5f));
            Assert.That(anchor, Is.LessThan(0.002f));
            Assert.That(axis, Is.LessThan(0.1f));
            Assert.That(bladeGap, Is.LessThan(0.002f));
        }

        /// <summary>
        /// A rotor like KSP's EM-32 on an 18 t tank on the ground. KSP limits the rotor's own axis to a range from 0.01
        /// to 0.1 degrees with a limit spring of 1e-8, which PhysX treats as almost free.
        /// </summary>
        [Test]
        public void ABladeRotorSpinsAtFortyEightRadiansPerSecondOnTheGround()
        {
            Ground();
            Rigidbody tank = Body(Box(Vector3.up, new Vector3(2.0f, 2.0f, 2.0f)), Vector3.up, 18.0f, 12.0f, true);
            Rigidbody probe = Part(2.1f * Vector3.up, 0.15f, 0.015f, true);
            Rigidbody rotor = Part(2.35f * Vector3.up, 0.1f, 0.016f, true);
            Link(probe, tank, probe.position, Vector3.right);

            var joint = Link(rotor, probe, rotor.position, Vector3.up, true);
            joint.secondaryAxis = Vector3.right;
            Limits(joint, 0.01f, 0.1f);
            joint.angularXLimitSpring = new SoftJointLimitSpring { spring = 1e-8f, damper = 1e-8f };
            Drive(joint, 1000.0f, 70.0f);

            var blades = new ConfigurableJoint[2];
            for (int i = 0; i < blades.Length; i++)
            {
                float offset = i == 0 ? -1.25f : 1.25f;
                Rigidbody blade = Part(rotor.position + offset * Vector3.right, 0.075f, 0.025f, true);
                blades[i] = Link(blade, rotor, rotor.position, Vector3.right);
            }

            float anchor = 0.0f, axis = 0.0f, bladeGap = 0.0f, tilt = 0.0f, peakSpeed = 0.0f, speed = 0.0f;
            for (int i = 0; i < 300; i++)
            {
                // KSP's rotors ramp their target speed up; this one takes a second.
                joint.targetAngularVelocity = Mathf.Min(48.0f, 48.0f * i * Step) * Vector3.right;
                Simulate();
                anchor = Mathf.Max(anchor, AnchorGap(joint));
                axis = Mathf.Max(axis, AxisGap(joint));
                tilt = Mathf.Max(tilt, Vector3.Angle(Vector3.up, tank.rotation * Vector3.up));
                foreach (ConfigurableJoint blade in blades)
                {
                    bladeGap = Mathf.Max(bladeGap, AnchorGap(blade));
                }

                speed = RotorSpeed(rotor, probe, Vector3.up);
                peakSpeed = Mathf.Max(peakSpeed, speed);
            }

            TestContext.WriteLine($"Speed {speed:F4} rad/s, peak speed {peakSpeed:F4} rad/s, peak anchor gap {anchor:E4} m, "
                + $"peak axis deflection {axis:E4} deg, peak blade anchor gap {bladeGap:E4} m, peak base tilt {tilt:E4} deg.");
            Assert.That(float.IsFinite(speed + anchor + axis + bladeGap + tilt), Is.True);
            Assert.That(speed, Is.EqualTo(48.0f).Within(0.5f));
            Assert.That(peakSpeed, Is.LessThanOrEqualTo(48.5f));
            Assert.That(anchor, Is.LessThan(0.002f));
            Assert.That(axis, Is.LessThan(0.1f));
            Assert.That(bladeGap, Is.LessThan(0.002f));
            Assert.That(tilt, Is.LessThan(1.0f));
        }
    }
}
