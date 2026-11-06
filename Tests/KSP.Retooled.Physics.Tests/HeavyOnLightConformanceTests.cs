using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>
    /// A heavy stack held up by very light bodies that are pinned to the world and rest on the ground, the way launch clamps
    /// hold a rocket. The light bodies see forces far beyond what their own mass could take, and the scene has to stay calm.
    /// </summary>
    public class HeavyOnLightConformanceTests : PhysicsScene
    {
        private const float StackMass = 125.0f;
        private const float ClampMass = 0.1f;
        private const float ClampBreakForce = 12000.0f;
        private const float FootGap = 0.002f;
        private const float TowerHeight = 7.5f;
        private const int ClampCount = 3;

        // The whole rocket drops onto its clamps here, and PhysX itself peaks near 2.4 m/s doing it.
        private const float RocketSettleSpeed = 3.0f;

        private Quaternion tilt = Quaternion.identity;
        private Rigidbody stack;
        private readonly Rigidbody[] clamps = new Rigidbody[ClampCount];

        private void BuildPad(Vector3 start)
        {
            GameObject ground = Ground();
            ground.name = "pad";

            stack = Box(new Vector3(0.0f, TowerHeight, 0.0f), new Vector3(2.5f, 12.0f, 2.5f)).AddComponent<Rigidbody>();
            stack.mass = StackMass;
            stack.linearVelocity = start;

            for (int i = 0; i < ClampCount; i++)
            {
                float angle = i * 2.0f * Mathf.PI / ClampCount;
                Vector3 side = new Vector3(Mathf.Cos(angle), 0.0f, Mathf.Sin(angle));

                // The body's origin sits at the top of the tower and its foot, far below, is the only part near the ground.
                GameObject clamp = Track(new GameObject("clamp"));
                clamp.transform.position = stack.position + side * 2.0f;

                GameObject foot = new GameObject("foot");
                foot.transform.SetParent(clamp.transform, false);
                foot.transform.localPosition = new Vector3(0.0f, FootGap + 0.5f - TowerHeight, 0.0f);
                foot.AddComponent<BoxCollider>().size = new Vector3(1.0f, 1.0f, 1.0f);

                Rigidbody body = clamp.AddComponent<Rigidbody>();
                body.mass = ClampMass;
                body.centerOfMass = Vector3.zero;
                body.inertiaTensor = new Vector3(7.0f, 0.1f, 7.0f);
                body.linearVelocity = start;
                clamps[i] = body;

                ConfigurableJoint toStack = clamp.AddComponent<ConfigurableJoint>();
                toStack.connectedBody = stack;
                toStack.autoConfigureConnectedAnchor = false;
                toStack.anchor = Vector3.zero;
                toStack.connectedAnchor = side * 2.0f;
                toStack.xMotion = toStack.yMotion = toStack.zMotion = ConfigurableJointMotion.Locked;
                toStack.angularXMotion = toStack.angularYMotion = toStack.angularZMotion = ConfigurableJointMotion.Locked;
                toStack.breakForce = ClampBreakForce;
                toStack.breakTorque = ClampBreakForce;

                Physics.IgnoreCollision(foot.GetComponent<Collider>(), stack.GetComponent<Collider>());
            }
        }

        private void PinToTheWorld(Rigidbody clamp)
        {
            ConfigurableJoint joint = clamp.gameObject.AddComponent<ConfigurableJoint>();
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            joint.configuredInWorldSpace = false;
            joint.autoConfigureConnectedAnchor = false;
            joint.connectedAnchor = clamp.transform.position;
        }

        private float TopSpeed()
        {
            float top = stack.linearVelocity.magnitude;
            foreach (Rigidbody clamp in clamps)
            {
                top = Mathf.Max(top, clamp.linearVelocity.magnitude);
            }

            return top;
        }

        /// <summary>A part joint as stock sets it up: everything limited far away, and drives doing the holding.</summary>
        private ConfigurableJoint PartJoint(Rigidbody host, Rigidbody target, Vector3 anchor, float stiffness)
        {
            Vector3 worldAnchor = tilt * anchor;
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(worldAnchor);
            joint.connectedAnchor = target.transform.InverseTransformPoint(worldAnchor);
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 1.0f };
            joint.highAngularXLimit = new SoftJointLimit { limit = 177.0f };
            joint.lowAngularXLimit = new SoftJointLimit { limit = -177.0f };
            joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
            joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
            JointDrive linear = new JointDrive { positionSpring = 1e20f, maximumForce = 1e20f };
            joint.xDrive = joint.yDrive = joint.zDrive = linear;
            JointDrive angular = new JointDrive { positionSpring = stiffness, maximumForce = 1e20f };
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularXDrive = joint.angularYZDrive = angular;
            return joint;
        }

        private Rigidbody Part(Vector3 position, Vector3 size, float mass)
        {
            GameObject box = Box(tilt * position, size);
            box.transform.rotation = tilt;
            Rigidbody body = box.AddComponent<Rigidbody>();
            body.mass = mass;
            body.linearVelocity = tilt * new Vector3(0.0f, -0.15f, 0.0f);
            return body;
        }

        [Test]
        public void ARocketOnPartJointsStaysCalmWhenItsClampsArrive([Values] bool tilted)
        {
            // The pad's up in KSP's world, where no axis points anywhere in particular.
            tilt = tilted ? Quaternion.FromToRotation(Vector3.up, new Vector3(-0.8231264f, -0.0016914f, 0.567856f)) : Quaternion.identity;
            Physics.gravity = tilt * Physics.gravity;

            GameObject pad = Ground();
            pad.name = "pad";
            pad.transform.SetPositionAndRotation(tilt * pad.transform.position, tilt);

            float y = 10.0f;
            Rigidbody bottom = Part(new Vector3(0.0f, y, 0.0f), new Vector3(2.5f, 8.0f, 2.5f), 18.0f);
            Rigidbody middle = Part(new Vector3(0.0f, y + 8.1f, 0.0f), new Vector3(2.5f, 8.0f, 2.5f), 18.0f);
            Rigidbody coupler = Part(new Vector3(0.0f, y + 12.3f, 0.0f), new Vector3(1.2f, 0.2f, 1.2f), 0.4f);
            Rigidbody upper = Part(new Vector3(0.0f, y + 13.4f, 0.0f), new Vector3(1.2f, 1.8f, 1.2f), 2.3f);
            Rigidbody tip = Part(new Vector3(0.0f, y + 14.5f, 0.0f), new Vector3(0.4f, 0.3f, 0.4f), 0.075f);

            for (int i = 0; i < 3; i++)
            {
                Vector3 offset = Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(1.0f, 0.0f, 0.0f);
                PartJoint(middle, bottom, new Vector3(0.0f, y + 4.05f, 0.0f) + offset, 90000.0f);
            }

            PartJoint(coupler, middle, new Vector3(0.0f, y + 12.2f, 0.0f), 24000.0f);
            PartJoint(upper, coupler, new Vector3(0.0f, y + 12.45f, 0.0f), 24000.0f);
            PartJoint(tip, upper, new Vector3(0.0f, y + 14.35f, 0.0f), 24000.0f);

            Rigidbody[] sides = new Rigidbody[3];
            for (int i = 0; i < 3; i++)
            {
                Vector3 side = Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(3.0f, 0.0f, 0.0f);
                sides[i] = Part(new Vector3(side.x, y, side.z), new Vector3(1.2f, 6.0f, 1.2f), 9.0f);
                Rigidbody strut = Part(new Vector3(side.x * 0.6f, y, side.z * 0.6f), new Vector3(0.2f, 0.2f, 0.2f), 0.1f);
                PartJoint(strut, bottom, new Vector3(side.x * 0.45f, y, side.z * 0.45f), 24000.0f);
                PartJoint(sides[i], strut, new Vector3(side.x * 0.7f, y, side.z * 0.7f), 24000.0f);

                GameObject clampObject = Track(new GameObject("clamp"));
                Vector3 clampOrigin = new Vector3(side.x * 2.0f, y + 2.0f, side.z * 2.0f);
                clampObject.transform.SetPositionAndRotation(tilt * clampOrigin, tilt);
                GameObject foot = new GameObject("foot");
                foot.transform.SetParent(clampObject.transform, false);
                foot.transform.localPosition = new Vector3(0.0f, -(y + 2.0f) + 0.5f + FootGap, 0.0f);
                foot.AddComponent<BoxCollider>().size = Vector3.one;
                Rigidbody clamp = clampObject.AddComponent<Rigidbody>();
                clamp.mass = ClampMass;
                clamp.centerOfMass = Vector3.zero;
                clamp.inertiaTensor = new Vector3(7.0f, 0.1f, 7.0f);
                clamp.linearVelocity = tilt * new Vector3(0.0f, -0.15f, 0.0f);
                clamps[i] = clamp;
                PartJoint(clamp, sides[i], clampOrigin, 24000.0f);
            }

            float top = 0.0f;
            Rigidbody[] all = { bottom, middle, coupler, upper, tip, sides[0], sides[1], sides[2], clamps[0], clamps[1], clamps[2] };

            for (int step = 0; step < 200; step++)
            {
                if (step == 3)
                {
                    foreach (Rigidbody clamp in clamps)
                    {
                        PinToTheWorld(clamp);
                    }
                }

                Simulate();
                foreach (Rigidbody body in all)
                {
                    top = Mathf.Max(top, body.linearVelocity.magnitude);
                }
            }

            Assert.That(top, Is.LessThan(RocketSettleSpeed));
        }

        [Test]
        public void ClampsPinnedLateStayCalmWhenTheirFeetLand()
        {
            BuildPad(new Vector3(0.0f, -0.15f, 0.0f));

            float top = 0.0f;
            for (int step = 0; step < 200; step++)
            {
                if (step == 3)
                {
                    foreach (Rigidbody clamp in clamps)
                    {
                        PinToTheWorld(clamp);
                    }
                }

                Simulate();
                top = Mathf.Max(top, TopSpeed());
            }

            foreach (Rigidbody clamp in clamps)
            {
                Assert.That(clamp.GetComponents<ConfigurableJoint>().Length, Is.EqualTo(2), "a clamp lost a joint");
            }

            Assert.That(top, Is.LessThan(2.0f));
        }
    }
}
