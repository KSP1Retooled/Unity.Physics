using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Test load sharing across the three joints used by large stock stack nodes, including each joint's break
    /// load.
    /// </summary>
    public class StackJointConformanceTests : PhysicsScene
    {
        private const float Mass = 10.0f;
        private const float NodeHeight = 4.5f;
        private const float Radius = 1.0f;

        private static readonly Vector3 Node = new Vector3(0.0f, NodeHeight, 0.0f);

        /// <summary>Three joints from the body to the world, set up the way a stock part joint is.</summary>
        private static ConfigurableJoint[] HangFromNode(Rigidbody body)
        {
            ConfigurableJoint[] joints = new ConfigurableJoint[3];
            for (int i = 0; i < 3; i++)
            {
                Vector3 anchor = Node + Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(Radius, 0.0f, 0.0f);
                ConfigurableJoint joint = body.gameObject.AddComponent<ConfigurableJoint>();
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = body.transform.InverseTransformPoint(anchor);
                joint.connectedAnchor = anchor;
                joint.axis = Vector3.up;
                joint.secondaryAxis = Vector3.forward;
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
                joint.linearLimit = new SoftJointLimit { limit = 1.0f };
                joint.highAngularXLimit = new SoftJointLimit { limit = 177.0f };
                joint.lowAngularXLimit = new SoftJointLimit { limit = 177.0f };
                joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
                joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
                joint.xDrive = joint.yDrive = joint.zDrive = new JointDrive { positionSpring = 1e20f, maximumForce = 1e20f };
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularXDrive = joint.angularYZDrive = new JointDrive { positionSpring = 120000.0f, maximumForce = 1e20f };
                joints[i] = joint;
            }

            return joints;
        }

        private static Vector3 Weight => new Vector3(0.0f, Mass * Gravity, 0.0f);

        [Test]
        public void ABodyHungOnItsAxisLoadsEachJointEqually()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 4.0f, 0.0f), Mass);
            ConfigurableJoint[] joints = HangFromNode(body);

            Simulate(50);

            foreach (ConfigurableJoint joint in joints)
            {
                Assert.That(joint.currentForce.magnitude, Is.EqualTo(Weight.magnitude / 3.0f).Within(5).Percent);
            }
        }

        [Test]
        public void JointsHoldingAnOffsetLoadBalanceItsWeightAndMoment()
        {
            Vector3 centre = new Vector3(1.5f, 4.0f, 0.5f);
            Rigidbody body = DynamicBox(centre, Mass);
            ConfigurableJoint[] joints = HangFromNode(body);

            Simulate(50);

            Vector3 force = Vector3.zero;
            Vector3 moment = Vector3.zero;
            foreach (ConfigurableJoint joint in joints)
            {
                Vector3 anchor = joint.transform.TransformPoint(joint.anchor);
                force += joint.currentForce;
                moment += Vector3.Cross(anchor - Node, joint.currentForce) + joint.currentTorque;
            }

            Vector3 weightMoment = Vector3.Cross(body.worldCenterOfMass - Node, Weight);
            Assert.That(force.magnitude, Is.EqualTo(Weight.magnitude).Within(5).Percent);
            string detail = $"moment {moment:F2} weight {weightMoment:F2} com {body.worldCenterOfMass:F3}";
            foreach (ConfigurableJoint joint in joints)
            {
                detail += $" | f {joint.currentForce:F2} t {joint.currentTorque:F2} at {joint.transform.TransformPoint(joint.anchor):F2}";
            }

            Assert.That(moment.magnitude, Is.EqualTo(weightMoment.magnitude).Within(5).Percent, detail);
            Assert.That(Quaternion.Angle(Quaternion.identity, body.rotation), Is.LessThan(0.5f));
        }
    }
}
