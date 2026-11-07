using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Two heavy booster columns stand on the pad with a core hung between them, the way a big asparagus craft sits before
    /// launch. Light decoupler and truss parts carry the core, and struts that never break tie the three together. The ground
    /// closes a loop through both columns, so how the load splits depends on how stiff each path is, and a light part must
    /// not end up carrying far more than its share.
    /// </summary>
    public class TwinBoosterConformanceTests : PhysicsScene
    {
        private const float TankMass = 450.0f;
        private const float EngineMass = 15.0f;
        private const float CoreMass = 100.0f;
        private const float TrussMass = 0.375f;
        private const float DecouplerMass = 0.05f;
        private const int TanksPerColumn = 6;
        private const float TankHeight = 7.5f;
        private const float Width = 3.75f;
        private const float ColumnX = 5.0f;
        private const float BridgeHeight = 25.0f;

        private const float StackStiffness = 120000.0f;
        private const float StackBreakForce = 53333.0f;
        private const float BridgeStiffness = 90000.0f;
        private const float BridgeBreakForce = 1760.0f;
        private const float StrutStiffness = 24000.0f;

        private readonly List<ConfigurableJoint> bridgeJoints = new List<ConfigurableJoint>();
        private readonly List<Rigidbody> bodies = new List<Rigidbody>();

        /// <summary>A part joint as stock sets it up: everything limited far away, and drives doing the holding.</summary>
        private static ConfigurableJoint PartJoint(Rigidbody host, Rigidbody target, Vector3 anchor, float stiffness, float breakForce)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(anchor);
            joint.connectedAnchor = target.transform.InverseTransformPoint(anchor);
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
            joint.breakForce = breakForce;
            joint.breakTorque = breakForce;
            return joint;
        }

        /// <summary>Stock joins two stacked parts with several joints spread over the node, not one at its centre.</summary>
        private static void StackJoint(Rigidbody upper, Rigidbody lower, float y, float x)
        {
            for (int i = 0; i < 3; i++)
            {
                Vector3 offset = Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(1.0f, 0.0f, 0.0f);
                PartJoint(upper, lower, new Vector3(x, y, 0.0f) + offset, StackStiffness, StackBreakForce);
            }
        }

        private Rigidbody Part(Vector3 position, Vector3 size, float mass)
        {
            Rigidbody body = Box(position, size).AddComponent<Rigidbody>();
            body.mass = mass;
            bodies.Add(body);
            return body;
        }

        /// <summary>A column standing on its engine, and the tank at the height where the core hangs from it.</summary>
        private Rigidbody[] Column(float x)
        {
            Rigidbody engine = Part(new Vector3(x, 0.75f, 0.0f), new Vector3(Width, 1.5f, Width), EngineMass);
            Rigidbody[] tanks = new Rigidbody[TanksPerColumn];
            Rigidbody below = engine;
            float bottom = 1.5f;
            for (int i = 0; i < TanksPerColumn; i++)
            {
                tanks[i] = Part(new Vector3(x, bottom + TankHeight * 0.5f, 0.0f), new Vector3(Width, TankHeight, Width), TankMass);
                StackJoint(tanks[i], below, bottom, x);
                below = tanks[i];
                bottom += TankHeight;
            }

            return tanks;
        }

        /// <summary>
        /// A decoupler on the core and a truss on the decoupler, with the column's tank on the truss: the light parts are the
        /// only tree path from a column to the core.
        /// </summary>
        private void Bridge(Rigidbody core, Rigidbody tank, float side)
        {
            float coreEdge = Width * 0.5f;
            Rigidbody decoupler = Part(new Vector3(side * (coreEdge + 0.2f), BridgeHeight, 0.0f), new Vector3(0.2f, 0.4f, 0.4f), DecouplerMass);
            Rigidbody truss = Part(new Vector3(side * (coreEdge + 0.8f), BridgeHeight, 0.0f), new Vector3(0.8f, 0.8f, 0.8f), TrussMass);
            bridgeJoints.Add(PartJoint(decoupler, core, new Vector3(side * (coreEdge + 0.05f), BridgeHeight, 0.0f), BridgeStiffness, BridgeBreakForce));
            bridgeJoints.Add(PartJoint(truss, decoupler, new Vector3(side * (coreEdge + 0.35f), BridgeHeight, 0.0f), BridgeStiffness, BridgeBreakForce));
            bridgeJoints.Add(PartJoint(tank, truss, new Vector3(side * (ColumnX - Width * 0.5f - 0.05f), BridgeHeight, 0.0f), BridgeStiffness, BridgeBreakForce));
        }

        /// <summary>A strut joins two parts that aren't neighbours in the tree, and stock never lets it break.</summary>
        private static void Strut(Rigidbody host, Rigidbody target, Vector3 anchor)
        {
            PartJoint(host, target, anchor, StrutStiffness, float.PositiveInfinity);
        }

        [Test]
        public void ACoreHungBetweenTwoBoostersKeepsItsLightParts()
        {
            Ground();

            Rigidbody[] left = Column(-ColumnX);
            Rigidbody[] right = Column(ColumnX);
            Rigidbody core = Part(new Vector3(0.0f, BridgeHeight, 0.0f), new Vector3(Width, 20.0f, Width), CoreMass);

            Bridge(core, left[3], -1.0f);
            Bridge(core, right[3], 1.0f);

            float strutX = ColumnX - Width * 0.5f - 0.6f;
            foreach (float side in new[] { -1.0f, 1.0f })
            {
                Rigidbody tank = side < 0.0f ? left[2] : right[2];
                Strut(tank, core, new Vector3(side * strutX, 17.0f, 0.0f));
                tank = side < 0.0f ? left[4] : right[4];
                Strut(tank, core, new Vector3(side * strutX, 33.0f, 0.0f));
            }

            Strut(left[5], right[5], new Vector3(0.0f, 1.5f + TankHeight * 5.5f, 0.0f));

            float peak = 0.0f;
            float top = 0.0f;
            for (int step = 0; step < 300; step++)
            {
                Simulate();
                foreach (ConfigurableJoint joint in bridgeJoints)
                {
                    if (joint != null)
                    {
                        peak = Mathf.Max(peak, joint.currentForce.magnitude);
                    }
                }

                foreach (Rigidbody body in bodies)
                {
                    top = Mathf.Max(top, body.linearVelocity.magnitude);
                }
            }

            // Each bridge holds about half the core's weight, near 490 kN, and its joints break at 1760.
            int broken = bridgeJoints.FindAll(joint => joint == null).Count;
            Assert.That(broken, Is.Zero, $"{broken} bridge joints broke; the peak force on one was {peak:F0} kN");
            Assert.That(top, Is.LessThan(0.5f), $"the craft moved at {top:F2} m/s; the peak bridge force was {peak:F0} kN");
        }
    }
}
