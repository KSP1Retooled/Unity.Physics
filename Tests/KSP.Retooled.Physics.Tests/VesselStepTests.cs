using System.Collections.Generic;
using NUnit.Framework;
using Unity.Physics;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Whole steps with vessels on, from ordinary Rigidbodies and ConfigurableJoints: the backend compiles the joints into
    /// vessels, and the direct solver solves them with the vessel kernel's own rows. Where a test compares, it runs the same
    /// scene with vessels off for reference.
    /// </summary>
    public unsafe class VesselStepTests : PhysicsScene
    {
        private const float PartMass = 4.0f;

        /// <summary>How many vessels the backend handed to the last step.</summary>
        private static int VesselCount() => StepKernel.Current->VesselSet.VesselCount;

        private bool wasVessels;
        private bool wasForest;

        [SetUp]
        public void TurnVesselsOn()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            VesselPipeline.Initialize();
            VesselForest.Initialize();
            wasVessels = VesselPipeline.Enabled;
            wasForest = VesselForest.Enabled;

            // These crafts are small, so let them compile into vessels.
            VesselPipeline.MinimumParts = 2;
            Fresh(true);
        }

        [TearDown]
        public void RestoreSwitch()
        {
            VesselPipeline.Enabled = wasVessels;
            VesselForest.Enabled = wasForest;
            VesselPipeline.MinimumParts = 0;
        }

        private void Fresh(bool vessels)
        {
            TearDownScene();
            SetUpScene();
            VesselPipeline.Enabled = vessels;
            VesselForest.Enabled = vessels;
            VesselForest.ForgetIslands();
        }

        /// <summary>
        /// A stock part joint: three joints over a stack node, anchors and twist locked, swing held by drives. The backend
        /// makes one hard edge of them.
        /// </summary>
        private static ConfigurableJoint[] StackJoint(Rigidbody host, Rigidbody target, Vector3 node, float breakForce = float.PositiveInfinity)
        {
            var joints = new ConfigurableJoint[3];
            for (int i = 0; i < 3; i++)
            {
                Vector3 offset = Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(0.3f, 0.0f, 0.0f);
                ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = target;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = host.transform.InverseTransformPoint(node + offset);
                joint.connectedAnchor = target != null ? target.transform.InverseTransformPoint(node + offset) : node + offset;
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = ConfigurableJointMotion.Locked;
                joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
                joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
                joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularYZDrive = new JointDrive { positionSpring = 20000.0f, positionDamper = 200.0f, maximumForce = 1e20f };
                joint.breakForce = breakForce;
                joints[i] = joint;
            }

            return joints;
        }

        /// <summary>A column of parts that hangs from the world by stack joints, top part first.</summary>
        private List<Rigidbody> HangingColumn(int count, float breakForce = float.PositiveInfinity)
        {
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < count; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, 10.0f - i, 0.0f), new Vector3(0.8f, 1.0f, 0.8f)).AddComponent<Rigidbody>();
                body.mass = PartMass;
                bodies.Add(body);
                StackJoint(body, i > 0 ? bodies[i - 1] : null, new Vector3(0.0f, 10.5f - i, 0.0f), i > 0 ? breakForce : float.PositiveInfinity);
            }

            return bodies;
        }

        [Test]
        public void AHangingStackCarriesItsWeightThroughItsJoints()
        {
            const int count = 5;
            List<Rigidbody> bodies = HangingColumn(count);
            Simulate();
            Assert.That(VesselCount(), Is.GreaterThan(0), "the column is a vessel");

            // The column falls asleep before long, and a sleeping vessel's joints keep the forces they last had.
            Simulate(49);
            for (int i = 1; i < count; i++)
            {
                Vector3 sum = Vector3.zero;
                foreach (ConfigurableJoint joint in bodies[i].GetComponents<ConfigurableJoint>())
                {
                    sum += joint.currentForce;
                }

                float weight = (count - i) * PartMass * -Gravity;
                Debug.Log($"[VesselStepTests] hanging: node {i} carries {sum:F3}, weight below {weight:F2}");
                Assert.That(sum.magnitude, Is.EqualTo(weight).Within(0.02f * weight), $"node {i} carries what hangs below it");
            }
        }

        /// <summary>
        /// A horizontal arm of stack joints held out from a part hooked to the world: its tip droops far less with vessels
        /// on, where a stack joint is hard, than with today's joints.
        /// </summary>
        [Test]
        public void AStackArmDroopsFarLessThanToday()
        {
            var droop = new float[2];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(pass == 1);
                var bodies = new List<Rigidbody>();
                for (int i = 0; i < 5; i++)
                {
                    Rigidbody body = Box(new Vector3(i, 5.0f, 0.0f), new Vector3(1.0f, 0.8f, 0.8f)).AddComponent<Rigidbody>();
                    body.mass = PartMass;
                    body.sleepThreshold = 0.0f;
                    bodies.Add(body);
                    if (i == 0)
                    {
                        ConfigurableJoint hook = body.gameObject.AddComponent<ConfigurableJoint>();
                        hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
                        hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;
                    }
                    else
                    {
                        // The node's three anchors lie in the plane of the joint, across the arm.
                        ConfigurableJoint[] joints = StackJoint(body, bodies[i - 1], new Vector3(i - 0.5f, 5.0f, 0.0f));
                        foreach (ConfigurableJoint joint in joints)
                        {
                            joint.axis = Vector3.up;
                        }
                    }
                }

                Simulate(100);
                droop[pass] = -bodies[0].transform.InverseTransformPoint(bodies[4].position).y;
            }

            Debug.Log($"[VesselStepTests] arm: tip droop today {droop[0] * 1e3f:F3} mm, with vessels {droop[1] * 1e3f:F4} mm");
            Assert.That(droop[1], Is.LessThan(0.1f * droop[0]), "the vessel's arm hardly droops");
        }

        [Test]
        public void AStackJointBreaksAtItsBreakForce()
        {
            var brokeAt = new int[2];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(pass == 1);
                List<Rigidbody> bodies = HangingColumn(2, 100.0f);
                Simulate(10);
                brokeAt[pass] = -1;
                for (int i = 0; i < 200 && brokeAt[pass] < 0; i++)
                {
                    bodies[1].AddForce(new Vector3(0.0f, -2.0f * i, 0.0f));
                    Simulate();
                    if (bodies[1].GetComponents<ConfigurableJoint>().Length == 0)
                    {
                        brokeAt[pass] = i;
                    }
                }
            }

            Debug.Log($"[VesselStepTests] break: today at step {brokeAt[0]}, with vessels at step {brokeAt[1]}");
            Assert.That(brokeAt[1], Is.GreaterThan(0), "the joint breaks");
            Assert.That(Mathf.Abs(brokeAt[1] - brokeAt[0]), Is.LessThanOrEqualTo(3), "at the load it breaks at today");
        }

        [Test]
        public void AStackLandsSettlesAndSleeps()
        {
            Ground();
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < 4; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, 1.0f + i, 0.0f), new Vector3(0.8f, 1.0f, 0.8f)).AddComponent<Rigidbody>();
                body.mass = PartMass;
                bodies.Add(body);
                if (i > 0)
                {
                    StackJoint(body, bodies[i - 1], new Vector3(0.0f, 0.5f + i, 0.0f));
                }
            }

            int asleepAt = -1;
            for (int i = 0; i < 400 && asleepAt < 0; i++)
            {
                Simulate();
                if (bodies.TrueForAll(body => body.IsSleeping()))
                {
                    asleepAt = i;
                }
            }

            Debug.Log($"[VesselStepTests] landing: asleep at step {asleepAt}, bottom at {bodies[0].position:F3}, top at {bodies[3].position:F3}");
            Assert.That(asleepAt, Is.GreaterThan(0), "the stack falls asleep");
            Assert.That(bodies[0].position.y, Is.EqualTo(0.5f).Within(0.02f), "standing on the ground");
            Assert.That(bodies[3].position.y - bodies[0].position.y, Is.EqualTo(3.0f).Within(0.01f), "in one piece");
        }

        /// <summary>A hinge between two stacks: each stack is a vessel, both are one island, and the hinge turns.</summary>
        [Test]
        public void AHingeInTheMiddleOfACraftTurns()
        {
            List<Rigidbody> upper = HangingColumn(2);
            var lower = new List<Rigidbody>();
            for (int i = 0; i < 2; i++)
            {
                Rigidbody body = Box(new Vector3(1.0f + i, 9.0f, 0.0f), new Vector3(1.0f, 0.6f, 0.6f)).AddComponent<Rigidbody>();
                body.mass = PartMass;
                body.sleepThreshold = 0.0f;
                lower.Add(body);
            }

            ConfigurableJoint[] stack = StackJoint(lower[1], lower[0], new Vector3(1.5f, 9.0f, 0.0f));
            foreach (ConfigurableJoint joint in stack)
            {
                joint.axis = Vector3.right;
            }

            HingeJoint hinge = lower[0].gameObject.AddComponent<HingeJoint>();
            hinge.connectedBody = upper[1];
            hinge.anchor = new Vector3(-0.5f, 0.0f, 0.0f);
            hinge.axis = new Vector3(0.0f, 0.0f, 1.0f);
            Simulate(60);
            int vessels = VesselCount();
            float gap = (lower[0].transform.TransformPoint(hinge.anchor) - upper[1].transform.TransformPoint(hinge.connectedAnchor)).magnitude;
            Debug.Log($"[VesselStepTests] hinge: {vessels} vessels, lower arm end at {lower[1].position:F3}, hinge gap {gap:E2} m");
            Assert.That(vessels, Is.EqualTo(2), "a stack on each side of the hinge");
            Assert.That(lower[1].position.y, Is.LessThan(8.5f), "the lower arm swings down");
            Assert.That(gap, Is.LessThan(0.01f), "the hinge holds");
            Assert.That((lower[1].position - lower[0].position).magnitude, Is.EqualTo(1.0f).Within(1e-3f), "the lower arm stays in one piece");
        }

        /// <summary>Destroying the joints between the halves of a hanging column lets the lower half fall away.</summary>
        [Test]
        public void ADecoupledHalfFallsAway()
        {
            List<Rigidbody> bodies = HangingColumn(4);
            Simulate(10);
            foreach (ConfigurableJoint joint in bodies[2].GetComponents<ConfigurableJoint>())
            {
                Object.DestroyImmediate(joint);
            }

            float before = bodies[2].position.y;
            Simulate(25);
            Debug.Log($"[VesselStepTests] decouple: {VesselCount()} vessels, lower half fell {before - bodies[2].position.y:F3} m");
            Assert.That(before - bodies[2].position.y, Is.GreaterThan(0.5f), "the lower half falls");
            Assert.That((bodies[3].position - bodies[2].position).magnitude, Is.EqualTo(1.0f).Within(1e-3f), "in one piece");
            Assert.That((bodies[1].position - new Vector3(0.0f, 9.0f, 0.0f)).magnitude, Is.LessThan(0.01f), "the upper half still hangs");
        }

        /// <summary>The top part of a hanging column turns kinematic: the rest hangs from it as from the world.</summary>
        [Test]
        public void AColumnHangsFromAPartMadeKinematic()
        {
            List<Rigidbody> bodies = HangingColumn(3);
            Simulate(10);
            bodies[1].isKinematic = true;
            Simulate(30);
            Debug.Log($"[VesselStepTests] kinematic: {VesselCount()} vessels, bottom at {bodies[2].position:F3}");
            Assert.That((bodies[2].position - new Vector3(0.0f, 8.0f, 0.0f)).magnitude, Is.LessThan(0.01f), "the bottom part still hangs");
        }

        [Test]
        public void AFloatingStackKeepsItsMomentum()
        {
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < 4; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, 5.0f + i, 0.0f), new Vector3(0.8f, 1.0f, 0.8f)).AddComponent<Rigidbody>();
                body.mass = PartMass * (1 + i);
                body.useGravity = false;
                body.sleepThreshold = 0.0f;
                bodies.Add(body);
                if (i > 0)
                {
                    StackJoint(body, bodies[i - 1], new Vector3(0.0f, 4.5f + i, 0.0f));
                }
            }

            bodies[0].linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            bodies[3].angularVelocity = new Vector3(0.0f, 2.0f, 0.0f);
            Simulate();

            Vector3 Momentum()
            {
                Vector3 sum = Vector3.zero;
                foreach (Rigidbody body in bodies)
                {
                    sum += body.mass * body.linearVelocity;
                }

                return sum;
            }

            Vector3 before = Momentum();
            Simulate(100);
            Vector3 after = Momentum();
            Debug.Log($"[VesselStepTests] floating: momentum {before:F4} -> {after:F4}");
            Assert.That((after - before).magnitude, Is.LessThan(1e-3f * before.magnitude), "momentum is kept");
            Assert.That((bodies[3].position - bodies[0].position).magnitude, Is.EqualTo(3.0f).Within(1e-3f), "the stack stays rigid");
        }
    }
}
