using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The forest path with locking on: a tree edge that carries little locks, its two bodies moving as one section, and
    /// its load comes from inverse dynamics into the joint's own rows, so joint forces and breaking read as before. Each
    /// test runs the same scene with locking off for reference where it compares.
    /// </summary>
    public class VesselSectionTests : PhysicsScene
    {
        private bool wasEnabled;
        private bool wasLocking;

        [SetUp]
        public void RememberSwitch()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The forest is part of the Unity Physics backend.");
            }

            // The environment's switches apply when the first world attaches; read them now, so they can't land on top
            // of what this fixture sets.
            VesselForest.Initialize();
            wasEnabled = VesselForest.Enabled;
            wasLocking = VesselForest.Locking;
            VesselForest.Enabled = true;
        }

        [TearDown]
        public void RestoreSwitch()
        {
            VesselForest.Enabled = wasEnabled;
            VesselForest.Locking = wasLocking;
        }

        private void Fresh(bool locking)
        {
            TearDownScene();
            SetUpScene();
            VesselForest.Enabled = true;
            VesselForest.Locking = locking;

            // Islands keep their lock states by their first body, which the last scene's islands had too.
            VesselForest.ForgetIslands();
        }

        /// <summary>A joint with every axis locked, as a rigid attachment.</summary>
        private static ConfigurableJoint RigidJoint(Rigidbody host, Rigidbody target, Vector3 anchor)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(anchor);
            joint.connectedAnchor = target != null ? target.transform.InverseTransformPoint(anchor) : anchor;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            return joint;
        }

        /// <summary>A stock-like joint: anchors and twist locked, swing held by a drive spring.</summary>
        private static ConfigurableJoint SpringJoint(Rigidbody host, Rigidbody target, Vector3 anchor, float spring)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(anchor);
            joint.connectedAnchor = target.transform.InverseTransformPoint(anchor);
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = ConfigurableJointMotion.Locked;
            joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Free;
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularYZDrive = new JointDrive { positionSpring = spring, positionDamper = 0.0f, maximumForce = 1e20f };
            return joint;
        }

        private const int ColumnParts = 6;
        private const float PartMass = 4.0f;

        /// <summary>A column of parts hung from the world, each held to the one above by a rigid joint.</summary>
        private List<ConfigurableJoint> HangingColumn(out List<Rigidbody> bodies)
        {
            bodies = new List<Rigidbody>();
            var joints = new List<ConfigurableJoint>();
            for (int i = 0; i < ColumnParts; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, 10.0f - i, 0.0f), new Vector3(0.6f, 1.0f, 0.6f)).AddComponent<Rigidbody>();
                body.mass = PartMass;
                bodies.Add(body);
                joints.Add(RigidJoint(body, i > 0 ? bodies[i - 1] : null, new Vector3(0.0f, 10.5f - i, 0.0f)));
            }

            return joints;
        }

        private static void Sections(out long lastSections, out long lastLocked)
        {
            VesselForest.GetSectionCounts(out _, out _, out _, out _, out _, out _, out lastSections, out lastLocked);
        }

        [Test]
        public void ALockedColumnCarriesTheLoadsOfTheYieldingOne()
        {
            var loads = new Vector3[2][];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(pass == 1);
                List<ConfigurableJoint> joints = HangingColumn(out _);
                Simulate(60);
                loads[pass] = new Vector3[ColumnParts];
                for (int i = 0; i < ColumnParts; i++)
                {
                    loads[pass][i] = joints[i].currentForce;
                }

                if (pass == 1)
                {
                    Sections(out long sections, out long locked);
                    Debug.Log($"[VesselSectionTests] column: {sections} sections, {locked} locked pairs");
                    Assert.That(locked, Is.EqualTo(ColumnParts - 1), "every joint between two parts locks");
                }
            }

            for (int i = 0; i < ColumnParts; i++)
            {
                float weight = (ColumnParts - i) * PartMass * -Gravity;
                Debug.Log($"[VesselSectionTests] joint {i}: yielding {loads[0][i]:F3}, locked {loads[1][i]:F3}, weight below {weight:F2}");
                Assert.That((loads[1][i] - loads[0][i]).magnitude, Is.LessThan(0.01f * weight + 0.01f),
                    $"joint {i} carries the same load locked");
                Assert.That(loads[1][i].magnitude, Is.EqualTo(weight).Within(0.02f * weight), $"joint {i} carries what hangs below it");
            }
        }

        /// <summary>
        /// Two floating bodies held by a swing spring, twisted apart by a torque that rises and falls again: the pair locks
        /// while the torque is small, yields once it would bend the spring past the yield angle, and locks again as it
        /// falls. The relative turn follows the run without locking to within the angles, with no jump at either change.
        /// </summary>
        [Test]
        public void APairLocksYieldsAndLocksAgainUnderARampedLoad()
        {
            const int steps = 240;
            // The backend turns a drive's spring into a frequency at its own estimate of the joint's mass, so the rows hold
            // at about 17,600 N m/rad here; the forest reads its deflection from the rows.
            const float spring = 2000.0f;
            const float peak = 50.0f;
            var turn = new float[2][];
            var lockedAt = new bool[steps];
            var estimate = new float[steps];

            // The lock test reads each step's deflection alone here: this tests locking, yielding and locking again, not
            // the smoothing, which would hold the yield and the relock back by its time constant.
            int wasSmoothing = VesselForest.DeflectionSmoothingSteps;
            VesselForest.DeflectionSmoothingSteps = 0;
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(pass == 1);
                Rigidbody a = WeightlessBox(new Vector3(-0.5f, 5.0f, 0.0f), 2.0f);
                Rigidbody b = WeightlessBox(new Vector3(0.5f, 5.0f, 0.0f), 2.0f);
                SpringJoint(b, a, new Vector3(0.0f, 5.0f, 0.0f), spring);

                // Locked, the torques cancel inside the section and nothing moves, so the bodies would fall asleep.
                a.sleepThreshold = 0.0f;
                b.sleepThreshold = 0.0f;
                turn[pass] = new float[steps];
                for (int i = 0; i < steps; i++)
                {
                    float torque = peak * Mathf.Sin(Mathf.PI * i / steps);
                    b.AddTorque(new Vector3(0.0f, torque, 0.0f));
                    a.AddTorque(new Vector3(0.0f, -torque, 0.0f));
                    Simulate();
                    Quaternion relative = Quaternion.Inverse(a.rotation) * b.rotation;
                    turn[pass][i] = 2.0f * Mathf.Asin(Mathf.Min(1.0f, new Vector3(relative.x, relative.y, relative.z).magnitude));
                    if (pass == 0)
                    {
                        estimate[i] = (float)VesselForest.LastDeflection;
                    }

                    if (pass == 1)
                    {
                        Sections(out _, out long locked);
                        lockedAt[i] = locked > 0;
                    }
                }
            }

            VesselForest.DeflectionSmoothingSteps = wasSmoothing;
            int changes = 0;
            int firstYield = -1, relock = -1;
            for (int i = 1; i < steps; i++)
            {
                if (lockedAt[i] != lockedAt[i - 1])
                {
                    changes++;
                    if (!lockedAt[i] && firstYield < 0)
                    {
                        firstYield = i;
                    }
                    else if (lockedAt[i] && firstYield >= 0 && relock < 0)
                    {
                        relock = i;
                    }
                }
            }

            float largest = 0.0f, jump = 0.0f, peakTurn = 0.0f;
            for (int i = 0; i < steps; i++)
            {
                largest = Mathf.Max(largest, Mathf.Abs(turn[1][i] - turn[0][i]));
                peakTurn = Mathf.Max(peakTurn, Mathf.Abs(turn[0][i]));
                if (i > 0)
                {
                    jump = Mathf.Max(jump, Mathf.Abs(turn[1][i] - turn[1][i - 1]) - Mathf.Abs(turn[0][i] - turn[0][i - 1]));
                }
            }

            Debug.Log($"[VesselSectionTests] ramp, step 60/120: turn {turn[0][60] * 1e3f:F3}/{turn[0][120] * 1e3f:F3} mrad, " +
                $"forest's deflection estimate {estimate[60] * 1e3f:F3}/{estimate[120] * 1e3f:F3} mrad");
            Debug.Log($"[VesselSectionTests] ramp: locked at 1 {lockedAt[1]}, yields at {firstYield}, locks again at {relock}, " +
                $"{changes} changes; peak turn {peakTurn * 1e3f:F2} mrad, largest difference {largest * 1e3f:F3} mrad, " +
                $"largest extra step {jump * 1e3f:F3} mrad");
            Assert.That(peakTurn, Is.GreaterThan(VesselForest.YieldAngle), "the torque bends the spring past the yield angle");
            Assert.That(lockedAt[2], Is.True, "the pair locks while the torque is small");
            Assert.That(firstYield, Is.GreaterThan(0), "the pair yields");
            Assert.That(relock, Is.GreaterThan(firstYield), "the pair locks again");
            Assert.That(largest, Is.LessThan(VesselForest.YieldAngle + 0.5e-3f), "locked or not, the pair turns alike");
            Assert.That(jump, Is.LessThan(VesselForest.YieldAngle), "nothing jumps when the pair yields or locks: it bends by no more than it held back");
        }

        /// <summary>
        /// A swing spring with a strong damper under a steady torque: the deflection the forest reads for the lock tests is
        /// the spring's, not the spring and damper's together, which would read low by 1 + c / (h k).
        /// </summary>
        [Test]
        public void ADampedSpringReadsItsOwnDeflection()
        {
            Fresh(false);
            Rigidbody a = WeightlessBox(new Vector3(-0.5f, 5.0f, 0.0f), 2.0f);
            Rigidbody b = WeightlessBox(new Vector3(0.5f, 5.0f, 0.0f), 2.0f);
            ConfigurableJoint joint = SpringJoint(b, a, new Vector3(0.0f, 5.0f, 0.0f), 2000.0f);
            JointDrive drive = joint.angularYZDrive;
            drive.positionDamper = 200.0f;
            joint.angularYZDrive = drive;
            float turn = 0.0f;
            for (int i = 0; i < 300; i++)
            {
                b.AddTorque(new Vector3(0.0f, 10.0f, 0.0f));
                a.AddTorque(new Vector3(0.0f, -10.0f, 0.0f));
                Simulate();
                Quaternion relative = Quaternion.Inverse(a.rotation) * b.rotation;
                turn = 2.0f * Mathf.Asin(Mathf.Min(1.0f, new Vector3(relative.x, relative.y, relative.z).magnitude));
            }

            float read = (float)VesselForest.LastDeflection;
            Debug.Log($"[VesselSectionTests] damped: turn {turn * 1e3f:F3} mrad, read {read * 1e3f:F3} mrad");
            Assert.That(read, Is.EqualTo(turn).Within(0.05f * turn));
        }

        /// <summary>
        /// A part hung below another by a rigid joint with a break force, pulled down harder every step: the joint breaks
        /// while locked at the same load as without locking, and the part falls away.
        /// </summary>
        [Test]
        public void ALockedJointBreaksAtTheLoadItWouldBreakAt()
        {
            var brokeAt = new int[2];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(pass == 1);
                Rigidbody top = Box(new Vector3(0.0f, 10.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                top.mass = 20.0f;
                Rigidbody bottom = Box(new Vector3(0.0f, 9.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                bottom.mass = 2.0f;
                RigidJoint(top, null, new Vector3(0.0f, 10.5f, 0.0f));
                ConfigurableJoint joint = RigidJoint(bottom, top, new Vector3(0.0f, 9.5f, 0.0f));
                joint.breakForce = 200.0f;
                Simulate(10);
                if (pass == 1)
                {
                    Sections(out _, out long locked);
                    Assert.That(locked, Is.EqualTo(1), "the joint locks before the pull");
                }

                brokeAt[pass] = -1;
                for (int i = 0; i < 300 && brokeAt[pass] < 0; i++)
                {
                    bottom.AddForce(new Vector3(0.0f, -2.0f * i, 0.0f));
                    Simulate();
                    if (joint == null)
                    {
                        brokeAt[pass] = i;
                    }
                }

                Simulate(25);
                Assert.That(bottom.position.y, Is.LessThan(8.5f), "the part falls away once the joint breaks");
            }

            Debug.Log($"[VesselSectionTests] break: yielding at step {brokeAt[0]}, locked at step {brokeAt[1]}");
            Assert.That(brokeAt[0], Is.GreaterThan(0));
            Assert.That(brokeAt[1], Is.GreaterThan(0), "the locked joint breaks");
            Assert.That(Mathf.Abs(brokeAt[1] - brokeAt[0]), Is.LessThanOrEqualTo(2), "at the load it breaks at yielding");
        }

        /// <summary>A locked column on the ground falls asleep as the yielding one does, and wakes when pushed.</summary>
        [Test]
        public void ALockedColumnSleepsAndWakesAsTheYieldingOne()
        {
            var asleepAt = new int[2];
            var moved = new float[2];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(pass == 1);
                Ground();
                var bodies = new List<Rigidbody>();
                for (int i = 0; i < 4; i++)
                {
                    Rigidbody body = Box(new Vector3(0.0f, 0.5f + i, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                    body.mass = PartMass;
                    if (i > 0)
                    {
                        RigidJoint(body, bodies[i - 1], new Vector3(0.0f, i, 0.0f));
                    }

                    bodies.Add(body);
                }

                asleepAt[pass] = -1;
                for (int i = 0; i < 600 && asleepAt[pass] < 0; i++)
                {
                    Simulate();
                    if (bodies.TrueForAll(body => body.IsSleeping()))
                    {
                        asleepAt[pass] = i;
                    }
                }

                Vector3 before = bodies[3].position;
                bodies[3].AddForce(new Vector3(40.0f, 0.0f, 0.0f), ForceMode.Impulse);
                Simulate(5);
                moved[pass] = (bodies[3].position - before).magnitude;
                Assert.That(bodies[0].IsSleeping(), Is.False, "the push wakes the whole column");
            }

            Debug.Log($"[VesselSectionTests] sleep: yielding at step {asleepAt[0]}, locked at step {asleepAt[1]}; pushed top moved {moved[0]:F3} / {moved[1]:F3} m");
            Assert.That(asleepAt[0], Is.GreaterThan(0), "the yielding column falls asleep");
            Assert.That(asleepAt[1], Is.GreaterThan(0), "the locked column falls asleep");
            Assert.That(Mathf.Abs(asleepAt[1] - asleepAt[0]), Is.LessThan(30), "about as soon");
            Assert.That(moved[1], Is.EqualTo(moved[0]).Within(0.1f * moved[0] + 0.01f), "and moves alike when woken");
        }
    }
}
