using KSP.Retooled.Physics.Stepping;
using NUnit.Framework;
using Unity.Physics;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Stepping;
using UnityEngine;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    public unsafe class PistonEdgeTests : PhysicsScene
    {
        private const float Stroke = 2.4f;
        private const float Middle = Stroke / 2.0f;
        private const float Stop = 1.01f * Middle;
        private const float MediumExtension = 1.6f;
        private RecordedPhysicsSettings settings;

        [SetUp]
        public void Configure()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Piston edges require the Retooled backend.");
            }

            EnvironmentSwitches.Read();
            settings = RecordedPhysicsSettings.Read();
            VesselSwitches.MinimumParts = 2;
        }

        [TearDown]
        public void RestoreSettings()
        {
            if (BackendInfo.IsRetooled)
            {
                settings.Apply();
            }
        }

        private static StoreStepper* State => StepEntryPoints.Current;

        private Rigidbody Part(Vector3 position, float mass, bool gravity)
            => ServoEdgeTests.Body(Track(new GameObject("piston part")), position, mass, 0.2f * mass, gravity);

        internal static ConfigurableJoint Piston(Rigidbody moving, Rigidbody basis, Vector3 axis, float stroke)
        {
            ConfigurableJoint joint = ServoEdgeTests.Link(moving, basis, moving.position, axis);
            joint.anchor = Vector3.zero;
            joint.connectedAnchor = basis.transform.InverseTransformPoint(basis.position + 0.5f * stroke * axis);
            joint.secondaryAxis = axis == Vector3.up ? Vector3.forward : Vector3.up;
            joint.xMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 1.01f * stroke / 2.0f, bounciness = 0.2f, contactDistance = 1.0f };
            joint.linearLimitSpring = new SoftJointLimitSpring { spring = 100000.0f, damper = 100000.0f };
            Drive(joint, 100.0f);
            return joint;
        }

        internal static void Drive(ConfigurableJoint joint, float force, float damper = 2.0f)
            => joint.xDrive = new JointDrive { positionSpring = force * 100.0f, positionDamper = damper, maximumForce = force };

        private static void Target(ConfigurableJoint joint, float extension)
            => joint.targetPosition = (Middle - extension) * Vector3.right;

        private (Rigidbody[] bodies, ConfigurableJoint joint) Craft(Vector3 axis, float extension = Middle,
            float payload = 1.0f, bool gravity = false, bool reversed = false)
        {
            var bodies = new Rigidbody[5];
            void MakeBase()
            {
                bodies[0] = Part(-0.45f * axis, 1.0f, gravity);
                bodies[1] = Part(Vector3.zero, 0.8f, gravity);
            }
            void MakePayload()
            {
                bodies[2] = Part(extension * axis, 0.5f, gravity);
                bodies[3] = Part((extension + 0.4f) * axis, 0.5f * payload, gravity);
                bodies[4] = Part((extension + 0.8f) * axis, 0.5f * payload, gravity);
            }

            if (reversed)
            {
                MakePayload();
                MakeBase();
            }
            else
            {
                MakeBase();
                MakePayload();
            }

            ServoEdgeTests.Link(bodies[1], bodies[0], bodies[1].position, Vector3.right);
            ServoEdgeTests.Link(bodies[3], bodies[2], bodies[3].position, Vector3.right);
            ServoEdgeTests.Link(bodies[4], bodies[3], bodies[4].position, Vector3.right);
            ConfigurableJoint joint = Piston(bodies[2], bodies[1], axis, Stroke);
            Target(joint, extension);
            return (bodies, joint);
        }

        private static (float extension, float side, float tilt) Measure(ConfigurableJoint joint)
        {
            Vector3 offset = joint.transform.TransformPoint(joint.anchor)
                - joint.connectedBody.transform.TransformPoint(joint.connectedAnchor);
            Vector3 axis = joint.transform.TransformDirection(joint.axis);
            float along = Vector3.Dot(offset, axis);
            Quaternion relative = Quaternion.Inverse(joint.connectedBody.rotation) * joint.transform.rotation;
            float tilt = 2.0f * Mathf.Atan2(new Vector3(relative.x, relative.y, relative.z).magnitude, Mathf.Abs(relative.w)) * Mathf.Rad2Deg;
            return (Middle + along, (offset - along * axis).magnitude, tilt);
        }

        private static VesselEdge PistonEdge(out int count)
        {
            VesselEdge edge = default;
            count = 0;
            VesselSet set = State->VesselSet;
            for (int v = 0; v < set.VesselCount; v++)
            {
                VesselRecord vessel = set.Vessels[v];
                for (int i = vessel.FirstPart; i < vessel.FirstPart + vessel.PartCount; i++)
                {
                    if (set.Edges[i].State == VesselEdgeState.FreeAxis && set.Edges[i].ServoAxis == ServoAxis.LinearX)
                    {
                        edge = set.Edges[i];
                        count++;
                    }
                }
            }

            return edge;
        }

        private static (int drive, int linear, int angular) Rows(ConfigurableJoint piston)
        {
            int drive = 0, linear = 0, angular = 0;
            int index = (int)JointRecord(piston)->index;
            foreach (var joint in State->World.Joints)
            {
                if (joint.Entity.Index != index || joint.Entity.Version != 1)
                {
                    continue;
                }

                foreach (Constraint row in joint.Constraints.GetConstraints())
                {
                    drive += row.Type == ConstraintType.PositionMotor ? row.Dimension : 0;
                    linear += row.Type == ConstraintType.Linear ? row.Dimension : 0;
                    angular += row.Type == ConstraintType.Angular ? row.Dimension : 0;
                }
            }

            return (drive, linear, angular);
        }

        [Test]
        public void DriveAndStopRowsFollowSettingsWithoutRecompiling()
        {
            var (_, joint) = Craft(Vector3.right, MediumExtension);
            Simulate();

            VesselSet set = State->VesselSet;
            Assert.That(set.VesselCount, Is.EqualTo(1));
            VesselEdge edge = PistonEdge(out int count);
            Assert.That(count, Is.EqualTo(1));
            Assert.That(set.Vessels[0].PartCount, Is.EqualTo(5));
            Assert.That(edge.MayLock, Is.False);
            Assert.That(edge.TwistStiffness, Is.EqualTo(double.PositiveInfinity));
            Assert.That(edge.SwingStiffness, Is.EqualTo(double.PositiveInfinity));

            Assert.That(Rows(joint), Is.EqualTo((1, 0, 0)));

            uint version = State->RecorderVesselCompilationVersion;
            Target(joint, MediumExtension + 0.01f);
            Drive(joint, 70.0f);
            SoftJointLimit limit = joint.linearLimit;
            limit.limit = MediumExtension - Middle + 0.03f;
            joint.linearLimit = limit;
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.EqualTo(version));
            Assert.That(Rows(joint), Is.EqualTo((1, 1, 0)));

            joint.xDrive = default;
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.EqualTo(version));
            Assert.That(Rows(joint), Is.EqualTo((0, 1, 0)));

            joint.xMotion = ConfigurableJointMotion.Free;
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.EqualTo(version));
            Assert.That(Rows(joint), Is.EqualTo((0, 0, 0)));

            joint.xMotion = ConfigurableJointMotion.Locked;
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.GreaterThan(version));
            PistonEdge(out count);
            Assert.That(count, Is.Zero);
            version = State->RecorderVesselCompilationVersion;
            joint.xMotion = ConfigurableJointMotion.Free;
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.GreaterThan(version));
            PistonEdge(out count);
            Assert.That(count, Is.EqualTo(1));
        }

        [Test]
        public void AnUnlimitedDriveDoesNotGroupThePistonWithAParallelJoint()
        {
            var (bodies, joint) = Craft(Vector3.right);
            Drive(joint, 1e16f);
            ServoEdgeTests.Link(bodies[2], bodies[1], bodies[2].position, Vector3.right);
            Simulate();

            PistonEdge(out int count);
            Assert.That(count, Is.EqualTo(1));
            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(1));
            Assert.That(State->VesselSet.Vessels[0].ClosureCount, Is.EqualTo(1));
        }

        private (float extension, float side, float tilt, float drift, Vector3 tip) Hold(bool vertical, bool reversed, float step)
        {
            Vector3 axis = vertical ? Vector3.up : Vector3.right;
            float goal = vertical ? 1.75f : 2.1f;
            float start = vertical ? 0.3f : goal;
            var (bodies, joint) = Craft(axis, start, payload: vertical ? 8.0f : 12.0f, gravity: true, reversed: reversed);
            ServoEdgeTests.Link(bodies[0], null, bodies[0].position, Vector3.right);
            Drive(joint, 150.0f, vertical ? 180.0f : 75.0f);
            float command = start;
            int steps = Mathf.RoundToInt((vertical ? 7.0f : 2.0f) / step);
            int holdStart = steps - Mathf.RoundToInt(0.5f / step);
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            for (int i = 0; i < steps; i++)
            {
                command = Mathf.MoveTowards(command, goal, 0.3f * step);
                Target(joint, command);
                UnityEngine.Physics.Simulate(step);
                if (i >= holdStart)
                {
                    float extension = Measure(joint).extension;
                    min = Mathf.Min(min, extension);
                    max = Mathf.Max(max, extension);
                }
            }

            var pose = Measure(joint);
            VesselEdge edge = PistonEdge(out int count);
            Assert.That(count, Is.EqualTo(1));
            Assert.That(edge.FrameInChild, Is.EqualTo(!reversed));

            return (pose.extension, pose.side, pose.tilt, max - min, bodies[4].position);
        }

        [TestCase(true, false)]
        [TestCase(false, false)]
        [TestCase(false, true)]
        public void ALoadedPistonTracksAndHoldsACommand(bool vertical, bool reversed)
        {
            var measured = Hold(vertical, reversed, Step);
            DestroyAll();
            var reference = Hold(vertical, reversed, Step / 10.0f);
            float extensionError = Mathf.Abs(measured.extension - reference.extension);
            float sideError = Mathf.Abs(measured.side - reference.side);
            float tiltError = Mathf.Abs(measured.tilt - reference.tilt);
            float tipError = Vector3.Distance(measured.tip, reference.tip);
            TestContext.WriteLine($"Vertical {vertical}, reversed {reversed}: extension {measured.extension:F6} m, "
                + $"side {measured.side:E4} m, tilt {measured.tilt:E4} deg, hold drift {measured.drift:E4} m. Absolute errors against 2 ms: "
                + $"extension {extensionError:E4} m, side {sideError:E4} m, tilt {tiltError:E4} deg, tip {tipError:E4} m.");
            Assert.That(measured.extension, Is.EqualTo(vertical ? 1.75f : 2.1f).Within(0.03f));
            // The piston's edge has no row stiffness cap. With the cap, the sideways load tilted it about 0.04 degrees.
            Assert.That(measured.side, Is.LessThan(5e-5f));
            Assert.That(measured.tilt, Is.LessThan(0.005f));
            Assert.That(measured.drift, Is.LessThan(0.003f));
            Assert.That(extensionError, Is.LessThan(0.03f));
            Assert.That(tipError, Is.LessThan(0.04f));
        }

        [TestCase(false, false, false)]
        [TestCase(false, false, true)]
        [TestCase(false, true, false)]
        [TestCase(false, true, true)]
        [TestCase(true, false, false)]
        [TestCase(true, false, true)]
        [TestCase(true, true, false)]
        [TestCase(true, true, true)]
        public void ThePistonStopsAtEitherEnd(bool hard, bool atSpeed, bool upper)
        {
            float sign = upper ? 1.0f : -1.0f;
            float start = Middle + sign * (Stop - (atSpeed ? 0.07f : 0.15f));
            var (bodies, joint) = Craft(Vector3.right, start, payload: 3.0f);
            if (hard)
            {
                joint.linearLimitSpring = default;
            }

            Simulate(3);
            if (atSpeed)
            {
                for (int i = 2; i < bodies.Length; i++)
                {
                    bodies[i].linearVelocity = 3.0f * sign * Vector3.right;
                }
            }

            Target(joint, Middle + sign * (Stop + 0.35f));
            float peak = float.NegativeInfinity;
            for (int i = 0; i < 65; i++)
            {
                Simulate();
                var pose = Measure(joint);
                peak = Mathf.Max(peak, sign * (pose.extension - Middle));
                Assert.That(Mathf.Abs(pose.extension - Middle), Is.LessThanOrEqualTo(Stop + (hard ? 0.016f : 0.032f)), $"step {i}");
                Assert.That(pose.side, Is.LessThan(0.003f));
            }

            Assert.That(peak, Is.GreaterThan(Stop - 0.004f));
            Assert.That(Mathf.Abs(Vector3.Dot(joint.currentForce, joint.transform.TransformDirection(joint.axis))), Is.GreaterThan(5.0f));
        }

        [Test]
        public void AHardStopThatStartsSlackCanCatchTheDrive()
        {
            var (_, joint) = Craft(Vector3.right, Middle + Stop - 0.07f);
            joint.linearLimitSpring = default;
            Simulate();
            Drive(joint, 1000.0f);
            Target(joint, Middle + Stop + 0.35f);
            long before = BoundaryCounters.Current.StopRetries;
            Simulate();
            float firstExtension = Measure(joint).extension;
            long retries = BoundaryCounters.Current.StopRetries - before;
            TestContext.WriteLine($"Hard stop: first-step excess {firstExtension - Middle - Stop:E4} m, {retries} retries.");
            Assert.That(retries, Is.GreaterThan(0));
            Assert.That(firstExtension, Is.LessThan(Middle + Stop + 0.002f));

            Simulate(12);
            Assert.That(Measure(joint).extension, Is.EqualTo(Middle + Stop).Within(0.025f));
        }

        [Test]
        public void AForceLimitedDriveCannotLiftThePayload()
        {
            var (bodies, joint) = Craft(Vector3.up, 0.6f, payload: 12.0f, gravity: true);
            ServoEdgeTests.Link(bodies[0], null, bodies[0].position, Vector3.right);
            Drive(joint, 20.0f);
            Target(joint, 1.8f);
            Simulate(40);
            Assert.That(Measure(joint).extension, Is.LessThan(0.1f));
        }

        [Test]
        public void TheForceCapSetsFloatingAccelerationWithoutReportingDriveForce()
        {
            var (bodies, joint) = Craft(Vector3.right, payload: 10.0f);
            const float force = 8.0f;
            Drive(joint, force);
            Target(joint, 1.9f);
            Simulate(8);
            float reducedMass = 1.0f / (1.0f / (bodies[0].mass + bodies[1].mass)
                + 1.0f / (bodies[2].mass + bodies[3].mass + bodies[4].mass));
            float speed = bodies[2].linearVelocity.x - bodies[1].linearVelocity.x;
            Assert.That(speed, Is.EqualTo(force / reducedMass * 8.0f * Step).Within(0.015f));
            Assert.That(joint.currentForce.magnitude, Is.LessThan(0.02f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AnExtendingPistonKeepsAlignmentAndConservesFloatingMomentum(bool spinning)
        {
            var (bodies, joint) = Craft(Vector3.right, 0.4f, payload: 3.0f);
            Vector3 spin = spinning ? Vector3.forward : Vector3.zero;
            foreach (Rigidbody body in bodies)
            {
                body.angularVelocity = spin;
                body.linearVelocity = new Vector3(0.18f, -0.12f, 0.25f) + Vector3.Cross(spin, body.worldCenterOfMass);
            }

            var before = ServoEdgeTests.Momentum(bodies);
            float peakSide = 0.0f, peakTilt = 0.0f;
            for (int i = 1; i <= 70; i++)
            {
                Target(joint, Mathf.Min(1.55f, 0.4f + i * 1.15f * Step));
                Simulate();
                var pose = Measure(joint);
                peakSide = Mathf.Max(peakSide, pose.side);
                peakTilt = Mathf.Max(peakTilt, pose.tilt);
            }

            var after = ServoEdgeTests.Momentum(bodies);
            float linear = (after.linear - before.linear).magnitude;
            float angular = (after.angular - before.angular).magnitude;
            TestContext.WriteLine($"Spinning {spinning}: linear drift {linear:E4}, angular drift {angular:E4}, "
                + $"peak side {peakSide:E4} m, peak tilt {peakTilt:E4} deg.");
            Assert.That(Measure(joint).extension, Is.EqualTo(1.55f).Within(0.06f));
            Assert.That(peakSide, Is.LessThan(0.005f));
            Assert.That(peakTilt, Is.LessThan(0.35f));
            Assert.That(linear, Is.LessThan(0.003f));
            Assert.That(angular, Is.LessThan(0.06f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void ASeriesPistonAndHingeMoveTogether(bool pistonFirst)
        {
            var (bodies, piston) = Craft(Vector3.right, 0.65f, payload: 2.0f);
            Rigidbody hingeChild = Part(pistonFirst ? 2.3f * Vector3.right : -0.9f * Vector3.right, 0.7f, false);
            ConfigurableJoint hinge = ServoEdgeTests.Link(hingeChild, pistonFirst ? bodies[4] : bodies[0],
                hingeChild.position, Vector3.right, true);
            if (!pistonFirst)
            {
                Rigidbody support = Part(-1.35f * Vector3.right, 0.8f, false);
                ServoEdgeTests.Link(support, hingeChild, support.position, Vector3.right);
                ServoEdgeTests.Link(support, null, support.position, Vector3.right);
            }
            else
            {
                Rigidbody tip = Part(2.7f * Vector3.right + 0.15f * Vector3.up, 0.8f, false);
                ServoEdgeTests.Link(tip, hingeChild, tip.position, Vector3.right);
                ServoEdgeTests.Link(bodies[0], null, bodies[0].position, Vector3.right);
            }

            ServoEdgeTests.Drive(hinge, 35.0f, 7.0f, 45.0f);
            for (int i = 1; i <= 100; i++)
            {
                Target(piston, Mathf.Min(1.65f, 0.65f + 0.9f * i * Step));
                hinge.targetRotation = Quaternion.AngleAxis(Mathf.Min(50.0f, i * 1.0f), Vector3.right);
                Simulate();
            }

            PistonEdge(out int count);
            Assert.That(count, Is.EqualTo(1));
            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(1));
            Assert.That(Measure(piston).extension, Is.EqualTo(1.65f).Within(0.04f));
            Assert.That(Measure(piston).side, Is.LessThan(0.003f));
            Assert.That(Measure(piston).tilt, Is.LessThan(0.25f));
            Assert.That(Quaternion.Angle(hingeChild.rotation, hinge.connectedBody.rotation), Is.EqualTo(50.0f).Within(3.0f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AStrutLoopLeavesThePistonOutOfTheClosures(bool strutFirst)
        {
            var (bodies, joint) = Craft(Vector3.right);
            if (strutFirst)
            {
                Object.DestroyImmediate(joint);
            }

            ServoEdgeTests.Link(bodies[4], bodies[0], bodies[4].position, Vector3.right);
            if (strutFirst)
            {
                joint = Piston(bodies[2], bodies[1], Vector3.right, Stroke);
            }

            Target(joint, 1.8f);
            Simulate(60);
            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(1));
            Assert.That(State->VesselSet.Vessels[0].ClosureCount, Is.EqualTo(strutFirst ? 0 : 1));
            PistonEdge(out int count);
            Assert.That(count, Is.EqualTo(strutFirst ? 0 : 1));
            Assert.That(Rows(joint).drive, Is.EqualTo(1));
            Assert.That(Measure(joint).extension, Is.EqualTo(Middle).Within(0.003f));
            Assert.That(Measure(joint).side, Is.LessThan(0.003f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void DeletingThePistonOrItsPayloadDuringMotionLeavesFiniteMotion(bool deletePart)
        {
            var (bodies, joint) = Craft(Vector3.right, 0.5f, payload: 3.0f);
            Target(joint, 1.9f);
            Simulate(6);
            Assert.That(Measure(joint).extension, Is.GreaterThan(0.6f));
            Assert.That(Mathf.Abs(bodies[2].linearVelocity.x - bodies[1].linearVelocity.x), Is.GreaterThan(0.2f));
            Object.DestroyImmediate(deletePart ? (Object)bodies[4].gameObject : joint);
            Simulate(25);
            foreach (Rigidbody body in bodies)
            {
                if (body != null)
                {
                    Assert.That(body.position.magnitude + body.linearVelocity.magnitude + body.angularVelocity.magnitude, Is.LessThan(100.0f));
                }
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AConstraintLoadIsReportedAndCanBreakThePiston(bool torque)
        {
            var (bodies, joint) = Craft(Vector3.right, 2.1f, payload: 10.0f, gravity: !torque);
            ServoEdgeTests.Link(bodies[0], null, bodies[0].position, Vector3.right);
            void Load()
            {
                if (torque)
                {
                    bodies[2].AddTorque(30.0f * Vector3.right);
                }
                Simulate();
            }

            for (int i = 0; i < 10; i++)
            {
                Load();
            }

            Assert.That((torque ? joint.currentTorque : joint.currentForce).magnitude, Is.GreaterThan(10.0f));
            if (torque)
            {
                joint.breakTorque = 5.0f;
            }
            else
            {
                joint.breakForce = 5.0f;
            }

            for (int i = 0; i < 20; i++)
            {
                Load();
            }

            Assert.That(joint == null, Is.True);
        }
    }
}
