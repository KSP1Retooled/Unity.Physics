using System;
using System.Linq;
using KSP.Retooled.Physics.Stepping;
using NUnit.Framework;
using Unity.Physics;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Stepping;
using UnityEngine;
using Object = UnityEngine.Object;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    public unsafe class ServoEdgeTests : PhysicsScene
    {
        private RecordedPhysicsSettings settings;

        [SetUp]
        public void Configure()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Servo edges require the Retooled backend.");
            }

            EnvironmentSwitches.Read();
            settings = RecordedPhysicsSettings.Read();
            VesselSwitches.Enabled = true;
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

        internal static Rigidbody Body(GameObject part, Vector3 position, float mass, float inertia, bool gravity = false)
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
            => Body(Track(new GameObject("servo part")), position, mass, inertia, gravity);

        internal static ConfigurableJoint Link(Rigidbody owner, Rigidbody connected, Vector3 point, Vector3 axis, bool servo = false)
        {
            ConfigurableJoint joint = owner.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = connected;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = owner.transform.InverseTransformPoint(point);
            joint.connectedAnchor = connected == null ? point : connected.transform.InverseTransformPoint(point);
            joint.axis = axis;
            joint.secondaryAxis = Vector3.up;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = servo ? ConfigurableJointMotion.Free : ConfigurableJointMotion.Locked;
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            return joint;
        }

        internal static void Drive(ConfigurableJoint joint, float spring, float damper, float maxForce)
        {
            joint.angularXDrive = new JointDrive { positionSpring = spring, positionDamper = damper, maximumForce = maxForce };
        }

        private static void Limits(ConfigurableJoint joint, float low, float high)
        {
            joint.lowAngularXLimit = new SoftJointLimit { limit = low };
            joint.highAngularXLimit = new SoftJointLimit { limit = high };
            joint.angularXMotion = ConfigurableJointMotion.Limited;
        }

        private static void Target(ConfigurableJoint joint, float degrees)
            => joint.targetRotation = Quaternion.AngleAxis(degrees, Vector3.right);

        private static float Twist(ConfigurableJoint joint)
        {
            Quaternion relative = Quaternion.Inverse(joint.transform.rotation) * joint.connectedBody.rotation;
            float twist = 2.0f * Mathf.Atan2(Vector3.Dot(new Vector3(relative.x, relative.y, relative.z), joint.axis), relative.w);
            return Mathf.DeltaAngle(0.0f, twist * Mathf.Rad2Deg);
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

        private static int FreeAxes()
        {
            VesselSet set = State->VesselSet;
            int count = 0;
            for (int v = 0; v < set.VesselCount; v++)
            {
                VesselRecord vessel = set.Vessels[v];
                for (int i = vessel.FirstPart; i < vessel.FirstPart + vessel.PartCount; i++)
                {
                    if (set.Edges[i].State == VesselEdgeState.FreeAxis)
                    {
                        count++;
                        Assert.That(set.Edges[i].MayLock, Is.False);
                        Assert.That(set.Edges[i].TwistStiffness, Is.Zero);
                        Assert.That(set.Edges[i].TwistDamping, Is.Zero);
                        Assert.That(set.Edges[i].SwingStiffness, Is.EqualTo(double.PositiveInfinity));
                    }
                }
            }
            return count;
        }

        private static int Rows(ConstraintType type)
        {
            int rows = 0;
            foreach (var joint in State->World.Joints)
            {
                foreach (Constraint constraint in joint.Constraints.GetConstraints())
                {
                    if (constraint.Type == type)
                    {
                        rows += constraint.Dimension;
                    }
                }
            }
            return rows;
        }

        private Rigidbody[] Coaxial()
        {
            var bodies = new[] { Part(-Vector3.right), Part(Vector3.zero), Part(Vector3.right), Part(2.0f * Vector3.right) };
            Link(bodies[1], bodies[0], bodies[1].position, Vector3.right);
            Link(bodies[3], bodies[2], bodies[2].position, Vector3.right);
            return bodies;
        }

        [Test]
        public void DriveAndLimitRowsFollowSettingsWithoutRecompiling()
        {
            Rigidbody[] bodies = Coaxial();
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Drive(joint, 20.0f, 8.0f, 50.0f);
            Target(joint, 60.0f);
            Simulate(80);

            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(1));
            Assert.That(State->VesselSet.Vessels[0].PartCount, Is.EqualTo(4));
            Assert.That(FreeAxes(), Is.EqualTo(1));
            Assert.That(Rows(ConstraintType.RotationMotor), Is.EqualTo(1));
            Assert.That(Rows(ConstraintType.Linear), Is.Zero);
            Assert.That(Rows(ConstraintType.Angular), Is.Zero);

            uint version = State->RecorderVesselCompilationVersion;
            Limits(joint, 0.0f, 120.0f);
            Target(joint, 61.0f);
            Drive(joint, 15.0f, 6.0f, 45.0f);
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.EqualTo(version));
            Assert.That(Rows(ConstraintType.Angular), Is.Zero);

            Limits(joint, 0.0f, 62.0f);
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.EqualTo(version));
            Assert.That(Rows(ConstraintType.Angular), Is.EqualTo(1));
            joint.angularXMotion = ConfigurableJointMotion.Locked;
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.GreaterThan(version));
            Assert.That(FreeAxes(), Is.Zero);
            version = State->RecorderVesselCompilationVersion;
            joint.angularXMotion = ConfigurableJointMotion.Free;
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.GreaterThan(version));
            Assert.That(FreeAxes(), Is.EqualTo(1));

            version = State->RecorderVesselCompilationVersion;
            joint.anchor += 0.01f * Vector3.right;
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.GreaterThan(version));
            version = State->RecorderVesselCompilationVersion;
            joint.connectedBody = bodies[0];
            Simulate();
            Assert.That(State->RecorderVesselCompilationVersion, Is.GreaterThan(version));
        }

        [Test]
        public void AnUnpoweredHingeTurnsWithoutRecompilingTheVessel()
        {
            Rigidbody[] bodies = Coaxial();
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Simulate();
            Assert.That(FreeAxes(), Is.EqualTo(1));
            uint version = State->RecorderVesselCompilationVersion;

            bodies[2].AddTorque(0.4f * Vector3.right, ForceMode.Impulse);
            Simulate(10);
            Assert.That(Mathf.Abs(Twist(joint)), Is.GreaterThan(5.0f));
            Assert.That(State->RecorderVesselCompilationVersion, Is.EqualTo(version));
            Assert.That(FreeAxes(), Is.EqualTo(1));
            Assert.That(joint.currentForce.magnitude, Is.LessThan(0.01f));
        }

        [Test]
        public void WithoutVesselsTheServoBuildsOrdinaryRows()
        {
            VesselSwitches.Enabled = false;
            ForestSwitches.Enabled = true;
            Rigidbody[] bodies = Coaxial();
            var servo = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Drive(servo, 20.0f, 8.0f, 50.0f);
            Target(servo, 60.0f);
            Simulate();
            Assert.That(State->VesselSet.VesselCount, Is.Zero);
            int index = (int)JointRecord(servo)->index;
            int rows = 0;
            foreach (var joint in State->World.Joints)
            {
                if (joint.Entity.Index == index)
                {
                    Assert.That(joint.ForestRole, Is.EqualTo(JointForestRole.None));
                    foreach (Constraint constraint in joint.Constraints.GetConstraints())
                    {
                        rows += constraint.Dimension;
                    }
                }
            }
            Assert.That(rows, Is.EqualTo(6));
        }

        [Test]
        public void ADriveWithoutContactsReachesItsTargetInTheFusedSolve()
        {
            SolverSwitches.FusedVesselParts = 1;
            Rigidbody[] bodies = Coaxial();
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Drive(joint, 20.0f, 8.0f, 50.0f);
            Target(joint, 60.0f);
            long before = IslandWorkCounters.FusedIslandSolves;
            Simulate(180);

            Assert.That(IslandWorkCounters.FusedIslandSolves - before, Is.EqualTo(180));
            Assert.That(Rows(ConstraintType.RotationMotor), Is.EqualTo(1));
            Assert.That(Twist(joint), Is.EqualTo(60.0f).Within(0.1f));
        }

        private (Rigidbody[] bodies, ConfigurableJoint joint) Arm(bool reversed, bool gravity)
        {
            Vector3 pivot = new Vector3(0.5f, 0.0f, 0.0f);
            Rigidbody root = null, support = null, arm = null, tip = null;
            void MakeBase()
            {
                root = Part(Vector3.zero, 500.0f, 2000.0f);
                support = Part(-Vector3.right, 20.0f, 50.0f);
            }
            void MakeArm()
            {
                arm = Part(Vector3.right, 2.0f, 0.4f, gravity);
                tip = Part(2.0f * Vector3.right, 2.0f, 0.4f, gravity);
            }
            if (reversed)
            {
                MakeArm();
                MakeBase();
            }
            else
            {
                MakeBase();
                MakeArm();
            }

            Link(support, root, -pivot, Vector3.forward);
            Link(tip, arm, 3.0f * pivot, Vector3.forward);
            var joint = reversed ? Link(root, arm, pivot, Vector3.forward, true) : Link(arm, root, pivot, Vector3.forward, true);
            return (new[] { root, support, arm, tip }, joint);
        }

        private (float angle, float anchor, float axis, Vector3 tip) Hold(bool reversed, float step)
        {
            var (bodies, joint) = Arm(reversed, true);
            Link(bodies[0], null, Vector3.zero, Vector3.forward);
            Drive(joint, 800.0f, 135.0f, 500.0f);
            Target(joint, reversed ? 55.0f : -55.0f);
            for (int i = 0; i < Mathf.RoundToInt(2.0f / step); i++)
            {
                UnityEngine.Physics.Simulate(step);
            }

            Assert.That(FreeAxes(), Is.EqualTo(1));
            Assert.That(joint.currentForce.magnitude, Is.GreaterThan(20.0f));
            return ((reversed ? 1.0f : -1.0f) * Twist(joint), AnchorGap(joint), AxisGap(joint), bodies[3].position);
        }

        [TestCase(false)]
        [TestCase(true)]
        public void SpringDriveHoldsAnArmUnderGravity(bool reversed)
        {
            var measured = Hold(reversed, Step);
            VesselSet set = State->VesselSet;
            State->FinishPendingRefit();
            Vector3 root = State->World.Bodies[set.PartBodies[set.Vessels[0].FirstPart]].WorldFromBody.pos;
            Assert.That(root.x, reversed ? Is.GreaterThan(0.7f) : Is.LessThan(0.3f));
            DestroyAll();
            var reference = Hold(reversed, Step / 10.0f);

            TestContext.WriteLine($"Gravity hold, reversed {reversed}: angle {measured.angle:F6} deg, "
                + $"anchor {measured.anchor:E4} m, axis {measured.axis:E4} deg. Absolute errors against 2 ms: "
                + $"angle {Mathf.Abs(measured.angle - reference.angle):E4} deg, anchor {Mathf.Abs(measured.anchor - reference.anchor):E4} m, "
                + $"axis {Mathf.Abs(measured.axis - reference.axis):E4} deg, tip {Vector3.Distance(measured.tip, reference.tip):E4} m.");
            Assert.That(measured.angle, Is.EqualTo(55.0f).Within(4.0f));
            Assert.That(measured.anchor, Is.LessThan(0.003f));
            Assert.That(measured.axis, Is.LessThan(0.2f));
            Assert.That(Vector3.Distance(measured.tip, reference.tip), Is.LessThan(0.03f));
        }

        [Test]
        public void ATargetSweepsBeyondAFullTurn()
        {
            Rigidbody[] bodies = Coaxial();
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Drive(joint, 80.0f, 6.0f, 50.0f);
            float accumulated = 0.0f, previous = Twist(joint), maxAxis = 0.0f;
            for (int i = 1; i <= 240; i++)
            {
                Target(joint, 2.0f * i);
                Simulate();
                float angle = Twist(joint);
                float increment = Mathf.DeltaAngle(previous, angle);
                Assert.That(Mathf.Abs(increment), Is.LessThan(6.0f), $"step {i}");
                accumulated += increment;
                previous = angle;
                maxAxis = Mathf.Max(maxAxis, AxisGap(joint));
            }

            Assert.That(accumulated, Is.EqualTo(480.0f).Within(15.0f));
            Assert.That(maxAxis, Is.LessThan(0.3f));
            Assert.That(AnchorGap(joint), Is.LessThan(0.003f));
        }

        [TestCase(120.0f, false)]
        [TestCase(120.0f, true)]
        [TestCase(177.0f, false)]
        public void WideLimitStopsTheHinge(float high, bool atSpeed)
        {
            Rigidbody[] bodies = Coaxial();
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Drive(joint, 30.0f, 8.0f, 50.0f);
            Limits(joint, 0.0f, high);
            Target(joint, high - 55.0f);
            Simulate(75);
            if (atSpeed)
            {
                bodies[0].angularVelocity = bodies[1].angularVelocity = 9.0f * Vector3.right;
                bodies[2].angularVelocity = bodies[3].angularVelocity = -9.0f * Vector3.right;
            }
            Target(joint, high + 20.0f);
            float max = 0.0f;
            for (int i = 0; i < 90; i++)
            {
                Simulate();
                float angle = Twist(joint);
                if (angle < -90.0f)
                {
                    angle += 360.0f;
                }
                Assert.That(angle, Is.InRange(-1.0f, high + 2.0f), $"step {i}");
                max = Mathf.Max(max, angle);
            }
            Assert.That(max, Is.GreaterThan(high - 2.0f));
            Assert.That(AnchorGap(joint), Is.LessThan(0.002f));
        }

        [Test]
        public void AWeakDriveCannotLiftTheArm()
        {
            var (bodies, joint) = Arm(false, true);
            Link(bodies[0], null, Vector3.zero, Vector3.forward);
            Drive(joint, 800.0f, 100.0f, 2.0f);
            Target(joint, -55.0f);
            Simulate(40);
            Assert.That(bodies[3].position.y, Is.LessThan(-0.4f));
        }

        [Test]
        public void TheTorqueCapSetsTheAccelerationOfAFloatingPayload()
        {
            Rigidbody[] bodies = Coaxial();
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Drive(joint, 0.0f, 50.0f, 1.5f);
            joint.targetAngularVelocity = 20.0f * Vector3.right;
            Simulate(10);

            float reducedInertia = 0.5f * (bodies[0].inertiaTensor.x + bodies[1].inertiaTensor.x);
            float expected = 1.5f / reducedInertia * 10 * Step;
            float speed = bodies[2].angularVelocity.x - bodies[1].angularVelocity.x;
            Assert.That(speed, Is.EqualTo(expected).Within(0.06f));
        }

        internal static (Vector3 linear, Vector3 angular) Momentum(Rigidbody[] bodies)
        {
            Vector3 linear = Vector3.zero, angular = Vector3.zero;
            foreach (Rigidbody body in bodies)
            {
                Vector3 momentum = body.mass * body.linearVelocity;
                Quaternion inertiaFrame = body.rotation * body.inertiaTensorRotation;
                linear += momentum;
                angular += Vector3.Cross(body.worldCenterOfMass, momentum)
                    + inertiaFrame * Vector3.Scale(body.inertiaTensor, Quaternion.Inverse(inertiaFrame) * body.angularVelocity);
            }
            return (linear, angular);
        }

        private (Vector3 linear, Vector3 angular, Vector3 tip, float angle) TurnInSpace(float step)
        {
            var (bodies, joint) = Arm(false, false);
            foreach (Rigidbody body in bodies)
            {
                body.linearVelocity = new Vector3(0.2f, -0.1f, 0.3f);
            }
            var before = Momentum(bodies);
            Drive(joint, 80.0f, 20.0f, 150.0f);
            Target(joint, 100.0f);
            for (int i = 0; i < Mathf.RoundToInt(1.2f / step); i++)
            {
                UnityEngine.Physics.Simulate(step);
            }
            Assert.That(FreeAxes(), Is.EqualTo(1));
            var after = Momentum(bodies);
            return (after.linear - before.linear, after.angular - before.angular, bodies[3].position, Twist(joint));
        }

        [Test]
        public void AServoConservesFloatingMomentum()
        {
            var measured = TurnInSpace(Step);
            DestroyAll();
            var reference = TurnInSpace(Step / 10.0f);
            TestContext.WriteLine($"Floating turn: linear drift {measured.linear.magnitude:E4} kg m/s, "
                + $"angular drift {measured.angular.magnitude:E4} kg m²/s. Errors against 2 ms: "
                + $"linear {(measured.linear - reference.linear).magnitude:E4} kg m/s, "
                + $"angular {(measured.angular - reference.angular).magnitude:E4} kg m²/s, "
                + $"angle {Mathf.Abs(Mathf.DeltaAngle(reference.angle, measured.angle)):E4} deg, tip {Vector3.Distance(measured.tip, reference.tip):E4} m.");
            Assert.That(measured.angle, Is.GreaterThan(60.0f));
            Assert.That(Mathf.Abs(Mathf.DeltaAngle(reference.angle, measured.angle)), Is.LessThan(2.0f));
            Assert.That(measured.linear.magnitude, Is.LessThan(0.003f));
            Assert.That(measured.angular.magnitude, Is.LessThan(0.15f));
            Assert.That(Vector3.Distance(measured.tip, reference.tip), Is.LessThan(0.1f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void ABladeRotorSpinsAtFortyRadiansPerSecond(bool limited)
        {
            Rigidbody[] bodies = Coaxial();
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
            // KSP's rotors give their velocity drive a spring of 1e-10 rather than zero.
            Drive(joint, 1e-10f, 80.0f, 1000.0f);
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
            TestContext.WriteLine($"Rotor limited {limited}, speed {speed:F4} rad/s, "
                + $"peak speed {peakSpeed:F4} rad/s, peak anchor gap {anchor:E4} m, peak axis deflection {axis:E4} deg, "
                + $"peak blade anchor gap {bladeGap:E4} m.");
            Assert.That(float.IsFinite(speed + anchor + axis + bladeGap), Is.True);
            Assert.That(peakSpeed, Is.GreaterThan(39.0f).And.LessThanOrEqualTo(40.5f));
            Assert.That(speed, Is.EqualTo(40.0f).Within(0.5f));
            Assert.That(anchor, Is.LessThan(0.002f));
            Assert.That(axis, Is.LessThan(0.1f));
            Assert.That(bladeGap, Is.LessThan(0.002f));
        }

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
            Drive(joint, 1e-10f, 1000.0f, 70.0f);

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

            TestContext.WriteLine($"Speed {speed:F4} rad/s, peak speed {peakSpeed:F4} rad/s, "
                + $"peak anchor gap {anchor:E4} m, peak axis deflection {axis:E4} deg, peak blade anchor gap {bladeGap:E4} m, "
                + $"peak base tilt {tilt:E4} deg.");
            Assert.That(float.IsFinite(speed + anchor + axis + bladeGap + tilt), Is.True);
            Assert.That(speed, Is.EqualTo(48.0f).Within(0.5f));
            Assert.That(peakSpeed, Is.LessThanOrEqualTo(48.5f));
            Assert.That(anchor, Is.LessThan(0.002f));
            Assert.That(axis, Is.LessThan(0.1f));
            Assert.That(bladeGap, Is.LessThan(0.002f));
            Assert.That(tilt, Is.LessThan(1.0f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AServoReportsAndBreaksUnderALockedAxisLoad(bool torque)
        {
            Rigidbody[] bodies = Coaxial();
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Drive(joint, 0.0f, 20.0f, 50.0f);
            joint.targetAngularVelocity = 3.0f * Vector3.right;
            void Load()
            {
                if (torque)
                {
                    bodies[2].AddTorque(30.0f * Vector3.up);
                }
                else
                {
                    bodies[2].AddForce(30.0f * Vector3.up);
                }
                Simulate();
            }

            for (int i = 0; i < 10; i++)
            {
                Load();
            }
            Assert.That((torque ? joint.currentTorque : joint.currentForce).magnitude, Is.GreaterThan(3.0f));
            if (torque)
            {
                joint.breakTorque = 1.0f;
            }
            else
            {
                joint.breakForce = 1.0f;
            }
            for (int i = 0; i < 20; i++)
            {
                Load();
            }
            Assert.That(joint == null, Is.True);
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AStrutAcrossTheServoFormsAStableLoop(bool strutFirst)
        {
            Rigidbody[] bodies = Coaxial();
            void Strut() => Link(bodies[3], bodies[0], Vector3.zero, Vector3.right);
            if (strutFirst)
            {
                Strut();
            }
            var joint = Link(bodies[2], bodies[1], Vector3.right, Vector3.right, true);
            Drive(joint, 20.0f, 6.0f, 10.0f);
            Target(joint, 20.0f);
            if (!strutFirst)
            {
                Strut();
            }
            Simulate(60);
            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(1));
            Assert.That(State->VesselSet.Vessels[0].ClosureCount, Is.EqualTo(strutFirst ? 0 : 1));
            Assert.That(FreeAxes(), Is.EqualTo(strutFirst ? 0 : 1));
            Assert.That(Rows(ConstraintType.RotationMotor), Is.EqualTo(1));
            Assert.That(Mathf.Abs(Twist(joint)), Is.LessThan(1.0f));
            Assert.That(AnchorGap(joint), Is.LessThan(0.002f));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void DeletingAComponentDuringMotionLeavesFiniteMotion(bool deletePart)
        {
            var (bodies, joint) = Arm(false, false);
            Drive(joint, 30.0f, 8.0f, 30.0f);
            Target(joint, 80.0f);
            Simulate(12);
            Assert.That(Mathf.Abs(Twist(joint)), Is.GreaterThan(2.0f));
            Object.DestroyImmediate(deletePart ? (Object)bodies[3].gameObject : joint);
            Simulate(25);
            foreach (Rigidbody body in bodies.Where(body => body != null))
            {
                Assert.That(body.position.magnitude + body.linearVelocity.magnitude + body.angularVelocity.magnitude, Is.LessThan(100.0f));
            }
        }

        [TestCase(2)]
        [TestCase(8)]
        public void SerialServosEachFollowTheirOwnTarget(int count)
        {
            Rigidbody parent = Part(Vector3.zero);
            var joints = new ConfigurableJoint[count];
            for (int i = 0; i <= count; i++)
            {
                Rigidbody next = Part((2 * i + 1) * Vector3.right);
                Link(next, parent, parent.position, Vector3.right);
                if (i < count)
                {
                    parent = Part((2 * i + 2) * Vector3.right);
                    joints[i] = Link(parent, next, parent.position, Vector3.right, true);
                    Drive(joints[i], 30.0f, 8.0f, 40.0f);
                    Target(joints[i], i % 2 == 0 ? 25.0f : -25.0f);
                }
            }
            Simulate(120);
            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(1));
            Assert.That(FreeAxes(), Is.EqualTo(count));
            for (int i = 0; i < count; i++)
            {
                Assert.That(Twist(joints[i]), Is.EqualTo(i % 2 == 0 ? 25.0f : -25.0f).Within(3.0f));
                Assert.That(AnchorGap(joints[i]), Is.LessThan(0.002f));
                Assert.That(AxisGap(joints[i]), Is.LessThan(0.1f));
            }
        }
    }
}
