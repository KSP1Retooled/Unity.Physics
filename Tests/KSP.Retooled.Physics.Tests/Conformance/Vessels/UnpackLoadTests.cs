using System;
using System.Collections.Generic;
using System.Linq;
using KSP.Retooled.Physics.Commands;
using KSP.Retooled.Physics.Interop;
using KSP.Retooled.Physics.Stepping;
using KSP.Retooled.Physics.Tests.Benchmarks;
using NUnit.Framework;
using Unity.Mathematics;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using UnityEngine;
using Object = UnityEngine.Object;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    public class UnpackLoadTests : PhysicsScene
    {
        private const float DriveForce = 1e20f;
        private static readonly Vector3 PegasusOrigin = new Vector3(-1122.0312f, -506.4113f, -1363.5506f);
        private int previousMinimumParts;
        private int previousFusedParts;
        private bool previousLocking;
        private double previousLinearCap;
        private double previousAngularCap;
        private Mesh padMesh;

        [SetUp]
        public void RememberSolverSettings()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("These cases select the Retooled solver paths.");
            }

            EnvironmentSwitches.Read();
            previousMinimumParts = VesselSwitches.MinimumParts;
            previousFusedParts = SolverSwitches.FusedVesselParts;
            previousLocking = ForestSwitches.Locking;
            previousLinearCap = VesselRowSwitches.LinearCap;
            previousAngularCap = VesselRowSwitches.AngularCap;
            VesselSwitches.MinimumParts = 2;
            SolverSwitches.FusedVesselParts = 1;
            UnityEngine.Physics.gravity = Vector3.zero;
        }

        [TearDown]
        public void RestoreSolverSettings()
        {
            VesselRowSwitches.LinearCap = previousLinearCap > 0.0 ? previousLinearCap : -1.0;
            VesselRowSwitches.AngularCap = previousAngularCap > 0.0 ? previousAngularCap : -1.0;
            VesselSwitches.MinimumParts = previousMinimumParts;
            SolverSwitches.FusedVesselParts = previousFusedParts;
            ForestSwitches.Locking = previousLocking;
            if (padMesh != null)
            {
                DestroyAll();
                Object.DestroyImmediate(padMesh);
                padMesh = null;
            }
        }

        [TestCase(false, false)]
        [TestCase(true, false)]
        [TestCase(true, true)]
        public unsafe void StockAttachmentReportsZeroUnderWeightPullAndCouple(bool vessels, bool fused)
        {
            SelectSolver(vessels, fused);
            Rigidbody support = Part("support", Vector3.up * 5.0f, 10.0f, Vector3.one * 2.0f);
            Rigidbody body = Part("hanging part", Vector3.up * 4.0f, 10.0f, Vector3.one * 2.0f);
            Pin(support);
            body.isKinematic = false;
            ConfigurableJoint joint = Joint(body, support, support.position, Vector3.right, Vector3.up);
            ConfigureAttachment(joint, 20000.0f);
            joint.breakForce = 50.0f;
            joint.breakTorque = 50.0f;

            for (int step = 0; step < 120; step++)
            {
                body.AddForce(10.0f * Gravity * Vector3.up);
                if (step >= 30 && step < 80)
                {
                    body.AddForce(20000.0f * Vector3.right);
                }
                else if (step >= 80)
                {
                    body.AddTorque(1000.0f * Vector3.forward);
                }

                Simulate();
                Assert.That(joint != null, Is.True, $"step {step}");
                Assert.That(joint.currentForce, Is.EqualTo(Vector3.zero), $"step {step}");
                Assert.That(joint.currentTorque, Is.EqualTo(Vector3.zero), $"step {step}");
            }

            AssertSolver(vessels);
        }

        [TestCase(false, false, 0)]
        [TestCase(true, false, 0)]
        [TestCase(true, true, 0)]
        [TestCase(true, true, 1)]
        [TestCase(true, true, 2)]
        public unsafe void LockedAttachmentReportsAndBreaksOnItsReaction(bool vessels, bool fused, int axis)
        {
            SelectSolver(vessels, fused);
            Rigidbody support = Part("support", Vector3.up * 5.0f, 10.0f, Vector3.one * 2.0f);
            Rigidbody body = Part("part", support.position, 10.0f, Vector3.one * 2.0f);
            Pin(support);
            body.isKinematic = false;
            ConfigurableJoint joint = LockedJoint(body, support, support.position, Vector3.right, Vector3.up);
            Vector3 force = axis == 0 ? 10.0f * Gravity * Vector3.up : Vector3.zero;
            Vector3 torque = axis == 1 ? 1000.0f * Vector3.forward : axis == 2 ? 1000.0f * Vector3.right : Vector3.zero;
            for (int step = 0; step < 34; step++)
            {
                if (step == 28)
                {
                    joint.breakForce = axis == 0 ? 50.0f : float.PositiveInfinity;
                    joint.breakTorque = axis == 0 ? float.PositiveInfinity : 50.0f;
                }

                body.AddForce(force);
                body.AddTorque(torque);
                Simulate();
                if (step == 27)
                {
                    AssertSolver(vessels);
                    AssertVector(-force, joint.currentForce, 0.5f, "locked force");
                    AssertVector(-torque, joint.currentTorque, 2.0f, "locked torque");
                }
            }

            Assert.That(joint == null, Is.True);
        }

        [TestCase(false, false)]
        [TestCase(true, false)]
        [TestCase(true, true)]
        public void LinearStopReportsTheReactionBeyondTheDrive(bool vessels, bool fused)
        {
            SelectSolver(vessels, fused);
            Rigidbody support = Part("support", Vector3.up * 5.0f, 10.0f, Vector3.one * 2.0f);
            Rigidbody body = Part("part", support.position, 10.0f, Vector3.one * 2.0f);
            body.isKinematic = false;
            ConfigurableJoint joint = LockedJoint(body, support, support.position, Vector3.right, Vector3.up);
            joint.yMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 0.05f };
            joint.yDrive = new JointDrive { positionSpring = 100.0f, maximumForce = DriveForce };
            joint.breakForce = 5000.0f;
            const float load = 2098.1f;

            for (int step = 0; step < 80; step++)
            {
                body.AddForce(load * Vector3.down);
                Simulate();
            }

            Assert.That(support.position.y - body.position.y, Is.EqualTo(0.05f).Within(0.002f));
            AssertVector((load - 5.0f) * Vector3.up, joint.currentForce, 5.0f, "linear limit reaction");
            joint.breakForce = 200.0f;
            body.AddForce(load * Vector3.down);
            Simulate();
            Assert.That(joint == null, Is.True);
        }

        [TestCase(false, false, 1, 1.0f, false, false)]
        [TestCase(true, false, 1, 1.0f, false, false)]
        [TestCase(true, true, 1, 1.0f, false, false)]
        [TestCase(false, false, 3, 1.0f, false, false)]
        [TestCase(true, true, 3, 1.0f, false, false)]
        [TestCase(false, false, 1, 1.0f, true, false)]
        [TestCase(true, true, 1, 1.0f, true, false)]
        [TestCase(true, false, 1, 1.0f, false, true)]
        [TestCase(true, true, 1, 1.0f, false, true)]
        [TestCase(false, false, 1, 5.0f, false, false)]
        [TestCase(true, false, 1, 5.0f, false, false)]
        [TestCase(true, true, 1, 5.0f, false, false)]
        [TestCase(false, false, 3, 5.0f, false, false)]
        [TestCase(true, false, 3, 5.0f, false, false)]
        [TestCase(true, true, 3, 5.0f, false, false)]
        [TestCase(true, false, 1, 5.0f, false, true)]
        [TestCase(true, true, 1, 5.0f, false, true)]
        [TestCase(false, false, 1, 10.0f, false, false)]
        [TestCase(true, false, 1, 10.0f, false, false)]
        [TestCase(true, true, 1, 10.0f, false, false)]
        [TestCase(false, false, 3, 10.0f, false, false)]
        [TestCase(true, false, 3, 10.0f, false, false)]
        [TestCase(true, true, 3, 10.0f, false, false)]
        [TestCase(true, false, 1, 10.0f, false, true)]
        [TestCase(true, true, 1, 10.0f, false, true)]
        [TestCase(false, false, 1, 10.0f, true, false)]
        [TestCase(true, false, 1, 10.0f, true, false)]
        [TestCase(true, true, 1, 10.0f, true, false)]
        public unsafe void AngularStopReportsTheReactionBeyondTheDrive(bool vessels, bool fused, int count, float angle, bool twist, bool closure)
        {
            // The plain rows and the vessel use different code to recover an angular stop's load.
            SelectSolver(vessels, fused);
            Rigidbody support = Part("support", Vector3.up * 5.0f, 10.0f, Vector3.one * 20.0f);
            Rigidbody body = Part("part", support.position, 10.0f, Vector3.one * 20.0f);
            support.isKinematic = body.isKinematic = false;
            support.maxAngularVelocity = body.maxAngularVelocity = 100.0f;
            support.angularDamping = body.angularDamping = 0.0f;
            var joints = new ConfigurableJoint[count];
            float spring = twist && angle == 10.0f ? 10.0f : 2000.0f;
            if (closure)
            {
                ConfigurableJoint tree = Joint(body, support, support.position, Vector3.right, Vector3.up);
                ConfigureAttachment(tree, spring);
                tree.breakTorque = 200.0f;
            }

            for (int i = 0; i < count; i++)
            {
                joints[i] = Joint(body, support, support.position, Vector3.right, Vector3.up);
                ConfigureAttachment(joints[i], spring, angle);
            }

            if (angle == 1.0f && !twist)
            {
                Simulate();
                // Unity's ConfigurableJoint setter clamps small swing angles before sending the joint command.
                // Send the original angle through that command to exercise the backend below Unity's clamp.
                float supplied = JointData.AxisOf(JointRecord(joints[0]), StoreConstants.AxisAngularYZ)->max * Mathf.Rad2Deg;
                TestContext.Out.WriteLine($"Unity supplied a {supplied:G6} degree swing stop.");
                foreach (ConfigurableJoint joint in joints)
                {
                    var args = new CommandArgs
                    {
                        value = StoreConstants.AxisAngularYZ,
                        scalar = angle * Mathf.Deg2Rad,
                        scalar2 = angle * Mathf.Deg2Rad,
                    };
                    JointCommands.Run((uint)JointCommand.Set6DofLimit, JointRecord(joint), &args, StepEntryPoints.Current->Globals);
                }
            }

            Vector3 axis = twist ? Vector3.right : Vector3.forward;
            float limitTorque = (count + (closure ? 1 : 0)) * spring * angle * Mathf.Deg2Rad;
            float highTorque = count == 3 && angle == 10.0f ? 1600.0f : 1000.0f;
            ConfigurableJoint loader = null;
            if (count == 3 && angle == 10.0f)
            {
                // A saturated velocity drive supplies the same torque in every solver substep. AddTorque
                // supplies its whole impulse at the start of the outer step, then the stiff triplet springs pull back.
                loader = LoadMotor(body, support, axis, 0.0f, false);
            }

            const int averageSteps = 8;
            var reaction = new Vector3[count];
            for (int step = 0; step < 120; step++)
            {
                float torque = step < 40 ? limitTorque * 0.1f : highTorque;
                if (loader == null)
                {
                    body.AddTorque(torque * axis);
                    support.AddTorque(-torque * axis);
                }
                else if (step == 0 || step == 40)
                {
                    loader.angularYZDrive = new JointDrive { positionDamper = 1000.0f, maximumForce = torque };
                }
                Simulate();
                if (step >= 120 - averageSteps)
                {
                    if (closure)
                    {
                        var set = StepEntryPoints.Current->VesselSet;
                        Assert.That(set.EdgeLocked[set.Vessels[0].FirstPart + 1], Is.False, "the closure stop must keep the tree path live");
                    }
                    for (int i = 0; i < count; i++)
                    {
                        reaction[i] += joints[i].currentTorque / averageSteps;
                    }
                }

                if (step == 39)
                {
                    AssertSolver(vessels);
                    if (vessels)
                    {
                        Assert.That(StepEntryPoints.Current->VesselSet.Vessels[0].ClosureCount, Is.EqualTo(closure ? 1 : 0));
                    }
                    else if (count == 3)
                    {
                        int groupBlocks = 0;
                        foreach (var built in StepEntryPoints.Current->World.Joints)
                        {
                            groupBlocks += built.Entity.Version == JointGroup.EntityVersion ? 1 : 0;
                        }
                        Assert.That(groupBlocks, Is.EqualTo(2), "the triplet shares four rows");
                    }

                    foreach (ConfigurableJoint joint in joints)
                    {
                        Assert.That(joint.currentTorque, Is.EqualTo(Vector3.zero));
                    }
                }
            }

            float flex = Quaternion.Angle(support.rotation, body.rotation);
            Assert.That(flex, Is.EqualTo(angle).Within(angle == 1.0f ? 0.06f : 0.2f));
            TestContext.Out.WriteLine($"Settled angle {flex:G6} degrees, mean reaction {reaction[0].magnitude:G6} N m per joint.");
            for (int i = 0; i < count; i++)
            {
                AssertVector(-(highTorque - limitTorque) / count * axis, reaction[i], twist ? 1.0f : 8.0f, "mean angular limit reaction");
                joints[i].breakTorque = 50.0f;
            }

            if (loader == null)
            {
                body.AddTorque(highTorque * axis);
                support.AddTorque(-highTorque * axis);
            }
            Simulate();
            Assert.That(joints.All(joint => joint == null), Is.True);
        }

        [Test]
        public unsafe void ConfiguredDriveStiffnessHoldsBetweenTwoMovingBodies([Values(0, 2, 3)] int solver,
            [Values] bool unequal, [Values(10.0f, 180.0f)] float limit, [Values(0, 1, 2, 3, 4, 5, 6)] int drive)
        {
            bool vessels = solver >= 2;
            SelectSolver(vessels, solver == 3);
            Vector3 point = Vector3.up * 5.0f;
            Rigidbody support = Part("support", point, unequal ? 40.0f : 10.0f, Vector3.one * (unequal ? 80.0f : 20.0f));
            Rigidbody body = Part("part", point, 10.0f, Vector3.one * 20.0f);
            support.isKinematic = body.isKinematic = false;
            support.angularDamping = body.angularDamping = 0.0f;
            support.linearDamping = body.linearDamping = 0.0f;

            const float spring = 2000.0f;
            bool linear = drive == 3 || drive == 6;
            bool closure = drive == 4;
            bool acceleration = drive >= 5;
            int members = drive == 2 ? 3 : closure ? 2 : 1;
            float deflection = linear ? 0.02f : 2.0f * Mathf.Deg2Rad;
            float reduced = linear ? 1.0f / (1.0f / body.mass + 1.0f / support.mass)
                : 1.0f / (1.0f / body.inertiaTensor.z + 1.0f / support.inertiaTensor.z);
            float load = members * spring * deflection * (acceleration ? reduced : 1.0f);
            for (int i = 0; i < members; i++)
            {
                ConfigurableJoint joint = Joint(body, support, point, Vector3.right, Vector3.up);
                float memberLimit = closure && i == 1 ? (limit == 10.0f ? 180.0f : 10.0f) : limit;
                ConfigureAttachment(joint, spring, memberLimit);
                var settings = new JointDrive
                {
                    positionSpring = spring,
                    positionDamper = 100.0f,
                    maximumForce = DriveForce,
                    useAcceleration = acceleration,
                };
                joint.angularXDrive = joint.angularYZDrive = settings;
                if (drive == 1 || closure)
                {
                    settings.positionSpring *= 0.5f;
                    joint.angularXDrive = settings;
                }
                if (linear)
                {
                    joint.xDrive = settings;
                }

                joint.breakTorque = load * 0.2f / members;
            }

            float settled = 0.0f;
            if (acceleration)
            {
                // Keep the input constant over the substeps, even for a drive with a high natural frequency.
                LoadMotor(body, support, linear ? Vector3.right : Vector3.forward, load, linear);
            }
            const int steps = 240;
            const int samples = 20;
            for (int step = 0; step < steps; step++)
            {
                if (!acceleration && linear)
                {
                    body.AddForce(load * Vector3.right);
                    support.AddForce(load * Vector3.left);
                }
                else if (!acceleration)
                {
                    body.AddTorque(load * Vector3.forward);
                    support.AddTorque(-load * Vector3.forward);
                }
                Simulate();
                if (step >= steps - samples)
                {
                    settled += (linear ? body.position.x - support.position.x
                        : Quaternion.Angle(support.rotation, body.rotation) * Mathf.Deg2Rad) / samples;
                }
            }

            AssertSolver(vessels && !linear && !acceleration);
            if (vessels && closure)
            {
                Assert.That(StepEntryPoints.Current->VesselSet.Vessels[0].ClosureCount, Is.EqualTo(1));
            }
            float scale = acceleration ? reduced : 1.0f;
            TestContext.Out.WriteLine($"Expected {deflection:G6}, settled {settled:G6}, inferred drive spring {load / (members * settled * scale):G6}.");
            Assert.That(settled, Is.EqualTo(deflection).Within(deflection * 0.015f));
        }

        private static ConfigurableJoint LoadMotor(Rigidbody body, Rigidbody support, Vector3 axis, float load, bool linear)
        {
            ConfigurableJoint motor = Joint(body, support, support.position, Vector3.right, Vector3.up);
            motor.xMotion = motor.yMotion = motor.zMotion = ConfigurableJointMotion.Free;
            motor.angularXMotion = motor.angularYMotion = motor.angularZMotion = ConfigurableJointMotion.Free;
            var drive = new JointDrive { positionDamper = 1000.0f, maximumForce = load };
            if (linear)
            {
                motor.targetVelocity = -100.0f * axis;
                motor.xDrive = drive;
            }
            else
            {
                motor.targetAngularVelocity = 100.0f * axis;
                motor.angularYZDrive = drive;
            }
            return motor;
        }

        private static void SelectSolver(bool vessels, bool fused)
        {
            VesselSwitches.MinimumParts = vessels ? 2 : int.MaxValue;
            SolverSwitches.FusedVesselParts = fused ? 1 : int.MaxValue;
            ForestSwitches.Locking = false;
        }

        private static void Pin(Rigidbody support)
        {
            support.isKinematic = false;
            FixedJoint pin = support.gameObject.AddComponent<FixedJoint>();
            pin.breakForce = pin.breakTorque = float.PositiveInfinity;
        }

        private static unsafe void AssertSolver(bool vessels)
        {
            Assert.That(StepEntryPoints.Current->VesselSet.VesselCount, Is.EqualTo(vessels ? 1 : 0));
        }

        [TestCase(true)]
        [TestCase(false)]
        public unsafe void AReleasedSectionDoesNotLoadOpposingHardRows(bool fused)
        {
            // The case checks each hard row's exact share in the step after the release. With the row caps, the two 81 t
            // parts sit on a critically damped spring with a period of about four steps, so the shares lag behind.
            VesselRowSwitches.LinearCap = -1.0;
            VesselRowSwitches.AngularCap = -1.0;
            SolverSwitches.FusedVesselParts = fused ? 1 : int.MaxValue;
            const float x = 1280.0f;
            Box(new Vector3(x, -0.5f, 0.0f), new Vector3(10.0f, 1.0f, 10.0f));
            Rigidbody support = Part("support", new Vector3(x, 0.5f, 0.0f), 81.0f, 13.5f * Vector3.one, true);
            Rigidbody tank = Part("tank", new Vector3(x, 2.0f, 0.0f), 81.0f, 13.5f * Vector3.one);
            Vector3 anchor = new Vector3(x, 1.0f, 0.0f);
            ConfigurableJoint primary = Joint(tank, support, anchor, Vector3.up, Vector3.right);
            ConfigureAttachment(primary, 1500.0f);

            // The reverse joint stays separate. Its horizontal anchor gap is one float spacing at x = 1280 m.
            ConfigurableJoint strut = LockedJoint(support, tank, anchor, Vector3.up, Vector3.right);
            strut.connectedAnchor += Vector3.right / 8192.0f;
            Simulate(3);

            support.isKinematic = tank.isKinematic = false;
            // The first load releases the locked edge in step two, before the smoothed load reaches this limit.
            primary.breakForce = primary.breakTorque = 18.0f;
            for (int step = 1; step <= 5; step++)
            {
                float acceleration = Gravity * step / 16.0f;
                support.AddForce(Vector3.up * acceleration, ForceMode.Acceleration);
                tank.AddForce(Vector3.up * acceleration, ForceMode.Acceleration);
                long fusedBefore = IslandWorkCounters.FusedIslandSolves;
                Simulate();

                var state = StepEntryPoints.Current;
                var set = state->VesselSet;
                Assert.That(set.VesselCount, Is.EqualTo(1));
                Assert.That(set.EdgeLocked[set.Vessels[0].FirstPart + 1], Is.EqualTo(step == 1), $"step {step}");
                Assert.That(IslandWorkCounters.FusedIslandSolves - fusedBefore, Is.EqualTo(fused ? 1 : 0));
                Assert.That(state->Store->jointBreakCount, Is.Zero, $"step {step}");
                float halfWeight = -tank.mass * acceleration * 0.5f;
                AssertVector(halfWeight * Vector3.up, FullJointLoad(primary).FullForce, 0.3f, $"attachment force at step {step}");
                Assert.That(primary.currentForce, Is.EqualTo(Vector3.zero));
                AssertVector(halfWeight * Vector3.down, strut.currentForce, 0.3f, $"strut force at step {step}");
                Assert.That(primary.currentTorque.magnitude, Is.LessThan(0.3f));
                Assert.That(strut.currentTorque.magnitude, Is.LessThan(0.3f));

                if (step == 1)
                {
                    primary.breakForce = primary.breakTorque = 2200.0f;
                }
            }
        }

        [TestCase(false, false, 180.0f)]
        [TestCase(true, false, 180.0f)]
        [TestCase(true, true, 180.0f)]
        [TestCase(false, false, 5.0f)]
        [TestCase(true, false, 5.0f)]
        [TestCase(true, true, 5.0f)]
        [TestCase(false, false, 10.0f)]
        [TestCase(true, false, 10.0f)]
        [TestCase(true, true, 10.0f)]
        public void PegasusKeepsItsBracedAdaptersAtOneG(bool vessels, bool fused, float angle)
        {
            SelectSolver(vessels, fused);
            PegasusPad();
            VesselForestBench.Graph graph = VesselForestBench.Load("pegasus");
            var joints = new List<ConfigurableJoint>();
            Rigidbody[] bodies = BuildPegasus(graph, joints, angle);
            int attachmentCount = joints.Count;
            AddStruts(graph, bodies, joints, false);
            AddStruts(graph, bodies, joints, true);
            var attachments = joints.Take(attachmentCount).ToArray();
            var links = attachments.Select(joint => (joint.gameObject.name, joint.connectedBody.name)).ToArray();
            int[] adapters = { 239, 292, 348, 401 };
            Simulate(3);

            SetKinematic(bodies, false);
            foreach (ConfigurableJoint joint in attachments)
            {
                joint.breakForce = joint.breakTorque = 2200.0f;
            }

            var broken = new HashSet<int>();
            for (int step = 1; step <= 40; step++)
            {
                Accelerate(bodies, Gravity * Mathf.Min(1.0f, step / 16.0f));
                Simulate();
                for (int i = 0; i < attachments.Length; i++)
                {
                    if (attachments[i] == null && broken.Add(i))
                    {
                        TestContext.Out.WriteLine($"Step {step}: attachment {links[i].Item1} to {links[i].Item2} broke.");
                    }
                }
            }

            TestContext.Out.WriteLine($"{broken.Count} of {attachments.Length} attachments broke.");
            foreach (int adapter in adapters)
            {
                ConfigurableJoint[] members = attachments.Where((_, i) => links[i].Item1 == $"part {adapter}").ToArray();
                Assert.That(members, Has.Length.GreaterThan(0));
                Assert.That(members.All(joint => joint != null), Is.True, $"part {adapter}");
            }

            Assert.That(bodies[0].position.y, Is.GreaterThan(PegasusOrigin.y));
        }

        [TestCase(true, true, false)]
        [TestCase(true, false, false)]
        [TestCase(false, false, false)]
        [TestCase(true, true, true)]
        [TestCase(true, false, true)]
        [TestCase(false, false, true)]
        public unsafe void AHeavyVesselKeepsBoundedLoadsAsItsSectionsRelease(bool vessels, bool fused, bool cycleStruts)
        {
            VesselSwitches.MinimumParts = vessels ? 2 : int.MaxValue;
            SolverSwitches.FusedVesselParts = fused ? 1 : int.MaxValue;
            GameObject pad = PegasusPad();

            VesselForestBench.Graph graph = VesselForestBench.Load("pegasus");
            var joints = new List<ConfigurableJoint>();
            Rigidbody[] bodies = BuildPegasus(graph, joints);
            int attachments = joints.Count;
            AddStruts(graph, bodies, joints, false);
            int permanent = joints.Count;
            AddStruts(graph, bodies, joints, true);
            AddNearbyVessel();

            var packedPositions = bodies.Select(body => body.position).ToArray();
            SetKinematic(bodies, false);
            for (int step = 0; step < 40; step++)
            {
                Accelerate(bodies, Gravity);
                Simulate();
            }

            SetKinematic(bodies, true);
            for (int i = 0; i < bodies.Length; i++)
            {
                bodies[i].position = packedPositions[i];
                bodies[i].rotation = Quaternion.identity;
            }

            var random = new Unity.Mathematics.Random(54321);
            foreach (ConfigurableJoint joint in joints)
            {
                joint.connectedAnchor += 0.00012f * (Vector3)random.NextFloat3Direction();
            }

            Simulate(3);
            // Start over a pad with uneven contacts. Use weak joints so that some edges release during the ramp.
            pad.transform.position += 0.04f * Vector3.up;
            UnityEngine.Physics.SyncTransforms();
            SetKinematic(bodies, false);
            foreach (Rigidbody body in bodies)
            {
                body.linearVelocity = Vector3.zero;
            }

            for (int i = 0; i < permanent; i++)
            {
                ConfigurableJoint joint = joints[i];
                joint.breakForce = joint.breakTorque = 2200.0f;
                if (i < attachments)
                {
                    ConfigureAttachment(joint, joint.angularXDrive.positionSpring);
                }
            }

            float fullWeight = -Gravity * graph.Mass.Sum();
            for (int step = 1; step <= 5; step++)
            {
                if (cycleStruts && step == 1)
                {
                    for (int i = permanent; i < joints.Count; i++)
                    {
                        Object.DestroyImmediate(joints[i]);
                    }

                    joints.RemoveRange(permanent, joints.Count - permanent);
                }
                else if (cycleStruts && step == 4)
                {
                    AddStruts(graph, bodies, joints, true);
                }

                Accelerate(bodies, Gravity * step / 16.0f);
                long fusedBefore = IslandWorkCounters.FusedIslandSolves;
                Simulate();

                var state = StepEntryPoints.Current;
                var set = state->VesselSet;
                long fusedSolves = IslandWorkCounters.FusedIslandSolves - fusedBefore;
                Assert.That(set.VesselCount > 0, Is.EqualTo(vessels), $"step {step}");
                Assert.That(fusedSolves > 0, Is.EqualTo(fused), $"step {step}");

                double peakForce = 0.0;
                double peakTorque = 0.0;
                foreach (ConfigurableJoint joint in joints)
                {
                    if (joint != null)
                    {
                        peakForce = math.max(peakForce, joint.currentForce.magnitude);
                        peakTorque = math.max(peakTorque, joint.currentTorque.magnitude);
                    }
                }

                // A joint can already have been destroyed by the time managed code reads the step's loads.
                for (int v = 0; v < set.VesselCount; v++)
                {
                    var vessel = set.Vessels[v];
                    for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                    {
                        peakForce = math.max(peakForce, math.length(set.EdgeLoads[part].Force));
                        peakTorque = math.max(peakTorque, math.length(set.EdgeLoads[part].Torque));
                    }

                    for (int closure = vessel.FirstClosure; closure < vessel.FirstClosure + vessel.ClosureCount; closure++)
                    {
                        peakForce = math.max(peakForce, math.length(set.ClosureLoads[closure].Force));
                        peakTorque = math.max(peakTorque, math.length(set.ClosureLoads[closure].Torque));
                    }
                }

                uint breaks = state->Store->jointBreakCount;
                TestContext.Out.WriteLine(FormattableString.Invariant(
                    $"Step {step}, force {peakForce:G6} kN, torque {peakTorque:G6} kN m, breaks {breaks}, fused solves {fusedSolves}"));
                Assert.That(peakForce, Is.LessThan(fullWeight), $"force at step {step}");
                Assert.That(peakTorque, Is.LessThan(fullWeight * 10.0f), $"torque at step {step}");
                if (step > 1)
                {
                    Assert.That(breaks, Is.LessThan(35u), $"breaks at step {step}");
                }
            }
        }

        private GameObject PegasusPad()
        {
            padMesh = new Mesh
            {
                vertices = new[]
                {
                    new Vector3(-45, 0, -45), new Vector3(-45, 0, 45), new Vector3(45, 0, -45),
                    new Vector3(45, 0, 45), new Vector3(-0.7f, 0.0f, 0.13f),
                },
                triangles = new[] { 0, 1, 4, 1, 3, 4, 3, 2, 4, 2, 0, 4 },
            };
            GameObject pad = Track(new GameObject("mesh pad"));
            pad.transform.position = PegasusOrigin;
            pad.AddComponent<MeshCollider>().sharedMesh = padMesh;
            return pad;
        }

        private Rigidbody[] BuildPegasus(VesselForestBench.Graph graph, List<ConfigurableJoint> joints, float angle = 180.0f)
        {
            int count = graph.Parent.Length;
            var bodies = new Rigidbody[count];
            // This graph has masses, poses and springs. Approximate the large stack nodes from their stiffness.
            var triplets = new HashSet<int>(Enumerable.Range(2, count - 2).OrderByDescending(i => graph.Swing[i]).Take(114)) { 1 };
            Vector3 shift = 2.273f * Vector3.up;
            for (int i = 0; i < count; i++)
            {
                float mass = graph.Mass[i];
                Vector3 position = graph.Position[i] + shift + PegasusOrigin;
                Rigidbody body = Part($"part {i}", position, mass, graph.Inertia[i]);
                bodies[i] = body;
                if (mass == 4.0f && graph.Position[i].y < -0.8f)
                {
                    var foot = body.gameObject.AddComponent<BoxCollider>();
                    foot.size = new Vector3(1.2f, 2.0f, 1.2f);
                    foot.center = Vector3.up * (1.0f - (position.y - PegasusOrigin.y) + (i % 5 - 2) * 0.0004f);
                }

                int parent = graph.Parent[i];
                if (parent < 0)
                {
                    continue;
                }

                Vector3 axis = graph.Joint[i] - graph.Position[parent];
                if (axis.sqrMagnitude < 1e-6f)
                {
                    axis = graph.Position[i] - graph.Position[parent];
                }

                axis = axis.sqrMagnitude < 1e-6f ? Vector3.up : axis.normalized;
                Vector3 across = Vector3.Cross(axis, Mathf.Abs(axis.y) > 0.9f ? Vector3.right : Vector3.up);
                Vector3 point = graph.Joint[i] + shift + PegasusOrigin;
                int jointCount = triplets.Contains(i) ? 3 : 1;
                for (int j = 0; j < jointCount; j++)
                {
                    Vector3 offset = jointCount == 3 ? Quaternion.AngleAxis(120.0f * j, axis) * across : Vector3.zero;
                    ConfigurableJoint joint = Joint(body, bodies[parent], point + offset, axis, across);
                    ConfigureAttachment(joint, graph.Swing[i], angle);
                    joints.Add(joint);
                }
            }

            return bodies;
        }

        private static void AddStruts(VesselForestBench.Graph graph, Rigidbody[] bodies, List<ConfigurableJoint> joints, bool cycling)
        {
            // Cycle half of the graph's braces to exercise removal and recreation while the island is active.
            for (int i = cycling ? 1 : 0; i < graph.ClosureA.Length; i += 2)
            {
                Rigidbody child = bodies[graph.ClosureA[i]], parent = bodies[graph.ClosureB[i]];
                joints.Add(LockedJoint(child, parent, child.position, Vector3.right, Vector3.up));
            }
        }

        private void AddNearbyVessel()
        {
            Rigidbody nearby = WeightlessBox(PegasusOrigin + new Vector3(0.0f, 170.0f, 90.0f), 6.0f);
            Rigidbody wing = WeightlessBox(PegasusOrigin + new Vector3(5.0f, 170.0f, 90.0f), 1.0f);
            LockedJoint(wing, nearby, wing.position, Vector3.right, Vector3.up);
            nearby.linearVelocity = wing.linearVelocity = 45.0f * Vector3.right;
        }

        private Rigidbody Part(string name, Vector3 position, float mass, Vector3 inertia, bool hasCollider = false)
        {
            var gameObject = Track(new GameObject(name));
            gameObject.transform.position = position;
            if (hasCollider)
            {
                gameObject.AddComponent<BoxCollider>();
            }

            Rigidbody body = gameObject.AddComponent<Rigidbody>();
            body.mass = mass;
            body.inertiaTensor = inertia;
            body.useGravity = false;
            body.sleepThreshold = 0.0f;
            body.isKinematic = true;
            return body;
        }

        private static ConfigurableJoint Joint(Rigidbody child, Rigidbody parent, Vector3 point, Vector3 axis, Vector3 across)
        {
            var joint = child.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = parent;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = child.transform.InverseTransformPoint(point);
            joint.connectedAnchor = parent.transform.InverseTransformPoint(point);
            joint.axis = axis;
            joint.secondaryAxis = across;
            joint.breakForce = joint.breakTorque = float.PositiveInfinity;
            return joint;
        }

        private static ConfigurableJoint LockedJoint(Rigidbody child, Rigidbody parent, Vector3 point, Vector3 axis, Vector3 across)
        {
            ConfigurableJoint joint = Joint(child, parent, point, axis, across);
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            return joint;
        }

        private static void ConfigureAttachment(ConfigurableJoint joint, float spring, float angle = 180.0f)
        {
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 1.0f };
            joint.linearLimitSpring = new SoftJointLimitSpring();
            var angularLimit = new SoftJointLimit { limit = angle };
            joint.lowAngularXLimit = new SoftJointLimit { limit = -angle };
            joint.highAngularXLimit = joint.angularYLimit = joint.angularZLimit = angularLimit;
            joint.angularXLimitSpring = joint.angularYZLimitSpring = new SoftJointLimitSpring();
            joint.xDrive = joint.yDrive = joint.zDrive = new JointDrive { positionSpring = DriveForce, maximumForce = DriveForce };
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularXDrive = joint.angularYZDrive = new JointDrive { positionSpring = spring, maximumForce = DriveForce };
            joint.targetPosition = joint.targetVelocity = joint.targetAngularVelocity = Vector3.zero;
            joint.targetRotation = Quaternion.identity;
        }

        private static void SetKinematic(Rigidbody[] bodies, bool enabled)
        {
            foreach (Rigidbody body in bodies)
            {
                body.isKinematic = enabled;
            }
        }

        private static void Accelerate(Rigidbody[] bodies, float acceleration)
        {
            foreach (Rigidbody body in bodies)
            {
                body.AddForce(Vector3.up * acceleration, ForceMode.Acceleration);
            }
        }
    }
}
