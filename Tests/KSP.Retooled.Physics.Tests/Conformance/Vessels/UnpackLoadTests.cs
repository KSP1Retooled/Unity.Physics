using System;
using System.Collections.Generic;
using System.Linq;
using KSP.Retooled.Physics.Stepping;
using KSP.Retooled.Physics.Tests.Benchmarks;
using NUnit.Framework;
using Unity.Mathematics;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using UnityEngine;
using Object = UnityEngine.Object;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    public class UnpackLoadTests : PhysicsScene
    {
        private const float DriveForce = 1e20f;
        private static readonly Vector3 PegasusOrigin = new Vector3(-1122.0312f, -506.4113f, -1363.5506f);
        private bool previousVessels;
        private bool previousFused;
        private int previousMinimumParts;
        private int previousFusedParts;
        private Mesh padMesh;

        [SetUp]
        public void RememberSolverSettings()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("These cases select the Retooled solver paths.");
            }

            EnvironmentSwitches.Read();
            previousVessels = VesselSwitches.Enabled;
            previousFused = SolverSwitches.VesselFused;
            previousMinimumParts = VesselSwitches.MinimumParts;
            previousFusedParts = SolverSwitches.FusedVesselParts;
            VesselSwitches.MinimumParts = 2;
            SolverSwitches.FusedVesselParts = 1;
            UnityEngine.Physics.gravity = Vector3.zero;
        }

        [TearDown]
        public void RestoreSolverSettings()
        {
            VesselSwitches.Enabled = previousVessels;
            SolverSwitches.VesselFused = previousFused;
            VesselSwitches.MinimumParts = previousMinimumParts;
            SolverSwitches.FusedVesselParts = previousFusedParts;
            if (padMesh != null)
            {
                DestroyAll();
                Object.DestroyImmediate(padMesh);
                padMesh = null;
            }
        }

        [TestCase(true)]
        [TestCase(false)]
        public unsafe void AReleasedSectionDoesNotLoadOpposingHardRows(bool fused)
        {
            VesselSwitches.Enabled = true;
            SolverSwitches.VesselFused = fused;
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
                AssertVector(halfWeight * Vector3.up, primary.currentForce, 0.3f, $"attachment force at step {step}");
                AssertVector(halfWeight * Vector3.down, strut.currentForce, 0.3f, $"strut force at step {step}");
                Assert.That(primary.currentTorque.magnitude, Is.LessThan(0.3f));
                Assert.That(strut.currentTorque.magnitude, Is.LessThan(0.3f));

                if (step == 1)
                {
                    primary.breakForce = primary.breakTorque = 2200.0f;
                }
            }
        }

        [TestCase(true, true, false)]
        [TestCase(true, false, false)]
        [TestCase(false, false, false)]
        [TestCase(true, true, true)]
        [TestCase(true, false, true)]
        [TestCase(false, false, true)]
        public unsafe void AHeavyVesselKeepsBoundedLoadsAsItsSectionsRelease(bool vessels, bool fused, bool cycleStruts)
        {
            VesselSwitches.Enabled = vessels;
            SolverSwitches.VesselFused = fused;
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

        private Rigidbody[] BuildPegasus(VesselForestBench.Graph graph, List<ConfigurableJoint> joints)
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
                    ConfigureAttachment(joint, graph.Swing[i]);
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

        private static void ConfigureAttachment(ConfigurableJoint joint, float spring)
        {
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 1.0f };
            joint.linearLimitSpring = new SoftJointLimitSpring();
            var angularLimit = new SoftJointLimit { limit = 180.0f };
            joint.lowAngularXLimit = joint.highAngularXLimit = joint.angularYLimit = joint.angularZLimit = angularLimit;
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
