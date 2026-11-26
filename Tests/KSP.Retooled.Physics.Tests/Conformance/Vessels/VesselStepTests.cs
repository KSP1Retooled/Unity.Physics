using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Diagnostics;
using KSP.Retooled.Physics.Stepping;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Physics;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Dynamics.Vessels.Model;
using UnityEngine;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Forest;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// Test whole steps with backend-compiled vessels, comparing with ordinary Unity joints where applicable.
    /// </summary>
    public unsafe class VesselStepTests : PhysicsScene
    {
        private const float PartMass = 4.0f;

        /// <summary>How many vessels the backend handed to the last step.</summary>
        private static int VesselCount() => StepEntryPoints.Current->VesselSet.VesselCount;

        private bool wasVessels;
        private bool wasForest;
        private int wasFusedParts;

        /// <summary>
        /// Fused-job part threshold for these tests. int.MaxValue keeps substep solving; zero uses the default.
        /// </summary>
        protected virtual int FusedParts => int.MaxValue;

        [SetUp]
        public void TurnVesselsOn()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            EnvironmentSwitches.Read();
            EnvironmentSwitches.Read();
            wasVessels = VesselSwitches.Enabled;
            wasForest = ForestSwitches.Enabled;

            // These crafts are small, so let them compile into vessels.
            VesselSwitches.MinimumParts = 2;
            wasFusedParts = SolverSwitches.FusedVesselParts;
            if (FusedParts > 0)
            {
                SolverSwitches.FusedVesselParts = FusedParts;
            }

            Fresh(true);
        }

        [TearDown]
        public void RestoreSwitch()
        {
            VesselSwitches.Enabled = wasVessels;
            ForestSwitches.Enabled = wasForest;
            VesselSwitches.MinimumParts = 0;
            SolverSwitches.FusedVesselParts = wasFusedParts;
        }

        private void Fresh(bool vessels)
        {
            TearDownScene();
            SetUpScene();
            VesselSwitches.Enabled = vessels;
            ForestSwitches.Enabled = vessels;
            ForestDiagnostics.ForgetIslands();
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
        /// Compare horizontal stack-arm droop with vessel constraints and ordinary joints.
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

        [TestCase(true)]
        [TestCase(false)]
        public void AFallingVesselCanHitALaunchClamp(bool pinned)
        {
            BoundaryCaps previousCaps = ForestSwitches.Caps;
            try
            {
                // Use a short boundary solve to reproduce the pinned failure with one clamp.
                // The loose clamp also reproduces the spike with the default budget.
                BoundaryCaps caps = previousCaps;
                caps.PivotRounds = pinned ? 2 : 0;
                ForestSwitches.Caps = caps;

                foreach (bool vessels in new[] { false, true })
                {
                    Fresh(vessels);
                    ForestSwitches.Enabled = false;
                    Ground();
                    GameObject tower = Box(new Vector3(0.865f, 3.78f, 0.1f), new Vector3(0.5f, 7.5f, 0.5f));
                    tower.transform.rotation = Quaternion.Euler(0.0f, 13.0f, 3.0f);
                    Rigidbody clamp = tower.AddComponent<Rigidbody>();
                    clamp.mass = 0.1f;
                    clamp.inertiaTensor = new Vector3(6.95f, 0.1f, 6.95f);
                    clamp.sleepThreshold = 0.0f;
                    clamp.maxAngularVelocity = 50.0f;
                    if (pinned)
                    {
                        var pin = tower.AddComponent<ConfigurableJoint>();
                        pin.autoConfigureConnectedAnchor = false;
                        pin.anchor = Vector3.up * 3.75f;
                        pin.connectedAnchor = tower.transform.TransformPoint(pin.anchor);
                        pin.xMotion = pin.yMotion = pin.zMotion = ConfigurableJointMotion.Locked;
                        pin.angularXMotion = pin.angularYMotion = pin.angularZMotion = ConfigurableJointMotion.Locked;
                    }

                    Vector3 start = clamp.position;
                    List<Rigidbody> craft = StandingColumn(7, 0.0f, 9.0f);
                    foreach (Rigidbody part in craft)
                    {
                        part.mass = 18.0f;
                        part.sleepThreshold = 0.0f;
                        part.maxAngularVelocity = 50.0f;
                    }

                    Simulate();
                    Assert.That(VesselCount(), Is.EqualTo(vessels ? 1 : 0));
                    float clampSpeed = 0.0f, clampTravel = 0.0f, craftSpeed = 0.0f;
                    int clampPeakStep = 0;
                    for (int step = 2; step <= 220; step++)
                    {
                        Simulate();
                        float speed = clamp.linearVelocity.magnitude;
                        if (speed > clampSpeed)
                        {
                            clampSpeed = speed;
                            clampPeakStep = step;
                        }

                        clampTravel = Mathf.Max(clampTravel, Vector3.Distance(start, clamp.position));
                        foreach (Rigidbody part in craft)
                        {
                            craftSpeed = Mathf.Max(craftSpeed, part.linearVelocity.magnitude);
                        }
                    }

                    Debug.Log($"[VesselStepTests] pinned={pinned}, vessels={vessels}, clamp {clampSpeed:F3} m/s at step {clampPeakStep}, " +
                        $"clamp travel {clampTravel:F3} m, craft {craftSpeed:F3} m/s, bottom {craft[0].position:F3}");
                    Assert.That(clampSpeed, Is.LessThan(pinned ? 5.0f : 20.0f), "the clamp stays at impact speeds");
                    Assert.That(craftSpeed, Is.LessThan(20.0f), "the craft stays at falling speeds");
                    if (pinned)
                    {
                        Assert.That(clampTravel, Is.LessThan(0.1f), "the world joint holds the clamp");
                    }

                    Assert.That(craft[0].position.y, Is.LessThan(1.0f), "the craft reaches the pad");
                }
            }
            finally
            {
                ForestSwitches.Caps = previousCaps;
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void ASlidingVesselDoesNotKickAPinnedClamp(bool mounted)
        {
            Fresh(true);
            ForestSwitches.Enabled = false;
            GameObject tower = Box(new Vector3(0.66f, 3.75f, 0.05f), new Vector3(0.5f, 7.5f, 0.5f));
            tower.transform.rotation = Quaternion.Euler(0.0f, 3.0f, 0.0f);
            Rigidbody clamp = tower.AddComponent<Rigidbody>();
            clamp.mass = 0.1f;
            clamp.inertiaTensor = new Vector3(6.95f, 0.1f, 6.95f);
            clamp.sleepThreshold = 0.0f;
            var pin = tower.AddComponent<ConfigurableJoint>();
            pin.autoConfigureConnectedAnchor = false;
            pin.anchor = Vector3.up * 3.75f;
            pin.connectedAnchor = tower.transform.TransformPoint(pin.anchor);
            pin.xMotion = pin.yMotion = pin.zMotion = ConfigurableJointMotion.Locked;
            pin.angularXMotion = pin.angularYMotion = pin.angularZMotion = ConfigurableJointMotion.Locked;
            Quaternion inverseStart = Quaternion.Inverse(clamp.rotation);

            // A separate dynamic clamp makes the contact island unfused. Adding a mount makes the clamp a
            // vessel too, so the same contact can run in the fused job.
            if (mounted)
            {
                Rigidbody mount = Track(new GameObject("clamp mount")).AddComponent<Rigidbody>();
                mount.position = pin.connectedAnchor;
                mount.mass = 0.001f;
                StackJoint(mount, clamp, pin.connectedAnchor);
            }

            List<Rigidbody> craft = StandingColumn(7, 0.0f, 7.0f);
            foreach (Rigidbody part in craft)
            {
                part.mass = 129.0f / craft.Count;
                part.linearVelocity = new Vector3(0.4f, -1.0f, 0.0f);
                part.sleepThreshold = 0.0f;
            }

            long fusedBefore = IslandWorkCounters.FusedIslandSolves;
            float peakSpeed = 0.0f, peakGap = 0.0f, peakAngle = 0.0f;
            int contactSteps = 0;
            const int steps = 48;
            for (int step = 0; step < steps; step++)
            {
                // Keep the craft in contact as it slides down.
                foreach (Rigidbody part in craft)
                {
                    part.AddForce(Vector3.right * 2.0f, ForceMode.Acceleration);
                }

                BoundaryCounters.Reset();
                Simulate();
                BoundaryCounters boundary = BoundaryCounters.Current;
                if (boundary.Rows > 6 * boundary.Solves)
                {
                    contactSteps++;
                }

                peakSpeed = Mathf.Max(peakSpeed, clamp.linearVelocity.magnitude);
                peakGap = Mathf.Max(peakGap, Vector3.Distance(tower.transform.TransformPoint(pin.anchor), pin.connectedAnchor));
                Quaternion error = inverseStart * clamp.rotation;
                float angle = 2.0f * Mathf.Atan2(new Vector3(error.x, error.y, error.z).magnitude, Mathf.Abs(error.w));
                peakAngle = Mathf.Max(peakAngle, angle * Mathf.Rad2Deg);
            }

            bool fused = mounted && SolverSwitches.FusedVesselParts <= craft.Count + 2;
            Assert.That(VesselCount(), Is.EqualTo(mounted ? 2 : 1));
            Assert.That(IslandWorkCounters.FusedIslandSolves - fusedBefore, Is.EqualTo(fused ? steps : 0));
            Assert.That(contactSteps, Is.GreaterThanOrEqualTo(25));
            Assert.That(peakSpeed, Is.LessThan(0.9f), "clamp speed in m/s");
            Assert.That(peakGap, Is.LessThan(0.027f), "anchor gap in metres");
            Assert.That(peakAngle, Is.LessThan(0.15f), "angle error in degrees");
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

        /// <summary>
        /// A column of parts held together by stack joints, upright, with the centre of its bottom part at the given x
        /// and z and at the height <paramref name="bottom"/>.
        /// </summary>
        private List<Rigidbody> StandingColumn(int count, float x, float bottom = 1.0f, float z = 0.0f)
        {
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < count; i++)
            {
                Rigidbody body = Box(new Vector3(x, bottom + i, z), new Vector3(0.8f, 1.0f, 0.8f)).AddComponent<Rigidbody>();
                body.mass = PartMass;
                bodies.Add(body);
                if (i > 0)
                {
                    StackJoint(body, bodies[i - 1], new Vector3(x, bottom - 0.5f + i, z));
                }
            }

            return bodies;
        }

        /// <summary>
        /// Verify every shared island body-index entry returns to -1 after solving mixed grounded islands.
        /// </summary>
        [Test]
        public void TheIslandsLeaveTheSharedIndexOfBodiesEmpty()
        {
            // The three-part column stays on Unity joints, and the five-part column is a vessel.
            VesselSwitches.MinimumParts = 4;
            Ground();
            StandingColumn(5, 0.0f);
            StandingColumn(3, 3.0f);
            Box(new Vector3(6.0f, 1.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
            Simulate(10);
            Assert.That(VesselCount(), Is.EqualTo(1), "only the five-part column is a vessel");

            int motionCount = StepEntryPoints.Current->World.MotionVelocities.Length;
            int* index = StepEntryPoints.Current->IslandSolverCache.CompactIndex(motionCount);
            Assert.That(index != null, "the islands share an index");
            for (int body = 0; body < motionCount; body++)
            {
                Assert.That(index[body], Is.EqualTo(-1), $"body {body} has no number between steps");
            }
        }

        /// <summary>
        /// Compare landing positions and loads for small compiled vessels and ordinary Unity joints under a slow
        /// initial spin.
        /// Higher spin amplifies their different joint compliance, so this test uses the shared landing tolerance.
        /// </summary>
        [Test]
        public void AFleetOfTinyCraftLandsTheSameAsOnUnityJoints()
        {
            const int craft = 12;
            const int steps = 60;
            var positions = new Vector3[2][];
            var loads = new Vector3[2][];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(true);
                VesselSwitches.MinimumParts = pass == 0 ? 1000 : 0;
                Ground();
                var bodies = new List<Rigidbody>();
                for (int c = 0; c < craft; c++)
                {
                    // Four rows of three craft, all on the ground, which is 20 m square.
                    foreach (Rigidbody body in StandingColumn(2 + c % 6, -6.0f + 4.0f * (c % 4), 1.2f, -4.0f + 4.0f * (c / 4)))
                    {
                        body.angularVelocity = new Vector3(0.01f * c, 0.0f, 0.05f);
                        bodies.Add(body);
                    }
                }

                // Count the vessels in the first step: a vessel that sleeps is not handed to the step.
                Simulate();
                Assert.That(VesselCount(), pass == 0 ? Is.EqualTo(0) : Is.EqualTo(craft),
                    $"in pass {pass}, the craft are vessels only in the second pass, and then every one of them");
                Simulate(steps - 1);
                positions[pass] = bodies.ConvertAll(b => b.position).ToArray();
                loads[pass] = bodies.ConvertAll(b =>
                {
                    Vector3 sum = Vector3.zero;
                    foreach (ConfigurableJoint joint in b.GetComponents<ConfigurableJoint>())
                    {
                        sum += joint.currentForce;
                    }

                    return sum;
                }).ToArray();
            }

            float largestPose = 0.0f, largestLoad = 0.0f;
            for (int i = 0; i < positions[0].Length; i++)
            {
                float pose = (positions[1][i] - positions[0][i]).magnitude;
                float load = (loads[1][i] - loads[0][i]).magnitude;
                largestPose = Mathf.Max(largestPose, pose);
                largestLoad = Mathf.Max(largestLoad, load);
                Assert.That(pose, Is.LessThan(1e-3f), $"part {i} lands in the same place");
                Assert.That(load, Is.LessThan(0.01f * loads[0][i].magnitude + 0.01f), $"part {i}'s joints carry the same load");
            }

            Debug.Log($"[VesselStepTests] tiny fleet: largest difference from Unity joints {largestPose:E2} m in pose, {largestLoad:E2} N in load");
        }
    }
}
