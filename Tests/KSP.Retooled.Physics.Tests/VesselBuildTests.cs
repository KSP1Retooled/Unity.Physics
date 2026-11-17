using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using Unity.Physics;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Test backend vessel compilation, emitted constraints and joint load/break conversion independently of the
    /// solver.
    /// </summary>
    public unsafe class VesselBuildTests : PhysicsScene
    {
        private const float PartHeight = 1.0f;
        private const float PartMass = 2.0f;

        private bool wasEnabled;

        [SetUp]
        public void TurnVesselsOn()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            VesselPipeline.Initialize();
            wasEnabled = VesselPipeline.Enabled;
            VesselPipeline.Enabled = true;

            // These crafts are small, so let them compile into vessels.
            VesselPipeline.MinimumParts = 2;
        }

        [TearDown]
        public void RestoreSwitch()
        {
            VesselPipeline.Enabled = wasEnabled;
            VesselPipeline.MinimumParts = 0;
        }

        private static StepState* State => StepKernel.Current;

        /// <summary>A stock part joint: three joints over a stack node, which the backend groups into one unit.</summary>
        private static ConfigurableJoint[] StackJoint(Rigidbody host, Rigidbody target, Vector3 node, float breakForce = float.PositiveInfinity)
        {
            var joints = new ConfigurableJoint[3];
            for (int i = 0; i < 3; i++)
            {
                Vector3 anchor = node + Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(0.3f, 0.0f, 0.0f);
                ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = target;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = host.transform.InverseTransformPoint(anchor);
                joint.connectedAnchor = target != null ? target.transform.InverseTransformPoint(anchor) : anchor;
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

        /// <summary>A surface attachment: one joint at one point, with its swing held by a drive.</summary>
        private static ConfigurableJoint SurfaceJoint(Rigidbody host, Rigidbody target, Vector3 point, float spring)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(point);
            joint.connectedAnchor = target.transform.InverseTransformPoint(point);
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = ConfigurableJointMotion.Locked;
            joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
            joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularYZDrive = new JointDrive { positionSpring = spring, positionDamper = 0.1f * spring, maximumForce = 1e20f };
            return joint;
        }

        /// <summary>A strut: one joint with every axis locked, between parts that aren't next to each other.</summary>
        private static ConfigurableJoint Strut(Rigidbody host, Rigidbody target)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            return joint;
        }

        /// <summary>Parts stacked from the top down, each joined to the one above by a stack joint.</summary>
        private List<Rigidbody> Column(int count, float top = 10.0f)
        {
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < count; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, top - PartHeight * (i + 0.5f), 0.0f), new Vector3(0.6f, PartHeight, 0.6f))
                    .AddComponent<Rigidbody>();
                body.mass = PartMass;
                body.useGravity = false;
                bodies.Add(body);
                if (i > 0)
                {
                    StackJoint(body, bodies[i - 1], new Vector3(0.0f, top - PartHeight * i, 0.0f));
                }
            }

            return bodies;
        }

        /// <summary>The Rigidbody of each part of each vessel in the step's set, found by the position of its built body.</summary>
        private static List<List<Rigidbody>> VesselsOf(List<Rigidbody> bodies)
        {
            VesselSet set = State->VesselSet;

            // The step returns while the refit moves the built bodies to their solved poses.
            State->FinishPendingRefit();
            PhysicsWorld world = State->World;
            var vessels = new List<List<Rigidbody>>();
            for (int v = 0; v < set.VesselCount; v++)
            {
                var parts = new List<Rigidbody>();
                VesselRecord vessel = set.Vessels[v];
                for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    float3 position = world.Bodies[set.PartBodies[part]].WorldFromBody.pos;
                    parts.Add(bodies.Find(body => math.distance(position, body.position) < 1e-4f));
                }

                vessels.Add(parts);
            }

            return vessels;
        }

        private static VesselRecord OnlyVessel()
        {
            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(1), "vessels");
            return State->VesselSet.Vessels[0];
        }

        [Test]
        public void AColumnIsOneVesselInParentFirstOrder()
        {
            List<Rigidbody> bodies = Column(4);
            Simulate();

            VesselRecord vessel = OnlyVessel();
            Assert.That(vessel.PartCount, Is.EqualTo(4));
            Assert.That(vessel.ClosureCount, Is.Zero);
            List<Rigidbody> parts = VesselsOf(bodies)[0];
            CollectionAssert.AreEquivalent(bodies, parts);
            for (int part = 0; part < vessel.PartCount; part++)
            {
                VesselEdge edge = State->VesselSet.Edges[vessel.FirstPart + part];
                Assert.That(edge.Parent, Is.LessThan(part), $"part {part}'s parent comes before it");
                Assert.That(part == 0 ? edge.Parent == -1 : edge.Parent >= 0, Is.True, $"only the root has no parent ({part})");
                if (part > 0)
                {
                    // Three joints spread around the node hold every rotation through their hard linear rows.
                    Assert.That(edge.TwistStiffness, Is.EqualTo(double.PositiveInfinity), "twist");
                    Assert.That(edge.SwingStiffness, Is.EqualTo(double.PositiveInfinity), "swing");
                    Assert.That(edge.MayLock, Is.True);
                }
            }
        }

        [Test]
        public void AVesselsJointsBuildNoUnityJoints()
        {
            Column(4);
            Simulate();

            // The step drops the contact pairs between jointed parts itself, so it needs no joint without constraints.
            Assert.That(State->World.Joints.Length, Is.Zero, "built joints");
        }

        /// <summary>Records the bodies that touch the body that this component is on.</summary>
        [ExecuteAlways]
        internal class TouchLog : MonoBehaviour
        {
            public readonly HashSet<Rigidbody> Touched = new HashSet<Rigidbody>();

            private void OnCollisionEnter(Collision collision)
            {
                Touched.Add(collision.rigidbody);
            }

            private void OnCollisionStay(Collision collision)
            {
                Touched.Add(collision.rigidbody);
            }
        }

        /// <summary>
        /// Three parts in a column that overlap their neighbours by a fifth of their height, and a fourth part on the
        /// side of the top part that overlaps the middle part. The fourth part is jointed only to the top part.
        /// </summary>
        private List<Rigidbody> OverlappingParts(bool stackCollides)
        {
            const float top = 10.0f;
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < 4; i++)
            {
                Vector3 position = i < 3 ? new Vector3(0.0f, top - PartHeight * (i + 0.5f), 0.0f) : new Vector3(0.5f, top - PartHeight * 1.5f, 0.0f);
                Rigidbody body = Box(position, new Vector3(0.6f, 1.2f * PartHeight, 0.6f)).AddComponent<Rigidbody>();
                body.mass = PartMass;
                body.useGravity = false;
                body.gameObject.AddComponent<TouchLog>();
                bodies.Add(body);
            }

            for (int i = 1; i < 3; i++)
            {
                foreach (ConfigurableJoint joint in StackJoint(bodies[i], bodies[i - 1], new Vector3(0.0f, top - PartHeight * i, 0.0f)))
                {
                    joint.enableCollision = stackCollides;
                }
            }

            SurfaceJoint(bodies[3], bodies[0], new Vector3(0.5f, top - PartHeight, 0.0f), 1000.0f);
            return bodies;
        }

        [Test]
        public void JointedPartsDoNotCollideButOtherPartsOfTheVesselDo()
        {
            List<Rigidbody> bodies = OverlappingParts(false);
            Simulate(3);

            Assert.That(VesselsOf(bodies).Count, Is.EqualTo(1), "one vessel");
            HashSet<Rigidbody> middle = bodies[1].GetComponent<TouchLog>().Touched;
            Assert.That(middle.Contains(bodies[0]), Is.False, "the middle part doesn't touch the part that it is jointed to above");
            Assert.That(middle.Contains(bodies[2]), Is.False, "the middle part doesn't touch the part that it is jointed to below");
            Assert.That(middle.Contains(bodies[3]), Is.True, "the middle part touches the side part, which it isn't jointed to");
            Assert.That(bodies[3].GetComponent<TouchLog>().Touched.Contains(bodies[0]), Is.False, "the side part doesn't touch its parent");
        }

        [Test]
        public void AVesselJointThatEnablesCollisionLetsItsPartsCollide()
        {
            List<Rigidbody> bodies = OverlappingParts(true);
            Simulate(3);

            Assert.That(VesselsOf(bodies).Count, Is.EqualTo(1), "one vessel");
            HashSet<Rigidbody> middle = bodies[1].GetComponent<TouchLog>().Touched;
            Assert.That(middle.Contains(bodies[0]), Is.True, "the stack joint above lets the parts collide");
            Assert.That(middle.Contains(bodies[2]), Is.True, "the stack joint below lets the parts collide");
            Assert.That(bodies[3].GetComponent<TouchLog>().Touched.Contains(bodies[0]), Is.False, "the side part doesn't touch its parent");
        }

        [Test]
        public void AStrutClosesALoop()
        {
            List<Rigidbody> bodies = Column(4);
            Strut(bodies[0], bodies[3]);
            Simulate();

            VesselRecord vessel = OnlyVessel();
            Assert.That(vessel.ClosureCount, Is.EqualTo(1));
            VesselClosure closure = State->VesselSet.Closures[vessel.FirstClosure];
            List<Rigidbody> parts = VesselsOf(bodies)[0];
            CollectionAssert.AreEquivalent(new[] { bodies[0], bodies[3] }, new[] { parts[closure.PartA], parts[closure.PartB] });
            Assert.That(closure.LinearStiffness, Is.EqualTo(double.PositiveInfinity));
            Assert.That(closure.AngularStiffness, Is.EqualTo(double.PositiveInfinity));
        }

        [Test]
        public void AHingeInTheMiddleMakesTwoVessels()
        {
            List<Rigidbody> bodies = new List<Rigidbody>();
            for (int i = 0; i < 4; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, 10.0f - PartHeight * (i + 0.5f), 0.0f), new Vector3(0.6f, PartHeight, 0.6f))
                    .AddComponent<Rigidbody>();
                body.useGravity = false;
                bodies.Add(body);
            }

            StackJoint(bodies[1], bodies[0], new Vector3(0.0f, 9.0f, 0.0f));
            HingeJoint hinge = bodies[2].gameObject.AddComponent<HingeJoint>();
            hinge.connectedBody = bodies[1];
            hinge.anchor = new Vector3(0.0f, 0.5f, 0.0f);
            StackJoint(bodies[3], bodies[2], new Vector3(0.0f, 7.0f, 0.0f));
            Simulate();

            List<List<Rigidbody>> vessels = VesselsOf(bodies);
            Assert.That(vessels.Count, Is.EqualTo(2));
            CollectionAssert.AreEquivalent(new[] { bodies[0], bodies[1] }, vessels.Find(v => v.Contains(bodies[0])));
            CollectionAssert.AreEquivalent(new[] { bodies[2], bodies[3] }, vessels.Find(v => v.Contains(bodies[3])));

            // The hinge is no vessel joint, so it is the only built joint with constraints.
            int constrained = 0;
            foreach (Unity.Physics.Joint joint in State->World.Joints)
            {
                constrained += joint.Constraints.Length > 0 ? 1 : 0;
            }

            Assert.That(constrained, Is.GreaterThan(0), "the hinge is built");
        }

        [Test]
        public void AJointToTheWorldStaysAUnityJoint()
        {
            List<Rigidbody> bodies = Column(3);
            StackJoint(bodies[0], null, new Vector3(0.0f, 10.0f, 0.0f));
            Simulate();

            VesselRecord vessel = OnlyVessel();
            Assert.That(vessel.PartCount, Is.EqualTo(3), "the world is no part");
            int constrained = 0;
            foreach (Unity.Physics.Joint joint in State->World.Joints)
            {
                constrained += joint.Constraints.Length > 0 ? 1 : 0;
            }

            Assert.That(constrained, Is.GreaterThan(0), "the joint to the world is built");
        }

        [Test]
        public void ChangingTheJointsGivesNewIdsAndNothingElseDoes()
        {
            List<Rigidbody> bodies = Column(4);
            Simulate();
            ulong first = OnlyVessel().Id;

            Simulate();
            bodies[2].position += new Vector3(0.1f, 0.0f, 0.0f);
            Simulate();
            Assert.That(OnlyVessel().Id, Is.EqualTo(first), "stepping and moving a part keep the vessel");

            ConfigurableJoint strut = Strut(bodies[0], bodies[3]);
            Simulate();
            ulong strutted = OnlyVessel().Id;
            Assert.That(strutted, Is.Not.EqualTo(first), "a new strut makes a new vessel");
            Assert.That(OnlyVessel().ClosureCount, Is.EqualTo(1));

            Object.DestroyImmediate(strut);
            Simulate();
            Assert.That(OnlyVessel().Id, Is.EqualTo(strutted), "a removed strut stays in its vessel as a tombstone for a while");
            Simulate(VesselPipeline.TombstoneSteps);
            ulong unstrutted = OnlyVessel().Id;
            Assert.That(OnlyVessel().ClosureCount, Is.Zero, "the tombstone has expired");
            Assert.That(unstrutted, Is.Not.EqualTo(strutted).And.Not.EqualTo(first), "Ids are never reused");

            // Decoupling: the joints between parts 1 and 2 go. The column stays one vessel with a broken edge for a
            // while, then it becomes two vessels.
            foreach (ConfigurableJoint joint in bodies[2].GetComponents<ConfigurableJoint>())
            {
                Object.DestroyImmediate(joint);
            }

            Simulate();
            Assert.That(OnlyVessel().Id, Is.EqualTo(unstrutted), "the removed edge is a tombstone");
            Simulate(VesselPipeline.TombstoneSteps);
            List<List<Rigidbody>> vessels = VesselsOf(bodies);
            Assert.That(vessels.Count, Is.EqualTo(2));
            for (int v = 0; v < 2; v++)
            {
                Assert.That(State->VesselSet.Vessels[v].Id, Is.Not.EqualTo(unstrutted));
            }
        }

        /// <summary>
        /// Verify a rebuilt craft gets a new vessel ID even when both body record addresses are reused.
        /// </summary>
        [Test]
        public void ACraftBuiltAgainInTheMemoryOfAnOldCraftIsANewVessel()
        {
            const int rounds = 100;
            int reused = 0;
            List<Rigidbody> bodies = Column(2);
            for (int round = 0; round < rounds; round++)
            {
                Simulate();
                ulong oldId = OnlyVessel().Id;
                var oldRecords = new HashSet<long>();
                for (int i = 0; i < State->Store->bodyCount; i++)
                {
                    oldRecords.Add((long)State->Store->bodies[i].owner);
                }

                for (int i = 0; i < bodies.Count; i++)
                {
                    Object.DestroyImmediate(bodies[i].gameObject);
                    bodies[i] = Box(new Vector3(0.0f, 10.0f - PartHeight * (i + 0.5f), 0.0f), new Vector3(0.6f, PartHeight, 0.6f))
                        .AddComponent<Rigidbody>();
                    bodies[i].mass = PartMass;
                    bodies[i].useGravity = false;
                }

                StackJoint(bodies[1], bodies[0], new Vector3(0.0f, 10.0f - PartHeight, 0.0f));
                int same = 0;
                for (int i = 0; i < State->Store->bodyCount; i++)
                {
                    same += oldRecords.Contains((long)State->Store->bodies[i].owner) ? 1 : 0;
                }

                reused += same == 2 ? 1 : 0;
                Simulate();
                Assert.That(OnlyVessel().Id, Is.Not.EqualTo(oldId), $"round {round}: the craft that was built again is a new vessel");
            }

            Debug.Log($"[VesselBuildTests] the new craft had both of the old craft's body records in {reused} of {rounds} rounds");
        }

        /// <summary>
        /// Verify edge caches are invalidated when recompilation reorders sleeping vessels without changing part
        /// count.
        /// </summary>
        [Test]
        public void EdgesCompiledWhileEveryVesselSleepsAreFilledForTheirOwnJoints()
        {
            const int parts = 6;
            var bodies = new List<Rigidbody>();
            for (int column = 0; column < 2; column++)
            {
                float height = column == 0 ? PartHeight : 2.0f * PartHeight;
                float x = 5.0f * column;
                for (int i = 0; i < parts; i++)
                {
                    Rigidbody body = Box(new Vector3(x, 20.0f - height * (i + 0.5f), 0.0f), new Vector3(0.6f, height, 0.6f))
                        .AddComponent<Rigidbody>();
                    body.mass = PartMass;
                    body.useGravity = false;
                    if (i > 0)
                    {
                        StackJoint(body, bodies[bodies.Count - 1], new Vector3(x, 20.0f - height * i, 0.0f));
                    }

                    bodies.Add(body);
                }
            }

            Simulate(2);
            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(2), "two vessels take part");

            void SleepAll()
            {
                foreach (Rigidbody body in bodies)
                {
                    body.Sleep();
                }
            }

            SleepAll();
            Simulate();
            Assert.That(State->VesselSet.VesselCount, Is.Zero, "no vessel takes part while its parts sleep");

            foreach (ConfigurableJoint joint in bodies[3].GetComponents<ConfigurableJoint>())
            {
                Object.DestroyImmediate(joint);
            }

            for (int step = 0; step <= VesselPipeline.TombstoneSteps; step++)
            {
                SleepAll();
                Simulate();
                Assert.That(State->VesselSet.VesselCount, Is.Zero, $"step {step}: no vessel takes part");
            }

            Assert.That(State->Vessels.Vessels.Length, Is.EqualTo(3), "the first column became two vessels when its tombstone expired");
            Assert.That(State->Vessels.PartStoreBodies.Length, Is.EqualTo(2 * parts), "the number of parts stays the same");

            foreach (Rigidbody body in bodies)
            {
                body.WakeUp();
            }

            Simulate();
            Assert.That(State->VesselSet.VesselCount, Is.EqualTo(3), "the vessels take part again");
            var kept = new VesselEdge[2 * parts];
            for (int part = 0; part < kept.Length; part++)
            {
                kept[part] = State->VesselSet.Edges[part];
            }

            // Compile again, which keeps every vessel in its place and computes every edge.
            JointTopology.Changed();
            Simulate();
            for (int part = 0; part < kept.Length; part++)
            {
                VesselEdge fresh = State->VesselSet.Edges[part];
                Assert.That(kept[part].Parent, Is.EqualTo(fresh.Parent), $"part {part}'s parent");
                Assert.That(kept[part].ChildAnchor, Is.EqualTo(fresh.ChildAnchor), $"part {part}'s anchor on itself");
                Assert.That(kept[part].ParentAnchor, Is.EqualTo(fresh.ParentAnchor), $"part {part}'s anchor on its parent");
            }
        }

        [Test]
        public void AStrutDestroyedAndCreatedAgainInOneFrameKeepsItsVessel()
        {
            List<Rigidbody> bodies = Column(4);
            ConfigurableJoint strut = Strut(bodies[0], bodies[3]);
            Simulate();
            VesselRecord before = OnlyVessel();

            Object.DestroyImmediate(strut);
            Strut(bodies[0], bodies[3]);
            Simulate();

            VesselRecord after = OnlyVessel();
            Assert.That(after.Id, Is.EqualTo(before.Id), "the vessel keeps its Id");
            Assert.That(after.StructureVersion, Is.EqualTo(before.StructureVersion), "the vessel keeps its structure");
            Assert.That(after.ClosureCount, Is.EqualTo(1));
            Assert.That(State->VesselSet.Closures[after.FirstClosure].LinearStiffness, Is.EqualTo(double.PositiveInfinity), "the new strut holds");
        }

        [Test]
        public void AStrutDestroyedAndCreatedAgainInTheNextFrameKeepsItsVessel()
        {
            List<Rigidbody> bodies = Column(4);
            ConfigurableJoint strut = Strut(bodies[0], bodies[3]);
            Simulate();
            VesselRecord before = OnlyVessel();

            Object.DestroyImmediate(strut);
            Simulate();
            VesselRecord gone = OnlyVessel();
            Assert.That(gone.Id, Is.EqualTo(before.Id), "the vessel keeps its Id while the strut is gone");
            Assert.That(gone.ClosureCount, Is.EqualTo(1), "the closure of the strut stays as a tombstone");
            VesselClosure tombstone = State->VesselSet.Closures[gone.FirstClosure];
            Assert.That(tombstone.LinearStiffness, Is.Zero, "a tombstone has no stiffness");
            Assert.That(tombstone.AngularStiffness, Is.Zero);

            Strut(bodies[0], bodies[3]);
            Simulate();
            VesselRecord after = OnlyVessel();
            Assert.That(after.Id, Is.EqualTo(before.Id), "the vessel keeps its Id when the strut comes back");
            Assert.That(State->VesselSet.Closures[after.FirstClosure].LinearStiffness, Is.EqualTo(double.PositiveInfinity), "the new strut holds");
        }

        [Test]
        public void ARemovedEdgeIsBrokenInPlaceUntilItsTombstoneExpires()
        {
            List<Rigidbody> bodies = Column(4);
            Simulate();
            ulong id = OnlyVessel().Id;
            foreach (ConfigurableJoint joint in bodies[3].GetComponents<ConfigurableJoint>())
            {
                Object.DestroyImmediate(joint);
            }

            Simulate();
            VesselRecord vessel = OnlyVessel();
            Assert.That(vessel.Id, Is.EqualTo(id));
            Assert.That(vessel.PartCount, Is.EqualTo(4), "the loose part stays in the vessel for now");
            int broken = 0;
            for (int part = 0; part < vessel.PartCount; part++)
            {
                broken += State->VesselSet.Edges[vessel.FirstPart + part].State == VesselEdgeState.Broken ? 1 : 0;
            }

            Assert.That(broken, Is.EqualTo(1), "its edge is broken");

            Simulate(VesselPipeline.TombstoneSteps);
            Assert.That(OnlyVessel().PartCount, Is.EqualTo(3), "after the tombstone expires, the loose part is no longer a part");
            Assert.That(OnlyVessel().Id, Is.Not.EqualTo(id));
        }

        /// <summary>
        /// Adds a body far from the crafts with a joint to the world. The new joint changes the joint topology, so the
        /// next step compiles the vessels again, but it joins no vessel.
        /// </summary>
        private Rigidbody AnchoredBody()
        {
            Rigidbody body = Box(new Vector3(50.0f, 10.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
            body.useGravity = false;
            body.gameObject.AddComponent<FixedJoint>();
            return body;
        }

        /// <summary>The edge of the part whose body is the given one.</summary>
        private static VesselEdge EdgeOf(List<Rigidbody> bodies, Rigidbody body)
        {
            VesselRecord vessel = OnlyVessel();
            int part = VesselsOf(bodies)[0].IndexOf(body);
            Assert.That(part, Is.GreaterThan(0), "the body is a part with a parent");
            return State->VesselSet.Edges[vessel.FirstPart + part];
        }

        [Test]
        public void ASecondUnitBetweenTheSameBodiesKeepsItsPlaceWhenTheVesselIsKept()
        {
            List<Rigidbody> bodies = Column(4);

            // This strut has a different frame, so it forms a closure beside the stack-joint edge. Only the
            // strut can break.
            ConfigurableJoint strut = Strut(bodies[3], bodies[2]);
            strut.axis = Vector3.forward;
            strut.breakForce = 777.0f;
            Simulate();
            VesselRecord before = OnlyVessel();
            Assert.That(before.ClosureCount, Is.EqualTo(1), "the strut is a closure");
            Assert.That(EdgeOf(bodies, bodies[3]).BreakForce, Is.EqualTo(double.PositiveInfinity), "the stack joints are the edge");

            AnchoredBody();
            Simulate();
            VesselRecord after = OnlyVessel();
            Assert.That(after.Id, Is.EqualTo(before.Id), "the vessel is kept");
            Assert.That(after.StructureVersion, Is.EqualTo(before.StructureVersion), "its structure is the same");
            Assert.That(EdgeOf(bodies, bodies[3]).BreakForce, Is.EqualTo(double.PositiveInfinity), "the stack joints are still the edge");
        }

        [Test]
        public void AnEdgeWhoseJointsAreReplacedChangesTheStructure()
        {
            List<Rigidbody> bodies = Column(4);
            Simulate();
            VesselRecord before = OnlyVessel();

            // The game destroys a part's attachment and makes it again in the same frame.
            foreach (ConfigurableJoint joint in bodies[3].GetComponents<ConfigurableJoint>())
            {
                Object.DestroyImmediate(joint);
            }

            StackJoint(bodies[3], bodies[2], new Vector3(0.0f, 10.0f - 3.0f * PartHeight, 0.0f));
            Simulate();
            VesselRecord after = OnlyVessel();
            Assert.That(after.Id, Is.EqualTo(before.Id), "the vessel is kept");
            Assert.That(after.PartCount, Is.EqualTo(4));
            Assert.That(EdgeOf(bodies, bodies[3]).State, Is.Not.EqualTo(VesselEdgeState.Broken), "the new joints hold the edge");
            Assert.That(after.StructureVersion, Is.Not.EqualTo(before.StructureVersion),
                "the lock state of the old joints doesn't carry over to the new ones");
        }

        [Test]
        public void AJointThatMovesInTheStoreKeepsItsEdge()
        {
            // The anchored body's joint is first in the store. Destroying it moves the store's last joint, one of the
            // lowest part's stack joints, into its place, and that joint then leads its group.
            Rigidbody anchored = AnchoredBody();
            List<Rigidbody> bodies = Column(4);
            Simulate();
            VesselRecord before = OnlyVessel();

            Object.DestroyImmediate(anchored.GetComponent<FixedJoint>());
            Simulate();
            VesselRecord after = OnlyVessel();
            Assert.That(after.Id, Is.EqualTo(before.Id), "the vessel is kept");
            Assert.That(after.StructureVersion, Is.EqualTo(before.StructureVersion), "the same joints hold every edge");
        }

        [Test]
        public void AServoThatGetsATargetSplitsItsVesselAtOnce()
        {
            List<Rigidbody> bodies = Column(4);
            Simulate();
            ulong id = OnlyVessel().Id;

            // A target takes the joints out of the vessel. They still hold the parts together, as a Unity Physics
            // joint, so the vessel splits in two at once instead of keeping a tombstone.
            foreach (ConfigurableJoint joint in bodies[2].GetComponents<ConfigurableJoint>())
            {
                joint.targetRotation = Quaternion.Euler(0.0f, 10.0f, 0.0f);
            }

            Simulate();
            List<List<Rigidbody>> vessels = VesselsOf(bodies);
            Assert.That(vessels.Count, Is.EqualTo(2), "two vessels");
            CollectionAssert.AreEquivalent(new[] { bodies[0], bodies[1] }, vessels.Find(v => v.Contains(bodies[0])));
            CollectionAssert.AreEquivalent(new[] { bodies[2], bodies[3] }, vessels.Find(v => v.Contains(bodies[3])));
            for (int v = 0; v < 2; v++)
            {
                Assert.That(State->VesselSet.Vessels[v].Id, Is.Not.EqualTo(id), "both vessels are new");
            }
        }

        [Test]
        public void ASettingThatChangesWhileItsVesselSleepsReachesItsEdge()
        {
            Rigidbody parent = Box(new Vector3(0.0f, 10.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
            Rigidbody child = Box(new Vector3(1.0f, 10.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
            parent.useGravity = child.useGravity = false;
            ConfigurableJoint joint = SurfaceJoint(child, parent, new Vector3(0.5f, 10.0f, 0.0f), 3000.0f);
            Simulate(2);
            Assert.That(State->VesselSet.Edges[OnlyVessel().FirstPart + 1].SwingStiffness, Is.EqualTo(3000.0).Within(1e-3));

            // Setting a drive wakes the joint's bodies, so put them to sleep after the change.
            joint.angularYZDrive = new JointDrive { positionSpring = 5000.0f, positionDamper = 50.0f, maximumForce = 1e20f };
            parent.Sleep();
            child.Sleep();
            Simulate();
            if (State->VesselSet.IsCreated)
            {
                Assert.Inconclusive("the vessel didn't sleep through the change");
            }

            parent.WakeUp();
            Simulate();
            VesselRecord vessel = OnlyVessel();
            Assert.That(State->VesselSet.Edges[vessel.FirstPart + 1].SwingStiffness, Is.EqualTo(5000.0).Within(1e-3),
                "the edge is computed again when the vessel wakes");
        }

        [Test]
        public void ASettingRefillsOnlyTheEdgesOfItsJoint()
        {
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < 3; i++)
            {
                Rigidbody body = Box(new Vector3(1.0f * i, 10.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                body.useGravity = false;
                bodies.Add(body);
            }

            SurfaceJoint(bodies[1], bodies[0], new Vector3(0.5f, 10.0f, 0.0f), 1000.0f);
            ConfigurableJoint second = SurfaceJoint(bodies[2], bodies[1], new Vector3(1.5f, 10.0f, 0.0f), 2000.0f);
            Simulate(2);

            // Mark the cached edge of the first joint. The mark reaches the step only if the fill job reuses that edge.
            VesselRecord vessel = OnlyVessel();
            int first = vessel.FirstPart + VesselsOf(bodies)[0].IndexOf(bodies[1]);
            VesselBuild build = State->Vessels;
            VesselEdge marked = build.EdgeCache[first];
            marked.SwingDamping = 12345.0;
            build.EdgeCache[first] = marked;

            uint version = JointSettings.Version;
            second.angularYZDrive = new JointDrive { positionSpring = 4000.0f, positionDamper = 400.0f, maximumForce = 1e20f };
            Assert.That(JointSettings.Version, Is.Not.EqualTo(version), "a new drive moves the version");
            Simulate();
            Assert.That(EdgeOf(bodies, bodies[2]).SwingStiffness, Is.EqualTo(4000.0).Within(1e-3), "the changed joint's edge is new");
            Assert.That(EdgeOf(bodies, bodies[1]).SwingDamping, Is.EqualTo(12345.0), "the other edge is reused");

            version = JointSettings.Version;
            second.angularYZDrive = new JointDrive { positionSpring = 4000.0f, positionDamper = 400.0f, maximumForce = 1e20f };
            second.breakForce = second.breakForce;
            Assert.That(JointSettings.Version, Is.EqualTo(version), "sending the same settings again changes nothing");
        }

        /// <summary>
        /// Verify a three-part craft retains Unity joints when the minimum vessel size is four.
        /// </summary>
        [Test]
        public void ACraftBelowTheMinimumSizeStaysOnUnityJoints()
        {
            VesselPipeline.MinimumParts = 4;
            List<Rigidbody> bodies = Column(3);
            bodies[2].linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            Simulate(20);

            Assert.That(State->VesselSet.IsCreated, Is.False, "a craft below the minimum size is no vessel");
            int constrained = 0;
            foreach (Unity.Physics.Joint joint in State->World.Joints)
            {
                constrained += joint.Constraints.Length > 0 ? 1 : 0;
            }

            Assert.That(constrained, Is.GreaterThanOrEqualTo(2), "both stack joints are built as Unity Physics joints");
            for (int i = 1; i < 3; i++)
            {
                Assert.That(Vector3.Distance(bodies[i].position, bodies[i - 1].position), Is.EqualTo(PartHeight).Within(1e-3f),
                    $"the joint between parts {i - 1} and {i} holds");
            }

            float meanX = (bodies[0].linearVelocity.x + bodies[1].linearVelocity.x + bodies[2].linearVelocity.x) / 3.0f;
            Assert.That(meanX, Is.EqualTo(1.0f / 3.0f).Within(1e-3f), "the craft keeps the momentum of the push");
        }

        /// <summary>A craft of the default minimum size, two parts and one joint, is a vessel.</summary>
        [Test]
        public void ACraftOfTheMinimumSizeIsAVessel()
        {
            VesselPipeline.MinimumParts = 0;
            Column(VesselPipeline.DefaultMinimumParts);
            Simulate();
            Assert.That(OnlyVessel().PartCount, Is.EqualTo(VesselPipeline.DefaultMinimumParts));
        }

        [Test]
        public void ADrivesSpringReachesItsEdge()
        {
            Rigidbody parent = Box(new Vector3(0.0f, 10.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
            Rigidbody child = Box(new Vector3(1.0f, 10.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
            parent.useGravity = child.useGravity = false;
            ConfigurableJoint joint = SurfaceJoint(child, parent, new Vector3(0.5f, 10.0f, 0.0f), 3000.0f);
            Simulate();
            VesselRecord vessel = OnlyVessel();
            ulong id = vessel.Id;
            VesselEdge edge = State->VesselSet.Edges[vessel.FirstPart + 1];
            Assert.That(edge.TwistStiffness, Is.EqualTo(double.PositiveInfinity), "the locked twist is hard");
            Assert.That(edge.SwingStiffness, Is.EqualTo(3000.0).Within(1e-3), "a single joint's swing is its drive");
            Assert.That(edge.SwingDamping, Is.EqualTo(300.0).Within(1e-3));

            joint.angularYZDrive = new JointDrive { positionSpring = 5000.0f, positionDamper = 50.0f, maximumForce = 1e20f };
            joint.breakForce = 1234.0f;
            Simulate();
            vessel = OnlyVessel();
            edge = State->VesselSet.Edges[vessel.FirstPart + 1];
            Assert.That(vessel.Id, Is.EqualTo(id), "a new drive is no new structure");
            Assert.That(edge.SwingStiffness, Is.EqualTo(5000.0).Within(1e-3));
            Assert.That(edge.SwingDamping, Is.EqualTo(50.0).Within(1e-3));
            Assert.That(edge.BreakForce, Is.EqualTo(1234.0).Within(1e-3));
        }

        /// <summary>The rotation from a Rigidbody's motion frame to the world: its rotation, then its inertia's.</summary>
        private static Quaternion MotionFrame(Rigidbody body)
        {
            return body.rotation * body.inertiaTensorRotation;
        }

        [Test]
        public void AnchorsAndRestAreInTheMotionFrames()
        {
            Rigidbody parent = Box(new Vector3(0.0f, 10.0f, 0.0f), new Vector3(2.0f, 1.0f, 0.5f)).AddComponent<Rigidbody>();
            Rigidbody child = Box(new Vector3(1.6f, 10.4f, 0.3f), new Vector3(0.4f, 1.5f, 0.7f)).AddComponent<Rigidbody>();
            parent.transform.rotation = Quaternion.Euler(10.0f, 40.0f, -20.0f);
            child.transform.rotation = Quaternion.Euler(-35.0f, 5.0f, 60.0f);
            parent.useGravity = child.useGravity = false;
            parent.centerOfMass = new Vector3(0.3f, -0.1f, 0.05f);
            child.centerOfMass = new Vector3(-0.1f, 0.2f, 0.1f);
            parent.inertiaTensorRotation = Quaternion.Euler(0.0f, 0.0f, 30.0f);
            child.inertiaTensorRotation = Quaternion.Euler(20.0f, 10.0f, 0.0f);
            Vector3 point = new Vector3(0.9f, 10.2f, 0.1f);
            ConfigurableJoint joint = SurfaceJoint(child, parent, point, 1000.0f);
            UnityEngine.Physics.SyncTransforms();
            Simulate();

            VesselRecord vessel = OnlyVessel();
            List<Rigidbody> parts = VesselsOf(new List<Rigidbody> { parent, child })[0];
            VesselEdge edge = State->VesselSet.Edges[vessel.FirstPart + 1];
            Rigidbody edgeParent = parts[edge.Parent];
            Rigidbody edgeChild = parts[1];

            // Each anchor, put back into the world from its part's motion frame, is the joint's point.
            Vector3 fromParent = edgeParent.worldCenterOfMass + MotionFrame(edgeParent) * (Vector3)(float3)edge.ParentAnchor;
            Vector3 fromChild = edgeChild.worldCenterOfMass + MotionFrame(edgeChild) * (Vector3)(float3)edge.ChildAnchor;
            Vector3 jointPoint = child.transform.TransformPoint(joint.anchor);
            AssertVector(jointPoint, fromParent, 1e-4f, "parent anchor");
            AssertVector(jointPoint, fromChild, 1e-4f, "child anchor");

            // The parts are at rest, so the parent's motion frame turned by the rest orientation is the child's.
            Quaternion rest = new Quaternion((float)edge.Rest.x, (float)edge.Rest.y, (float)edge.Rest.z, (float)edge.Rest.w);
            Quaternion child2 = MotionFrame(edgeParent) * rest;
            Assert.That(Quaternion.Angle(child2, MotionFrame(edgeChild)), Is.LessThan(0.01f), "rest orientation");

            // The edge's twist axis, put into the world from the parent's motion frame, is the joint's x axis.
            float3x3 frame = (float3x3)edge.Frame;
            Vector3 twist = MotionFrame(edgeParent) * (Vector3)frame.c0;
            Vector3 jointAxis = child.transform.TransformDirection(joint.axis).normalized;
            Assert.That(Vector3.Angle(twist, jointAxis), Is.LessThan(0.05f), "twist axis");
        }

        /// <summary>How many of the store's joints are broken.</summary>
        private static int BrokenJoints()
        {
            NativeStore.WorldData* store = State->Store;
            int broken = 0;
            for (uint i = 0; i < store->jointCount; i++)
            {
                broken += (store->joints[i]->state & NativeStore.JointState.Broken) != 0 ? 1 : 0;
            }

            return broken;
        }

        [Test]
        public void AnEdgesLoadReachesItsJointsAndBreaksThem()
        {
            List<Rigidbody> bodies = Column(2);
            ConfigurableJoint[] joints = bodies[1].GetComponents<ConfigurableJoint>();
            foreach (ConfigurableJoint joint in joints)
            {
                joint.breakForce = 100.0f;
            }

            Simulate();
            VesselRecord vessel = OnlyVessel();
            VesselSet set = State->VesselSet;
            List<Rigidbody> parts = VesselsOf(bodies)[0];
            int childPart = vessel.FirstPart + 1;

            // The parent pulls the child up with 60 N. The three joints share it, and each carries 20 N, under its limit.
            var pull = new double3(0.0, 60.0, 0.0);
            set.EdgeLoads[childPart] = new VesselEdgeLoad { Force = pull };
            State->ReadVesselLoads(Step);
            Vector3 onChild = parts[1] == bodies[1] ? (Vector3)(float3)pull : -(Vector3)(float3)pull;
            foreach (ConfigurableJoint joint in joints)
            {
                // A joint reports what acts on its own body, and every joint is on the lower part.
                AssertVector(onChild / 3.0f, joint.currentForce, 1e-3f, "each joint's share");
            }

            Assert.That(BrokenJoints(), Is.Zero, "nothing breaks under its limit");

            // Far past the limit, the smoothed load passes it at once, and all three joints break together.
            set.EdgeLoads[childPart] = new VesselEdgeLoad { Force = new double3(0.0, 60000.0, 0.0) };
            State->ReadVesselLoads(Step);
            Assert.That(BrokenJoints(), Is.EqualTo(3), "the edge's joints broke");
            Assert.That(State->Store->jointBreakCount, Is.EqualTo(3), "the breaks are reported");

            // The broken joints are no edge any more. The edge stays in the vessel as a tombstone for a while, and then
            // the column comes apart into two parts that are no vessel.
            Simulate();
            Assert.That(State->VesselSet.Edges[childPart].State, Is.EqualTo(VesselEdgeState.Broken), "the edge is broken in place");
            Simulate(VesselPipeline.TombstoneSteps);
            Assert.That(State->VesselSet.VesselCount, Is.Zero);
        }

        [Test]
        public void OverloadedEdgesBreakTheirJointsAmongManyJoints()
        {
            // Twelve joints exercise both the eight-wide break flag loop and its scalar tail.
            List<Rigidbody> bodies = Column(5);
            foreach (ConfigurableJoint joint in Object.FindObjectsByType<ConfigurableJoint>(FindObjectsSortMode.None))
            {
                joint.breakForce = 1e6f;
            }

            Simulate();
            VesselRecord vessel = OnlyVessel();
            Assert.That(State->Store->jointCount, Is.EqualTo(12), "the column's joints");
            Assert.That(BrokenJoints(), Is.Zero, "nothing breaks before the loads");
            VesselSet set = State->VesselSet;
            List<Rigidbody> parts = VesselsOf(bodies)[0];
            var tear = new VesselEdgeLoad { Force = new double3(0.0, 1e12, 0.0) };
            set.EdgeLoads[vessel.FirstPart + parts.IndexOf(bodies[2])] = tear;
            set.EdgeLoads[vessel.FirstPart + parts.IndexOf(bodies[4])] = tear;
            State->ReadVesselLoads(Step);
            Assert.That(BrokenJoints(), Is.EqualTo(6), "the two edges' joints broke");
            Assert.That(State->Store->jointBreakCount, Is.EqualTo(6), "the breaks are reported");

            // Each broken joint holds the third part or the fifth part to the part above it.
            NativeStore.WorldData* store = State->Store;
            for (uint i = 0; i < store->jointCount; i++)
            {
                NativeStore.JointData* joint = store->joints[i];
                bool broken = (joint->state & NativeStore.JointState.Broken) != 0;
                ulong lower = joint->body0->index == joint->body1->index ? 0 : math.max(joint->body0->index, joint->body1->index);
                Assert.That(broken, Is.EqualTo(lower == 2 || lower == 4), $"joint {i} between store bodies {joint->body0->index} and {joint->body1->index}");
            }
        }

        [Test]
        public void AGroupsFrameFromTheSolvedMotionIsTheWrittenBackFrame()
        {
            // A tilted column that spins, so that every pose has a rotation that is not exact in floats.
            List<Rigidbody> bodies = Column(3);
            Quaternion tilt = Quaternion.Euler(23.0f, 41.0f, 7.0f);
            foreach (Rigidbody body in bodies)
            {
                body.transform.SetPositionAndRotation(tilt * body.position, tilt);
                body.angularVelocity = new Vector3(0.3f, 1.7f, -0.9f);
            }

            Simulate(5);
            OnlyVessel();

            // The leader frame read from solved motion must match the written-back pose bit for bit.
            NativeStore.WorldData* store = State->Store;
            PhysicsWorld world = State->World;
            int checkedJoints = 0;
            for (uint i = 0; i < store->jointCount; i++)
            {
                NativeStore.JointData* joint = store->joints[i];
                quaternion fromMotion = VesselLoadsJob.LeaderFrame(joint, world.MotionDatas, State->BuiltIndices.AsArray());
                quaternion writtenBack = JointConstraints.WorldRotation0(joint);
                Assert.That(fromMotion.value.Equals(writtenBack.value), Is.True, $"joint {i}: {fromMotion} and {writtenBack}");
                checkedJoints++;
            }

            Assert.That(checkedJoints, Is.EqualTo(6), "the column's joints");
        }

        [Test]
        public void AStrutsLoadActsOnPartB()
        {
            List<Rigidbody> bodies = Column(3);
            ConfigurableJoint strut = Strut(bodies[2], bodies[0]);
            Simulate();
            VesselRecord vessel = OnlyVessel();
            VesselSet set = State->VesselSet;
            VesselClosure closure = set.Closures[vessel.FirstClosure];
            List<Rigidbody> parts = VesselsOf(bodies)[0];
            var force = new double3(5.0, 0.0, 0.0);
            set.ClosureLoads[vessel.FirstClosure] = new VesselEdgeLoad { Force = force };
            State->ReadVesselLoads(Step);

            // The strut's owner is the lowest part. The load acts on part B, so the owner feels it if it is part B.
            Vector3 expected = parts[closure.PartB] == bodies[2] ? (Vector3)(float3)force : -(Vector3)(float3)force;
            AssertVector(expected, strut.currentForce, 1e-4f, "the strut's force");
        }

        [Test]
        public void WithVesselsOffNothingIsCompiled()
        {
            VesselPipeline.Enabled = false;
            Column(3);
            Simulate();
            Assert.That(State->VesselSet.IsCreated, Is.False);
            int constrained = 0;
            foreach (Unity.Physics.Joint joint in State->World.Joints)
            {
                constrained += joint.Constraints.Length > 0 ? 1 : 0;
            }

            Assert.That(constrained, Is.GreaterThanOrEqualTo(2), "both stack joints are built");
        }
    }
}
