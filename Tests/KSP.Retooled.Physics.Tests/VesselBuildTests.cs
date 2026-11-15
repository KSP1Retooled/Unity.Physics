using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using Unity.Physics;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The backend's side of persistent vessels: which vessels it compiles from the joints, the edges and closures that it
    /// hands to the step, the Unity Physics joints that it still builds, and how it turns the loads that the solve writes
    /// into the joints' forces and breaks. These tests read what the backend built, so they don't need the solver's side.
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
        }

        [TearDown]
        public void RestoreSwitch()
        {
            VesselPipeline.Enabled = wasEnabled;
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
        public void AVesselsJointsBuildOnlyCollisionFilters()
        {
            Column(4);
            Simulate();

            // The stack joints don't let their bodies collide, so each of the three units keeps one joint without constraints.
            NativeArrayCheck(State->World.Joints, 3);
        }

        private static void NativeArrayCheck(Unity.Collections.NativeArray<Unity.Physics.Joint> joints, int expected)
        {
            Assert.That(joints.Length, Is.EqualTo(expected), "built joints");
            foreach (Unity.Physics.Joint joint in joints)
            {
                Assert.That(joint.Constraints.Length, Is.Zero, "a vessel's joint keeps no constraints");
                Assert.That(joint.EnableCollision, Is.False);
            }
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
            ulong unstrutted = OnlyVessel().Id;
            Assert.That(unstrutted, Is.Not.EqualTo(strutted).And.Not.EqualTo(first), "Ids are never reused");

            // Decoupling: the joints between parts 1 and 2 go, and the column becomes two vessels.
            foreach (ConfigurableJoint joint in bodies[2].GetComponents<ConfigurableJoint>())
            {
                Object.DestroyImmediate(joint);
            }

            Simulate();
            List<List<Rigidbody>> vessels = VesselsOf(bodies);
            Assert.That(vessels.Count, Is.EqualTo(2));
            for (int v = 0; v < 2; v++)
            {
                Assert.That(State->VesselSet.Vessels[v].Id, Is.Not.EqualTo(unstrutted));
            }
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

            // The broken joints are no edge any more, so the column comes apart into two parts that are no vessel.
            Simulate();
            Assert.That(State->VesselSet.VesselCount, Is.Zero);
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
