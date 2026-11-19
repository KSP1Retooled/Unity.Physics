using System.Collections.Generic;
using System.Diagnostics;
using TestScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using NUnit.Framework;
using Unity.Collections;
using UnityEngine;
using EnginePhysics = UnityEngine.Physics;
using Object = UnityEngine.Object;

namespace KSP.Retooled.Physics.Tests.Conformance.Articulations
{
    public sealed class ArticulationConformanceTests : TestScene
    {
        [Test]
        public void PendulumMatchesSmallStepReference()
        {
            ArticulationBody root = Root(true);
            ArticulationBody link = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            const int steps = 25;
            float position = 0f;
            float velocity = 0f;
            for (int i = 0; i < steps; i++)
            {
                velocity += Step * (-Gravity * Mathf.Cos(position) / 1.25f);
                position += Step * velocity;
                Simulate();
            }

            Record("pendulum", link.jointPosition[0], link.jointVelocity[0], AnchorError(link));
            Assert.That(link.jointPosition[0], Is.EqualTo(position).Within(0.035f));
            Assert.That(link.jointVelocity[0], Is.EqualTo(velocity).Within(0.12f));
            Assert.That(AnchorError(link), Is.LessThan(0.0003f));
        }

        [Test]
        public void LongChainKeepsItsAnchorsUnderGravity()
        {
            ArticulationBody[] chain = Chain(100, 0.15f);
            Simulate(100);
            float maxError = 0f;
            for (int i = 1; i < chain.Length; i++)
            {
                maxError = Mathf.Max(maxError, AnchorError(chain[i]));
            }

            Vector3 tip = chain[chain.Length - 1].transform.position;
            Record("long-chain", tip.x, tip.y, tip.z, maxError);
            Assert.That(maxError, Is.LessThan(0.003f));
            Assert.That(tip.y, Is.LessThan(-1f));
        }

        [Test]
        public void FixedLinkHasNoCoordinatesAndHoldsItsOffset()
        {
            ArticulationBody root = Root(true);
            ArticulationBody link = Link(root, Vector3.forward, ArticulationJointType.FixedJoint);
            Simulate(30);
            Record("fixed", link.dofCount, AnchorError(link), link.transform.position.z);
            Assert.That(link.dofCount, Is.Zero);
            AssertVector(Vector3.forward, link.transform.position, 0.0003f);
        }

        [TestCase(ArticulationJointType.PrismaticJoint, 0.4f, 0.2f)]
        [TestCase(ArticulationJointType.RevoluteJoint, 35f, 15f)]
        [TestCase(ArticulationJointType.SphericalJoint, 35f, 15f)]
        public void PositionDriveRespectsJointLimits(ArticulationJointType type, float target, float upper)
        {
            ArticulationBody root = Root(true);
            ArticulationBody link = Link(root, Vector3.forward, type, type == ArticulationJointType.SphericalJoint ? 1 : 0);
            link.useGravity = false;
            switch (type)
            {
                case ArticulationJointType.PrismaticJoint:
                    link.linearLockX = ArticulationDofLock.LimitedMotion;
                    link.xDrive = Drive(target, -upper, upper);
                    break;
                case ArticulationJointType.RevoluteJoint:
                    link.twistLock = ArticulationDofLock.LimitedMotion;
                    link.xDrive = Drive(target, -upper, upper);
                    break;
                case ArticulationJointType.SphericalJoint:
                    link.twistLock = ArticulationDofLock.LimitedMotion;
                    link.xDrive = Drive(target, -upper, upper);
                    break;
            }

            Simulate(120);
            float expected = type == ArticulationJointType.PrismaticJoint ? upper : upper * Mathf.Deg2Rad;
            Record(type.ToString(), link.dofCount, link.jointPosition[0], link.jointVelocity[0]);
            Assert.That(link.dofCount, Is.EqualTo(1));
            Assert.That(link.jointPosition[0], Is.EqualTo(expected).Within(0.025f));
            Assert.That(Mathf.Abs(link.jointVelocity[0]), Is.LessThan(0.08f));
        }

        [TestCase(ArticulationDriveType.Force)]
        [TestCase(ArticulationDriveType.Acceleration)]
        [TestCase(ArticulationDriveType.Target)]
        public void ImmovableArmHoldsPosition(ArticulationDriveType driveType)
        {
            ArticulationBody root = Root(true);
            ArticulationBody link = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            ArticulationDrive drive = Drive(30f, -180f, 180f);
            drive.stiffness = 1000f;
            drive.damping = 80f;
            drive.driveType = driveType;
            link.xDrive = drive;
            Simulate(150);
            Record(driveType.ToString(), link.jointPosition[0], link.jointVelocity[0], AnchorError(link));
            AssertVector(Vector3.zero, root.transform.position, 0.0001f);
            Assert.That(link.jointPosition[0], Is.EqualTo(Mathf.PI / 6f).Within(0.025f));
            Assert.That(Mathf.Abs(link.jointVelocity[0]), Is.LessThan(0.05f));
        }

        [TestCase(ArticulationDriveType.Force)]
        [TestCase(ArticulationDriveType.Velocity)]
        public void VelocityDriveTracksItsTarget(ArticulationDriveType driveType)
        {
            ArticulationBody root = Root(true);
            ArticulationBody link = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            link.useGravity = false;
            link.xDrive = new ArticulationDrive
            {
                targetVelocity = 40f,
                damping = 25f,
                forceLimit = 10000f,
                driveType = driveType,
            };
            Simulate(60);
            Record(driveType + "-velocity", link.jointPosition[0], link.jointVelocity[0]);
            Assert.That(link.jointVelocity[0], Is.EqualTo(40f * Mathf.Deg2Rad).Within(0.05f));
        }

        [Test]
        public void LowForceLimitCapsDriveAcceleration()
        {
            ArticulationBody root = Root(true);
            ArticulationBody link = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            link.useGravity = false;
            ArticulationDrive drive = Drive(90f, -180f, 180f);
            drive.forceLimit = 0.5f;
            link.xDrive = drive;
            Simulate(10);
            Record("limited-drive", link.jointPosition[0], link.jointVelocity[0], link.driveForce[0]);
            Assert.That(link.jointVelocity[0], Is.EqualTo(0.08f).Within(0.025f));
            Assert.That(Mathf.Abs(link.driveForce[0]), Is.EqualTo(0.5f).Within(0.08f));
        }

        [Test]
        public void PackedStateFollowsLinkAndAxisOrder()
        {
            ArticulationBody root = Root(true);
            ArticulationBody first = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            ArticulationBody second = Link(first, Vector3.forward, ArticulationJointType.SphericalJoint, 7);
            first.useGravity = second.useGravity = false;
            Simulate();

            var starts = new List<int>();
            root.GetDofStartIndices(starts);
            Assert.That(starts, Is.EqualTo(new[] { 0, 0, 1 }));
            root.SetJointPositions(new List<float> { 0.2f, 0.1f, 0.3f, -0.4f });
            root.SetJointVelocities(new List<float> { 0.5f, 0.1f, 0.6f, -0.7f });
            var positions = new List<float>();
            var velocities = new List<float>();
            Assert.That(root.GetJointPositions(positions), Is.EqualTo(4));
            Assert.That(root.GetJointVelocities(velocities), Is.EqualTo(4));
            Record("packed", positions[0], positions[1], positions[2], positions[3], velocities[0], velocities[1], velocities[2], velocities[3]);
            Assert.That(first.jointPosition[0], Is.EqualTo(0.2f).Within(0.0001f));
            Assert.That(second.dofCount, Is.EqualTo(3));
            Assert.That(second.jointPosition[2], Is.EqualTo(-0.4f).Within(0.0001f));
            Assert.That(second.jointVelocity[1], Is.EqualTo(0.6f).Within(0.0001f));

            var forces = new List<float> { 1f, 2f, 3f, 4f };
            root.SetJointForces(forces);
            var read = new List<float>();
            Assert.That(root.GetJointForces(read), Is.EqualTo(forces.Count));
            Assert.That(read, Is.EqualTo(forces));
            Assert.That(root.GetJointAccelerations(read), Is.EqualTo(forces.Count));
        }

        [Test]
        public void FloatingBaseBulkAndJacobian()
        {
            ArticulationBody root = Root(false, Vector3.one);
            ArticulationBody link = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            root.useGravity = link.useGravity = false;
            Simulate();
            link.jointPosition = new ArticulationReducedSpace(0.1f);
            link.jointVelocity = new ArticulationReducedSpace(0.2f);
            root.linearVelocity = new Vector3(0.3f, 0.4f, 0.5f);
            var values = new List<float>();
            var starts = new List<int>();
            var jacobian = new ArticulationJacobian();
            root.GetDofStartIndices(starts);
            root.GetDenseJacobian(ref jacobian);
            int count = root.GetJointPositions(values);
            Assert.That(count, Is.EqualTo(7));
            Assert.That(starts, Is.EqualTo(new[] { 0, 6 }));
            Assert.That(values.GetRange(0, 6), Is.All.Zero);
            Assert.That(values[6], Is.EqualTo(0.1f).Within(0.0001f));
            TestContext.WriteLine($"{EnginePhysics.GetCurrentIntegrationInfo().name} floating positions: {string.Join(", ", values)}; starts: {string.Join(", ", starts)}");
            int velocityCount = root.GetJointVelocities(values);
            Assert.That(velocityCount, Is.EqualTo(7));
            Assert.That(values.GetRange(0, 6), Is.All.Zero);
            Assert.That(values[6], Is.EqualTo(0.2f).Within(0.0001f));
            root.SetJointPositions(new List<float> { 0f, 0f, 0f, 0f, 0f, 0f, -0.15f });
            Assert.That(link.jointPosition[0], Is.EqualTo(-0.15f).Within(0.0001f));
            TestContext.WriteLine($"floating velocities: {string.Join(", ", values)}");
            Record("floating", count, velocityCount, jacobian.rows, jacobian.columns);
            Assert.That(jacobian.rows, Is.EqualTo(12));
            Assert.That(jacobian.columns, Is.EqualTo(7));
            Assert.That(jacobian[0, 0], Is.EqualTo(1f).Within(0.001f));
        }

        [Test]
        public void InverseDynamicsOfOneLink()
        {
            ArticulationBody root = Root(true);
            ArticulationBody link = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            link.mass = 2f;
            Simulate();
            link.jointPosition = new ArticulationReducedSpace(0f);
            link.jointVelocity = new ArticulationReducedSpace(0f);
            Simulate();
            link.jointPosition = new ArticulationReducedSpace(0f);
            link.jointVelocity = new ArticulationReducedSpace(0f);

            var gravity = new List<float>();
            var coriolis = new List<float>();
            var external = new List<float>();
            var jacobian = new ArticulationJacobian();
            Assert.That(root.GetJointGravityForces(gravity), Is.EqualTo(1));
            Assert.That(root.GetJointCoriolisCentrifugalForces(coriolis), Is.EqualTo(1));
            link.AddForce(new Vector3(0f, 3f, 0f));
            Assert.That(root.GetJointExternalForces(external, Step), Is.EqualTo(1));
            ArticulationReducedSpace force = link.GetJointForcesForAcceleration(new ArticulationReducedSpace(2f));
            root.GetDenseJacobian(ref jacobian);
            Record("inverse", gravity[0], coriolis[0], external[0], force[0], jacobian.rows, jacobian.columns,
                jacobian[1, 0], jacobian[3, 0]);
            Assert.That(gravity[0], Is.EqualTo(2f * Gravity).Within(0.04f));
            Assert.That(coriolis[0], Is.EqualTo(0f).Within(0.005f));
            Assert.That(external[0], Is.EqualTo(3f).Within(0.03f));
            // The PhysX backend returns the requested acceleration in this scene. The Retooled result
            // includes the inertia of 0.25 kg m² and the 2 kg mass at a 1 m radius.
            float expectedForce = EnginePhysics.GetCurrentIntegrationInfo().name == "PhysX" ? 2f : 4.5f;
            Assert.That(force[0], Is.EqualTo(expectedForce).Within(0.06f));
            Assert.That(jacobian.rows, Is.EqualTo(6));
            Assert.That(jacobian.columns, Is.EqualTo(1));
            Assert.That(jacobian[1, 0], Is.EqualTo(-1f).Within(0.003f));
            Assert.That(jacobian[3, 0], Is.EqualTo(1f).Within(0.003f));
        }

        [Test]
        public void TwoLinkArmReportsCentrifugalTorque()
        {
            ArticulationBody root = Root(true);
            ArticulationBody first = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            ArticulationBody second = Link(first, Vector3.down, ArticulationJointType.RevoluteJoint);
            first.useGravity = second.useGravity = false;
            Simulate();
            first.jointVelocity = new ArticulationReducedSpace(1f);
            var coriolis = new List<float>();
            Assert.That(root.GetJointCoriolisCentrifugalForces(coriolis), Is.EqualTo(2));
            Record("centrifugal", coriolis[0], coriolis[1]);
            Assert.That(coriolis[0], Is.EqualTo(0f).Within(0.02f));
            Assert.That(coriolis[1], Is.EqualTo(1f).Within(0.02f));
        }

        [Test]
        public void LinkCollidesWithGroundAndCanBeQueried()
        {
            Ground();
            ArticulationBody root = Root(false, new Vector3(0f, 2f, 0f));
            ArticulationBody link = Link(root, Vector3.zero, ArticulationJointType.FixedJoint);
            BoxCollider collider = link.gameObject.AddComponent<BoxCollider>();
            collider.providesContacts = true;
            int enters = 0;
            void CountContacts(PhysicsScene scene, NativeArray<ContactPairHeader>.ReadOnly contacts)
            {
                foreach (ContactPairHeader header in contacts)
                {
                    for (int i = 0; i < header.pairCount; i++)
                    {
                        ContactPair pair = header.GetContactPair(i);
                        if (pair.isCollisionEnter && (pair.collider == collider || pair.otherCollider == collider))
                        {
                            enters++;
                        }
                    }
                }
            }

            EnginePhysics.ContactEvent += CountContacts;
            try
            {
                Simulate(120);
            }
            finally
            {
                EnginePhysics.ContactEvent -= CountContacts;
            }
            bool hit = EnginePhysics.Raycast(link.transform.position + Vector3.up * 4f, Vector3.down, out RaycastHit result, 10f);
            Record("ground", root.transform.position.y, link.transform.position.y, AnchorError(link), hit ? result.distance : -1f, enters);
            Assert.That(hit, Is.True);
            Assert.That(result.collider, Is.SameAs(collider));
            Assert.That(result.articulationBody, Is.SameAs(link));
            Assert.That(enters, Is.GreaterThan(0));
            Assert.That(link.transform.position.y, Is.EqualTo(0.5f).Within(0.08f));
            Assert.That(AnchorError(link), Is.LessThan(0.0005f));
        }

        [Test]
        public void NonAdjacentLinksCollide()
        {
            ArticulationBody root = Root(true);
            ArticulationBody left = Link(root, Vector3.left * 0.8f, ArticulationJointType.PrismaticJoint);
            ArticulationBody right = Link(root, Vector3.right * 0.8f, ArticulationJointType.PrismaticJoint);
            left.useGravity = right.useGravity = false;
            left.linearLockX = right.linearLockX = ArticulationDofLock.FreeMotion;
            left.gameObject.AddComponent<BoxCollider>();
            right.gameObject.AddComponent<BoxCollider>();
            Simulate();
            left.jointVelocity = new ArticulationReducedSpace(1f);
            right.jointVelocity = new ArticulationReducedSpace(-1f);
            Simulate(35);
            float separation = right.transform.position.x - left.transform.position.x;
            Record("self-contact", separation, left.jointVelocity[0], right.jointVelocity[0]);
            Assert.That(separation, Is.GreaterThan(0.92f));
        }

        [Test]
        public void SleepWakeAndChangeLinks()
        {
            ArticulationBody root = Root(true);
            ArticulationBody branch = Link(root, Vector3.forward, ArticulationJointType.RevoluteJoint);
            branch.useGravity = false;
            Simulate(35);
            Assert.That(root.IsSleeping(), Is.True);
            root.WakeUp();
            Assert.That(root.IsSleeping(), Is.False);
            root.Sleep();
            Assert.That(branch.IsSleeping(), Is.True);
            branch.AddTorque(Vector3.right, ForceMode.Impulse);
            Simulate();
            Assert.That(root.IsSleeping(), Is.False);
            Assert.That(branch.jointVelocity[0], Is.GreaterThan(0.5f));

            ArticulationBody leaf = Link(branch, Vector3.forward, ArticulationJointType.RevoluteJoint);
            leaf.useGravity = false;
            Simulate();
            var state = new List<float>();
            Assert.That(root.GetJointPositions(state), Is.EqualTo(2));
            Object.DestroyImmediate(leaf.gameObject);
            Simulate();
            Assert.That(root.GetJointPositions(state), Is.EqualTo(1));
            Record("sleep-and-links", branch.jointPosition[0], branch.jointVelocity[0], state.Count);
        }

        [TestCase(7)]
        [TestCase(100)]
        [Explicit("Measures steady-state step cost for both backends")]
        public void MeasureStepCost(int degreesOfFreedom)
        {
            int count = degreesOfFreedom == 7 ? 8 : 100;
            ArticulationBody[] chain = Chain(count, 0.15f);
            const float dt = 0.005f;
            for (int i = 0; i < 300; i++)
            {
                EnginePhysics.Simulate(dt);
            }

            var times = new double[5];
            for (int repeat = 0; repeat < times.Length; repeat++)
            {
                var watch = Stopwatch.StartNew();
                for (int i = 0; i < 400; i++)
                {
                    EnginePhysics.Simulate(dt);
                }

                times[repeat] = watch.Elapsed.TotalMilliseconds * 1000 / 400;
            }

            System.Array.Sort(times);
            TestContext.WriteLine($"{EnginePhysics.GetCurrentIntegrationInfo().name}, {count} links, {count - 1} DOFs, step us {string.Join(", ", times)}");
            Assert.That(AnchorError(chain[chain.Length - 1]), Is.LessThan(0.003f));
        }

        private ArticulationBody[] Chain(int count, float length)
        {
            var chain = new ArticulationBody[count];
            chain[0] = Root(true);
            for (int i = 1; i < chain.Length; i++)
            {
                chain[i] = Link(chain[i - 1], Vector3.forward * length, ArticulationJointType.RevoluteJoint);
            }

            return chain;
        }

        private ArticulationBody Root(bool fixedBase, Vector3 position = default)
        {
            var gameObject = Track(new GameObject("articulation root"));
            gameObject.transform.position = position;
            ArticulationBody body = gameObject.AddComponent<ArticulationBody>();
            body.immovable = fixedBase;
            return body;
        }

        private static ArticulationBody Link(ArticulationBody parent, Vector3 offset, ArticulationJointType type, int sphericalAxes = 7)
        {
            var gameObject = new GameObject("articulation link");
            gameObject.SetActive(false);
            gameObject.transform.SetParent(parent.transform, false);
            gameObject.transform.localPosition = offset;
            ArticulationBody body = gameObject.AddComponent<ArticulationBody>();
            body.jointType = type;
            body.matchAnchors = false;
            body.anchorPosition = -offset;
            body.parentAnchorPosition = Vector3.zero;
            body.anchorRotation = body.parentAnchorRotation = Quaternion.identity;
            body.linearDamping = body.angularDamping = body.jointFriction = 0f;
            body.automaticCenterOfMass = body.automaticInertiaTensor = false;
            body.centerOfMass = Vector3.zero;
            body.inertiaTensor = Vector3.one * 0.25f;
            if (type == ArticulationJointType.RevoluteJoint)
            {
                body.twistLock = ArticulationDofLock.FreeMotion;
            }
            else if (type == ArticulationJointType.PrismaticJoint)
            {
                body.linearLockX = ArticulationDofLock.FreeMotion;
                body.linearLockY = body.linearLockZ = ArticulationDofLock.LockedMotion;
            }

            if (type == ArticulationJointType.SphericalJoint)
            {
                body.twistLock = AxisLock(sphericalAxes, 1);
                body.swingYLock = AxisLock(sphericalAxes, 2);
                body.swingZLock = AxisLock(sphericalAxes, 4);
            }

            gameObject.SetActive(true);
            return body;
        }

        private static ArticulationDofLock AxisLock(int axes, int bit)
        {
            return (axes & bit) == 0 ? ArticulationDofLock.LockedMotion : ArticulationDofLock.FreeMotion;
        }

        private static ArticulationDrive Drive(float target, float lower, float upper)
        {
            return new ArticulationDrive
            {
                target = target,
                lowerLimit = lower,
                upperLimit = upper,
                stiffness = 100f,
                damping = 18f,
                forceLimit = 10000f,
            };
        }

        private static float AnchorError(ArticulationBody link)
        {
            return Vector3.Distance(link.transform.TransformPoint(link.anchorPosition),
                link.transform.parent.TransformPoint(link.parentAnchorPosition));
        }

        private static void Record(string name, params float[] values)
        {
            TestContext.WriteLine($"{EnginePhysics.GetCurrentIntegrationInfo().name} {name} {string.Join(", ", values)}");
        }
    }
}
