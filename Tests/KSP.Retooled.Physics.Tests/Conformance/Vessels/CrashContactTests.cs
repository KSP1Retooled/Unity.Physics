using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using CollisionLog = KSP.Retooled.Physics.Tests.Conformance.Contacts.CollisionEventConformanceTests.CollisionLog;
using System.Collections.Generic;
using KSP.Retooled.Physics.Interop;
using NUnit.Framework;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Configuration;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// A part that hits the ground faster than its crash speed is destroyed by the game in the collision callback.
    /// Its contact must not stop the vessel behind it in the step before that callback.
    /// </summary>
    public class CrashContactTests : PhysicsScene
    {
        private const int Parts = 10;
        private const float PartHeight = 1.0f;
        private const float PartMass = 2.0f;
        private const float Speed = 44.0f;

        // The column starts this far above the ground. It travels 0.88 m in a step, so the second step begins 0.12 m
        // above the ground, and without a limit the contact stops the column in that step.
        private const float StartGap = 1.0f;

        private bool wasEnabled;
        private bool reuseCallbacks;

        [SetUp]
        public void TurnVesselsOn()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Crash speeds are part of the Unity Physics backend.");
            }

            EnvironmentSwitches.Read();
            wasEnabled = VesselSwitches.Enabled;
            VesselSwitches.Enabled = true;
            VesselSwitches.MinimumParts = 2;
            reuseCallbacks = UnityEngine.Physics.reuseCollisionCallbacks;
            UnityEngine.Physics.reuseCollisionCallbacks = false;
        }

        [TearDown]
        public void RestoreSwitches()
        {
            VesselSwitches.Enabled = wasEnabled;
            VesselSwitches.MinimumParts = 0;
            UnityEngine.Physics.reuseCollisionCallbacks = reuseCallbacks;
        }

        private static void StackJoint(Rigidbody host, Rigidbody target, Vector3 node)
        {
            for (int i = 0; i < 3; i++)
            {
                Vector3 anchor = node + Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(0.3f, 0.0f, 0.0f);
                ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = target;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = host.transform.InverseTransformPoint(anchor);
                joint.connectedAnchor = target.transform.InverseTransformPoint(anchor);
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = ConfigurableJointMotion.Locked;
                joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
                joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
                joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularYZDrive = new JointDrive { positionSpring = 20000.0f, positionDamper = 200.0f, maximumForce = 1e20f };
            }
        }

        /// <summary>
        /// A weightless column of jointed parts above the ground, listed from the top down and falling at
        /// <see cref="Speed"/>. The bottom part logs its collisions.
        /// </summary>
        private List<Rigidbody> FallingColumn(float crashSpeed, out CollisionLog log, bool newShape = false)
        {
            Ground();
            float top = StartGap + Parts * PartHeight;
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < Parts; i++)
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

            // The crash speed is set before the backend has a shape for the collider.
            Rigidbody bottom = bodies[Parts - 1];
            if (crashSpeed > 0.0f)
            {
                Assert.That(CrashContacts.SetCrashSpeed(bottom.GetComponent<Collider>(), crashSpeed), Is.True, "crash speed set");
            }

            Simulate();
            if (newShape)
            {
                ReplaceShape(bottom.GetComponent<Collider>());
            }

            log = bottom.gameObject.AddComponent<CollisionLog>();
            foreach (Rigidbody body in bodies)
            {
                body.linearVelocity = new Vector3(0.0f, -Speed, 0.0f);
            }

            return bodies;
        }

        private static unsafe Shape* ShapeOf(Collider collider)
        {
            return GroupTable.ShapeOf(NativeLibrary.RetooledPhysics_Globals(), EntityId.ToULong(collider.GetEntityId()));
        }

        /// <summary>Disables and enables a collider with a step between, so that the backend makes it a new shape.</summary>
        private static unsafe void ReplaceShape(Collider collider)
        {
            collider.enabled = false;
            Simulate();
            Assert.That(ShapeOf(collider) == null, "the disabled collider has no shape");
            collider.enabled = true;
            Simulate();
            Assert.That(ShapeOf(collider) != null, "the enabled collider has a shape");
        }

        /// <summary>Steps until the bottom part reports its collision, and returns the number of steps taken.</summary>
        private static int SimulateToImpact(CollisionLog log)
        {
            for (int step = 1; step <= 10; step++)
            {
                Simulate();
                if (log.Enters.Count > 0)
                {
                    return step;
                }
            }

            Assert.Fail("the column never reported a collision");
            return 0;
        }

        [Test]
        public void WithoutACrashSpeedTheFirstPartStopsTheColumn()
        {
            List<Rigidbody> bodies = FallingColumn(0.0f, out CollisionLog log);

            SimulateToImpact(log);

            Assert.That(log.Enters[0].relativeVelocity.magnitude, Is.EqualTo(Speed).Within(1.0f));
            Assert.That(bodies[0].linearVelocity.y, Is.GreaterThan(-5.0f), "the top part's speed after the impact step");
        }

        [Test]
        public void ADoomedPartTakesOnlyItsOwnMomentumFromTheColumn()
        {
            List<Rigidbody> bodies = FallingColumn(20.0f, out CollisionLog log);
            log.DeactivateOnEnter = true;

            SimulateToImpact(log);

            // The part may pass on its own momentum, which slows the ten equal parts by a tenth.
            float slowest = -Speed * (1.0f - 1.0f / Parts);
            Assert.That(log.Enters[0].relativeVelocity.magnitude, Is.EqualTo(Speed).Within(1.0f), "the report keeps the impact speed");
            Assert.That(log.gameObject.activeSelf, Is.False);
            Assert.That(bodies[0].linearVelocity.y, Is.LessThan(slowest + 1.0f), "the top part's speed after the impact step");
            Assert.That(bodies[Parts - 2].linearVelocity.y, Is.LessThan(slowest + 1.0f), "the next part's speed after the impact step");
        }

        [Test]
        public void ACrashSpeedOutlivesTheCollidersShape()
        {
            List<Rigidbody> bodies = FallingColumn(20.0f, out CollisionLog log, newShape: true);
            log.DeactivateOnEnter = true;

            SimulateToImpact(log);

            Assert.That(bodies[0].linearVelocity.y, Is.LessThan(-Speed * (1.0f - 1.0f / Parts) + 1.0f), "the top part's speed after the impact step");
        }

        [Test]
        public unsafe void TheBackendForgetsTheCrashSpeedOfADestroyedCollider()
        {
            StoreGlobals* globals = NativeLibrary.RetooledPhysics_Globals();
            Collider collider = Box(Vector3.up, Vector3.one).GetComponent<Collider>();
            ulong engineId = EntityId.ToULong(collider.GetEntityId());
            CrashContacts.SetCrashSpeed(collider, 20.0f);
            Simulate();

            Object.DestroyImmediate(collider.gameObject);
            Simulate();
            Assert.That(OrphanIds(globals), Has.Member(engineId), "the destroyed collider's speed waits for the check");

            CollisionGroups.ForgetDestroyedColliders(globals);
            Assert.That(OrphanIds(globals), Has.No.Member(engineId));
        }

        private static unsafe List<ulong> OrphanIds(StoreGlobals* globals)
        {
            var ids = new ulong[IgnoreTable.GroupsOf(globals)->Orphans + 1];
            int count;
            fixed (ulong* first = ids)
            {
                count = GroupTable.ListOrphans(globals, first, ids.Length);
            }

            return new List<ulong>(ids).GetRange(0, count);
        }

        [Test]
        public void BelowItsCrashSpeedAPartStopsTheColumn()
        {
            List<Rigidbody> bodies = FallingColumn(100.0f, out CollisionLog log);

            SimulateToImpact(log);

            Assert.That(bodies[0].linearVelocity.y, Is.GreaterThan(-5.0f), "the top part's speed after the impact step");
        }

        [Test]
        public void AFastPartThatTheGameKeepsComesToRestOnTheGround()
        {
            List<Rigidbody> bodies = FallingColumn(20.0f, out CollisionLog log);
            Rigidbody bottom = bodies[Parts - 1];

            SimulateToImpact(log);

            // Gravity brings the column back down after the push-out, so that it has a resting height.
            foreach (Rigidbody body in bodies)
            {
                body.useGravity = true;
            }

            float lowest = bottom.position.y;
            for (int step = 0; step < 300; step++)
            {
                Simulate();
                lowest = Mathf.Min(lowest, bottom.position.y);
            }

            // The limit lets the part sink for one step, which is 0.88 m of travel.
            Assert.That(lowest, Is.GreaterThan(PartHeight * 0.5f - Speed * Step - 0.1f), "the deepest the bottom part sank");
            Assert.That(bottom.position.y, Is.EqualTo(PartHeight * 0.5f).Within(0.1f), "the bottom part's resting height");
            Assert.That(bodies[0].linearVelocity.magnitude, Is.LessThan(0.5f), "the top part's speed at rest");
        }
    }
}
