using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// What the step keeps from one step to the next while the joints stay as they are: the same scene stepped once as it
    /// is and once with everything worked out again every step moves exactly the same, through joints added, removed,
    /// broken and set up again, bodies made kinematic and moved by hand, and contacts coming and going.
    /// </summary>
    public class PipelineCacheTests : PhysicsScene
    {
        private const int Parts = 6;
        private const float PartHeight = 2.0f;
        private const float PartMass = 4.0f;

        private bool wasEnabled;

        [SetUp]
        public void RememberSwitch()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The caches are part of the Unity Physics backend.");
            }

            wasEnabled = VesselForest.Enabled;

            // The columns here have six parts. Let them compile into vessels when vessels are on.
            VesselPipeline.MinimumParts = 2;
        }

        [TearDown]
        public void RestoreSwitch()
        {
            VesselForest.Enabled = wasEnabled;
            VesselPipeline.MinimumParts = 0;
        }

        /// <summary>A stock part joint: three joints over a stack node, which the step builds as one group.</summary>
        private static void StackJoint(Rigidbody host, Rigidbody target, Vector3 node)
        {
            for (int i = 0; i < 3; i++)
            {
                Vector3 anchor = node + Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(0.5f, 0.0f, 0.0f);
                ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = target;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = host.transform.InverseTransformPoint(anchor);
                joint.connectedAnchor = target != null ? target.transform.InverseTransformPoint(anchor) : anchor;
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
                joint.linearLimit = new SoftJointLimit { limit = 1.0f };
                joint.highAngularXLimit = new SoftJointLimit { limit = 177.0f };
                joint.lowAngularXLimit = new SoftJointLimit { limit = -177.0f };
                joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
                joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
                joint.xDrive = joint.yDrive = joint.zDrive = new JointDrive { positionSpring = 1e20f, maximumForce = 1e20f };
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularXDrive = joint.angularYZDrive = new JointDrive { positionSpring = 20000.0f, maximumForce = 1e20f };
            }
        }

        /// <summary>A strut: one locked joint between two parts that aren't next to each other, which closes a loop.</summary>
        private static ConfigurableJoint Strut(Rigidbody host, Rigidbody target)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            return joint;
        }

        private List<Rigidbody> Column(float bottom, bool hung)
        {
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < Parts; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, bottom + PartHeight * (i + 0.5f), 0.0f), new Vector3(1.0f, PartHeight, 1.0f))
                    .AddComponent<Rigidbody>();
                body.mass = PartMass;
                bodies.Add(body);
                if (i > 0)
                {
                    StackJoint(body, bodies[i - 1], new Vector3(0.0f, bottom + PartHeight * i, 0.0f));
                }
            }

            if (hung)
            {
                StackJoint(bodies[Parts - 1], null, new Vector3(0.0f, bottom + PartHeight * Parts, 0.0f));
            }

            return bodies;
        }

        private static void StepOnce(bool everyStepAfresh)
        {
            if (everyStepAfresh)
            {
                JointTopology.Changed();
            }

            Simulate();
        }

        /// <summary>
        /// A hung column taken through every change the caches have to notice, returning each part's pose at the end. With
        /// <paramref name="everyStepAfresh"/> nothing is kept from one step to the next.
        /// </summary>
        private (Vector3[], Quaternion[]) RunChanges(bool forest, bool everyStepAfresh)
        {
            TearDownScene();
            SetUpScene();
            VesselForest.Enabled = forest;

            // Islands keep their pairs' lock states by their first body, which the run before had too.
            VesselForest.ForgetIslands();
            List<Rigidbody> bodies = Column(5.0f, true);
            bodies[0].linearVelocity = new Vector3(1.0f, 0.0f, 0.5f);
            ConfigurableJoint strut = null;
            FixedJoint weak = null;
            for (int step = 0; step < 120; step++)
            {
                switch (step)
                {
                    case 5:
                        weak = bodies[1].gameObject.AddComponent<FixedJoint>();
                        weak.connectedBody = bodies[4];
                        weak.breakForce = 50.0f;
                        break;
                    case 10:
                        strut = Strut(bodies[0], bodies[3]);
                        break;
                    case 20:
                        Object.DestroyImmediate(strut);
                        break;
                    case 30:
                        bodies[2].isKinematic = true;
                        break;
                    case 40:
                        bodies[2].isKinematic = false;
                        break;
                    case 45:
                        bodies[5].position += new Vector3(0.2f, 0.0f, 0.0f);
                        break;
                    case 55:
                        strut = Strut(bodies[1], bodies[5]);
                        break;
                    case 65:
                        strut.connectedBody = bodies[4];
                        break;
                    case 75:
                        bodies[3].gameObject.SetActive(false);
                        break;
                    case 85:
                        bodies[3].gameObject.SetActive(true);
                        break;
                    case 95:
                        bodies[0].WakeUp();
                        bodies[0].angularVelocity = new Vector3(0.0f, 2.0f, 0.0f);
                        break;
                }

                StepOnce(everyStepAfresh);
            }

            // With vessels on, the stack joints are hard, so the weak joint across them carries almost nothing and holds.
            Assert.That(weak == null, Is.EqualTo(!VesselPipeline.Enabled), "the weak joint breaks only where the stack joints flex");
            return Poses(bodies);
        }

        /// <summary>
        /// A column that moves at 2,000 m/s onto a 5 cm plate. When vessels are on, the column is a vessel, so this
        /// checks that the vessel's contact rows reach one step ahead.
        /// </summary>
        [Test]
        public void AFastColumnDoesNotPassThroughAThinPlate()
        {
            Box(new Vector3(0.0f, -0.025f, 0.0f), new Vector3(20.0f, 0.05f, 20.0f));
            List<Rigidbody> bodies = Column(5.0f, false);
            foreach (Rigidbody body in bodies)
            {
                body.linearVelocity = new Vector3(0.0f, -2000.0f, 0.0f);
            }

            Simulate(20);

            Assert.That(bodies[0].position.y, Is.GreaterThan(0.0f));
        }

        /// <summary>A column dropped onto the ground, where contacts appear on an island whose joints were kept.</summary>
        private (Vector3[], Quaternion[]) RunLanding(bool forest, bool everyStepAfresh)
        {
            TearDownScene();
            SetUpScene();
            VesselForest.Enabled = forest;
            Ground();
            List<Rigidbody> bodies = Column(0.3f, false);
            bodies[Parts - 1].linearVelocity = new Vector3(0.5f, 0.0f, 0.2f);
            for (int step = 0; step < 80; step++)
            {
                StepOnce(everyStepAfresh);
            }

            return Poses(bodies);
        }

        private static (Vector3[], Quaternion[]) Poses(List<Rigidbody> bodies)
        {
            var positions = new Vector3[bodies.Count];
            var rotations = new Quaternion[bodies.Count];
            for (int i = 0; i < bodies.Count; i++)
            {
                positions[i] = bodies[i].position;
                rotations[i] = bodies[i].rotation;
            }

            return (positions, rotations);
        }

        private static void AssertSame((Vector3[], Quaternion[]) kept, (Vector3[], Quaternion[]) afresh)
        {
            for (int i = 0; i < kept.Item1.Length; i++)
            {
                Assert.That(kept.Item1[i], Is.EqualTo(afresh.Item1[i]), $"part {i}'s position");
                Assert.That(kept.Item2[i], Is.EqualTo(afresh.Item2[i]), $"part {i}'s rotation");
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void KeptJointsMoveAsJointsWorkedOutAfresh(bool forest)
        {
            // The direct solver keeps each island's forest, and with locking its pairs' lock states, under the island's first
            // body, which a new scene of the same shape takes over. A run first gives both compared runs the same start.
            RunChanges(forest, true);
            AssertSame(RunChanges(forest, false), RunChanges(forest, true));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AContactOnAKeptIslandLandsAsOnAFreshOne(bool forest)
        {
            // As above: the first scene of this shape lands a few ulps from every later one, kept or afresh.
            RunLanding(forest, true);
            AssertSame(RunLanding(forest, false), RunLanding(forest, true));
        }

        [Test]
        public void EveryChangeToTheJointsMovesTheirVersion()
        {
            Rigidbody first = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            Rigidbody second = DynamicBox(new Vector3(0.0f, 7.0f, 0.0f));
            uint version = JointTopology.Version;

            void Moved(string what)
            {
                // Unity hands some changes to the backend only when it next steps.
                Simulate();
                Assert.That(JointTopology.Version, Is.Not.EqualTo(version), what);
                version = JointTopology.Version;
            }

            ConfigurableJoint joint = first.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = second;
            Moved("adding a joint");

            // A joint free to slide is no part of a group or the forest, so tuning it leaves both alone.
            joint.angularXMotion = ConfigurableJointMotion.Locked;
            Simulate();
            Assert.That(JointTopology.Version, Is.EqualTo(version), "locking a turn of a joint that slides keeps the version");

            joint.xMotion = ConfigurableJointMotion.Locked;
            joint.yMotion = ConfigurableJointMotion.Locked;
            joint.zMotion = ConfigurableJointMotion.Locked;
            Moved("locking the anchors, which makes it hold its bodies as a part joint does");

            joint.axis = Vector3.up;
            Moved("turning its frames");

            joint.targetRotation = Quaternion.Euler(10.0f, 0.0f, 0.0f);
            Moved("giving it a target away from rest, which leaves the forest");

            joint.targetRotation = Quaternion.Euler(20.0f, 0.0f, 0.0f);
            Simulate();
            Assert.That(JointTopology.Version, Is.EqualTo(version), "a servo's next target keeps the version");

            joint.targetRotation = Quaternion.identity;
            Moved("bringing its target back to rest");

            joint.connectedBody = null;
            Moved("joining it to the world");

            second.isKinematic = true;
            Moved("making a body kinematic");

            Object.DestroyImmediate(joint);
            Moved("removing a joint");

            Rigidbody third = DynamicBox(new Vector3(0.0f, 9.0f, 0.0f));
            Moved("adding a body");

            Object.DestroyImmediate(third.gameObject);
            Moved("removing a body");

            second.position += Vector3.one;
            first.linearVelocity = Vector3.one;
            Simulate();
            Assert.That(JointTopology.Version, Is.EqualTo(version), "moving bodies and stepping keep the version");
        }

        /// <summary>
        /// A box rests on the ground, so that the step keeps the impulse of its contact for the next step's warm start.
        /// Then the box is destroyed and a new box is made in its place. The store often gives the new box's shape the
        /// memory of the old box's shape, and the step keeps the impulses by the shapes' addresses. The new box's
        /// contact is new, so it must start without an impulse. The test repeats this, so that the store reuses the
        /// memory at least once, and it counts how often that happened.
        /// </summary>
        [Test]
        public unsafe void ANewShapeInTheMemoryOfARemovedShapeStartsWithoutItsImpulse()
        {
            const int rounds = 20;
            Ground();
            Rigidbody box = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            int reused = 0;
            for (int round = 0; round < rounds; round++)
            {
                Simulate(3);
                var before = new HashSet<(long, long)>();
                foreach (ManifoldRecord record in StepKernel.Current->Manifolds)
                {
                    Assert.That(record.Impulse, Is.GreaterThan(0.0f), "the box rests on the ground");
                    before.Add(((long)record.ShapeA, (long)record.ShapeB));
                }

                Object.DestroyImmediate(box.gameObject);
                box = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
                Simulate();
                Unity.Collections.NativeList<ManifoldRecord> manifolds = StepKernel.Current->Manifolds;
                Assert.That(manifolds.Length, Is.GreaterThan(0), "the new box touches the ground");
                foreach (ManifoldRecord record in manifolds)
                {
                    reused += before.Contains(((long)record.ShapeA, (long)record.ShapeB)) ? 1 : 0;
                    Assert.That(record.WarmImpulse, Is.Zero, $"round {round}: the new box's contact starts without an impulse");
                }
            }

            Debug.Log($"[PipelineCacheTests] a new box's contact had the shapes' addresses of the old box's contact in {reused} of {rounds} rounds");
        }
    }
}
