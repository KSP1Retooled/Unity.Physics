using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Unity.Physics.Store.Versions;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Contacts;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Compare cached and fully rebuilt pipeline results through joint, body and contact changes, requiring
    /// identical motion.
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

            wasEnabled = ForestSwitches.Enabled;

            // The columns here have six parts. Let them compile into vessels when vessels are on.
            VesselSwitches.MinimumParts = 2;
        }

        [TearDown]
        public void RestoreSwitch()
        {
            ForestSwitches.Enabled = wasEnabled;
            VesselSwitches.MinimumParts = 0;
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
        /// Run a hanging column through cache-invalidating changes and return final poses.
        /// everyStepAfresh disables reuse between steps.
        /// </summary>
        private (Vector3[], Quaternion[]) RunChanges(bool forest, bool everyStepAfresh)
        {
            TearDownScene();
            SetUpScene();
            ForestSwitches.Enabled = forest;

            // Islands keep their pairs' lock states by their first body, which the run before had too.
            ForestDiagnostics.ForgetIslands();
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
            Assert.That(weak == null, Is.EqualTo(!VesselSwitches.Enabled), "the weak joint breaks only where the stack joints flex");
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
            ForestSwitches.Enabled = forest;
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

        /// <summary>
        /// Run a bounds-cache comparison scene with landing, manual movement, rotation, collider replacement and
        /// kinematic motion.
        /// Return final poses and sorted query hits.
        /// </summary>
        private ((Vector3[], Quaternion[]) poses, List<string> hits) RunBounds(bool boundsCache)
        {
            TearDownScene();
            SetUpScene();
            StepState.BoundsCacheEnabled = boundsCache;
            ForestDiagnostics.ForgetIslands();
            GameObject ground = Ground();
            List<Rigidbody> bodies = Column(0.3f, false);
            bodies[Parts - 1].linearVelocity = new Vector3(0.5f, 0.0f, 0.2f);
            Rigidbody loose = DynamicBox(new Vector3(3.0f, 2.0f, 0.0f));
            Rigidbody carried = DynamicBox(new Vector3(-3.0f, 1.0f, 0.0f));
            carried.isKinematic = true;
            bodies.Add(loose);
            bodies.Add(carried);

            // The runs make new colliders, so the hits name each collider by its body.
            var names = new Dictionary<Collider, string> { [ground.GetComponent<Collider>()] = "ground" };
            for (int i = 0; i < bodies.Count; i++)
            {
                names[bodies[i].GetComponent<Collider>()] = $"body {i}";
            }

            var hits = new List<string>();
            for (int step = 0; step < 90; step++)
            {
                switch (step)
                {
                    case 10:
                        loose.position += new Vector3(0.5f, 0.0f, 0.0f);
                        break;
                    case 20:
                        loose.GetComponent<BoxCollider>().size = new Vector3(2.0f, 0.5f, 1.0f);
                        break;
                    case 30:
                        loose.rotation = Quaternion.Euler(0.0f, 30.0f, 0.0f);
                        break;
                }

                carried.MovePosition(new Vector3(-3.0f + 0.05f * step, 1.0f, 0.0f));
                StepOnce(false);

                var found = new List<string>();
                foreach (RaycastHit hit in UnityEngine.Physics.RaycastAll(new Vector3(-6.0f, 1.0f, 0.1f), Vector3.right, 12.0f))
                {
                    found.Add($"ray {names[hit.collider]} {hit.point.x:R} {hit.point.y:R} {hit.point.z:R} {hit.distance:R}");
                }

                foreach (Collider collider in UnityEngine.Physics.OverlapBox(new Vector3(0.0f, 1.5f, 0.0f), new Vector3(3.0f, 1.0f, 1.0f)))
                {
                    found.Add($"box {names[collider]}");
                }

                found.Sort(string.CompareOrdinal);
                hits.Add($"step {step}: {string.Join(", ", found)}");
            }

            return (Poses(bodies), hits);
        }

        /// <summary>
        /// Require bitwise-identical motion and query hits with the dynamic bounds cache enabled and disabled.
        /// </summary>
        [Test]
        public void TheBoundsCacheChangesNoPoseAndNoHit()
        {
            // RETOOLED_BOUNDS_CACHE=0 turns the cache off for a whole run, so give the later tests the switch as it was.
            bool wasEnabled = StepState.BoundsCacheEnabled;
            try
            {
                // As above: the first scene of this shape can land a few ulps from every later one.
                RunBounds(false);
                ((Vector3[], Quaternion[]) poses, List<string> hits) cached = RunBounds(true);
                ((Vector3[], Quaternion[]) poses, List<string> hits) computed = RunBounds(false);
                AssertSame(cached.poses, computed.poses);
                for (int step = 0; step < computed.hits.Count; step++)
                {
                    Assert.That(cached.hits[step], Is.EqualTo(computed.hits[step]));
                }
            }
            finally
            {
                StepState.BoundsCacheEnabled = wasEnabled;
            }
        }

        /// <summary>
        /// Verify filter invalidation when a disabled collider is recreated on a new layer at the same address.
        /// </summary>
        [Test]
        public unsafe void ABoxWhoseColliderComesBackOnAnotherLayerIsFoundOnThatLayer()
        {
            const int layer = 29;
            bool wasEnabled = StepState.BoundsCacheEnabled;
            bool wasChecked = Unity.Physics.DynamicBoundsCache.CheckEnabled;
            try
            {
                StepState.BoundsCacheEnabled = true;
                Unity.Physics.DynamicBoundsCache.CheckEnabled = true;
                Rigidbody body = WeightlessBox(new Vector3(0.0f, 1.0f, 0.0f));
                BoxCollider box = body.GetComponent<BoxCollider>();
                Simulate(2);

                long mismatches = Unity.Physics.DynamicBoundsCache.Counts->Mismatches;
                long hits = Unity.Physics.DynamicBoundsCache.Counts->Hits;
                var misses = new List<int>();
                for (int round = 0; round < 16; round++)
                {
                    int target = round % 2 == 0 ? layer : 0;
                    box.enabled = false;
                    Simulate(1);
                    body.gameObject.layer = target;
                    box.enabled = true;
                    Simulate(3);
                    if (!UnityEngine.Physics.Raycast(new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, 10.0f, 1 << target))
                    {
                        misses.Add(round);
                    }
                }

                Assert.That(Unity.Physics.DynamicBoundsCache.Counts->Hits, Is.GreaterThan(hits), "the box took its bounds from the cache");
                Assert.That(misses, Is.Empty, "the rounds in which the ray missed the box on its new layer");
                Assert.That(Unity.Physics.DynamicBoundsCache.Counts->Mismatches - mismatches, Is.EqualTo(0), "no entry kept another filter");
            }
            finally
            {
                StepState.BoundsCacheEnabled = wasEnabled;
                Unity.Physics.DynamicBoundsCache.CheckEnabled = wasChecked;
            }
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
        /// Verify that a new contact starts cold when shapes reuse the addresses of a destroyed warm contact.
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
