using System.Collections.Generic;
using NUnit.Framework;
using Unity.Physics;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The limits on a vessel island's boundary block, with vessels on: the cap on contact points against the ground, the
    /// cap on contact manifolds between parts of the same vessel, and the warm start of the boundary's pivoting from the
    /// last solve.
    /// </summary>
    public class BoundaryCapTests : PhysicsScene
    {
        private const float PartMass = 4.0f;

        private bool wasVessels;
        private bool wasForest;
        private BoundaryCaps wasCaps;

        [SetUp]
        public void TurnVesselsOn()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            VesselPipeline.Initialize();
            VesselForest.Initialize();
            wasVessels = VesselPipeline.Enabled;
            wasForest = VesselForest.Enabled;
            wasCaps = VesselForest.Caps;
            Fresh(default);
        }

        [TearDown]
        public void RestoreSwitch()
        {
            VesselPipeline.Enabled = wasVessels;
            VesselForest.Enabled = wasForest;
            VesselForest.Caps = wasCaps;
        }

        private void Fresh(BoundaryCaps caps)
        {
            TearDownScene();
            SetUpScene();
            VesselPipeline.Enabled = true;
            VesselForest.Enabled = true;
            VesselForest.Caps = caps;
            VesselForest.ForgetIslands();
        }

        /// <summary>
        /// A stock part joint between two parts that lie side by side along x: three joints over the node, anchors and
        /// twist locked, swing held by drives. The backend makes one hard edge of them.
        /// </summary>
        private static void StackJoint(Rigidbody host, Rigidbody target, Vector3 node, Vector3 axis)
        {
            for (int i = 0; i < 3; i++)
            {
                Vector3 offset = Quaternion.AngleAxis(120.0f * i, axis) * (Vector3.Cross(axis, Vector3.forward).sqrMagnitude > 0.1f
                    ? Vector3.Cross(axis, Vector3.forward).normalized * 0.2f
                    : Vector3.up * 0.2f);
                ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = target;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = host.transform.InverseTransformPoint(node + offset);
                joint.connectedAnchor = target != null ? target.transform.InverseTransformPoint(node + offset) : node + offset;
                joint.axis = axis;
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
        /// A long craft that lies on a large ground: a row of box parts along x, each joined to the one before it. Each
        /// part touches the ground with a manifold of four points.
        /// </summary>
        private List<Rigidbody> LyingCraft(int count)
        {
            Box(new Vector3(0.0f, -0.5f, 0.0f), new Vector3(400.0f, 1.0f, 400.0f));
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < count; i++)
            {
                Rigidbody body = Box(new Vector3(i - 0.5f * count, 0.3f, 0.0f), new Vector3(1.0f, 0.6f, 0.6f)).AddComponent<Rigidbody>();
                body.mass = PartMass;
                body.sleepThreshold = 0.0f;
                bodies.Add(body);
                if (i > 0)
                {
                    StackJoint(body, bodies[i - 1], new Vector3(i - 0.5f * count - 0.5f, 0.3f, 0.0f), Vector3.right);
                }
            }

            return bodies;
        }

        /// <summary>
        /// A craft of 100 parts that lies on the ground has 400 contact points, more than the cap of 256. The cap takes
        /// every part's deepest point before any part's second point, so every part keeps some of its points, and the
        /// boundary keeps no more than the 448 rows that 256 points in manifolds of 4 would bring. The craft rests on the
        /// ground without sinking or shaking.
        /// </summary>
        [Test]
        public void ALongCraftOnTheGroundKeepsItsContactPointsUnderTheCap()
        {
            List<Rigidbody> bodies = LyingCraft(100);
            Simulate(100);
            BoundaryCounters.Reset();
            float lowest = float.MaxValue, highest = float.MinValue, fastest = 0.0f;
            for (int step = 0; step < 50; step++)
            {
                Simulate();
                foreach (Rigidbody body in bodies)
                {
                    lowest = Mathf.Min(lowest, body.position.y);
                    highest = Mathf.Max(highest, body.position.y);
                    fastest = Mathf.Max(fastest, body.linearVelocity.magnitude);
                }
            }

            BoundaryCounters counts = BoundaryCounters.Current;
            double rows = counts.Rows / (double)System.Math.Max(1, counts.Solves);
            Debug.Log($"[BoundaryCapTests] lying craft: {counts.Solves} solves, {rows:F0} rows a solve, most outside points " +
                $"{counts.MostOutsidePoints}, held points {counts.HeldPoints}, manifolds left with no point " +
                $"{counts.EmptiedOutsideManifolds}, rounds a solve {counts.Rounds / (double)System.Math.Max(1, counts.Solves):F2} " +
                $"(most {counts.MostRounds}, {counts.CappedSolves} at the cap, {counts.UnsettledSolves} without an answer); " +
                $"centres from {lowest:F4} to {highest:F4} m, fastest {fastest:F4} m/s");
            Assert.That(counts.Solves, Is.GreaterThan(0), "the craft's island solves a boundary");
            Assert.That(counts.MostOutsidePoints, Is.LessThanOrEqualTo(BoundaryCaps.DefaultOutsideContactPoints), "the cap holds");
            Assert.That(rows, Is.LessThanOrEqualTo(448.0), "the limit on rows holds");
            Assert.That(counts.HeldPoints, Is.GreaterThan(0), "the cap held points out");
            Assert.That(counts.EmptiedOutsideManifolds, Is.EqualTo(0), "every part keeps some of its points");
            Assert.That(lowest, Is.GreaterThan(0.29f), "no part sinks");
            Assert.That(highest, Is.LessThan(0.31f), "no part lifts");
            Assert.That(fastest, Is.LessThan(0.02f), "no part shakes");
        }

        /// <summary>
        /// Two columns of parts that hang side by side from one part hooked to the world, joined at the top, so that they
        /// are one vessel. A push on the second column presses it against the first, so that every pair of parts across
        /// the columns is a self-contact, more than the cap of 8 manifolds. The columns stay apart and still.
        /// </summary>
        [Test]
        public void PressedColumnsKeepTheirSelfContactsUnderTheCap()
        {
            const int count = 12;
            var left = new List<Rigidbody>();
            var right = new List<Rigidbody>();
            for (int i = 0; i < count; i++)
            {
                for (int side = 0; side < 2; side++)
                {
                    float x = side == 0 ? -0.45f : 0.45f;
                    Rigidbody body = Box(new Vector3(x, 20.0f - i, 0.0f), new Vector3(0.9f, 1.0f, 0.8f)).AddComponent<Rigidbody>();
                    body.mass = PartMass;
                    body.sleepThreshold = 0.0f;
                    List<Rigidbody> column = side == 0 ? left : right;
                    column.Add(body);
                    if (i > 0)
                    {
                        StackJoint(body, column[i - 1], new Vector3(x, 20.5f - i, 0.0f), Vector3.up);
                    }
                }
            }

            StackJoint(left[0], null, new Vector3(-0.45f, 20.5f, 0.0f), Vector3.up);
            StackJoint(right[0], left[0], new Vector3(0.0f, 20.0f, 0.0f), Vector3.right);

            float deepest = 0.0f, fastest = 0.0f;
            BoundaryCounters.Reset();
            for (int step = 0; step < 150; step++)
            {
                foreach (Rigidbody body in right)
                {
                    body.AddForce(new Vector3(-20.0f, 0.0f, 0.0f));
                }

                Simulate();
                for (int i = 1; i < count && step >= 100; i++)
                {
                    deepest = Mathf.Max(deepest, 0.9f - (right[i].position.x - left[i].position.x));
                    fastest = Mathf.Max(fastest, right[i].linearVelocity.magnitude, left[i].linearVelocity.magnitude);
                }
            }

            BoundaryCounters counts = BoundaryCounters.Current;
            Debug.Log($"[BoundaryCapTests] pressed columns: {counts.Solves} solves, most self-contact manifolds " +
                $"{counts.MostSelfManifolds}, held manifolds {counts.HeldSelfManifolds}, rounds a solve " +
                $"{counts.Rounds / (double)System.Math.Max(1, counts.Solves):F2}; deepest overlap {deepest * 1e3f:F2} mm, " +
                $"fastest {fastest:F4} m/s");
            Assert.That(counts.Solves, Is.GreaterThan(0), "the craft's island solves a boundary");
            Assert.That(counts.MostSelfManifolds, Is.EqualTo(BoundaryCaps.DefaultSelfContactManifolds), "the cap holds");
            Assert.That(counts.HeldSelfManifolds, Is.GreaterThan(0), "the cap held manifolds out");
            Assert.That(deepest, Is.LessThan(0.01f), "the columns stay apart");
            Assert.That(fastest, Is.LessThan(0.05f), "the columns hang still");
        }

        /// <summary>
        /// A craft of 8 parts lies on the ground, and next to each part a loose box sits 3 cm deep in the ground, touching
        /// the part, so that the boxes are in the craft's island. With a cap of 4 outside points, the contacts of most
        /// boxes are held out of the boundary. The scene runs twice: once with held rows that take no impulse of their
        /// own, as before, and once with held rows that are too deep getting a push-out of their own. The boxes come
        /// further out of the ground in the second run.
        /// </summary>
        [Test]
        public void HeldContactsThatAreTooDeepArePushedOut()
        {
            const float depth = 0.03f;
            var deepest = new float[2];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(new BoundaryCaps { OutsideContactPoints = 4, IdleHeldRows = pass == 0 });
                LyingCraft(8);
                var boxes = new List<Rigidbody>();
                for (int i = 0; i < 8; i++)
                {
                    Rigidbody box = Box(new Vector3(i - 4.0f, 0.25f - depth, 0.6f), new Vector3(0.8f, 0.5f, 0.6f)).AddComponent<Rigidbody>();
                    box.mass = PartMass;
                    box.sleepThreshold = 0.0f;
                    boxes.Add(box);
                }

                BoundaryCounters.Reset();
                Simulate(3);
                foreach (Rigidbody box in boxes)
                {
                    deepest[pass] = Mathf.Max(deepest[pass], 0.25f - box.position.y);
                }

                BoundaryCounters counts = BoundaryCounters.Current;
                Debug.Log($"[BoundaryCapTests] deep held contacts, {(pass == 0 ? "idle" : "pushed")}: {counts.Solves} solves, " +
                    $"held points {counts.HeldPoints}, manifolds left with no point {counts.EmptiedOutsideManifolds}, held rows " +
                    $"pushed {counts.HeldPushes}; deepest box after 3 steps {deepest[pass] * 1e3f:F2} mm");
                Assert.That(counts.EmptiedOutsideManifolds, Is.GreaterThan(0), "the cap held whole manifolds out");
            }

            Assert.That(deepest[1], Is.LessThan(deepest[0] - 0.005f), "the push-out of the held rows lifts the boxes");
        }

        /// <summary>
        /// A craft of 8 parts lies on the ground, and next to each part a loose box rests on the ground, touching the part,
        /// so that the boxes are in the craft's island. With a cap of 4 outside points, the contacts of most boxes are
        /// held out of the boundary. A held row used to keep only its warm start, which is 95% of the contact's last
        /// impulse, so a box whose contacts were all held sank a little in every step until its points were the deepest
        /// and the cap took them again. The scene runs twice: once with held rows that take no impulse of their own, and
        /// once with held rows that push when their contact closes. The boxes stay up in the second run.
        /// </summary>
        [Test]
        public void BoxesWhoseContactsAreHeldDoNotSink()
        {
            var deepest = new float[2];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(new BoundaryCaps { OutsideContactPoints = 4, IdleHeldRows = pass == 0 });
                LyingCraft(8);
                var boxes = new List<Rigidbody>();
                for (int i = 0; i < 8; i++)
                {
                    Rigidbody box = Box(new Vector3(i - 4.0f, 0.25f, 0.6f), new Vector3(0.8f, 0.5f, 0.6f)).AddComponent<Rigidbody>();
                    box.mass = PartMass;
                    box.sleepThreshold = 0.0f;
                    boxes.Add(box);
                }

                Simulate(50);
                BoundaryCounters.Reset();
                for (int step = 0; step < 200; step++)
                {
                    Simulate();
                    foreach (Rigidbody box in boxes)
                    {
                        deepest[pass] = Mathf.Max(deepest[pass], 0.25f - box.position.y);
                    }
                }

                BoundaryCounters counts = BoundaryCounters.Current;
                Debug.Log($"[BoundaryCapTests] resting held boxes, {(pass == 0 ? "idle" : "pushing")}: {counts.Solves} solves, " +
                    $"held points {counts.HeldPoints}, manifolds left with no point {counts.EmptiedOutsideManifolds}, held rows " +
                    $"that pushed {counts.HeldSupports}; deepest box {deepest[pass] * 1e3f:F2} mm");
                Assert.That(counts.EmptiedOutsideManifolds, Is.GreaterThan(0), "the cap held whole manifolds out");
            }

            Assert.That(deepest[1], Is.LessThan(deepest[0]), "the held rows' pushes keep the boxes higher");
            Assert.That(deepest[1], Is.LessThan(0.002f), "no box sinks");
        }

        /// <summary>
        /// A long craft rests on the ground, with more contact points than the cap. Then gravity goes away. Each contact
        /// starts the next step with most of its last impulse, which pushes the craft up. The solve takes that impulse
        /// back from the points that it keeps, and it has to take it back from the held points too, or the held points
        /// throw the craft into the air. The scene runs twice: once with held rows that keep their warm start, as before,
        /// and once with held rows that give it back.
        /// </summary>
        [Test]
        public void HeldContactsGiveBackTheirWarmStartWhenTheyAreNotNeeded()
        {
            var fastest = new float[2];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(new BoundaryCaps { IdleHeldRows = pass == 0 });
                List<Rigidbody> bodies = LyingCraft(100);
                Simulate(100);
                UnityEngine.Physics.gravity = Vector3.zero;
                BoundaryCounters.Reset();
                for (int step = 0; step < 10; step++)
                {
                    Simulate();
                    foreach (Rigidbody body in bodies)
                    {
                        fastest[pass] = Mathf.Max(fastest[pass], body.linearVelocity.y);
                    }
                }

                BoundaryCounters counts = BoundaryCounters.Current;
                Debug.Log($"[BoundaryCapTests] gravity gone, {(pass == 0 ? "idle" : "given back")}: held points {counts.HeldPoints}, " +
                    $"held rows that gave back their warm start {counts.HeldReleases}; fastest upward speed {fastest[pass]:F4} m/s");
                Assert.That(counts.HeldPoints, Is.GreaterThan(0), "the cap held points out");
            }

            Assert.That(fastest[1], Is.LessThan(0.5f * fastest[0]), "giving back the warm start keeps the craft lower");
            Assert.That(fastest[1], Is.LessThan(0.03f), "the craft stays on the ground, apart from the push out of its resting depth");
        }

        /// <summary>
        /// The limit on self-contact manifolds takes its default of 8 for a setting of zero, and it never goes above 25,
        /// not even for a negative setting, which turns the other limits off.
        /// </summary>
        [Test]
        public void TheSelfContactLimitNeverGoesAbove25()
        {
            Assert.That(new BoundaryCaps().SelfManifoldLimit, Is.EqualTo(BoundaryCaps.DefaultSelfContactManifolds));
            Assert.That(new BoundaryCaps { SelfContactManifolds = 12 }.SelfManifoldLimit, Is.EqualTo(12));
            Assert.That(new BoundaryCaps { SelfContactManifolds = 40 }.SelfManifoldLimit, Is.EqualTo(BoundaryCaps.MaxSelfContactManifolds));
            Assert.That(new BoundaryCaps { SelfContactManifolds = -1 }.SelfManifoldLimit, Is.EqualTo(BoundaryCaps.MaxSelfContactManifolds));
            Assert.That(new BoundaryCaps { OutsideContactPoints = -1 }.OutsidePointLimit, Is.EqualTo(int.MaxValue));
        }

        /// <summary>
        /// The same craft on the ground, solved once with each solve of its boundary starting cold and once starting
        /// from the last solve's states. The warm start takes fewer pivoting rounds. Neither run has a cap on rounds, so
        /// that the counts are the rounds that the solve needs.
        /// </summary>
        [Test]
        public void AWarmStartTakesFewerPivotingRounds()
        {
            var rounds = new double[2];
            for (int pass = 0; pass < 2; pass++)
            {
                Fresh(new BoundaryCaps { PivotRounds = -1, ColdStart = pass == 0 });
                LyingCraft(30);
                Simulate(50);
                BoundaryCounters.Reset();
                Simulate(50);
                BoundaryCounters counts = BoundaryCounters.Current;
                rounds[pass] = counts.Rounds / (double)System.Math.Max(1, counts.Solves);
                Debug.Log($"[BoundaryCapTests] {(pass == 0 ? "cold" : "warm")}: {counts.Solves} solves, " +
                    $"{counts.Rows / (double)System.Math.Max(1, counts.Solves):F0} rows a solve, {rounds[pass]:F2} rounds a solve, " +
                    $"most {counts.MostRounds}, warm rows {counts.WarmRows}");
                Assert.That(counts.Solves, Is.GreaterThan(0), "the craft's island solves a boundary");
            }

            Assert.That(rounds[1], Is.LessThan(rounds[0]), "the warm start takes fewer rounds");
        }
    }
}
