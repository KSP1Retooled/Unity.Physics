using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Diagnostics;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Physics;
using Unity.Physics.Dynamics.Vessels.Sections;
using UnityEngine;
using Unity.Physics.Configuration;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// Strutted crafts built from Unity joints the way the game builds them, run through the vessel pipeline with locking
    /// on and compared with the same crafts on Unity's joint rows (vessels off). Inside a locked section, the struts must
    /// share the loads with the stack joints, as they do on the joint rows.
    /// </summary>
    public class StrutLoadTests : PhysicsScene
    {
        private bool wasVessels;
        private int wasMinimumParts;
        private bool wasStrutLoads;
        private bool wasFused;
        private int wasFusedParts;

        [SetUp]
        public void RememberSwitches()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            EnvironmentSwitches.Read();
            EnvironmentSwitches.Read();
            wasVessels = VesselSwitches.Enabled;
            wasMinimumParts = VesselSwitches.MinimumParts;
            wasStrutLoads = VesselSwitches.StrutLoadsEnabled;
            wasFused = SolverSwitches.VesselFused;
            wasFusedParts = SolverSwitches.FusedVesselParts;
            VesselSwitches.MinimumParts = 2;
        }

        [TearDown]
        public void RestoreSwitches()
        {
            VesselSwitches.Enabled = wasVessels;
            VesselSwitches.MinimumParts = wasMinimumParts;
            VesselSwitches.StrutLoadsEnabled = wasStrutLoads;
            SolverSwitches.VesselFused = wasFused;
            SolverSwitches.FusedVesselParts = wasFusedParts;
        }

        private void Fresh(bool vessels)
        {
            TearDownScene();
            SetUpScene();
            VesselSwitches.Enabled = vessels;
            ForestDiagnostics.ForgetIslands();
        }

        /// <summary>
        /// A stock part joint: three joints around a ring at a stack node, whose axis is <paramref name="axis"/>. The backend
        /// groups the three into one unit.
        /// </summary>
        private static List<ConfigurableJoint> StackJoint(Rigidbody host, Rigidbody target, Vector3 node, Vector3 axis, float breakForce)
        {
            var joints = new List<ConfigurableJoint>();
            Vector3 across = Vector3.Cross(axis, Mathf.Abs(axis.y) < 0.9f ? Vector3.up : Vector3.right).normalized;
            for (int i = 0; i < 3; i++)
            {
                Vector3 anchor = node + Quaternion.AngleAxis(120.0f * i, axis) * (0.3f * across);
                ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = target;
                joint.autoConfigureConnectedAnchor = false;
                joint.axis = host.transform.InverseTransformDirection(axis);
                joint.anchor = host.transform.InverseTransformPoint(anchor);
                joint.connectedAnchor = target.transform.InverseTransformPoint(anchor);
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = ConfigurableJointMotion.Locked;
                joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
                joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
                joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularYZDrive = new JointDrive { positionSpring = 20000.0f, positionDamper = 200.0f, maximumForce = 1e20f };
                joint.breakForce = breakForce;
                joint.breakTorque = breakForce;
                joints.Add(joint);
            }

            return joints;
        }

        /// <summary>A strut: one joint with every axis locked, between parts that aren't next to each other.</summary>
        private static ConfigurableJoint Strut(Rigidbody host, Rigidbody target, float breakForce)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            joint.breakForce = breakForce;
            joint.breakTorque = breakForce;
            return joint;
        }

        private const float CoreMass = 2.0f;
        private const float RadialMass = 8.0f;
        private const float RadialLength = 4.0f;

        /// <summary>The joints and bodies of the strutted craft.</summary>
        private sealed class Craft
        {
            public readonly List<Rigidbody> Bodies = new List<Rigidbody>();
            public readonly List<ConfigurableJoint> Joints = new List<ConfigurableJoint>();
            public readonly List<string> Names = new List<string>();
            public List<ConfigurableJoint> LeftRing;
            public ConfigurableJoint RightStrut;
        }

        /// <summary>
        /// A core of three parts standing on the ground, with a heavy radial part on each side of the top one. Each radial
        /// part hangs from the top part by a stack joint, and a strut runs from it down to the bottom part, which closes a
        /// loop through the core.
        /// </summary>
        private Craft BuildCraft(float ringBreak, float strutBreak, bool ground = true)
        {
            var craft = new Craft();
            float bottom = ground ? 0.0f : 5.0f;
            if (ground)
            {
                Ground();
            }

            for (int i = 0; i < 3; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, bottom + 0.5f + i, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                body.mass = CoreMass;
                craft.Bodies.Add(body);
                if (i > 0)
                {
                    Add(craft, StackJoint(body, craft.Bodies[i - 1], new Vector3(0.0f, bottom + i, 0.0f), Vector3.up, float.PositiveInfinity), $"core {i}");
                }
            }

            for (int side = 0; side < 2; side++)
            {
                float x = side == 0 ? 1.0f : -1.0f;
                Rigidbody radial = Box(new Vector3((0.5f + 0.5f * RadialLength) * x, bottom + 2.5f, 0.0f),
                    new Vector3(RadialLength, 0.8f, 0.8f)).AddComponent<Rigidbody>();
                radial.mass = RadialMass;
                craft.Bodies.Add(radial);
                List<ConfigurableJoint> ring = StackJoint(radial, craft.Bodies[2], new Vector3(0.5f * x, bottom + 2.5f, 0.0f),
                    new Vector3(x, 0.0f, 0.0f), ringBreak);
                Add(craft, ring, side == 0 ? "left ring" : "right ring");
                ConfigurableJoint strut = Strut(radial, craft.Bodies[0], strutBreak);
                Add(craft, new List<ConfigurableJoint> { strut }, side == 0 ? "left strut" : "right strut");
                if (side == 0)
                {
                    craft.LeftRing = ring;
                }
                else
                {
                    craft.RightStrut = strut;
                }
            }

            // Parts of one craft don't collide with each other, as in the game.
            for (int a = 0; a < craft.Bodies.Count; a++)
            {
                for (int b = a + 1; b < craft.Bodies.Count; b++)
                {
                    UnityEngine.Physics.IgnoreCollision(craft.Bodies[a].GetComponent<UnityEngine.Collider>(), craft.Bodies[b].GetComponent<UnityEngine.Collider>());
                }
            }

            return craft;
        }

        private static void Add(Craft craft, List<ConfigurableJoint> joints, string name)
        {
            foreach (ConfigurableJoint joint in joints)
            {
                craft.Joints.Add(joint);
                craft.Names.Add(name);
            }
        }

        /// <summary>The largest force on the joints of the craft whose name contains <paramref name="kind"/>.</summary>
        private static float LargestOf(Craft craft, string kind)
        {
            float largest = 0.0f;
            for (int j = 0; j < craft.Joints.Count; j++)
            {
                if (craft.Joints[j] != null && craft.Names[j].Contains(kind))
                {
                    largest = Mathf.Max(largest, craft.Joints[j].currentForce.magnitude);
                }
            }

            return largest;
        }

        /// <summary>The largest force on any joint of the craft that still exists, and the number of joints that broke.</summary>
        private static float LargestLoad(Craft craft, out int broken, out string which)
        {
            float largest = 0.0f;
            broken = 0;
            which = "";
            for (int j = 0; j < craft.Joints.Count; j++)
            {
                ConfigurableJoint joint = craft.Joints[j];
                if (joint == null)
                {
                    broken++;
                    continue;
                }

                float force = joint.currentForce.magnitude;
                if (force > largest)
                {
                    largest = force;
                    which = craft.Names[j];
                }
            }

            return largest;
        }

        /// <summary>
        /// The craft settles on the ground. With vessels on, after the edges lock, its largest joint load stays within a
        /// factor of two of the largest load on Unity's joint rows, and no joint breaks that doesn't break there. The rings'
        /// break force lies between the share that they carry with the struts and the whole moment of the radial parts,
        /// which the tree method puts on them.
        /// </summary>
        [Test]
        public void AStruttedCraftOnTheGroundCarriesItsLoadsAsOnTheJointRows()
        {
            const int settle = 150, measure = 50;
            // The tree method loads each ring with the whole radial part as a cantilever, about 260 N on a ring's joint. On
            // the joint rows, the strut takes nearly all of it and leaves about 5 N on the ring.
            const float ringBreak = 150.0f;
            var largest = new float[3];
            var broken = new int[3];
            var which = new string[3];
            var locked = new long[3];
            var rings = new float[3];
            var struts = new float[3];
            string[] passes = { "joint rows", "vessels", "vessels, tree method" };
            for (int pass = 0; pass < 3; pass++)
            {
                Fresh(pass > 0);
                VesselSwitches.StrutLoadsEnabled = pass != 2;
                Craft craft = BuildCraft(ringBreak, float.PositiveInfinity);
                Simulate(settle);
                largest[pass] = 0.0f;
                for (int i = 0; i < measure; i++)
                {
                    Simulate();
                    largest[pass] = Mathf.Max(largest[pass], LargestLoad(craft, out broken[pass], out which[pass]));
                    rings[pass] = Mathf.Max(rings[pass], LargestOf(craft, "ring"));
                    struts[pass] = Mathf.Max(struts[pass], LargestOf(craft, "strut"));
                }

                LargestLoad(craft, out broken[pass], out _);
                ForestDiagnostics.GetSectionCounts(out _, out _, out _, out _, out _, out _, out _, out locked[pass]);
                Debug.Log($"[StrutLoadTests] ground, {passes[pass]}: largest joint force {largest[pass]:F1} N on the {which[pass]}, " +
                    $"rings {rings[pass]:F1} N, struts {struts[pass]:F1} N, {broken[pass]} broken joints, {locked[pass]} locked edges");
            }

            Assert.That(locked[1], Is.GreaterThan(0), "the craft's edges lock");
            Assert.That(broken[0], Is.Zero, "nothing breaks on the joint rows");
            Assert.That(broken[1], Is.LessThanOrEqualTo(broken[0]), "nothing breaks with vessels that doesn't break on the joint rows");
            Assert.That(largest[1], Is.LessThan(2.0f * largest[0]), "the largest load stays within twice the joint rows' largest");
            Assert.That(largest[1], Is.GreaterThan(0.5f * largest[0]), "and above half of it");
            Assert.That(broken[2], Is.GreaterThan(0), "the tree method, which leaves the struts without load, breaks the rings");
        }

        /// <summary>
        /// A floating strutted craft whose edges have locked takes a blow that breaks a ring inside the locked section, and
        /// one that breaks a strut. In the steps after, every body keeps a finite pose and velocity, in the fused job and in
        /// the substeps' jobs.
        /// </summary>
        [Test]
        public void BreaksInsideALockedSectionLeaveFinitePoses([Values] bool fused)
        {
            SolverSwitches.VesselFused = fused;
            SolverSwitches.FusedVesselParts = fused ? 1 : int.MaxValue;
            Fresh(true);
            Craft craft = BuildCraft(400.0f, 400.0f, false);
            foreach (Rigidbody body in craft.Bodies)
            {
                body.useGravity = false;
                body.sleepThreshold = 0.0f;
            }

            Simulate(30);
            ForestDiagnostics.GetSectionCounts(out _, out _, out _, out _, out _, out _, out long sections, out long locked);
            Debug.Log($"[StrutLoadTests] floating, fused {fused}: {sections} sections, {locked} locked edges before the blows");
            Assert.That(locked, Is.GreaterThan(0), "the craft's edges lock before the blows");

            void CheckFinite(string when)
            {
                foreach (Rigidbody body in craft.Bodies)
                {
                    Vector3 p = body.position, v = body.linearVelocity, w = body.angularVelocity;
                    Quaternion q = body.rotation;
                    bool finite = float.IsFinite(p.x) && float.IsFinite(p.y) && float.IsFinite(p.z)
                        && float.IsFinite(q.x) && float.IsFinite(q.y) && float.IsFinite(q.z) && float.IsFinite(q.w)
                        && float.IsFinite(v.x) && float.IsFinite(v.y) && float.IsFinite(v.z)
                        && float.IsFinite(w.x) && float.IsFinite(w.y) && float.IsFinite(w.z);
                    Assert.That(finite, Is.True, $"{body.name} keeps a finite pose and velocity {when}: {p} {q} {v} {w}");
                }
            }

            // The left radial part is struck along the core, which shears its ring: the ring breaks inside the section.
            craft.Bodies[3].AddForce(new Vector3(0.0f, 4000.0f, 0.0f), ForceMode.Impulse);
            craft.Bodies[2].AddForce(new Vector3(0.0f, -1000.0f, 0.0f), ForceMode.Impulse);
            for (int i = 0; i < 40; i++)
            {
                Simulate();
                CheckFinite($"{i + 1} steps after the blow to the ring");
            }

            int ringsLeft = craft.LeftRing.FindAll(joint => joint != null).Count;
            Debug.Log($"[StrutLoadTests] floating, fused {fused}: {ringsLeft} of the left ring's joints left after its blow");
            Assert.That(ringsLeft, Is.Zero, "the left ring breaks");

            // The right radial part is pulled away from the bottom part, which stretches its strut.
            craft.Bodies[4].AddForce(new Vector3(0.0f, 1500.0f, 0.0f), ForceMode.Impulse);
            craft.Bodies[0].AddForce(new Vector3(0.0f, -1500.0f, 0.0f), ForceMode.Impulse);
            for (int i = 0; i < 40; i++)
            {
                Simulate();
                CheckFinite($"{i + 1} steps after the blow to the strut");
            }

            Debug.Log($"[StrutLoadTests] floating, fused {fused}: the right strut {(craft.RightStrut == null ? "broke" : "held")}");
            Assert.That(craft.RightStrut == null, Is.True, "the right strut breaks");
        }

        /// <summary>
        /// Pulls the radial parts of a floating craft apart until a ring yields, then lets the craft settle.
        /// The section that forms again must use the factor that it kept before the pull.
        /// </summary>
        [Test]
        public void ASectionThatLocksAgainKeepsItsFactor([Values] bool fused)
        {
            SolverSwitches.VesselFused = fused;
            SolverSwitches.FusedVesselParts = fused ? 1 : int.MaxValue;
            Fresh(true);
            VesselSwitches.StrutLoadsEnabled = true;
            Craft craft = BuildCraft(1000.0f, float.PositiveInfinity, ground: false);
            foreach (Rigidbody body in craft.Bodies)
            {
                body.useGravity = false;
                body.sleepThreshold = 0.0f;
            }

            long factorsAtStart = StrutLoads.Counts.Data.Factored;
            Simulate(30);
            ForestDiagnostics.GetSectionCounts(out _, out _, out _, out _, out _, out _, out long sectionsBefore, out long lockedBefore);
            Assert.That(sectionsBefore, Is.EqualTo(1), "the craft starts as one locked section");
            Assert.That(lockedBefore, Is.EqualTo(craft.Bodies.Count - 1));
            StrutLoadsCounters before = StrutLoads.Counts.Data;
            Assert.That(before.Factored - factorsAtStart, Is.GreaterThan(0), "the section has a factor before the pull");

            long locked = lockedBefore;
            for (int i = 1; i <= 200 && locked == lockedBefore; i++)
            {
                Vector3 pull = 20.0f * i * Vector3.right;
                craft.Bodies[3].AddForce(pull);
                craft.Bodies[4].AddForce(-pull);
                Simulate();
                ForestDiagnostics.GetSectionCounts(out _, out _, out _, out _, out _, out _, out _, out locked);
            }

            StrutLoadsCounters yielded = StrutLoads.Counts.Data;
            Assert.That(locked, Is.LessThan(lockedBefore), "the pull makes a ring yield");
            LargestLoad(craft, out int broken, out _);
            Assert.That(broken, Is.Zero, "the pull breaks no joint");

            Simulate(100);
            StrutLoadsCounters after = StrutLoads.Counts.Data;
            ForestDiagnostics.GetSectionCounts(out _, out _, out _, out _, out _, out _, out long sectionsAfter, out long lockedAfter);
            LargestLoad(craft, out broken, out _);
            Assert.That(broken, Is.Zero, "all joints survive the recovery");
            Assert.That(lockedAfter, Is.EqualTo(lockedBefore), "the ring locks again");
            Assert.That(sectionsAfter, Is.EqualTo(sectionsBefore));
            Assert.That(after.Reactivated - yielded.Reactivated, Is.GreaterThan(0), "a section uses a factor from before the split");
            Assert.That(after.Factored - yielded.Factored, Is.Zero, "the section that comes back needs no new factor");
        }

        /// <summary>
        /// The craft stands on the ground with joints too weak to hold it, so that most of them break in the same few steps
        /// after the edges lock, as in the bursts of breaks that the game showed. Every body keeps a finite pose and velocity.
        /// </summary>
        [Test]
        public void ABurstOfBreaksLeavesFinitePoses([Values] bool fused)
        {
            SolverSwitches.VesselFused = fused;
            SolverSwitches.FusedVesselParts = fused ? 1 : int.MaxValue;
            Fresh(true);
            Craft craft = BuildCraft(20.0f, 20.0f);
            foreach (ConfigurableJoint joint in craft.Joints)
            {
                joint.breakForce = 20.0f;
                joint.breakTorque = 20.0f;
            }

            int brokenAt = -1;
            for (int i = 0; i < 100; i++)
            {
                Simulate();
                LargestLoad(craft, out int broken, out _);
                if (broken > 0 && brokenAt < 0)
                {
                    brokenAt = i;
                }

                foreach (Rigidbody body in craft.Bodies)
                {
                    Vector3 p = body.position, v = body.linearVelocity, w = body.angularVelocity;
                    Quaternion q = body.rotation;
                    bool finite = float.IsFinite(p.x) && float.IsFinite(p.y) && float.IsFinite(p.z)
                        && float.IsFinite(q.x) && float.IsFinite(q.y) && float.IsFinite(q.z) && float.IsFinite(q.w)
                        && float.IsFinite(v.x) && float.IsFinite(v.y) && float.IsFinite(v.z)
                        && float.IsFinite(w.x) && float.IsFinite(w.y) && float.IsFinite(w.z);
                    Assert.That(finite, Is.True, $"{body.name} keeps a finite pose and velocity at step {i + 1}: {p} {q} {v} {w}");
                }
            }

            LargestLoad(craft, out int brokenJoints, out _);
            Debug.Log($"[StrutLoadTests] burst, fused {fused}: the first joint broke at step {brokenAt + 1}, {brokenJoints} of {craft.Joints.Count} joints broke");
            Assert.That(brokenJoints, Is.GreaterThan(craft.Joints.Count / 2), "most of the joints break");
        }
    }
}
