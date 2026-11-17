using KSP.Retooled.Physics.Diagnostics;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Unity.Physics.Configuration;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Compare forest and sparse solving, verifying matching motion, actual forest use and fallback above the
    /// boundary cap.
    /// </summary>
    public class VesselForestTests : PhysicsScene
    {
        private const int Parts = 12;
        private const float PartHeight = 2.0f;
        private const float PartMass = 4.0f;

        private bool wasEnabled;
        private bool wasVessels;
        private bool wasLocking;
        private int wasCap;

        [SetUp]
        public void RememberSwitch()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The forest is part of the Unity Physics backend.");
            }

            // The environment's switches apply when the first world attaches; read them now, so they can't land on top
            // of what this fixture sets.
            ForestSwitches.Initialize();
            VesselSwitches.Initialize();
            wasVessels = VesselSwitches.Enabled;

            // These test the forest path with Unity's joint rows, which vessels replace.
            VesselSwitches.Enabled = false;
            wasEnabled = ForestSwitches.Enabled;
            wasLocking = ForestSwitches.Locking;
            wasCap = ForestSwitches.BoundaryRowCap;

            // These compare the forest's solve with the sparse one, so nothing locks (VesselSectionTests covers locking).
            ForestSwitches.Locking = false;
        }

        [TearDown]
        public void RestoreSwitch()
        {
            ForestSwitches.Enabled = wasEnabled;
            VesselSwitches.Enabled = wasVessels;
            ForestSwitches.Locking = wasLocking;
            ForestSwitches.BoundaryRowCap = wasCap;
        }

        /// <summary>A stock part joint: limits far away, drives doing the holding, three of them over a stack node.</summary>
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

        /// <summary>A column of parts, the top one hung from the world or the bottom one standing on the ground.</summary>
        private List<Rigidbody> Column(bool standing, float sunk = 0.0f)
        {
            var bodies = new List<Rigidbody>();
            float bottom = standing ? 0.5f - sunk : 5.0f;
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

            if (!standing)
            {
                StackJoint(bodies[Parts - 1], null, new Vector3(0.0f, bottom + PartHeight * Parts, 0.0f));
            }

            return bodies;
        }

        /// <summary>Builds the scene, swings it, and returns every part's position after a few seconds.</summary>
        private Vector3[] Run(bool forest, bool standing, int steps, float sunk = 0.0f)
        {
            TearDownScene();
            SetUpScene();
            ForestSwitches.Enabled = forest;
            if (standing)
            {
                Ground();
            }

            List<Rigidbody> bodies = Column(standing, sunk);
            bodies[standing ? Parts - 1 : 0].linearVelocity = new Vector3(1.0f, 0.0f, 0.5f);
            Simulate(steps);
            var positions = new Vector3[Parts];
            for (int i = 0; i < Parts; i++)
            {
                positions[i] = bodies[i].position;
            }

            return positions;
        }

        private static float Largest(Vector3[] a, Vector3[] b)
        {
            float largest = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                largest = Mathf.Max(largest, (a[i] - b[i]).magnitude);
            }

            return largest;
        }

        [Test]
        public void AHangingColumnSwingsAsOnTheSparsePath()
        {
            Vector3[] sparse = Run(false, false, 100);
            ForestDiagnostics.GetCounts(out long solvesBefore, out _, out _);
            Vector3[] forest = Run(true, false, 100);
            ForestDiagnostics.GetCounts(out long solves, out long fallbacks, out long builds);
            float difference = Largest(sparse, forest);
            Debug.Log($"[VesselForestTests] hanging: forest solves {solves - solvesBefore}, fallbacks {fallbacks}, builds {builds}; " +
                $"largest difference {difference:E2} m, bottom moved {(sparse[0] - new Vector3(0, 6, 0)).magnitude:F3} m");
            Assert.Greater(solves, solvesBefore, "the forest path never ran");
            Assert.Less(difference, 1e-3f);
        }

        [Test]
        public void AStandingColumnRocksAsOnTheSparsePath()
        {
            Vector3[] sparse = Run(false, true, 100);
            ForestDiagnostics.GetCounts(out long solvesBefore, out _, out _);
            Vector3[] forest = Run(true, true, 100);
            ForestDiagnostics.GetCounts(out long solves, out long fallbacks, out long builds);
            float difference = Largest(sparse, forest);
            Debug.Log($"[VesselForestTests] standing: forest solves {solves - solvesBefore}, fallbacks {fallbacks}, builds {builds}; " +
                $"largest difference {difference:E2} m, top at {sparse[Parts - 1]:F3}");
            Assert.Greater(solves, solvesBefore, "the forest path never ran");
            Assert.Less(difference, 1e-2f);
        }

        /// <summary>
        /// The bottom part starts 20 cm into the ground, deep enough that the push-out moves the column, which the forest
        /// path solves against its own factor.
        /// </summary>
        [Test]
        public void AColumnSunkIntoTheGroundComesOutAsOnTheSparsePath()
        {
            Vector3[] sparse = Run(false, true, 10, 0.7f);
            Vector3[] forest = Run(true, true, 10, 0.7f);
            float difference = Largest(sparse, forest);
            Debug.Log($"[VesselForestTests] sunk: largest difference {difference:E2} m, bottom at {sparse[0]:F3} / {forest[0]:F3}");
            Assert.Greater(forest[0].y, 0.95f, "the bottom part came out of the ground");
            Assert.Less(difference, 1e-2f);
        }

        [Test]
        public void ATooBigBoundaryFallsBackToTheSparsePath()
        {
            ForestSwitches.BoundaryRowCap = 1;
            ForestDiagnostics.GetCounts(out long solvesBefore, out long fallbacksBefore, out _);
            Vector3[] capped = Run(true, true, 20);
            ForestDiagnostics.GetCounts(out long solves, out long fallbacks, out _);
            ForestSwitches.BoundaryRowCap = ForestSwitches.DefaultBoundaryRowCap;
            Vector3[] sparse = Run(false, true, 20);
            Assert.Greater(fallbacks, fallbacksBefore);
            Assert.Less(Largest(sparse, capped), 1e-4f);
        }
    }
}
