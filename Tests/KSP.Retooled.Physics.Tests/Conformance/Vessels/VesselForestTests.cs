using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Diagnostics;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Unity.Physics.Configuration;

namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// Checks motion and push-out with ordinary joints in the forest, including fallback above the boundary cap.
    /// </summary>
    public class VesselForestTests : PhysicsScene
    {
        private const int Parts = 12;
        private const float PartHeight = 2.0f;
        private const float PartMass = 4.0f;

        private int wasMinimumParts;
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
            EnvironmentSwitches.Read();

            // These tests exercise the forest with ordinary joint rows.
            wasMinimumParts = VesselSwitches.MinimumParts;
            VesselSwitches.MinimumParts = Parts + 1;
            wasLocking = ForestSwitches.Locking;
            wasCap = ForestSwitches.BoundaryRowCap;

            // Keep the joints yielding so that each run tests the forest with the same springs.
            ForestSwitches.Locking = false;
        }

        [TearDown]
        public void RestoreSwitch()
        {
            VesselSwitches.MinimumParts = wasMinimumParts;
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
        private Vector3[] Run(bool standing, int steps, float sunk = 0.0f)
        {
            TearDownScene();
            SetUpScene();
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

        private static void AssertJointSpacing(Vector3[] positions)
        {
            for (int i = 1; i < positions.Length; i++)
            {
                Assert.That(Vector3.Distance(positions[i], positions[i - 1]), Is.EqualTo(PartHeight).Within(1e-3f),
                    $"the joint between parts {i - 1} and {i} keeps their spacing");
            }
        }

        [Test]
        public void AHangingColumnSwingsAndStaysAttached()
        {
            ForestDiagnostics.GetCounts(out long solvesBefore, out _, out _);
            Vector3[] positions = Run(false, 100);
            ForestDiagnostics.GetCounts(out long solves, out _, out _);
            Assert.That(solves, Is.GreaterThan(solvesBefore));
            AssertJointSpacing(positions);
            Assert.That(positions[Parts - 1].y, Is.EqualTo(28.0f).Within(1e-3f), "the top stays near its anchor");
            // After 100 steps, the bottom has swung back to within a millimetre of its starting position on the
            // original reference run. Allow the original one-millimetre tolerance on top of that displacement.
            Assert.That(Vector3.Distance(positions[0], new Vector3(0.0f, 6.0f, 0.0f)), Is.LessThan(0.002f));
        }

        [Test]
        public void AStandingColumnRocksAndStaysOnTheGround()
        {
            ForestDiagnostics.GetCounts(out long solvesBefore, out _, out _);
            Vector3[] positions = Run(true, 100);
            ForestDiagnostics.GetCounts(out long solves, out _, out _);
            Assert.That(solves, Is.GreaterThan(solvesBefore));
            AssertJointSpacing(positions);
            Assert.That(positions[0].y, Is.EqualTo(1.0f).Within(1e-2f), "the bottom stays on the ground");
            // The original reference run puts the top here after 100 steps.
            var expectedTop = new Vector3(-0.095f, 23.001f, -0.021f);
            Assert.That(Vector3.Distance(positions[Parts - 1], expectedTop), Is.LessThan(1e-2f), "the top rocks sideways");
        }

        [Test]
        public void AColumnSunkIntoTheGroundComesOutInOnePiece()
        {
            Vector3[] positions = Run(true, 10, 0.7f);
            // The original reference run puts the foot here after 10 steps.
            var expectedFoot = new Vector3(0.003f, 0.998f, 0.002f);
            Assert.That(Vector3.Distance(positions[0], expectedFoot), Is.LessThan(1e-2f), "the bottom came out of the ground");
            AssertJointSpacing(positions);
        }

        [Test]
        public void ATooBigBoundaryFallsBackToTheSparsePath()
        {
            ForestSwitches.BoundaryRowCap = ForestSwitches.DefaultBoundaryRowCap;
            Vector3[] positions = Run(true, 20);
            ForestDiagnostics.GetCounts(out _, out long fallbacksBefore, out _);

            ForestSwitches.BoundaryRowCap = 1;
            Vector3[] capped = Run(true, 20);
            ForestDiagnostics.GetCounts(out _, out long fallbacks, out _);
            Assert.That(fallbacks, Is.GreaterThan(fallbacksBefore));
            for (int i = 0; i < Parts; i++)
            {
                Assert.That(Vector3.Distance(capped[i], positions[i]), Is.LessThan(1e-4f),
                    $"part {i} ends at the same position when the cap makes the forest fall back");
            }
        }
    }
}
