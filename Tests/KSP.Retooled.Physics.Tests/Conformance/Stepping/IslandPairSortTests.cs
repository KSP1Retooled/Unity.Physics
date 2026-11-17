using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Stepping
{
    /// <summary>
    /// Require serial and parallel island pair sorts to produce identical ordering and collision-disable flags.
    /// </summary>
    public class IslandPairSortTests : PhysicsScene
    {
        [SetUp]
        public void RequireRetooled()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The pair sort is part of the Unity Physics backend.");
            }
        }

        [TearDown]
        public void RestoreLimit()
        {
            Unity.Physics.DispatchPairSequencer.SerialIslandPairLimit = Unity.Physics.DispatchPairSequencer.DefaultSerialIslandPairLimit;
        }

        /// <summary>
        /// A stack of boxes on the ground, and two boxes held together by a joint that doesn't allow collision. They
        /// overlap, so a contact between them would push them apart.
        /// </summary>
        private Vector3[] Run(int limit)
        {
            Unity.Physics.DispatchPairSequencer.SerialIslandPairLimit = limit;
            Ground();
            var bodies = new Rigidbody[8];
            for (int i = 0; i < 6; i++)
            {
                bodies[i] = DynamicBox(new Vector3(0.1f * i, 0.5f + 1.01f * i, 0.0f));
            }

            bodies[6] = DynamicBox(new Vector3(4.0f, 0.5f, 0.0f));
            bodies[7] = DynamicBox(new Vector3(4.6f, 0.5f, 0.0f));
            FixedJoint joint = bodies[6].gameObject.AddComponent<FixedJoint>();
            joint.connectedBody = bodies[7];
            joint.enableCollision = false;

            Simulate(60);
            var positions = new Vector3[bodies.Length];
            for (int i = 0; i < bodies.Length; i++)
            {
                positions[i] = bodies[i].position;
            }

            TearDownScene();
            SetUpScene();
            return positions;
        }

        [Test]
        public void TheParallelChainMovesTheBodiesAlikeTwice()
        {
            Vector3[] first = Run(-1);
            Vector3[] second = Run(-1);

            for (int i = 0; i < first.Length; i++)
            {
                AssertVector(first[i], second[i], 1e-4f, $"body {i}");
            }
        }

        [Test]
        public void OneJobAndTheParallelChainMoveTheBodiesAlike()
        {
            Vector3[] serial = Run(Unity.Physics.DispatchPairSequencer.DefaultSerialIslandPairLimit);
            Vector3[] parallel = Run(-1);

            for (int i = 0; i < serial.Length; i++)
            {
                AssertVector(parallel[i], serial[i], 1e-4f, $"body {i}");
            }

            // The joined boxes keep their overlap, so no contact pushed them apart.
            Assert.That((serial[7] - serial[6]).magnitude, Is.EqualTo(0.6f).Within(1e-2f));
        }
    }
}
