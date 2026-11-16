using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Covers the reads that Unity sends around each step: the read-back of the poses into the Transforms after the step,
    /// on the main thread, and the reads of the poses of moved Transforms before the step, mostly on job workers. The
    /// plugin answers both kinds of GetPose itself through the rows.
    /// </summary>
    public class ReadBackTests : PhysicsScene
    {
        private Rigidbody SpinningBox(Vector3 position)
        {
            Rigidbody body = WeightlessBox(position);
            body.sleepThreshold = 0.0f;
            body.linearVelocity = new Vector3(1.0f, 0.5f, 0.0f);
            body.angularVelocity = new Vector3(0.3f, 1.0f, 2.0f);
            return body;
        }

        [Test]
        public void TheReadBackGivesEveryTransformItsBodysPose()
        {
            UnityEngine.Physics.autoSyncTransforms = false;
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < 64; i++)
            {
                Rigidbody body = SpinningBox(new Vector3(2.0f * i, 0.0f, 0.0f));
                new GameObject("model").transform.SetParent(body.transform, false);
                bodies.Add(body);
            }

            Simulate(5);

            for (int i = 0; i < bodies.Count; i++)
            {
                Rigidbody body = bodies[i];
                Assert.That(body.position.x, Is.GreaterThan(2.0f * i + 0.05f), "the body moved");
                Assert.That(body.transform.position, Is.EqualTo(body.position), body.name);
                Assert.That(Quaternion.Angle(body.transform.rotation, body.rotation), Is.LessThan(1e-3f), body.name);
            }
        }

        [Test]
        public void TheSyncBeforeTheStepReadsTheMovedBodiesPoses()
        {
            // Unity's sync reads the pose of each body whose Transform moved, on job workers, and sends the Transform's
            // pose only where they differ. Half of the Transforms move, and the other half are set to the poses that
            // their bodies already have.
            UnityEngine.Physics.autoSyncTransforms = false;
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < 256; i++)
            {
                bodies.Add(WeightlessBox(new Vector3(2.0f * (i % 16), 0.0f, 2.0f * (i / 16))));
            }

            Simulate(2);
            for (int i = 0; i < bodies.Count; i++)
            {
                Transform transform = bodies[i].transform;
                transform.position = (i & 1) == 0 ? transform.position + new Vector3(0.0f, 3.0f, 0.0f) : transform.position;
            }

            UnityEngine.Physics.SyncTransforms();

            for (int i = 0; i < bodies.Count; i++)
            {
                Assert.That(bodies[i].position, Is.EqualTo(bodies[i].transform.position), bodies[i].name);
                Assert.That(bodies[i].position.y, Is.EqualTo((i & 1) == 0 ? 3.0f : 0.0f), bodies[i].name);
            }
        }

        [Test]
        public void EveryReadRunsWithoutItsCategorysSpan()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
            foreach (BodyCommand command in Enum.GetValues(typeof(BodyCommand)))
            {
                Assert.That(BodyKernel.IsRead(command), Is.EqualTo(IsGet(command.ToString())), command.ToString());
            }

            foreach (ShapeCommand command in Enum.GetValues(typeof(ShapeCommand)))
            {
                bool read = IsGet(command.ToString()) || command == ShapeCommand.ShouldIgnoreCollision;
                Assert.That(ShapeKernel.IsRead(command), Is.EqualTo(read), command.ToString());
            }

            foreach (JointCommand command in Enum.GetValues(typeof(JointCommand)))
            {
                Assert.That(JointKernel.IsRead(command), Is.EqualTo(IsGet(command.ToString())), command.ToString());
            }

            foreach (VehicleCommand command in Enum.GetValues(typeof(VehicleCommand)))
            {
                Assert.That(VehicleKernel.IsRead(command), Is.EqualTo(IsGet(command.ToString())), command.ToString());
            }

            // The plugin answers the other world reads itself, so only these reach the kernel.
            Assert.That(WorldKernel.IsRead(WorldCommand.GetBasicStats), Is.True);
            Assert.That(WorldKernel.IsRead(WorldCommand.GetLastSimulationStats), Is.True);
            Assert.That(WorldKernel.IsRead(WorldCommand.GetAllShapesByLayerMask), Is.True);
            Assert.That(WorldKernel.IsRead(WorldCommand.CreateBody), Is.False);
            Assert.That(WorldKernel.IsRead(WorldCommand.DestroyBody), Is.False);
        }

        private static bool IsGet(string name)
        {
            return name.StartsWith("Get", StringComparison.Ordinal);
        }
    }
}
