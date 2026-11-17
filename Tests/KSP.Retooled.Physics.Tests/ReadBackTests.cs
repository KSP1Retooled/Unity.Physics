using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Test main-thread pose read-back and worker reads during transform synchronization.
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
            // Move half the transforms and leave the rest at their body poses to test sync's pose comparison.
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

        private static Vector3Int Bits(Vector3 v)
        {
            return new Vector3Int(BitConverter.SingleToInt32Bits(v.x), BitConverter.SingleToInt32Bits(v.y), BitConverter.SingleToInt32Bits(v.z));
        }

        [Test]
        public void SettingTheStoredPoseKeepsTheVelocityToTheLastBit()
        {
            // Setting a rotation converts the angular velocity to world space and back, which can change its last bits.
            // A pose that equals the stored one to the last bit leaves the body as it is.
            var bodies = new List<Rigidbody>();
            var random = new System.Random(3);
            for (int i = 0; i < 32; i++)
            {
                Rigidbody body = SpinningBox(new Vector3(2.0f * i, 0.0f, 0.0f));
                body.rotation = Quaternion.Euler(random.Next(360), random.Next(360), random.Next(360));
                body.angularVelocity = new Vector3((float)random.NextDouble(), (float)random.NextDouble(), (float)random.NextDouble()) * 3.0f;
                bodies.Add(body);
            }

            Simulate(3);
            foreach (Rigidbody body in bodies)
            {
                Vector3 angular = body.angularVelocity;
                Vector3 position = body.position;
                Quaternion rotation = body.rotation;

                body.position = position;
                body.rotation = rotation;

                Assert.That(Bits(body.angularVelocity), Is.EqualTo(Bits(angular)), body.name);
                Assert.That(body.position, Is.EqualTo(position), body.name);
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
