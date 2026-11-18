using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Bodies;
using Unity.Transforms;

namespace KSP.Retooled.Physics.Tests.Conformance.Stepping
{
    public unsafe class WriteBackTests
    {
        [TestCase(false, false)]
        [TestCase(true, false)]
        [TestCase(true, true)]
        public void InvalidSolvedMotionRestoresPreviousPoseAndStopsBody(bool invalidPose, bool sleeping)
        {
            float3 position = new float3(1f, 2f, 3f);
            quaternion rotation = quaternion.RotateX(0.8f);
            LocalTransform pose = LocalTransform.FromPositionRotation(position, rotation);
            PhysicsVelocity velocity = new PhysicsVelocity { Linear = new float3(1f), Angular = new float3(2f) };
            BodyLimits limits = new BodyLimits { MaxLinearVelocity = 50f, MaxAngularVelocity = 50f };
            BodyData body = new BodyData
            {
                pose = &pose,
                velocity = &velocity,
                limits = &limits,
                flags = sleeping ? BodyFlags.Sleeping : default,
            };

            using var indices = new NativeArray<int>(1, Allocator.TempJob);
            RigidTransform bodyFromMotion = new RigidTransform(quaternion.identity, new float3(0f, 0.1f, 0f));
            MotionData solvedMotion = new MotionData
            {
                BodyFromMotion = bodyFromMotion,
                WorldFromMotion = new RigidTransform(quaternion.identity, new float3(invalidPose ? float.NaN : 10f)),
            };
            MotionVelocity solvedVelocity = new MotionVelocity
            {
                LinearVelocity = new float3(3f),
                AngularVelocity = new float3(invalidPose ? 4f : float.PositiveInfinity),
            };
            using var motions = new NativeArray<MotionData>(new[] { solvedMotion }, Allocator.TempJob);
            using var velocities = new NativeArray<MotionVelocity>(new[] { solvedVelocity }, Allocator.TempJob);
            StoreStepperCounters counters = default;
            new StoreBodies.WriteBackJob
            {
                Store = &body,
                Counters = &counters,
                DynamicIndices = indices,
                MotionDatas = motions,
                MotionVelocities = velocities,
            }.Schedule(1, 1).Complete();

            Assert.That(pose.Position, Is.EqualTo(position));
            Assert.That(pose.Rotation, Is.EqualTo(rotation));
            Assert.That(velocity.Linear, Is.EqualTo(float3.zero));
            Assert.That(velocity.Angular, Is.EqualTo(float3.zero));
            Assert.That(motions[0].WorldFromMotion, Is.EqualTo(math.mul(new RigidTransform(rotation, position), bodyFromMotion)));
            Assert.That(velocities[0].LinearVelocity, Is.EqualTo(float3.zero));
            Assert.That(velocities[0].AngularVelocity, Is.EqualTo(float3.zero));
            Assert.That(counters.NonFiniteBodies, Is.EqualTo(1));
        }
    }
}
