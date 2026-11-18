using Unity.Collections;
using NUnit.Framework;
using Unity.Mathematics;
using static Unity.Physics.Math;
using PhysicsSolver = Unity.Physics.Solver;

namespace Unity.Physics.Tests.Dynamics.Joints
{
    /// <summary>Checks joint limits and impulse events.</summary>
    class JointStressTests
    {
        const float JointTimestep = 1.0f / 60.0f;

        [Test]
        public void FireImpulseEventTest()
        {
            var bodyAFromJoint = BodyFrame.Identity;
            bodyAFromJoint.Position = new float3(0, 0.5f, 0);
            var bodyBFromJoint = BodyFrame.Identity;
            bodyBFromJoint.Position = new float3(0, 0.5f, 0);

            var constraint = new Constraint
            {
                ConstrainedAxes = new bool3(true),
                Type = ConstraintType.Linear,
                Min = 0.5f,
                Max = 2.0f,
                SpringFrequency = Constraint.DefaultSpringFrequency,
                DampingRatio = Constraint.DefaultDampingRatio,
                MaxImpulse = float3.zero,
            };

            ConstraintBlock3 constraintblock = new ConstraintBlock3
            {
                A = constraint,
                Length = 1
            };

            var jointData = new Joint
            {
                AFromJoint = bodyAFromJoint.AsMTransform(),
                BFromJoint = bodyBFromJoint.AsMTransform(),
                Constraints = constraintblock
            };

            MotionVelocity velocityA = new MotionVelocity
            {
                LinearVelocity = new float3(10),
                AngularVelocity = new float3(10),
                InverseInertia = new float3(1),
                InverseMass = 1,
                GravityFactor = 1
            };
            MotionVelocity velocityB = new MotionVelocity
            {
                LinearVelocity = float3.zero,
                AngularVelocity = float3.zero,
                InverseInertia = float3.zero,
                InverseMass = 0.0f
            };

            MotionData motionData = new MotionData
            {
                WorldFromMotion = RigidTransform.identity,
                BodyFromMotion = RigidTransform.identity
            };

            using var world = new PhysicsWorld(0, 2, 1);
            var joints = world.Joints;
            jointData.BodyPair = new BodyIndexPair { BodyIndexA = 0, BodyIndexB = 1 };
            joints[0] = jointData;
            var motions = world.MotionDatas;
            motions[0] = motionData;
            motions[1] = motionData;
            var velocities = world.MotionVelocities;
            velocities[0] = velocityA;
            velocities[1] = velocityB;

            var input = CreateStepInput(world, JointTimestep);
            input.Gravity = new float3(0, -9.81f, 0);
            using var simulation = Unity.Physics.Simulation.Create();
            int activeFrames = 0;
            int quietFrames = 0;
            for (int frame = 0; frame < 15; frame++)
            {
                float startDistance = math.distance(
                    math.transform(motions[0].WorldFromMotion, bodyAFromJoint.Position),
                    math.transform(motions[1].WorldFromMotion, bodyBFromJoint.Position));
                float3 freeVelocity = velocities[0].LinearVelocity + input.Gravity * input.TimeStep;
                simulation.ResetSimulationContext(input);
                simulation.Step(input);
                float endDistance = math.distance(
                    math.transform(motions[0].WorldFromMotion, bodyAFromJoint.Position),
                    math.transform(motions[1].WorldFromMotion, bodyBFromJoint.Position));
                float3 appliedImpulse = (velocities[0].LinearVelocity - freeVelocity) / velocityA.InverseMass;
                int events = 0;
                foreach (var impulse in simulation.ImpulseEvents)
                {
                    Assert.That(impulse.BodyIndexA, Is.EqualTo(0));
                    Assert.That(impulse.BodyIndexB, Is.EqualTo(1));
                    Assert.That(impulse.Type, Is.EqualTo(ConstraintType.Linear));
                    Assert.That(math.distance(impulse.Impulse, appliedImpulse), Is.LessThan(1e-4f), $"Frame {frame}");
                    events++;
                }

                // A pivot crossing the limit does not imply that the solver applied an impulse that step.
                bool applied = math.length(appliedImpulse) > 1e-5f;
                Assert.That(events, Is.EqualTo(applied ? 1 : 0),
                    $"Frame {frame}, pivot separation changed from {startDistance} m to {endDistance} m, impulse {appliedImpulse}");
                if (frame <= 1)
                {
                    TestContext.WriteLine($"Frame {frame}: {startDistance:R} to {endDistance:R} m, impulse {appliedImpulse}, events {events}");
                }

                activeFrames += applied ? 1 : 0;
                quietFrames += applied ? 0 : 1;
            }

            Assert.That(activeFrames, Is.Positive);
            Assert.That(quietFrames, Is.Positive);
        }

        static SimulationStepInput CreateStepInput(in PhysicsWorld world, float timeStep)
        {
            return new SimulationStepInput
            {
                World = world,
                TimeStep = timeStep,
                NumSolverIterations = 4,
                DirectSolverSettings = PhysicsSolver.DirectSolverSettings.Default,
            };
        }

        [TestCase(0.5f, 1)]
        [TestCase(3.0f, 0)]
        public void LinearImpulseThresholdUsesWorldAxes(float threshold, int expectedEvents)
        {
            quaternion rotation = quaternion.RotateZ(math.PIHALF);
            var frame = new MTransform(rotation, float3.zero);
            using var world = new PhysicsWorld(0, 2, 1);
            var joints = world.Joints;
            joints[0] = new Joint
            {
                AFromJoint = frame,
                BFromJoint = frame,
                BodyPair = new BodyIndexPair { BodyIndexA = 0, BodyIndexB = 1 },
                Constraints = new ConstraintBlock3
                {
                    A = new Constraint
                    {
                        Type = ConstraintType.Linear,
                        ConstrainedAxes = new bool3(true, false, false),
                        Min = -0.5f,
                        Max = 0.5f,
                        SpringFrequency = Constraint.DefaultSpringFrequency,
                        DampingRatio = Constraint.DefaultDampingRatio,
                        MaxImpulse = new float3(float.MaxValue, threshold, float.MaxValue),
                    },
                    Length = 1,
                },
            };
            var motions = world.MotionDatas;
            motions[0] = new MotionData
            {
                WorldFromMotion = new RigidTransform(quaternion.identity, new float3(0, 0.5f, 0)),
                BodyFromMotion = RigidTransform.identity,
            };
            motions[1] = new MotionData { WorldFromMotion = RigidTransform.identity, BodyFromMotion = RigidTransform.identity };
            var velocities = world.MotionVelocities;
            velocities[0] = new MotionVelocity { LinearVelocity = new float3(0, 2, 0), InverseMass = 1 };
            velocities[1] = MotionVelocity.Zero;

            using var simulation = Unity.Physics.Simulation.Create();
            var input = CreateStepInput(world, JointTimestep);
            simulation.ResetSimulationContext(input);
            simulation.Step(input);

            float3 appliedImpulse = velocities[0].LinearVelocity - new float3(0, 2, 0);
            Assert.That(appliedImpulse.y, Is.EqualTo(-2.0f).Within(0.02f));
            int events = 0;
            foreach (var impulse in simulation.ImpulseEvents)
            {
                Assert.That(math.distance(impulse.Impulse, math.rotate(math.conjugate(rotation), appliedImpulse)), Is.LessThan(1e-4f));
                events++;
            }

            Assert.That(events, Is.EqualTo(expectedEvents));
        }

        [Test]
        public void TwistTest()
        {
            float timestep = 1.0f; // a large single step

            // Check that the twist constraint works in each axis.
            // Set up a constraint between a fixed and dynamic body, give the dynamic body
            // angular velocity about the limited axis, and verify that it stops at the limit
            for (int i = 0; i < 3; i++) // For each axis
            {
                for (int j = 0; j < 2; j++) // Negative / positive limit
                {
                    float3 axis = float3.zero;
                    axis[i] = 1.0f;

                    MotionVelocity velocityA = new MotionVelocity
                    {
                        LinearVelocity = float3.zero,
                        AngularVelocity = (j + j - 1) * axis,
                        InverseInertia = new float3(1),
                        InverseMass = 1
                    };

                    MotionVelocity velocityB = new MotionVelocity
                    {
                        LinearVelocity = float3.zero,
                        AngularVelocity = float3.zero,
                        InverseInertia = float3.zero,
                        InverseMass = 0.0f
                    };

                    MotionData motionA = new MotionData
                    {
                        WorldFromMotion = RigidTransform.identity,
                        BodyFromMotion = RigidTransform.identity
                    };

                    MotionData motionB = motionA;

                    const float angle = 0.5f;
                    float minLimit = (j - 1) * angle;
                    float maxLimit = j * angle;

                    var jointData = new Joint
                    {
                        AFromJoint = MTransform.Identity,
                        BFromJoint = MTransform.Identity
                    };
                    jointData.Constraints.A = Constraint.Twist(i, new FloatRange(minLimit, maxLimit));
                    jointData.Constraints.Length = 1;
                    using var world = new PhysicsWorld(0, 2, 1);
                    var joints = world.Joints;
                    jointData.BodyPair = new BodyIndexPair { BodyIndexA = 0, BodyIndexB = 1 };
                    joints[0] = jointData;
                    var motions = world.MotionDatas;
                    motions[0] = motionA;
                    motions[1] = motionB;
                    var velocities = world.MotionVelocities;
                    velocities[0] = velocityA;
                    velocities[1] = velocityB;

                    var input = CreateStepInput(world, timestep);
                    using var simulation = Unity.Physics.Simulation.Create();
                    simulation.ResetSimulationContext(input);
                    simulation.Step(input);

                    // Verify that the limits aren't exceeded
                    float finalAngle = CalculateTwistAngle(motions[0].WorldFromMotion.rot, i);
                    var tolerance = 1e-3f;
                    Assert.IsTrue(minLimit - tolerance <= finalAngle, $"Min limit {minLimit} of constraint is exceeded {finalAngle}");
                    Assert.IsTrue(finalAngle <= maxLimit + tolerance, $"Max limit {maxLimit} of constraint is exceeded {finalAngle}");
                }
            }
        }

        [Test]
        public void RaiseImpulseEventsFlagTest()
        {
            const int numCases = 9;

            NativeArray<float3> impulses = new NativeArray<float3>(numCases, Allocator.Temp);
            NativeArray<bool> shouldRaiseImpulseEvents = new NativeArray<bool>(numCases, Allocator.Temp);

            impulses[0] = new float3(math.INFINITY);                                                                shouldRaiseImpulseEvents[0] = false;
            impulses[1] = new float3(1.0f, math.INFINITY, math.INFINITY);                                           shouldRaiseImpulseEvents[1] = true;
            impulses[2] = new float3(1.0f, 2.0f, 3.0f);                                                             shouldRaiseImpulseEvents[2] = true;
            impulses[3] = new float3(System.Single.NegativeInfinity);                                               shouldRaiseImpulseEvents[3] = false;
            impulses[4] = new float3(1.0f, 2.0f, System.Single.NegativeInfinity);                                   shouldRaiseImpulseEvents[4] = true;
            impulses[5] = new float3(-1.0f, -2.0f, -3.0f);                                                          shouldRaiseImpulseEvents[5] = true;
            impulses[6] = new float3(-1.0f, System.Single.NegativeInfinity, System.Single.NegativeInfinity);        shouldRaiseImpulseEvents[6] = true;
            impulses[7] = new float3(math.INFINITY, math.INFINITY, System.Single.NegativeInfinity);                 shouldRaiseImpulseEvents[7] = false;
            impulses[8] = new float3(1.0f, -2.0f, 3.0f);                                                            shouldRaiseImpulseEvents[8] = true;

            Constraint constraint = default(Constraint);
            for (int i = 0; i < numCases; i++)
            {
                constraint = Constraint.BallAndSocket(impulses[i]);
                Assert.AreEqual(constraint.ShouldRaiseImpulseEvents, shouldRaiseImpulseEvents[i], $"Constraint: {constraint.Type} test failed. Expected value for raising impulse events is: {shouldRaiseImpulseEvents[i]} but the actual value is: {constraint.ShouldRaiseImpulseEvents}");

                constraint = Constraint.Twist(0, new FloatRange(-1.0f, 1.0f), impulses[i]);
                Assert.AreEqual(constraint.ShouldRaiseImpulseEvents, shouldRaiseImpulseEvents[i], $"Constraint: {constraint.Type} test failed. Expected value for raising impulse events is: {shouldRaiseImpulseEvents[i]} but the actual value is: {constraint.ShouldRaiseImpulseEvents}");

                constraint = Constraint.Cone(0, new FloatRange(-1.0f, 1.0f), impulses[i]);
                Assert.AreEqual(constraint.ShouldRaiseImpulseEvents, shouldRaiseImpulseEvents[i], $"Constraint: {constraint.Type} test failed. Expected value for raising impulse events is: {shouldRaiseImpulseEvents[i]} but the actual value is: {constraint.ShouldRaiseImpulseEvents}");

                constraint = Constraint.Cylindrical(0, new FloatRange(-1.0f, 1.0f), impulses[i]);
                Assert.AreEqual(constraint.ShouldRaiseImpulseEvents, shouldRaiseImpulseEvents[i], $"Constraint: {constraint.Type} test failed. Expected value for raising impulse events is: {shouldRaiseImpulseEvents[i]} but the actual value is: {constraint.ShouldRaiseImpulseEvents}");

                constraint = Constraint.FixedAngle(impulses[i]);
                Assert.AreEqual(constraint.ShouldRaiseImpulseEvents, shouldRaiseImpulseEvents[i], $"Constraint: {constraint.Type} test failed. Expected value for raising impulse events is: {shouldRaiseImpulseEvents[i]} but the actual value is: {constraint.ShouldRaiseImpulseEvents}");

                constraint = Constraint.Hinge(0, impulses[i]);
                Assert.AreEqual(constraint.ShouldRaiseImpulseEvents, shouldRaiseImpulseEvents[i], $"Constraint: {constraint.Type} test failed. Expected value for raising impulse events is: {shouldRaiseImpulseEvents[i]} but the actual value is: {constraint.ShouldRaiseImpulseEvents}");

                constraint = Constraint.LimitedDistance(new FloatRange(-1.0f, 1.0f), impulses[i]);
                Assert.AreEqual(constraint.ShouldRaiseImpulseEvents, shouldRaiseImpulseEvents[i], $"Constraint: {constraint.Type} test failed. Expected value for raising impulse events is: {shouldRaiseImpulseEvents[i]} but the actual value is: {constraint.ShouldRaiseImpulseEvents}");

                constraint = Constraint.Planar(0, new FloatRange(-1.0f, 1.0f), impulses[i]);
                Assert.AreEqual(constraint.ShouldRaiseImpulseEvents, shouldRaiseImpulseEvents[i], $"Constraint: {constraint.Type} test failed. Expected value for raising impulse events is: {shouldRaiseImpulseEvents[i]} but the actual value is: {constraint.ShouldRaiseImpulseEvents}");

                // Motorized constraints should never raise impulse events

                for (int j = 0; j < 3; j++)
                {
                    constraint = Constraint.MotorPlanar(impulses[i][j], 1.0f);
                    Assert.IsFalse(constraint.ShouldRaiseImpulseEvents, $"Motorized constraints should not raise impulse events. Test failed for: {constraint.Type}");

                    constraint = Constraint.MotorTwist(impulses[i][j], 1.0f);
                    Assert.IsFalse(constraint.ShouldRaiseImpulseEvents, $"Motorized constraints should not raise impulse events. Test failed for: {constraint.Type}");

                    constraint = Constraint.LinearVelocityMotor(impulses[i][j], 1.0f);
                    Assert.IsFalse(constraint.ShouldRaiseImpulseEvents, $"Motorized constraints should not raise impulse events. Test failed for: {constraint.Type}");

                    constraint = Constraint.AngularVelocityMotor(impulses[i][j], 1.0f);
                    Assert.IsFalse(constraint.ShouldRaiseImpulseEvents, $"Motorized constraints should not raise impulse events. Test failed for: {constraint.Type}");
                }
            }
        }
    }
}
