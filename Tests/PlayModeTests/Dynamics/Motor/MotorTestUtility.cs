using NUnit.Framework;
using Unity.Mathematics;
using PhysicsSolver = Unity.Physics.Solver;

//
// Tools to create and run a tiny simulation for a single body pair and joint, used by all of the tests
//
namespace Unity.Physics.Tests.Dynamics.Motors
{
    // Test runner tools and utilities
    public class MotorTestUtility
    {
        internal static void SetupMotionVelocity(out MotionVelocity velocityA, out MotionVelocity velocityB)
        {
            velocityA = new MotionVelocity
            {
                LinearVelocity = float3.zero,
                AngularVelocity = float3.zero,
                InverseInertia = new float3(6f, 6f, 6f),
                InverseMass = 1.0f,
                AngularExpansionFactor = 0.692820311f
            };

            velocityB = new MotionVelocity();
        }

        internal static void SetupMotionData(RigidTransform worldFromA, RigidTransform worldFromB,
            out MotionData motionA, out MotionData motionB)
        {
            motionA = new MotionData
            {
                WorldFromMotion = worldFromA,
                BodyFromMotion = RigidTransform.identity,
                LinearDamping = 0.0099f,
                AngularDamping = 0.05f
            };

            motionB = new MotionData //use default other than this position
            {
                WorldFromMotion = worldFromB,
                BodyFromMotion = RigidTransform.identity,
                LinearDamping = 0.0f,
                AngularDamping = 0.0f
            };
        }
    }

    //
    // Test runner methods
    //
    public class MotorTestRunner
    {
        internal const float Frequency = 60.0f;
        internal const float Timestep = 1.0f / Frequency;

        // Configures a new Joint based on input constraints between two Bodies
        internal static Joint CreateTestMotor(PhysicsJoint joint) => new Joint
        {
            AFromJoint = joint.BodyAFromJoint.AsMTransform(),
            BFromJoint = joint.BodyBFromJoint.AsMTransform(),
            Constraints = joint.m_Constraints
        };

        internal enum JointType
        {
            PositionMotor = 0,
            RotationMotor = 1,
            LinearVelocityMotor = 2,
            AngularVelocityMotor = 3
        }

        // For the given test configuration and motor types, simulate the motor
        // Returns: accumulated angular velocity, accumulated linear velocity, velocityA, velocityB, motionA, motionB
        // Verifies 2 values: the impulse applied to the linear velocity, the impulse applied to the angular velocity
        internal static void TestSimulateMotor(string testName, ref Joint jointData, JointType jointType,
            ref MotionVelocity velocityA, ref MotionVelocity velocityB, ref MotionData motionA, ref MotionData motionB,
            bool useGravity, float maxImpulse, float3 motorOrientation,
            int numFrames, int numStabilizingSteps,
            out float3 accumulateAngularVelocity, out float3 accumulateLinearVelocity, bool isException = false)
        {
            string failureMessage;
            var impulseThreshold = math.EPSILON;

            float3 gravity = float3.zero;
            if (useGravity) gravity = new float3(0.0f, -9.81f, 0.0f);

            accumulateAngularVelocity = float3.zero;
            accumulateLinearVelocity = float3.zero;

            using var world = new PhysicsWorld(0, 2, 1);
            var joints = world.Joints;
            jointData.BodyPair = new BodyIndexPair { BodyIndexA = 0, BodyIndexB = 1 };
            joints[0] = jointData;

            var velocities = world.MotionVelocities;
            velocityA.GravityFactor = velocityA.InverseMass > 0 ? 1f : 0f;
            velocityB.GravityFactor = velocityB.InverseMass > 0 ? 1f : 0f;
            velocities[0] = velocityA;
            velocities[1] = velocityB;

            var motions = world.MotionDatas;
            motions[0] = motionA;
            motions[1] = motionB;
            var stepInput = new SimulationStepInput
            {
                World = world,
                Gravity = gravity,
                TimeStep = Timestep,
                NumSolverIterations = 4,
                DirectSolverSettings = PhysicsSolver.DirectSolverSettings.Default,
            };
            using var context = Unity.Physics.Simulation.Create();
            context.ResetSimulationContext(stepInput);

            // Simulate frames
            for (int iFrame = 0; iFrame < numFrames + numStabilizingSteps; iFrame++)
            {
                var v0Ang = velocityA.AngularVelocity;
                var v0Lin = velocityA.LinearVelocity * motorOrientation;

                context.Step(stepInput);
                velocityA = velocities[0];
                velocityB = velocities[1];
                motionA = motions[0];
                motionB = motions[1];

                if (jointType == JointType.AngularVelocityMotor || jointType == JointType.RotationMotor)
                {
                    // Verify that the angular maxImpulse for the motor is never exceeded
                    var v1Ang = velocityA.AngularVelocity;
                    var undoApplyAngularImpulse = (v1Ang - v0Ang) / velocityA.InverseInertia;
                    var motorImpulse = math.lengthsq(undoApplyAngularImpulse);
                    var impulseMarginAng = motorImpulse - maxImpulse * maxImpulse;
                    if (isException) //then there is movement due to gravity and no motor impulse is applied
                    {
                        failureMessage =
                            $"{testName}: Angular Motor with impulse {math.sqrt(motorImpulse)} did not move when it should have)";
                        Assert.GreaterOrEqual(impulseMarginAng, impulseThreshold, failureMessage);
                    }
                    else
                    {
                        failureMessage =
                            $"{testName}: Angular Motor impulse {math.sqrt(motorImpulse)} exceeded maximum ({maxImpulse})";
                        Assert.LessOrEqual(impulseMarginAng, impulseThreshold, failureMessage);
                    }
                }

                if (jointType == JointType.LinearVelocityMotor || jointType == JointType.PositionMotor)
                {
                    // Verify that the linear maxImpulse for the motor is never exceeded, but only consider directions
                    // that the motor is acting on
                    var v1Lin = velocityA.LinearVelocity * motorOrientation;
                    var motorImpulseLin = math.lengthsq((v1Lin - v0Lin) / velocityA.InverseMass);
                    var impulseMarginLin = motorImpulseLin - maxImpulse * maxImpulse;
                    if (isException) //then there is movement due to gravity and no motor impulse is applied
                    {
                        failureMessage =
                            $"{testName}: Linear Motor with impulse {math.sqrt(motorImpulseLin)} did not move when it should have)";
                        Assert.GreaterOrEqual(impulseMarginLin, impulseThreshold, failureMessage);
                    }
                    else
                    {
                        failureMessage = $"{testName}: Linear Motor impulse {math.sqrt(motorImpulseLin)} exceeded maximum ({maxImpulse})";
                        Assert.LessOrEqual(impulseMarginLin, impulseThreshold, failureMessage);
                    }
                }

                // Only start to accumulate after a stabilizing velocity has been achieved
                if (iFrame > numStabilizingSteps - 1)
                {
                    accumulateAngularVelocity += velocityA.AngularVelocity;
                    accumulateLinearVelocity += velocityA.LinearVelocity;
                }
            }
        }
    }
}
