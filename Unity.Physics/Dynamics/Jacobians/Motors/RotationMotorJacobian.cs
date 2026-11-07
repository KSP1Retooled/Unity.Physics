using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Mathematics;
using static Unity.Physics.Math;

namespace Unity.Physics
{
    // Solve data for a constraint that limits one degree of angular freedom
    [NoAlias]
    struct RotationMotorJacobian
    {
        const float SmallestSinHalfAngle = 1e-6f;

        // Limited axis in motion A space
        public float3 AxisInMotionA;
        public float Target;

        public float MinAngle;
        public float MaxAngle;

        // Index of the limited axis
        public int AxisIndex;

        // Motion A orientation in world space
        public quaternion WorldFromMotionA;

        // Relative orientation of the motions before solving
        public quaternion MotionBFromA;

        // Rotation to joint space from motion space
        public quaternion MotionAFromJoint;
        public quaternion MotionBFromJoint;

        // Maximum impulse that can be applied to the motor before it caps out (not a breaking impulse)
        public float MaxImpulseOfMotor;

        // Accumulated impulse applied over the number of solver iterations
        public float AccumulatedImpulse;

        // Error before solving
        public float InitialError;

        // Fraction of the position error to correct per step
        public float Tau;

        // Fraction of the velocity error to correct per step
        public float Damping;

        // Build the Jacobian
        public void Build(
            MTransform aFromConstraint, MTransform bFromConstraint,
            MotionData motionA, MotionData motionB,
            Constraint constraint, float tau, float damping)
        {
            AxisIndex = constraint.ConstrainedAxis1D;
            AxisInMotionA = math.normalize(aFromConstraint.Rotation[AxisIndex]);
            Target = constraint.Target[AxisIndex];
            MinAngle = constraint.Min;
            MaxAngle = constraint.Max;
            Tau = tau;
            Damping = damping;
            MaxImpulseOfMotor = math.abs(constraint.MaxImpulse.x); //using as magnitude, y&z components are unused
            AccumulatedImpulse = 0.0f;

            MotionAFromJoint = new quaternion(aFromConstraint.Rotation);
            MotionBFromJoint = new quaternion(bFromConstraint.Rotation);

            Update(motionA, motionB);
        }

        public void Update(in MotionData motionA, in MotionData motionB)
        {
            MotionBFromA = math.mul(math.inverse(motionB.WorldFromMotion.rot), motionA.WorldFromMotion.rot);

            WorldFromMotionA = motionA.WorldFromMotion.rot;

            // Calculate the current error
            InitialError = CalculateError(MotionBFromA, out float tmp);
        }

        // Solve the Jacobian
        public void Solve(ref JacobianHeader jacHeader, ref MotionVelocity velocityA, ref MotionVelocity velocityB,
            Solver.StepInput stepInput)
        {
            // Predict the relative orientation at the end of the step
            quaternion futureMotionBFromA = JacobianUtilities.IntegrateOrientationBFromA(MotionBFromA,
                velocityA.AngularVelocity, velocityB.AngularVelocity, stepInput.Timestep);

            // Calculate the effective mass
            float3 axisInMotionB = math.mul(futureMotionBFromA, -AxisInMotionA);
            float effectiveMass;
            {
                float invEffectiveMass = math.csum(AxisInMotionA * AxisInMotionA * velocityA.InverseInertia +
                    axisInMotionB * axisInMotionB * velocityB.InverseInertia);
                effectiveMass = math.select(1.0f / invEffectiveMass, 0.0f, invEffectiveMass == 0.0f);
            }

            // Calculate the error, adjust by tau and damping, and apply an impulse to correct it
            float futureError = CalculateError(futureMotionBFromA, out float currentAngle);
            float correction = JacobianUtilities.CalculateCorrection(futureError, InitialError, Tau, Damping);

            var invTimestep = Solver.CalculateInvTimestep(stepInput.Timestep);
            float impulse = math.mul(effectiveMass, -correction) * invTimestep;
            impulse = JacobianUtilities.CapImpulse(impulse, ref AccumulatedImpulse, MaxImpulseOfMotor);

            // check if we hit a limit
            var correctedAngle = currentAngle - correction;
            var limitError = JacobianUtilities.CalculateError(correctedAngle, MinAngle, MaxAngle);
            if (math.abs(limitError) > 0)
            {
                impulse += math.mul(effectiveMass, -limitError) * invTimestep;
            }

            velocityA.ApplyAngularImpulse(impulse * AxisInMotionA);
            velocityB.ApplyAngularImpulse(impulse * axisInMotionB);
        }

        // Helper function
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float CalculateError(quaternion motionBFromA, out float currentAngle)
        {
            // Calculate the relative joint frame rotation
            quaternion jointBFromA = math.mul(math.mul(math.inverse(MotionBFromJoint), motionBFromA), MotionAFromJoint);

            // Quaternion.ToAngleAxis returns zero below about 0.6 mrad, where stiff joints rest, and nearly a full turn when the
            // relative rotation comes out negated. Nothing keeps it positive: its sign is a product of the bodies' rotations
            // and of conversions from the joint frames' matrices.
            float4 rotation = jointBFromA.value.w < 0.0f ? -jointBFromA.value : jointBFromA.value;

            // The share of the rotation about this axis, with the angle over sin(angle / 2) tending to 2 as the angle vanishes.
            float sinHalfAngle = math.length(rotation.xyz);
            float anglePerSin = sinHalfAngle > SmallestSinHalfAngle ? 2.0f * math.atan2(sinHalfAngle, rotation.w) / sinHalfAngle : 2.0f;
            float angle = anglePerSin * rotation[AxisIndex];

            // A rotation only knows its angle to a full turn, and a motor takes the way round nearest its target.
            float error = angle - Target;
            error -= 2.0f * math.PI * math.round(error / (2.0f * math.PI));

            currentAngle = Target + error;
            return error;
        }
    }
}
