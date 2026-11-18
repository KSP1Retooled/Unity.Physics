using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Numerics.Linear;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Physics.Dynamics.Joint.Translation;
using static Unity.Physics.Math;

namespace Unity.Physics
{
    public static partial class Solver
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static unsafe void AddAngularLimit1D(ref JacobianRows jacobianRows, Vector* JBlocks,
            bool bodyAIsStatic, bool bodyBIsStatic, in float3 hingeAxis, float currentAngle, float minAngle, float maxAngle,
            int startRow, in Vector l, in Vector u, ref Vector eps, ref NativeArray<LCP.MLCPIndexFlag> indexSetArray, ref bool isLCP,
            in Vector rhs, in StepInput stepInput, in JacobianHeader.DirectSolverRegularizationData jointRegularizationData,
            in LimitMotion motion)
        {
            const int startColumnA = 0;
            const int startColumnB = 6;

            var J = jacobianRows.Next();
            J.Clear();
            JBlocks[0] = J;

            if (!bodyAIsStatic)
            {
                SetSubVector3(J, hingeAxis, startColumnA + 3);
            }

            if (!bodyBIsStatic)
            {
                SetSubVector3(J, -hingeAxis, startColumnB + 3);
            }

            // Constraint error and gamma term:
            var gamma_phi_0 = 0f;

            if (math.abs(minAngle - maxAngle) < math.EPSILON)
            {
                gamma_phi_0 = -(jointRegularizationData.Gamma / stepInput.Timestep) * (currentAngle - minAngle);
            }
            else if (minAngle > -math.INFINITY || maxAngle < math.INFINITY)
            {
                float rate = motion.Predictive
                    ? RowRate(jacobianRows, startRow, motion.BodyIndexA, motion.BodyIndexB, motion.BodyAIsDynamic,
                        motion.BodyBIsDynamic, motion.MotionVelocities, motion.MotionDatas)
                    : 0.0f;

                gamma_phi_0 = SetLimitRow(currentAngle, minAngle, maxAngle, rate, jointRegularizationData,
                    stepInput.Timestep, startRow, in l, in u, ref indexSetArray, ref isLCP, motion.ConfigurableAngularStop);
            }

            // Assemble right hand side (rhs):

            // Note: \phi(q_) and gamma appear on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
            // After this assignment, we have rhs = -\gamma * \phi(q_)/dt
            // The remaining terms in rhs follow later below.
            rhs[startRow] = gamma_phi_0;

            // epsilon term
            eps[startRow] = jointRegularizationData.Epsilon;
        }

        /// <summary>
        /// What a limit's row needs to tell whether the substep will reach the limit. Without it, the row only acts once the
        /// limit is passed.
        /// </summary>
        struct LimitMotion
        {
            public bool Predictive;
            public bool ConfigurableAngularStop;
            public int BodyIndexA;
            public int BodyIndexB;
            public bool BodyAIsDynamic;
            public bool BodyBIsDynamic;
            public NativeArray<MotionVelocity> MotionVelocities;
            public NativeArray<MotionData> MotionDatas;
        }

        static LimitMotion PredictiveLimit(int bodyIndexA, int bodyIndexB, bool bodyAIsDynamic, bool bodyBIsDynamic,
            NativeArray<MotionVelocity> motionVelocities, NativeArray<MotionData> motionDatas, bool configurableAngularStop = false)
        {
            return new LimitMotion
            {
                Predictive = true,
                ConfigurableAngularStop = configurableAngularStop,
                BodyIndexA = bodyIndexA,
                BodyIndexB = bodyIndexB,
                BodyAIsDynamic = bodyAIsDynamic,
                BodyBIsDynamic = bodyBIsDynamic,
                MotionVelocities = motionVelocities,
                MotionDatas = motionDatas,
            };
        }

        /// <summary>
        /// Arming a row this far inside a limit, in metres or radians, stops a body resting on it from switching the row on
        /// and off.
        /// </summary>
        const float LimitArmingMargin = 0.001f;

        static bool IsNarrowConfigurableStop(ref JacobianHeader header, float minimum, float maximum)
        {
            return (minimum > -JointTranslation.BuildJointsJob.NearAngle || maximum < JointTranslation.BuildJointsJob.NearAngle)
                && (header.Flags & JacobianFlags.EnableImpulseEvents) != 0 && header.AccessImpulseEventSolverData().IsConfigurableLimit;
        }

        /// <summary>
        /// Select the active limit bound and return its gamma*phi/dt correction.
        /// Arm hard limits before crossing to stop at the boundary; activate soft-limit springs only after
        /// crossing.
        /// </summary>
        static float SetLimitRow(float distance, float minimum, float maximum, float rate,
            in JacobianHeader.DirectSolverRegularizationData regularization, float timestep, int row,
            in Vector l, in Vector u, ref NativeArray<LCP.MLCPIndexFlag> indexSetArray, ref bool isLCP, bool configurableAngularStop = false)
        {
            float reach = regularization.Hard ? distance + rate * timestep : distance;
            float margin = regularization.Hard ? LimitArmingMargin : 0.0f;

            isLCP = true;
            float error;
            // Keep the nearer side of a narrow hard angular stop available to the coupled solve.
            // Its impulse remains zero until the other rows would take the joint past the stop.
            bool upper = reach >= 0.5f * (minimum + maximum);
            bool close = upper ? distance >= 0.5f * maximum : distance <= 0.5f * minimum;
            bool arm = regularization.Hard && configurableAngularStop && close;
            if (reach > maximum - margin || (arm && upper))
            {
                error = distance - maximum;
                u[row] = 0;
            }
            else if (reach < minimum + margin || arm)
            {
                error = distance - minimum;
                l[row] = 0;
            }
            else
            {
                // disable the row.
                // @todo direct solver (DOTS-11060): completely discard row instead by reducing
                // row count and using SubMatrix of system matrix for solve. This also doesn't
                // force use of the LCP solver.
                l[row] = 0;
                u[row] = 0;
                indexSetArray[row] = LCP.MLCPIndexFlag.LowerTight;
                return 0.0f;
            }

            // A gap is closed whole: any less would brake the body on its way to the limit.
            bool inside = distance >= minimum && distance <= maximum;
            return -((inside ? 1.0f : regularization.Gamma) / timestep) * error;
        }

        /// <summary> How fast a row's constraint function is changing at the bodies' current velocities. </summary>
        internal static unsafe float RowRate(in JacobianRows jacobianRows, int row, int bodyIndexA, int bodyIndexB,
            bool bodyAIsDynamic, bool bodyBIsDynamic, in NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas)
        {
            const int startColumnA = 0;
            const int startColumnB = 6;

            float rate = 0.0f;
            if (bodyAIsDynamic)
            {
                ComputeRHSGImpulse(jacobianRows, motionVelocities[bodyIndexA], motionDatas[bodyIndexA], row, startColumnA, 1, &rate);
            }

            if (bodyBIsDynamic)
            {
                ComputeRHSGImpulse(jacobianRows, motionVelocities[bodyIndexB], motionDatas[bodyIndexB], row, startColumnB, 1, &rate);
            }

            return rate;
        }
    }
}
