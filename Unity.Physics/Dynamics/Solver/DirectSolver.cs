using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Numerics.Linear;
using Unity.Numerics.Linear.Dense.Primitives;
using UnityEngine;
using static Unity.Physics.Math;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        ///     Settings for the <see cref="SolverType">direct solver</see>, an advanced solver for accurate
        ///     physics simulation, enabling complex and realistic joint and contact behavior.
        /// </summary>
        [Serializable]
        public struct DirectSolverSettings
        {
            /// <summary>   The slip a contact will experience in the friction plane when simulated with the direct solver. </summary>
            public float ContactSlip
            {
                get  => m_ContactSlip;
                set => m_ContactSlip = value;
            }
            [SerializeField]
            [Tooltip("The slip a contact will experience in the friction plane when simulated with the direct solver.")]
            float m_ContactSlip;

            /// <summary>   The maximum stiffness a joint can have when simulated with the direct solver. </summary>
            public float MaximumJointStiffness
            {
                get  => m_MaximumJointStiffness;
                set => m_MaximumJointStiffness = value;
            }
            [SerializeField]
            [Tooltip("The maximum stiffness a joint can have when simulated with the direct solver.")]
            float m_MaximumJointStiffness;

            /// <summary>   The maximum damping a joint can have when simulated with the direct solver. </summary>
            public float MaximumJointDamping
            {
                get  => m_MaximumJointDamping;
                set => m_MaximumJointDamping = value;
            }
            [SerializeField]
            [Tooltip("The maximum damping a joint can have when simulated with the direct solver.")]
            float m_MaximumJointDamping;

            /// <summary>   The minimum slip a motor joint will experience when simulated with the direct solver. </summary>
            public float MinimumMotorSlip
            {
                get  => m_MinimumMotorSlip;
                set => m_MinimumMotorSlip = value;
            }
            [SerializeField]
            [Tooltip("The minimum slip a motor joint will experience when simulated with the direct solver.")]
            float m_MinimumMotorSlip;

            /// <summary>   The default direct solver settings. </summary>
            public static readonly DirectSolverSettings Default = new ()
            {
                ContactSlip = 0.001f,
                MaximumJointStiffness = 100000f,
                MaximumJointDamping = 10000f,
                MinimumMotorSlip = 0.0001f,
            };

        }

        #region BuildJacobians

        [BurstCompile]
        struct BuildDirectSolverJacobiansMapJob : IJob
        {
            public NativeStream.Reader JacobiansReader;
            public DispatchPairSequencer.SolverSchedulerInfo SolverSchedulerInfo;
            [ReadOnly] public NativeArray<DispatchPairSequencer.DispatchPair> PhasedDispatchPairs;

            public void Execute()
            {
                BuildDirectSolverJacobiansMap(ref SolverSchedulerInfo, JacobiansReader, PhasedDispatchPairs);
            }
        }

        [BurstCompile]
        struct ParallelBuildDirectSolverJacobiansMapJob : IJobParallelForDefer
        {
            public NativeStream.Reader JacobiansReader;

            [NativeDisableContainerSafetyRestriction]
            public DispatchPairSequencer.SolverSchedulerInfo SolverSchedulerInfo;

            [ReadOnly]
            public NativeArray<DispatchPairSequencer.DispatchPair> PhasedDispatchPairs;

            public void Execute(int workItemIndexOffset)
            {
                BuildDirectSolverJacobiansMap(workItemIndexOffset, ref SolverSchedulerInfo, JacobiansReader, PhasedDispatchPairs);
            }
        }

        /// <summary>
        /// Build Jacobians map required for direct solver.
        /// Does nothing if no direct solver joints or contacts are present.
        /// </summary>
        internal static void BuildDirectSolverJacobiansMap(ref DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo,
            in NativeStream.Reader jacobiansReader, in NativeArray<DispatchPairSequencer.DispatchPair> phasedDispatchPairs)
        {
            var numWorkItems = solverSchedulerInfo.DirectPairsIterativeScheduling.NumWorkItems[0];

            for (int workItemIndexOffset = 0; workItemIndexOffset < numWorkItems; ++workItemIndexOffset)
            {
                BuildDirectSolverJacobiansMap(workItemIndexOffset, ref solverSchedulerInfo, jacobiansReader, phasedDispatchPairs);
            }
        }

        static void BuildDirectSolverJacobiansMap(int workItemIndexOffset,
            ref DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo, in NativeStream.Reader jacobiansReader,
            in NativeArray<DispatchPairSequencer.DispatchPair> phasedDispatchPairs)
        {
            // Map the Jacobians to the direct solver dispatch pairs, so that we can access
            // the corresponding Jacobians out of order (not in stream order) when solving the direct solver islands.
            // Note: Alongside the Jacobian stream we walk through the dispatch pairs in the same order to check if we need
            // to invalidate some mappings in case the pair is invalid and thus shouldn't be processed by the solver.

            int workItemIndex = solverSchedulerInfo.DirectPairsIterativeScheduling.FirstWorkItemIndex.Value + workItemIndexOffset;
            int dispatchPairOffsetInDirectPairs =
                solverSchedulerInfo.DirectPairsIterativeScheduling.GetWorkItemReadOffset(workItemIndexOffset: workItemIndexOffset,
                    out int dispatchPairCount);
            var iterator = new JacobianIterator(jacobiansReader, workItemIndex);
            var firstDirectPairsIndex = solverSchedulerInfo.DirectPairsIterativeScheduling.FirstDispatchPairIndex.Value;
            var jacobianMappings = solverSchedulerInfo.DirectPairsDirectScheduling.PhasedDispatchPairJacobianMappings;

            for (int j = 0; j < dispatchPairCount; ++j)
            {
                var localPhasedDispatchPairIndex = dispatchPairOffsetInDirectPairs + j;
                var globalPhasedDispatchPairIndex = firstDirectPairsIndex + localPhasedDispatchPairIndex;
                var pair = phasedDispatchPairs[globalPhasedDispatchPairIndex];
                if (!pair.IsValid)
                {
                    // invalidate mapping for this pair to make sure it is skipped in the solver (see IslandJacobianIterator).
                    jacobianMappings[localPhasedDispatchPairIndex] = DispatchPairSequencer.DispatchPairJacobianMapping.Invalid;

                    continue;
                }
                // else:
                SafetyChecks.CheckAreEqualAndThrow(true, iterator.HasJacobiansLeft());

                var mapping = new DispatchPairSequencer.DispatchPairJacobianMapping
                {
                    ReaderState = iterator.ReaderState,
                    WorkItemIndex = workItemIndex,
                    NumBlocks = 1
                };

                var jacobian = iterator.ReadJacobianHeader();

                // Skip over all additional Jacobians belonging to the same constraint block, together
                // representing a single joint or contact.
                var blockLength = jacobian.ConstraintBlockInfo.Length;
                SafetyChecks.CheckAreEqualAndThrow(true, blockLength > 0);
                SafetyChecks.CheckAreEqualAndThrow(jacobian.ConstraintBlockInfo.Index, 0);
                for (int c = 1; c < blockLength; ++c)
                {
                    jacobian = iterator.ReadJacobianHeader();
                    SafetyChecks.CheckAreEqualAndThrow(jacobian.ConstraintBlockInfo.Index, c);
                }

                // A pair of composite bodies writes a contact Jacobian for each manifold, which all belong to this pair.
                while (pair.IsContact && iterator.HasJacobiansLeft())
                {
                    JacobianIterator next = iterator;
                    ref JacobianHeader nextHeader = ref next.ReadJacobianHeader();
                    if (nextHeader.BodyPair.BodyIndexA != pair.BodyIndexA || nextHeader.BodyPair.BodyIndexB != pair.BodyIndexB)
                    {
                        break;
                    }

                    iterator = next;
                    mapping.NumBlocks++;
                }

                jacobianMappings[localPhasedDispatchPairIndex] = mapping;
            }
        }

        #endregion //BuildJacobians

        internal const float ContactRowReach = ContactRowLimits.Reach;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int GetConstraintRows(ref JacobianHeader header)
        {
            switch (header.Type)
            {
                case JacobianType.Contact:
                    if (!header.IsContactDynamic || !header.Enabled)
                    {
                        // Nothing to do for contacts that don't involve a dynamic body
                        return 0;
                    }
                    // else:

                    // access contact point information
                    ref var contactJac = ref header.AccessBaseJacobian<ContactJacobian>();
                    // rows corresponds to number of (active!) normal constraints plus 2 linear friction constraints
                    // plus 1 angular friction constraint for the manifold if required
                    int activeNormalConstraints = 0;
                    for (int i = 0; i < contactJac.BaseJacobian.NumContacts; ++i)
                    {
                        ref var contactAngularJac = ref header.AccessAngularJacobian(i);
                        var contactDistance = contactAngularJac.ContactDistance;

                        activeNormalConstraints += math.select(1, 0, contactDistance > ContactRowReach);
                    }
                    // Note: we only add friction rows if the friction coefficient is larger zero and if there are any
                    // active normal contact constraints.
                    return activeNormalConstraints
                        + math.select(0, 3,
                            contactJac.CoefficientOfFriction > 0 && activeNormalConstraints > 0);
                case JacobianType.Trigger:
                    return 0;
                case JacobianType.LinearLimit: // up to 3 linear constraints (3 for ball & socket, 2 for free prismatic)
                    ref var limit = ref header.AccessBaseJacobian<LinearLimitJacobian>();
                    int rows;
                    if (math.all(limit.ConstrainedAxes))
                    {
                        // true value: ball & socket joint with 3 scalar constraint functions
                        // false value: spring / distance constraint with 1 scalar constraint function
                        rows = math.select(1, 3,
                            limit.MinDistance < math.EPSILON && limit.MaxDistance < math.EPSILON);
                    }
                    else
                    {
                        // true value: controlled / limited axis of prismatic, or line constraint with min/max distance,
                        //             both with 1 scalar constraint function
                        // false value: 2D line constraint with 2 scalar constraint functions
                        rows = math.select(2, 1, limit.Is1D // controlled / limited axis of prismatic
                            || !(limit.MinDistance < math.EPSILON && limit.MaxDistance < math.EPSILON)); // line constraint with min/max distance
                    }
                    return rows;
                case JacobianType.AngularLimit1D: // angular constraint on the hinge axis of a locked hinge
                    return 1;
                case JacobianType.AngularLimit2D: // free hinge's angular constraints (2 rows), potentially limited with min/max angle (1 row)
                    ref var angLimit2D = ref header.AccessBaseJacobian<AngularLimit2DJacobian>();
                    return math.select(1, 2,
                        math.abs(angLimit2D.MinAngle) < math.EPSILON && math.abs(angLimit2D.MaxAngle) < math.EPSILON);
                case JacobianType.AngularLimit3D: // fixed relative orientation constraints (prismatic's 3 angular)
                    return 3;
                case JacobianType.RotationMotor:
                    return 1; // A 1D rotation constraint on the controlled axis of a locked / limited hinge joint.
                case JacobianType.AngularVelocityMotor: // a motorized hinge's scalar velocity constraint
                    return 1;
                case JacobianType.PositionMotor:
                    return 1; // A 1D position constraint on the controlled axis of a locked / limited prismatic joint.
                case JacobianType.LinearVelocityMotor: // 3 linear constraints of a motorized prismatic
                    return 3;
                default:
                    SafetyChecks.ThrowNotImplementedException();
                    return 0;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float MulSubVector3(ref Vector a, in float3 b, int row)
        {
            return math.dot(new float3(a[row], a[row + 1], a[row + 2]), b);
        }

        /// <summary>
        /// Sets the 2-component sub-vector located at specified row in 'v' to 'a'.
        /// </summary>
        /// <param name="v"></param>
        /// <param name="a"></param>
        /// <param name="row"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void SetSubVector2(in Vector v, in float2 a, int row)
        {
            for (int i = 0; i < 2; ++i)
            {
                v[row + i] = a[i];
            }
        }

        /// <summary>
        /// Sets the 3-component sub-vector located at specified row in 'v' to 'a'.
        /// </summary>
        /// <param name="v"></param>
        /// <param name="a"></param>
        /// <param name="row"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void SetSubVector3(in Vector v, in float3 a, int row)
        {
            for (int i = 0; i < 3; ++i)
            {
                v[row + i] = a[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void AddConstraintSpaceScalarVelocity(ref float v0, ref Vector J, in float3 v, in float3 w, int column)
        {
            // linear part
            v0 += MulSubVector3(ref J, v, column);
            // angular part
            v0 += MulSubVector3(ref J, w, column + 3);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool ComputeKinematicVelocities(int bodyIndex, in NativeArray<MotionData> motionDatas,
            in NativeArray<MotionVelocity> motionVelocities, ref float3 linearVelocity, ref float3 angularVelocity)
        {
            var velA = motionVelocities[bodyIndex];
            bool isKinematic = velA.IsKinematic;
            if (isKinematic)
            {
                linearVelocity = velA.LinearVelocity;
                angularVelocity = math.rotate(motionDatas[bodyIndex].WorldFromMotion, velA.AngularVelocity);
            }

            return isKinematic;
        }


        /// <summary>
        /// Apply gyroscopic torque to the angular velocity in the provided MotionVelocity, using a stable integrator.
        ///
        /// <remarks>
        /// A simple modification of the velocity update step in the standard semi-implicit Euler
        /// integrator provides better stability. This integrator can be derived from Newton's 2nd law,
        /// by approximating the angular acceleration \dot(w) as (w_{k+1} - w_k)/h
        /// and considering an implicit form of the gyroscopic torque. With Newton's 2nd law for the
        /// angular quantities being
        ///
        ///     \tau = I_0 * \dot(w) + w \cross I_0 * w,
        ///
        /// where I_0 denotes the inertia tensor, w the angular velocity and \tau the external torque,
        /// we then obtain the following discretized form
        ///
        ///     I_0 * (w_{k+1} - w_k)/h = \tau - w_{k+1} \cross I_0 * w_k,
        ///
        /// which can be rewritten as
        ///
        ///     I_0 * (w_{k+1} - w_k) = h * \hat(L) * w_{k+1} + h * \tau,
        ///  => I_0 * w_{k+1} = I_0 * w_k + h * \hat(L) * w_{k+1} + h * \tau
        ///
        /// where w_k and w_{k+1} correspond to the angular velocity vectors at the current and next time
        /// step respectively, L = I_0 * w_k is the angular momentum in the current time step, and
        /// \hat(L) is the skew-symmetric matrix of L, with \hat(a) * b = a \cross b = -b \cross a.
        /// The above equation can be solved for w_{k+1} which yields
        ///
        ///     w_{k+1} = inv(I_0 - h * \hat(L)) * (I_0 * w_k + h * \tau).        (1)
        ///
        /// Note that the modified inertia tensor I_0 - h * \hat(L) is positive definite but not
        /// symmetric, since I_0 is symmetric positive definite and \hat(L) is skew-symmetric.
        /// A symmetric mass matrix can be obtained as follows.
        ///
        /// As shown in Claude Lacoursiere's PhD thesis (Equation 15.28), by expanding the next step's angular velocity
        /// w_{k+1} on the right hand side to 2nd order in time and rearranging the terms, we get the
        /// following update rule for the angular velocity:
        ///
        ///                                            I_0 * w_{k+1} = I_0 * w_k + h * \hat(L) * w_{k+1} + h * \tau
        ///  =>                                                      = I_0 * w_k + h * \hat(L) * (w_k + h * I_0^-1 * \hat(L) * w_{k+1} + h * I_0^-1 * \tau) + h * \tau
        ///  => (I_0 - h^2 * \hat(L) * inv(I_0) * \hat(L)) * w_{k+1} = (I_0 + h * \hat(L)) * w_k + h * (I_3 + h * \hat(L) * I_0^-1) * \tau
        ///  =>                                              w_{k+1} = I_s^-1 * ((I_0 + h * \hat(L)) * w_k + h * (I_3 + h * \hat(L) * I_0^-1) * \tau)
        ///
        /// where I_s = I_0 - h^2 * \hat(L) * I_0^-1 * \hat(L) is a stabilized inertia tensor that is
        /// symmetric and positive definite.
        ///
        /// This can be rearranged as follows:
        ///     w_{k+1} = I_s^-1 * (L - h * w_k \cross L) + h * I_s^-1 * \tau + h^2 * I_s^-1 * \hat(L) * I_0^-1 * \tau. (2)
        ///
        /// As equation (1) above, this new update rule for the angular velocity provides an integrator
        /// with strictly dissipative angular kinetic energy qualities, which improves the stability of
        /// free spinning objects, while also making use of a proper symmetric and positive definite
        /// inertia tensor, as is shown in Claude Lacoursiere's PhD thesis.
        ///
        /// Given that in the current case we have no external torque, i.e., \tau = 0, the update rule
        /// simplifies to
        ///      w_{k+1} = I_s^-1 * (L - h * w_k \cross L).
        /// </remarks>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ApplyGyroscopicTorque(ref MotionVelocity motionVelocity, float timestep)
        {
            // Apply gyroscopic torque and update angular velocity:

            // obtain the principal components (PCs) of the inertia tensor
            var inverseInertiaPCs = motionVelocity.InverseInertia;
            var inertiaPCs = 1 / inverseInertiaPCs;

            var I0 = new float3x3(inertiaPCs.x, 0, 0,
                0, inertiaPCs.y, 0,
                0, 0, inertiaPCs.z);

            float3 L = inertiaPCs * motionVelocity.AngularVelocity; // Warning: don't use mul here so that
                                                                    // we get the Hadamard product and not
                                                                    // the dot product, as desired!
            var hatL = DoubleMath.Cross(L);
#if false
            // Use equation (2):

            // Calculate stabilized inertia tensor I_s = I_0 - h^2 * \hat(L) * I_0^-1 * \hat(L).
            var invI0 = new float3x3(inverseInertiaPCs.x, 0, 0,
                0, inverseInertiaPCs.y, 0,
                0, 0, inverseInertiaPCs.z);
            var hatLinvI0 = math.mul(hatL, invI0);
            var stabilizationTerm = timestep * timestep * math.mul(hatLinvI0, hatL);
            var inertiaTensor = I0 - stabilizationTerm;
            var invInertiaTensor = math.inverse(inertiaTensor);

            var gyroTorque = math.cross(motionVelocity.AngularVelocity, L);
            motionVelocity.AngularVelocity = math.mul(invInertiaTensor, L - timestep * gyroTorque);
#else

            // Use equation (1):
            // Given that we don't require a symmetric mass matrix here, we will just use equation (1)
            // since it requires fewer calculations and produces the same results as equation (2).
            var invInertiaTensor = math.inverse(I0 - timestep * hatL);
            motionVelocity.AngularVelocity = math.mul(invInertiaTensor, L);
#endif
        }
        /// <summary>
        /// Add Jacobian for ball and socket between body A and B to Jacobian matrix J
        /// and return positional constraint error given the current transformation of the two bodies.
        /// </summary>
        /// <returns>positional constraint error given the current transformation of the two bodies</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float3 AddBallAndSocketJacobian(ref RigidTransform worldTransformBodyA, ref RigidTransform worldTransformBodyB,
            ref MTransform localTransformAnchorA, ref MTransform localTransformAnchorB,
            bool bodyAIsStatic, bool bodyBIsStatic,
            ref Vector J_1, ref Vector J_2, ref Vector J_3)
        {
            // p_A+s_A:
            float3 anchor_A = math.transform(worldTransformBodyA,
                localTransformAnchorA.Translation);

            // p_B+s_B:
            float3 anchor_B = math.transform(worldTransformBodyB,
                localTransformAnchorB.Translation);

            // Constraint: \phi(q) = p_A+s_A - (p_B+s_B) = 0
            // Jacobian: J = [id, -\hat(s_A), -id, \hat(s_B)] \in 3x12

            if (!bodyAIsStatic)
            {
                float3 s_A = anchor_A - worldTransformBodyA.pos;

                const int startColumnA = 0;

                var id = float3x3.identity;
                // Note: we can pass columns as rows into SetSubVector3 since id is symmetric
                SetSubVector3(J_1, id.c0, startColumnA);
                SetSubVector3(J_2, id.c1, startColumnA);
                SetSubVector3(J_3, id.c2, startColumnA);

                var nhat_s_A = DoubleMath.Cross(-s_A);
                SetSubVector3(J_1, new float3(nhat_s_A.c0[0], nhat_s_A.c1[0], nhat_s_A.c2[0]), startColumnA + 3);
                SetSubVector3(J_2, new float3(nhat_s_A.c0[1], nhat_s_A.c1[1], nhat_s_A.c2[1]), startColumnA + 3);
                SetSubVector3(J_3, new float3(nhat_s_A.c0[2], nhat_s_A.c1[2], nhat_s_A.c2[2]), startColumnA + 3);
            }

            if (!bodyBIsStatic)
            {
                float3 s_B = anchor_B - worldTransformBodyB.pos;

                const int startColumnB = 6;

                var nid = new float3x3(-1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f);
                // Note: we can pass columns as rows into SetSubVector3 since id is symmetric
                SetSubVector3(J_1, nid.c0, startColumnB);
                SetSubVector3(J_2, nid.c1, startColumnB);
                SetSubVector3(J_3, nid.c2, startColumnB);

                var hat_s_B = DoubleMath.Cross(s_B);
                SetSubVector3(J_1, new float3(hat_s_B.c0[0], hat_s_B.c1[0], hat_s_B.c2[0]), startColumnB + 3);
                SetSubVector3(J_2, new float3(hat_s_B.c0[1], hat_s_B.c1[1], hat_s_B.c2[1]), startColumnB + 3);
                SetSubVector3(J_3, new float3(hat_s_B.c0[2], hat_s_B.c1[2], hat_s_B.c2[2]), startColumnB + 3);
            }

            // Constraint error from last step, at coordinates q_:
            // \phi(q_) = p_A+s_A - (p_B+s_B) = anchor_A - anchor_B
            var phi_0 = anchor_A - anchor_B;
            return phi_0;
        }

        /// <summary>
        /// Add Jacobian for linear limit 2D between body A and B to Jacobian matrix J
        /// and return positional constraint error given the current transformation of the two bodies.
        /// </summary>
        /// <returns> positional constraint error given the current transformation of the two bodies </returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float2 AddLinearLimit2DJacobian(ref RigidTransform worldTransformBodyA,
            ref RigidTransform worldTransformBodyB,
            ref MTransform localTransformAnchorA, ref MTransform localTransformAnchorB,
            int axis0, int axis1,
            bool bodyAIsStatic, bool bodyBIsStatic,
            ref Vector J_1, ref Vector J_2)
        {
            // Constraint:  Two scalar dot-2 constraints,
            //              preventing any linear displacement on the plane
            //              defined by the free prismatic axis, n_B, which is pairwise
            //              orthogonal to the two perpendicular axes u_B and v_B in the
            //              body B frame. Here, n_B, u_B and v_B form the attachment frame
            //              of this joint fixed to body B.
            //
            //      \phi(q)_u = d_AB^t * u_B = 0
            //      \phi(q)_v = d_AB^t * v_B = 0
            //
            //  where d_AB = p_A+s_A - (p_B+s_B) (see ball & socket's constraint functions).

            // The two corresponding 1x12 Jacobian rows are as follows:
            //      J_u = [u_B^t, (u_B x (d_AB-s_A))^t, -u_B, (u_B x s_B)^t ]
            //      J_v = [v_B^t, (v_B x (d_AB-s_A))^t, -v_B, (v_B x s_B)^t ]

            // p_A+s_A:
            float3 anchor_A = math.transform(worldTransformBodyA,
                localTransformAnchorA.Translation);

            // p_B+s_B:
            float3 anchor_B = math.transform(worldTransformBodyB,
                localTransformAnchorB.Translation);

            var d_AB = anchor_A - anchor_B;

            var u_B = math.rotate(worldTransformBodyB, localTransformAnchorB.Rotation[axis0]);
            var v_B = math.rotate(worldTransformBodyB, localTransformAnchorB.Rotation[axis1]);

            if (!bodyAIsStatic)
            {
                const int startColumnA = 0;

                float3 s_A = anchor_A - worldTransformBodyA.pos;

                float3 angTmp = d_AB - s_A;
                float3 J_u_ang = math.cross(u_B, angTmp);
                float3 J_v_ang = math.cross(v_B, angTmp);

                SetSubVector3(J_1, u_B, startColumnA);
                SetSubVector3(J_1, J_u_ang, startColumnA + 3);

                SetSubVector3(J_2, v_B, startColumnA);
                SetSubVector3(J_2, J_v_ang, startColumnA + 3);
            }

            if (!bodyBIsStatic)
            {
                const int startColumnB = 6;

                float3 s_B = anchor_B - worldTransformBodyB.pos;

                float3 J_u_ang = math.cross(u_B, s_B);
                float3 J_v_ang = math.cross(v_B, s_B);

                SetSubVector3(J_1, -u_B, startColumnB);
                SetSubVector3(J_1, J_u_ang, startColumnB + 3);

                SetSubVector3(J_2, -v_B, startColumnB);
                SetSubVector3(J_2, J_v_ang, startColumnB + 3);
            }

            float2 phi_0 = new float2(math.mul(d_AB, u_B), math.mul(d_AB, v_B));

            return phi_0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static unsafe void ComputeRHSGImpulse(in JacobianRows jacobianRows, in MotionVelocity motionVelocity,
            in MotionData motionData, int startRow, int startColumn, int jacRowCount, float* rhsBodyVelocity)
        {
            // bring generalized velocity into constraint space to form rhsBodyVelocity = Ju_ (= JinvM * Mu_):
            var v = motionVelocity.LinearVelocity;
            var worldTransform = motionData.WorldFromMotion;
            var omega = math.mul(worldTransform.rot, motionVelocity.AngularVelocity);

            for (int i = 0; i < jacRowCount; ++i)
            {
                var rowIndex = startRow + i;
                var JBlock = jacobianRows.Row(rowIndex).Subvector(startColumn, 6);
                var JBlockLinear = new float3(JBlock[0], JBlock[1], JBlock[2]);
                var JBlockAngular = new float3(JBlock[3], JBlock[4], JBlock[5]);
                rhsBodyVelocity[i] += math.mul(JBlockLinear, v) + math.mul(JBlockAngular, omega);
            }
        }

        /// <summary>
        /// Shared inputs and outputs for Jacobian row construction. Sparse Jacobians write disjoint rows and may
        /// run in parallel;
        /// dense construction must preserve block and coupling order.
        /// </summary>
        unsafe struct RowBuildContext
        {
            public NativeArray<MotionVelocity> MotionVelocities;
            public NativeArray<MotionData> MotionDatas;
            public StepInput StepInput;
            public JacobianRows JacobianRows;
            public Vector Eps;
            public Vector L;
            public Vector U;
            public Vector RightHandSide;

            // Where a sparse solve keeps the push-out of each contact row that it moves bodies by rather than speeding them up,
            // and -1 on rows that aren't contacts. Null where every push-out is velocity.
            public float* Recovery;

            public NativeArray<LCP.MLCPIndexFlag> IndexSetArray;
            public bool IsLCP;

            // Contacts that follow the island's pairs are solved sparsely, with their normals and without friction.

            // The default regularization of the step's rows, worked out once by a caller that builds many Jacobians, or null.
            public RowRegularization* Regularization;
        }

        /// <summary>
        /// The regularization a Jacobian's rows take from the solver's settings and the substep alone, which is the same for
        /// every Jacobian of a substep.
        /// </summary>
        internal struct RowRegularization
        {
            public float JointEpsilon;
            public float JointGamma;
            public float ContactViscousEpsilon;

            public static RowRegularization Of(in StepInput stepInput)
            {
                RowRegularization terms;
                JacobianUtilities.ComputeDirectSolverViscoelasticRegularizationTerms(out terms.JointEpsilon, out terms.JointGamma,
                    stepInput.DirectSolverSettings.MaximumJointStiffness, stepInput.DirectSolverSettings.MaximumJointDamping,
                    stepInput.Timestep);
                JacobianUtilities.ComputeDirectSolverViscousRegularizationTerm(out terms.ContactViscousEpsilon,
                    stepInput.DirectSolverSettings.ContactSlip, stepInput.Timestep);
                return terms;
            }
        }

        /// <summary>
        /// Build a Jacobian's rows, regularization, bounds and right-hand side starting at startRow.
        /// JBlocks provides scratch space for jacRows rows.
        /// </summary>
        static unsafe void BuildJacobianRows(ref JacobianHeader header, int startRow, int jacRows, ref RowBuildContext context,
            Vector* JBlocks)
        {
            NativeArray<MotionVelocity> motionVelocities = context.MotionVelocities;
            NativeArray<MotionData> motionDatas = context.MotionDatas;
            ref StepInput stepInput = ref context.StepInput;
            JacobianRows jacobianRows = context.JacobianRows.From(startRow);
            ref Vector eps = ref context.Eps;
            ref Vector l = ref context.L;
            ref Vector u = ref context.U;
            ref Vector rhs = ref context.RightHandSide;
            ref NativeArray<LCP.MLCPIndexFlag> indexSetArray = ref context.IndexSetArray;
            ref bool isLCP = ref context.IsLCP;
            RowRegularization terms = context.Regularization != null ? *context.Regularization : RowRegularization.Of(stepInput);
            float defaultJointViscoelasticEpsilon = terms.JointEpsilon;
            float defaultJointViscoelasticGamma = terms.JointGamma;
            float contactViscousEpsilon = terms.ContactViscousEpsilon;

            JacobianHeader.DirectSolverRegularizationData jointRegularizationData = default;

                            int ixBodyA = header.BodyPair.BodyIndexA;
                            int ixBodyB = header.BodyPair.BodyIndexB;

                            // Always store value for body A and B in the 6-component sub-vectors at columns 0 and 6, respectively.
                            const int startColumnA = 0;
                            const int startColumnB = 6;

                            bool bodyAIsStatic = ixBodyA >= motionVelocities.Length;
                            bool bodyBIsStatic = ixBodyB >= motionVelocities.Length;

                            bool bodyAIsKinematic = false;
                            bool bodyBIsKinematic = false;
                            float3 kinematicLinearVelocityA = float3.zero;
                            float3 kinematicLinearVelocityB = float3.zero;
                            float3 kinematicAngularVelocityA = float3.zero;
                            float3 kinematicAngularVelocityB = float3.zero;
                            if (!bodyAIsStatic)
                            {
                                bodyAIsKinematic = ComputeKinematicVelocities(ixBodyA, motionDatas, motionVelocities,
                                    ref kinematicLinearVelocityA, ref kinematicAngularVelocityA);
                            }
                            if (!bodyBIsStatic)
                            {
                                bodyBIsKinematic = ComputeKinematicVelocities(ixBodyB, motionDatas, motionVelocities,
                                    ref kinematicLinearVelocityB, ref kinematicAngularVelocityB);
                            }

                            // get joint regularization parameters:
                            if (header.Type != JacobianType.Contact)
                            {
                                jointRegularizationData = header.AccessDirectSolverRegularizationData();
                            }

                            switch (header.Type)
                            {
                                case JacobianType.AngularVelocityMotor: // motorized hinge's actuated angular constraint
                                {
                                    ref AngularVelocityMotorJacobian angJacobian =
                                        ref header.AccessBaseJacobian<AngularVelocityMotorJacobian>();

                                    // hinge axis in world space
                                    var a = math.rotate(angJacobian.WorldFromA, angJacobian.AxisInMotionA);

                                    var J_1 = jacobianRows.Next();
                                    J_1.Clear();
                                    JBlocks[0] = J_1;
                                    if (!bodyAIsStatic)
                                    {
                                        SetSubVector3(J_1, a, startColumnA + 3);
                                    }

                                    if (!bodyBIsStatic)
                                    {
                                        SetSubVector3(J_1, -a, startColumnB + 3);
                                    }

                                    // Set target velocity v0 but no constraint error \phi(q_), since the constraint is
                                    // velocity based.
                                    float v0 = angJacobian.Target;

                                    // Note: v0 appears on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
                                    // After this assignment, we have rhs = v0, since \phi(q_) = 0, given that
                                    // this constraint is a velocity based constraint (no \phi position error required).
                                    // The remaining terms in rhs follow later below.
                                    rhs[startRow] = v0;

                                    // epsilon term
                                    eps[startRow] = jointRegularizationData.Epsilon;

                                    // set maximum motor impulse
                                    l[startRow] = -angJacobian.MaxImpulseOfMotor;
                                    u[startRow] = +angJacobian.MaxImpulseOfMotor;
                                    isLCP |= angJacobian.MaxImpulseOfMotor < float.MaxValue;

                                    break;
                                }
                                case JacobianType.AngularLimit1D:
                                {
                                    // angular constraint on the hinge axis of a locked or limited hinge
                                    ref AngularLimit1DJacobian angJacobian =
                                        ref header.AccessBaseJacobian<AngularLimit1DJacobian>();

                                    var hingeAxis = math.rotate(angJacobian.WorldFromMotionA, angJacobian.AxisInMotionA);
                                    var currentAngle = angJacobian.CalculateAngle(angJacobian.MotionBFromA);

                                    AddAngularLimit1D(ref jacobianRows, JBlocks,
                                        bodyAIsStatic, bodyBIsStatic, in hingeAxis, currentAngle,
                                        angJacobian.MinAngle, angJacobian.MaxAngle,
                                        startRow, in l, in u, ref eps, ref indexSetArray, ref isLCP,
                                        in rhs, in stepInput, in jointRegularizationData,
                                        PredictiveLimit(ixBodyA, ixBodyB, !bodyAIsStatic && !bodyAIsKinematic,
                                            !bodyBIsStatic && !bodyBIsKinematic, motionVelocities, motionDatas,
                                            IsNarrowConfigurableStop(ref header, angJacobian.MinAngle, angJacobian.MaxAngle)));

                                    break;
                                }
                                case JacobianType.AngularLimit2D: // free hinge's 2 angular constraints
                                {
                                    ref AngularLimit2DJacobian angJacobian =
                                        ref header.AccessBaseJacobian<AngularLimit2DJacobian>();

                                    if (math.abs(angJacobian.MinAngle) < math.EPSILON && math.abs(angJacobian.MaxAngle) < math.EPSILON)
                                    {
                                        // Constraint: two scalar dot-1 constraints
                                        //      \phi(q)_u = n_A^t * u_B = 0
                                        //      \phi(q)_v = n_A^t * v_B = 0
                                        // with n_A denoting the hinge rotation axis attached to body A and
                                        // u_B and v_B denoting two axes attached to body B which are perpendicular to each
                                        // other and are kept orthogonal to n_A through the two scalar constraints.

                                        // The two 1x12 Jacobian rows are as follows:
                                        //      J_u = [0_{1x3}, (\hat(n_A) * u_B)^t, 0_{1x3}, -(\hat(n_A) * u_B)^t ]
                                        //          = [0_{1x3}, (n_A x u_B)^t, 0_{1x3}, -(n_A x u_B)^t ]
                                        // Analogously:
                                        //      J_v = [0_{1x3}, (n_A x v_B)^t, 0_{1x3}, -(n_A x v_B)^t ]

                                        // @todo direct solver (DOTS-11060): compute n_A, u_B and v_B in BuildJacobian of AngularLimit2DJacobian so that we
                                        // don't need to store all this data (WorldFromA/B and AnchorFrameInBodyA/B) in the jacobian struct.
                                        var n_A = math.rotate(angJacobian.WorldFromA, angJacobian.AnchorFrameInBodyA.Rotation[angJacobian.FreeIndex]);
                                        var u_B = math.rotate(angJacobian.WorldFromB, angJacobian.AnchorFrameInBodyB.Rotation[angJacobian.ConstraintIndexX]);
                                        var v_B = math.rotate(angJacobian.WorldFromB, angJacobian.AnchorFrameInBodyB.Rotation[angJacobian.ConstraintIndexY]);

                                        var J_u = math.cross(n_A, u_B);
                                        var J_v = math.cross(n_A, v_B);

                                        var J_1 = jacobianRows.Next();
                                        var J_2 = jacobianRows.Next();
                                        J_1.Clear();
                                        J_2.Clear();
                                        JBlocks[0] = J_1;
                                        JBlocks[1] = J_2;

                                        if (!bodyAIsStatic)
                                        {
                                            SetSubVector3(J_1, J_u, startColumnA + 3);
                                            SetSubVector3(J_2, J_v, startColumnA + 3);
                                        }

                                        if (!bodyBIsStatic)
                                        {
                                            SetSubVector3(J_1, -J_u, startColumnB + 3);
                                            SetSubVector3(J_2, -J_v, startColumnB + 3);
                                        }

                                        // Constraint error from last step, at coordinates q_:
                                        //      \phi(q_)_u = n_A^t * u_B
                                        //      \phi(q_)_v = n_A^t * v_B
                                        // including a mapping to angles for correct constraint regularization via acos.
                                        float2 phi_0 = math.PIHALF - new float2(
                                            math.acos(math.mul(n_A, u_B)),
                                            math.acos(math.mul(n_A, v_B)));

                                        // Assemble right hand side (rhs):

                                        // Constraint error and gamma term
                                        var gamma_phi_0 = -(jointRegularizationData.Gamma / stepInput.Timestep) * phi_0;

                                        // Note: \phi(q_) and gamma appear on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
                                        // After this assignment, we have rhs = -\gamma * \phi(q_)/dt
                                        // The remaining terms in rhs follow later below.
                                        SetSubVector2(rhs, gamma_phi_0, startRow);

                                        // epsilon term
                                        SetSubVector2(eps, jointRegularizationData.Epsilon, startRow);
                                    }
                                    else
                                    {
                                        // min / max angle limits are specified: only add a single constraint row if we
                                        // are outside the limits.

                                        var axisAinB = math.mul(angJacobian.BFromA, angJacobian.AxisAinA);
                                        var rotationAxis = -math.cross(axisAinB, angJacobian.AxisBinB);
                                        var sinAngle = math.length(rotationAxis);
                                        var cosAngle = math.dot(axisAinB, angJacobian.AxisBinB);
                                        var currentAngle = math.atan2(sinAngle, cosAngle);
                                        // A cone has no lower angular limit. Keep the bound finite because an
                                        // unbounded row is treated as equality.
                                        float minAngle = angJacobian.MinAngle > 0.0f ? angJacobian.MinAngle : -float.MaxValue;
                                        bool narrow = IsNarrowConfigurableStop(ref header, minAngle, angJacobian.MaxAngle);

                                        if (sinAngle > math.EPSILON)
                                        {
                                            // The axes are in body B's motion space, and a row's Jacobian is in world space.
                                            rotationAxis = math.rotate(angJacobian.WorldFromB.rot, rotationAxis / sinAngle);
                                        }
                                        else
                                        {
                                            // axes are parallel, so we can choose any axis perpendicular to axisAinB
                                            // as rotation axis.
                                            rotationAxis = math.mul(angJacobian.WorldFromA.rot,
                                                angJacobian.AnchorFrameInBodyA.Rotation[angJacobian.ConstraintIndexX]);
                                            if (narrow)
                                            {
                                                float3 angularA = bodyAIsStatic ? float3.zero : math.rotate(angJacobian.WorldFromA.rot,
                                                    motionVelocities[ixBodyA].AngularVelocity);
                                                float3 angularB = bodyBIsStatic ? float3.zero : math.rotate(angJacobian.WorldFromB.rot,
                                                    motionVelocities[ixBodyB].AngularVelocity);
                                                float3 axis = math.rotate(angJacobian.WorldFromB.rot, angJacobian.AxisBinB);
                                                float3 swing = angularA - angularB;
                                                rotationAxis = math.normalizesafe(swing - math.dot(axis, swing) * axis, rotationAxis);
                                            }
                                        }

                                        AddAngularLimit1D(ref jacobianRows, JBlocks,
                                            bodyAIsStatic, bodyBIsStatic, in rotationAxis, currentAngle,
                                            minAngle, angJacobian.MaxAngle,
                                            startRow, in l, in u, ref eps, ref indexSetArray, ref isLCP,
                                            in rhs, in stepInput, in jointRegularizationData,
                                            PredictiveLimit(ixBodyA, ixBodyB, !bodyAIsStatic && !bodyAIsKinematic,
                                                !bodyBIsStatic && !bodyBIsKinematic, motionVelocities, motionDatas, narrow));
                                    }

                                    break;
                                }
                                case JacobianType.AngularLimit3D: // fixed relative orientation constraints
                                {
                                    ref AngularLimit3DJacobian angJacobian =
                                        ref header.AccessBaseJacobian<AngularLimit3DJacobian>();

                                    if (math.abs(angJacobian.MinAngle) < math.EPSILON && math.abs(angJacobian.MaxAngle) < math.EPSILON)
                                    {
                                        // Constraint: three scalar dot-1 constraints
                                        //      \phi(q)_u = n_A^t * u_B = 0
                                        //      \phi(q)_v = n_A^t * v_B = 0
                                        //      \phi(q)_n = u_A^t * v_B = 0
                                        // with n_A and u_A denoting two orthogonal axes attached to body A and
                                        // u_B and v_B denoting two orthogonal axes attached to body B.
                                        // In both body frames, the axes n_i, u_i and v_i form an orthogonal coordinate frame,
                                        // with i \in {A,B}.
                                        // This joint keeps the two bodies' coordinate frames aligned by ensuring
                                        // that the pairs of axes (n_A, u_B), (n_A, v_B), and (u_A, v_B) are kept orthogonal
                                        // to each other respectively.

                                        // The three 1x12 Jacobian rows are as follows:
                                        //      J_u = [0_{1x3}, (\hat(n_A) * u_B)^t, 0_{1x3}, -(\hat(n_A) * u_B)^t ]
                                        //          = [0_{1x3}, (n_A x u_B)^t, 0_{1x3}, -(n_A x u_B)^t ]
                                        // Analogously:
                                        //      J_v = [0_{1x3}, (n_A x v_B)^t, 0_{1x3}, -(n_A x v_B)^t ]
                                        //      J_n = [0_{1x3}, (u_A x v_B)^t, 0_{1x3}, -(u_A x v_B)^t ]

                                        // @todo direct solver (DOTS-11060): compute n_A, uA, u_B and v_B in BuildJacobian of AngularLimit3DJacobian so that we
                                        // don't need to store all this data (WorldFromA/B and AnchorFrameInBodyA/B) in the jacobian struct.
                                        var n_A = math.rotate(angJacobian.WorldFromA, angJacobian.AnchorFrameInBodyA.Rotation[0]);
                                        var u_A = math.rotate(angJacobian.WorldFromA, angJacobian.AnchorFrameInBodyA.Rotation[1]);

                                        var u_B = math.rotate(angJacobian.WorldFromB, angJacobian.AnchorFrameInBodyB.Rotation[1]);
                                        var v_B = math.rotate(angJacobian.WorldFromB, angJacobian.AnchorFrameInBodyB.Rotation[2]);

                                        var J_u = math.cross(n_A, u_B);
                                        var J_v = math.cross(n_A, v_B);
                                        var J_n = math.cross(u_A, v_B);

                                        var J_1 = jacobianRows.Next();
                                        var J_2 = jacobianRows.Next();
                                        var J_3 = jacobianRows.Next();
                                        J_1.Clear();
                                        J_2.Clear();
                                        J_3.Clear();
                                        JBlocks[0] = J_1;
                                        JBlocks[1] = J_2;
                                        JBlocks[2] = J_3;
                                        if (!bodyAIsStatic)
                                        {
                                            SetSubVector3(J_1, J_u, startColumnA + 3);
                                            SetSubVector3(J_2, J_v, startColumnA + 3);
                                            SetSubVector3(J_3, J_n, startColumnA + 3);
                                        }

                                        if (!bodyBIsStatic)
                                        {
                                            SetSubVector3(J_1, -J_u, startColumnB + 3);
                                            SetSubVector3(J_2, -J_v, startColumnB + 3);
                                            SetSubVector3(J_3, -J_n, startColumnB + 3);
                                        }

                                        // Constraint error from last step, at coordinates q_:
                                        //      \phi(q_)_u = n_A^t * u_B
                                        //      \phi(q_)_v = n_A^t * v_B
                                        //      \phi(q_)_n = u_A^t * v_B
                                        // including a mapping to angles for correct constraint regularization via acos.
                                        float3 phi_0 = math.PIHALF - new float3(
                                            math.acos(math.mul(n_A, u_B)),
                                            math.acos(math.mul(n_A, v_B)),
                                            math.acos(math.mul(u_A, v_B)));

                                        // Assemble right hand side (rhs):

                                        // Constraint error and gamma term
                                        var gamma_phi_0 = -(jointRegularizationData.Gamma / stepInput.Timestep) * phi_0;

                                        // Note: \phi(q_) and gamma appear on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
                                        // After this assignment, we have rhs = -\gamma * \phi(q_)/dt
                                        // The remaining terms in rhs follow later below.
                                        SetSubVector3(rhs, gamma_phi_0, startRow);

                                        // epsilon term
                                        SetSubVector3(eps, jointRegularizationData.Epsilon, startRow);
                                    }
                                    else
                                    {
                                        var deltaRotation = math.mul(math.inverse(angJacobian.RefBFromA), angJacobian.BFromA);
                                        ((Quaternion)deltaRotation).ToAngleAxis(out float currentAngle, out Vector3 hingeAxisVec3);
                                        float3 hingeAxis = hingeAxisVec3;

                                        // ToAngleAxis gives degrees about an axis in body A's space, so this angle's rate is
                                        // not the row's, and the limit acts only once it is passed.
                                        AddAngularLimit1D(ref jacobianRows, JBlocks,
                                            bodyAIsStatic, bodyBIsStatic, in hingeAxis, currentAngle,
                                            angJacobian.MinAngle, angJacobian.MaxAngle,
                                            startRow, in l, in u, ref eps, ref indexSetArray, ref isLCP,
                                            in rhs, in stepInput, in jointRegularizationData, default);
                                    }
                                    break;
                                }
                                case JacobianType.LinearLimit:
                                {
                                    ref LinearLimitJacobian limitJacobian =
                                        ref header.AccessBaseJacobian<LinearLimitJacobian>();

                                    // 3D case: ball & socket constraint
                                    var lockAll = math.all(limitJacobian.ConstrainedAxes);
                                    if (lockAll && limitJacobian.MinDistance < math.EPSILON && limitJacobian.MaxDistance < math.EPSILON)
                                    {
                                        // ball & socket case: 3 linear constraint functions
                                        var J_1 = jacobianRows.Next();
                                        var J_2 = jacobianRows.Next();
                                        var J_3 = jacobianRows.Next();
                                        J_1.Clear();
                                        J_2.Clear();
                                        J_3.Clear();
                                        JBlocks[0] = J_1;
                                        JBlocks[1] = J_2;
                                        JBlocks[2] = J_3;
                                        // Add Jacobian to J and obtain constraint error from last step.
                                        float3 phi_0 = AddBallAndSocketJacobian(ref limitJacobian.WorldFromA,
                                            ref limitJacobian.WorldFromB,
                                            ref limitJacobian.BodyFromConstraintA,
                                            ref limitJacobian.BodyFromConstraintB,
                                            bodyAIsStatic, bodyBIsStatic,
                                            ref J_1, ref J_2, ref J_3);

                                        // Assemble right hand side (rhs):

                                        // Constraint error and gamma term:
                                        var gamma_phi_0 = -(jointRegularizationData.Gamma / stepInput.Timestep) * phi_0;

                                        // Note: \phi(q_) and gamma appear on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
                                        // After this assignment, we have rhs = -\gamma * \phi(q_)/dt.
                                        // The remaining terms in rhs follow later below.
                                        SetSubVector3(rhs, gamma_phi_0, startRow);

                                        // epsilon term
                                        SetSubVector3(eps, jointRegularizationData.Epsilon, startRow);
                                    }
                                    // 2D case: line constraint
                                    else if (!lockAll && !limitJacobian.Is1D &&
                                        limitJacobian.MinDistance < math.EPSILON && limitJacobian.MaxDistance < math.EPSILON)
                                    {
                                        // 2D case: line constraint

                                        int axis0 = limitJacobian.ConstrainedAxes[0] ? 0 :
                                            (limitJacobian.ConstrainedAxes[1] ? 1 : -1);
                                        SafetyChecks.CheckAreEqualAndThrow(true, axis0 != -1);

                                        int axis1 = axis0 == 1 ? 2 : (limitJacobian.ConstrainedAxes[1] ? 1 : 2);

                                        SafetyChecks.CheckAreEqualAndThrow(true, (axis0 == 0 && (axis1 == 1 || axis1 == 2)) || (axis0 == 1 && axis1 == 2));

                                        // 2D case: linear limit 2D's 2 linear constraint functions

                                        var J_1 = jacobianRows.Next();
                                        var J_2 = jacobianRows.Next();
                                        J_1.Clear();
                                        J_2.Clear();
                                        JBlocks[0] = J_1;
                                        JBlocks[1] = J_2;

                                        var phi_0 = AddLinearLimit2DJacobian(ref limitJacobian.WorldFromA,
                                            ref limitJacobian.WorldFromB,
                                            ref limitJacobian.BodyFromConstraintA,
                                            ref limitJacobian.BodyFromConstraintB,
                                            axis0, axis1, bodyAIsStatic, bodyBIsStatic,
                                            ref J_1, ref J_2);

                                        // Assemble right hand side (rhs):

                                        // Constraint error and gamma term:
                                        var gamma_phi_0 = -(jointRegularizationData.Gamma / stepInput.Timestep) * phi_0;

                                        // Note: \phi(q_) and gamma appear on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
                                        // After this assignment, we have rhs = -\gamma * \phi(q_)/dt.
                                        // The remaining terms in rhs follow later below.
                                        SetSubVector2(rhs, gamma_phi_0, startRow);

                                        // epsilon term
                                        SetSubVector2(eps, jointRegularizationData.Epsilon, startRow);
                                    }
                                    // Cases: 1D constraint, 3D spring / distance constraint, or 2D line constraint with min/max distance
                                    else
                                    {
                                        var J = jacobianRows.Next();
                                        J.Clear();
                                        JBlocks[0] = J;
                                        // Constraint:  A scalar dot-2 constraint, preventing any linear displacement along
                                        //              the line defined by the axis n_B, which is defined in body B frame
                                        //              in the 1D constraint case, and which is defined dynamically as the
                                        //              direction between the two anchor points in the spring / distance
                                        //              constraint case.
                                        //
                                        //      \phi(q)_n = d_AB^t * n_B = 0
                                        //
                                        //  where d_AB = p_A+s_A - (p_B+s_B) (see ball & socket's constraint functions).

                                        // The corresponding 1x12 Jacobian row is as follows:
                                        //      J_n = [n_B^t, (n_B x (d_AB-s_A))^t, -n_B^t, (n_B x s_B)^t ]

                                        // p_A+s_A:
                                        float3 anchor_A = math.transform(limitJacobian.WorldFromA,
                                            limitJacobian.BodyFromConstraintA.Translation);

                                        // p_B+s_B:
                                        float3 anchor_B = math.transform(limitJacobian.WorldFromB,
                                            limitJacobian.BodyFromConstraintB.Translation);

                                        var d_AB = anchor_A - anchor_B;
                                        var n_B = new float3(0, 1, 0);

                                        if (limitJacobian.Is1D) // 1D case: controlled or limited prismatic axis
                                        {
                                            n_B = math.rotate(limitJacobian.WorldFromB, limitJacobian.AxisInB);
                                        }
                                        else if (lockAll) // 3D case: spring / distance constraint
                                        {
                                            var direction = d_AB;
                                            var distanceSq = math.lengthsq(direction);

                                            if (distanceSq > math.EPSILON)
                                            {
                                                direction /= math.sqrt(distanceSq);
                                            }
                                            else
                                            {
                                                direction = new float3(0, -1, 0);
                                            }

                                            n_B = direction;
                                        }
                                        else // 2D case: line constraint with min/max distance
                                        {
                                            // choose as axis the normalized vector rejection of the anchor point offset
                                            // vector onto the line defined by AxisInB
                                            var axis = math.rotate(limitJacobian.WorldFromB, limitJacobian.AxisInB);
                                            var direction = d_AB - math.dot(axis, d_AB) * axis;
                                            var distanceSq = math.lengthsq(direction);
                                            if (distanceSq > math.EPSILON)
                                            {
                                                direction /= math.sqrt(distanceSq);
                                            }
                                            else
                                            {
                                                // pick any direction perpendicular to the free axis
                                                int otherAxis = limitJacobian.ConstrainedAxes[0] ? 0 :
                                                    (limitJacobian.ConstrainedAxes[1] ? 1 : -1);
                                                SafetyChecks.CheckAreEqualAndThrow(true, otherAxis != -1);
                                                direction = math.rotate(limitJacobian.WorldFromB,
                                                    limitJacobian.BodyFromConstraintB.Rotation[otherAxis]);
                                            }

                                            n_B = direction;
                                        }

                                        var currentDistance = math.mul(d_AB, n_B);

                                        if (!bodyAIsStatic)
                                        {
                                            float3 s_A = anchor_A - limitJacobian.WorldFromA.pos;

                                            float3 angTmp = d_AB - s_A;
                                            float3 J_n_ang = math.cross(n_B, angTmp);

                                            SetSubVector3(J, n_B, startColumnA);
                                            SetSubVector3(J, J_n_ang, startColumnA + 3);
                                        }

                                        if (!bodyBIsStatic)
                                        {
                                            float3 s_B = anchor_B - limitJacobian.WorldFromB.pos;

                                            float3 J_n_ang = math.cross(n_B, s_B);

                                            SetSubVector3(J, -n_B, startColumnB);
                                            SetSubVector3(J, J_n_ang, startColumnB + 3);
                                        }

                                        // Assemble right hand side (rhs):

                                        // Constraint error and gamma term:
                                        float gamma_phi_0;
                                        if (math.abs(limitJacobian.MinDistance - limitJacobian.MaxDistance) < math.EPSILON)
                                        {
                                            gamma_phi_0 = -(jointRegularizationData.Gamma / stepInput.Timestep) *
                                                (currentDistance - limitJacobian.MaxDistance);
                                        }
                                        else
                                        {
                                            float rate = RowRate(jacobianRows, startRow, ixBodyA, ixBodyB,
                                                !bodyAIsStatic && !bodyAIsKinematic, !bodyBIsStatic && !bodyBIsKinematic,
                                                motionVelocities, motionDatas);

                                            gamma_phi_0 = SetLimitRow(currentDistance, limitJacobian.MinDistance,
                                                limitJacobian.MaxDistance, rate, jointRegularizationData, stepInput.Timestep,
                                                startRow, in l, in u, ref indexSetArray, ref isLCP);
                                        }

                                        // Note: \phi(q_) and gamma appear on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
                                        // After this assignment, we have rhs = -\gamma * \phi(q_)/dt.
                                        // The remaining terms in rhs follow later below.
                                        rhs[startRow] = gamma_phi_0;

                                        // epsilon term
                                        eps[startRow] = jointRegularizationData.Epsilon;
                                    }
                                    break;
                                }
                                case JacobianType.LinearVelocityMotor:
                                {
                                    ref LinearVelocityMotorJacobian motorJacobian =
                                        ref header.AccessBaseJacobian<LinearVelocityMotorJacobian>();

                                    // 2 constraints for LinearLimit2D plus one extra linear motor row for motorized prismatic axis.

                                    // LinearLimit2D rows:

                                    int axis1 = (motorJacobian.AxisIndex + 1) % 3;
                                    int axis2 = (motorJacobian.AxisIndex + 2) % 3;

                                    var J_1 = jacobianRows.Next();
                                    var J_2 = jacobianRows.Next();
                                    var J_3 = jacobianRows.Next();
                                    J_1.Clear();
                                    J_2.Clear();
                                    J_3.Clear();
                                    JBlocks[0] = J_1;
                                    JBlocks[1] = J_2;
                                    JBlocks[2] = J_3;
                                    // positional error term for two first rows
                                    var phi_0 = AddLinearLimit2DJacobian(ref motorJacobian.WorldFromA,
                                        ref motorJacobian.WorldFromB,
                                        ref motorJacobian.AnchorFrameInBodyA,
                                        ref motorJacobian.AnchorFrameInBodyB,
                                        axis1, axis2, bodyAIsStatic, bodyBIsStatic,
                                        ref J_1, ref J_2);

                                    // Linear motor row:

                                    var motorAxis = math.rotate(motorJacobian.WorldFromB, motorJacobian.AxisInB);
                                    if (!bodyAIsStatic)
                                    {
                                        SetSubVector3(J_3, motorAxis, startColumnA);
                                    }

                                    if (!bodyBIsStatic)
                                    {
                                        SetSubVector3(J_3, -motorAxis, startColumnB);
                                    }

                                    // Assemble right hand side (rhs):

                                    // Constraint error and gamma term, for two first rows (the LinearLimt2D part):

                                    // Use the default viscoelastic gamma for the first two rows which are not motorized.
                                    var gamma_phi_0 = -(defaultJointViscoelasticGamma / stepInput.Timestep) * phi_0;

                                    // For last row (startRow + 2), add motor target velocity to v0.
                                    // Note: for this row we have no constraint error term \phi(q_), since this constraint
                                    // is velocity based.
                                    float targetVelocity = math.mul(motorJacobian.Target, motorJacobian.AxisInB);

                                    SetSubVector3(rhs, new float3(gamma_phi_0.x, gamma_phi_0.y, targetVelocity), startRow);

                                    // epsilon term:
                                    // Note: we use the default viscoelastic epsilon for the first two rows which are
                                    // not motorized and the joint's (viscous) epsilon for the last row which is
                                    // motorized (velocity constraint).
                                    SetSubVector2(eps, defaultJointViscoelasticEpsilon, startRow);
                                    eps[startRow + 2] = jointRegularizationData.Epsilon;

                                    // set maximum motor impulse
                                    l[startRow + 2] = -motorJacobian.MaxImpulseOfMotor;
                                    u[startRow + 2] = +motorJacobian.MaxImpulseOfMotor;
                                    isLCP |= motorJacobian.MaxImpulseOfMotor < float.MaxValue;

                                    break;
                                }
                                case JacobianType.PositionMotor:
                                {
                                    ref PositionMotorJacobian motorJacobian =
                                        ref header.AccessBaseJacobian<PositionMotorJacobian>();

                                    var J = jacobianRows.Next();
                                    J.Clear();
                                    JBlocks[0] = J;
                                    // Constraint:  A scalar dot-2 constraint, preventing any linear displacement along
                                    //              the line defined by the axis n_B, defined in body B frame.
                                    //
                                    //      \phi(q)_n = d_BA^t * n_B = 0
                                    //
                                    //  where d_BA = p_B+s_B - (p_A+s_A) (see ball & socket's constraint functions).

                                    // The corresponding 1x12 Jacobian row is as follows:
                                    //      J_n = [-n_B^t, (n_B x s_A)^t, n_B^t, (n_B x (d_BA-s_B))^t]

                                    // p_B+s_B:
                                    float3 anchor_B = math.transform(motorJacobian.WorldFromB, motorJacobian.TargetInB);

                                    // p_A+s_A:
                                    float3 anchor_A = math.transform(motorJacobian.WorldFromA, motorJacobian.PivotAinA);

                                    var d_BA = anchor_B - anchor_A;

                                    var n_B = math.rotate(motorJacobian.WorldFromB, motorJacobian.AxisInB);

                                    if (!bodyAIsStatic)
                                    {
                                        float3 s_A = anchor_A - motorJacobian.WorldFromA.pos;

                                        float3 J_n_ang = math.cross(n_B, s_A);

                                        SetSubVector3(J, -n_B, startColumnA);
                                        SetSubVector3(J, J_n_ang, startColumnA + 3);
                                    }

                                    if (!bodyBIsStatic)
                                    {
                                        float3 s_B = anchor_B - motorJacobian.WorldFromB.pos;

                                        float3 angTmp = d_BA - s_B;
                                        float3 J_n_ang = math.cross(n_B, angTmp);

                                        SetSubVector3(J, n_B, startColumnB);
                                        SetSubVector3(J, J_n_ang, startColumnB + 3);
                                    }

                                    var phi_0 = math.mul(d_BA, n_B);

                                    // Assemble right hand side (rhs):

                                    // Constraint error and gamma term:
                                    var gamma_phi_0 = -(jointRegularizationData.Gamma / stepInput.Timestep) * phi_0;

                                    // Note: \phi(q_) and gamma appear on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
                                    // After this assignment, we have rhs = -\gamma * \phi(q_)/dt.
                                    // The remaining terms in rhs follow later below.
                                    rhs[startRow] = gamma_phi_0;

                                    // epsilon term
                                    eps[startRow] = jointRegularizationData.Epsilon;

                                    // set maximum motor impulse
                                    l[startRow] = -motorJacobian.MaxImpulseOfMotor;
                                    u[startRow] = +motorJacobian.MaxImpulseOfMotor;
                                    isLCP |= motorJacobian.MaxImpulseOfMotor < float.MaxValue;

                                    break;
                                }
                                case JacobianType.RotationMotor:
                                {
                                    ref RotationMotorJacobian motorJacobian =
                                        ref header.AccessBaseJacobian<RotationMotorJacobian>();

                                    var hingeAxis = math.rotate(motorJacobian.WorldFromMotionA, motorJacobian.AxisInMotionA);

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

                                    // Constraint error from last step, at coordinates q_:
                                    var phi_0 = motorJacobian.CalculateError(motorJacobian.MotionBFromA, out var currentAngle);

                                    var gamma = jointRegularizationData.Gamma;
                                    var epsilon = jointRegularizationData.Epsilon;

                                    var lowerForce = -motorJacobian.MaxImpulseOfMotor;
                                    var upperForce = +motorJacobian.MaxImpulseOfMotor;

                                    // check whether we hit the limits and choose the constraint regularization accordingly.
                                    // If we are within the limits, we use the user-specified regularization parameters.
                                    // Otherwise, we use the maximum joint stiffness and damping parameters to enforce
                                    // perfectly hard limits.
                                    var hasLimits = motorJacobian.MinAngle > -math.INFINITY || motorJacobian.MaxAngle < math.INFINITY;
                                    if (hasLimits)
                                    {
                                        if (currentAngle > motorJacobian.MaxAngle)
                                        {
                                            phi_0 = currentAngle - motorJacobian.MaxAngle;

                                            upperForce = 0;
                                            isLCP = true;

                                            epsilon = defaultJointViscoelasticEpsilon;
                                            gamma = defaultJointViscoelasticGamma;
                                        }
                                        else if (currentAngle < motorJacobian.MinAngle)
                                        {
                                            phi_0 = currentAngle - motorJacobian.MinAngle;

                                            lowerForce = 0;
                                            isLCP = true;

                                            epsilon = defaultJointViscoelasticEpsilon;
                                            gamma = defaultJointViscoelasticGamma;
                                        }
                                    }

                                    // Assemble right hand side (rhs):

                                    // Constraint error and gamma term
                                    var gamma_phi_0 = -(gamma / stepInput.Timestep) * phi_0;

                                    // Note: \phi(q_) and gamma appear on rhs as rhs = v0 - \gamma * \phi(q_)/dt - JinvM (...)
                                    // After this assignment, we have rhs = -\gamma * \phi(q_)/dt
                                    // The remaining terms in rhs follow later below.
                                    rhs[startRow] = gamma_phi_0;

                                    // epsilon term
                                    eps[startRow] = epsilon;

                                    // set maximum motor impulse
                                    l[startRow] = lowerForce;
                                    u[startRow] = upperForce;
                                    isLCP |= motorJacobian.MaxImpulseOfMotor < float.MaxValue;

                                    break;
                                }
                                case JacobianType.Contact:
                                {
                                    // access contact point information
                                    ref ContactJacobian contactJacobian = ref header.AccessBaseJacobian<ContactJacobian>();

                                    var numContacts = contactJacobian.BaseJacobian.NumContacts;
                                    var normal = contactJacobian.BaseJacobian.Normal;

                                    float3 s_A_avg = float3.zero;
                                    float3 s_B_avg = float3.zero;

                                    // @todo direct solver (DOTS-11060): could store body positions in ContactJacobian during BuildJacobian phase
                                    // to avoid random access in motionDatas
                                    var p_A = !bodyAIsStatic ? motionDatas[ixBodyA].WorldFromMotion.pos : float3.zero;
                                    var p_B = !bodyBIsStatic ? motionDatas[ixBodyB].WorldFromMotion.pos : float3.zero;

                                    // add normal rows for active normal constraints (contacts that are touching or penetrating)

                                    int activeNormalConstraints = 0;
                                    for (int i = 0; i < numContacts; ++i)
                                    {
                                        // Normal constraint function (gap function):
                                        //  \phi(q)_{gap} = (p_A+s_A - (p_B+s_B))^t * n = 0
                                        // where n is the contact normal pointing from B to A, and p_A + s_A and p_B + s_B
                                        // correspond to the contact positions in world space on the surfaces
                                        // of body A and B respectively, with p_i being the position of body i.
                                        // If the bodies intersect, the gap function is zero or negative, and otherwise
                                        // positive. This ensures that the constraint impulse lambda will be positive
                                        // when the contact pushes objects apart (increases the gap) and negative otherwise.
                                        // This is important for formulation of the contact constraint as part of an LCP,
                                        // where we enforce lambda >= 0 at all times.

                                        // Time derivative of the constraint function:
                                        //  \dot{\phi}(q)_{gap} = d/dt ((p_A+s_A - (p_B+s_B))^t * n)
                                        //                      = d/dt (p_A+s_A - (p_B+s_B))^t * n + (p_A+s_A - (p_B+s_B))^t * d/dt * n
                                        //                      ~ d/dt (p_A+s_A - (p_B+s_B))^t * n
                                        //                      = (v_A + \omega_A x s_A - v_B - \omega_B x s_B)^t * n
                                        //                      = n^t * v_A + n^t * (\omega_A x s_a) - n^t * v_B - n^t * \omega_B x s_B
                                        //                      = [n^t, (s_A x n)^t, -n^t, -(s_B x n)^t] * [v_A^t, \omega_A^t, v_B^t, \omega_B^t]^t
                                        // => Jacobian:
                                        //  J_{gap} = [n^t, (s_A x n)^t, -n^t, -(s_B x n)^t] \in 1x12

                                        ref var contactAngularJac = ref header.AccessAngularJacobian(i);
                                        var contactDistance = contactAngularJac.ContactDistance;

                                        if (contactDistance > ContactRowReach)
                                        {
                                            // skip inactive contacts
                                            continue;
                                        }

                                        // normal row index for active contact.
                                        var normalRow = startRow + activeNormalConstraints;

                                        var J_normal = jacobianRows.Next();
                                        J_normal.Clear();
                                        JBlocks[activeNormalConstraints] = J_normal;
                                        ++activeNormalConstraints;

                                        // Get world space contact point on surface of B.
                                        ref var contact = ref header.AccessContactPoint(i);
                                        var pointOnB = contact.Position;
                                        // Calculate world space contact point on surface of A.
                                        // Note: normal points from B to A, and the contactDistance corresponds to the
                                        // signed distance which is negative when the objects are intersecting.
                                        // So in order to get to the surface of A we need to calculate
                                        // "contact point on A" = "contact point on B" + "contact normal" * "signed distance".
                                        var pointOnA = contact.Position + normal * contactDistance;

                                        if (!bodyAIsStatic)
                                        {
                                            var s_A = pointOnA - p_A;
                                            s_A_avg += s_A;

                                            SetSubVector3(J_normal, normal, startColumnA);
                                            SetSubVector3(J_normal, math.cross(s_A, normal), startColumnA + 3);
                                        }

                                        if (!bodyBIsStatic)
                                        {
                                            var s_B = pointOnB - p_B;
                                            s_B_avg += s_B;

                                            SetSubVector3(J_normal, -normal, startColumnB);
                                            SetSubVector3(J_normal, -math.cross(s_B, normal), startColumnB + 3);
                                        }

                                        // Constraint error from last step, at coordinates q_:
                                        //      \phi(q_)_{gap} = (p_A+s_A - (p_B+s_B))^t * n
                                        //                     = (pointOnA - pointOnB)^t * n = contactDistance
                                        var phi_0 = contactDistance;

                                        float gamma_phi_0 = SparseContactRow(phi_0, motionVelocities, ixBodyA, ixBodyB,
                                            bodyAIsStatic || bodyAIsKinematic, bodyBIsStatic || bodyBIsKinematic,
                                            stepInput.Timestep, out float epsilon);
                                        eps[normalRow] = epsilon;
                                        if (context.Recovery != null)
                                        {
                                            bool againstFixed = bodyAIsStatic || bodyAIsKinematic || bodyBIsStatic || bodyBIsKinematic;
                                            float fastest = againstFixed
                                                ? stepInput.MaxStaticDepenetrationVelocity
                                                : stepInput.MaxDynamicDepenetrationVelocity;
                                            // Below the slop the velocity solve recovers only the slop's worth, however deep
                                            // the contact: the depth the capped push-out leaves comes out in later substeps.
                                            float pushOut = 0.0f;
                                            if (phi_0 < -ContactSlop)
                                            {
                                                float held = (ContactRecovery / stepInput.Timestep) * ContactSlop;
                                                pushOut = math.min(gamma_phi_0 - held, fastest);
                                                gamma_phi_0 = held;
                                            }

                                            context.Recovery[normalRow] = pushOut;
                                        }

                                        rhs[normalRow] = gamma_phi_0;

                                        // Allow removal of the warm impulse, but keep the contact's total
                                        // impulse nonnegative.
                                        l[normalRow] = -contactAngularJac.Jac.Impulse;

                                        // A crash contact shares what is left of its limit between its points, so a
                                        // part that the game is about to destroy cannot stop the vessel behind it.
                                        if (contactJacobian.MaxNormalImpulse > 0.0f)
                                        {
                                            float left = math.max(contactJacobian.MaxNormalImpulse - contactJacobian.NormalImpulseOverSubsteps, 0.0f);
                                            u[normalRow] = math.max(left / numContacts - contactAngularJac.Jac.Impulse, l[normalRow]);
                                        }

                                        // Since by design (see skip if inactive contact) the contact is already penetrating,
                                        // as initial guess we assume the variable to be free.

                                        // @todo direct solver (DOTS-11019): Look at current velocity along the contact normal
                                        // for the guess here and assume tight if the contact is separating.
                                        // This would avoid creating unnecessarily large M_FF principal submatrices when
                                        // starting the LCP solve, which saves some time (assuming Judice & Pires is
                                        // used as method in LCP.Solve).

                                        indexSetArray[normalRow] = LCP.MLCPIndexFlag.Free;
                                        isLCP = true;
                                    }

                                    // add friction rows if coefficient of friction larger zero and we have any penetrating
                                    // contacts
                                    if (contactJacobian.CoefficientOfFriction > 0 && activeNormalConstraints > 0)
                                    {
                                        var frictionStartRow = startRow + activeNormalConstraints;

                                        var J_frictionDir0 = jacobianRows.Next();
                                        var J_frictionDir1 = jacobianRows.Next();
                                        var J_frictionNormal = jacobianRows.Next();

                                        J_frictionDir0.Clear();
                                        J_frictionDir1.Clear();
                                        J_frictionNormal.Clear();

                                        JBlocks[activeNormalConstraints] = J_frictionDir0;
                                        JBlocks[activeNormalConstraints + 1] = J_frictionDir1;
                                        JBlocks[activeNormalConstraints + 2] = J_frictionNormal;
                                        // Choose friction axes
                                        Math.CalculatePerpendicularNormalized(normal,
                                            out float3 frictionDir0, out float3 frictionDir1);

                                        if (!bodyAIsStatic)
                                        {
                                            // Calculate average arm vector from individual arms s_A_i for contacts C_i
                                            s_A_avg /= contactJacobian.BaseJacobian.NumContacts;

                                            SetSubVector3(J_frictionDir0, frictionDir0, startColumnA);
                                            SetSubVector3(J_frictionDir1, frictionDir1, startColumnA);
                                            SetSubVector3(J_frictionDir0, math.cross(s_A_avg, frictionDir0), startColumnA + 3);
                                            SetSubVector3(J_frictionDir1, math.cross(s_A_avg, frictionDir1), startColumnA + 3);
                                            SetSubVector3(J_frictionNormal, normal, startColumnA + 3);
                                        }

                                        if (!bodyBIsStatic)
                                        {
                                            // Calculate average arm vector from individual arms s_B_i for contacts C_i
                                            s_B_avg /= contactJacobian.BaseJacobian.NumContacts;

                                            SetSubVector3(J_frictionDir0, -frictionDir0, startColumnB);
                                            SetSubVector3(J_frictionDir1, -frictionDir1, startColumnB);
                                            SetSubVector3(J_frictionDir0, -math.cross(s_B_avg, frictionDir0), startColumnB + 3);
                                            SetSubVector3(J_frictionDir1, -math.cross(s_B_avg, frictionDir1), startColumnB + 3);
                                            SetSubVector3(J_frictionNormal, -normal, startColumnB + 3);
                                        }

                                        float3 surfaceVelLocal = float3.zero;
                                        if (header.HasSurfaceVelocity)
                                        {
                                            var surfaceVel = header.AccessSurfaceVelocity();

                                            // linear surface velocity in contact space (shift along friction direction 0 and 1)
                                            surfaceVelLocal.x = math.dot(surfaceVel.LinearVelocity, frictionDir0);
                                            surfaceVelLocal.y = math.dot(surfaceVel.LinearVelocity, frictionDir1);
                                            // angular surface velocity in contact space (twist around contact normal)
                                            surfaceVelLocal.z = math.dot(surfaceVel.AngularVelocity, normal);
                                        }

                                        // set target velocity
                                        SetSubVector3(rhs, surfaceVelLocal, frictionStartRow);

                                        // model friction as velocity constraint by choosing viscous epsilon
                                        eps[frictionStartRow] = contactViscousEpsilon;
                                        eps[frictionStartRow + 1] = contactViscousEpsilon;
                                        eps[frictionStartRow + 2] = contactViscousEpsilon;

                                        // Earlier substeps' impulses count toward the friction cone for the whole step.
                                        float3 held = contactJacobian.FrictionImpulseOverSubsteps;
                                        SetSubVector3(l, -held, frictionStartRow);
                                        SetSubVector3(u, -held, frictionStartRow);

                                        // Start at the lower bound. The normal impulse determines the friction bounds.
                                        indexSetArray[frictionStartRow] = LCP.MLCPIndexFlag.LowerTight;
                                        indexSetArray[frictionStartRow + 1] = LCP.MLCPIndexFlag.LowerTight;
                                        indexSetArray[frictionStartRow + 2] = LCP.MLCPIndexFlag.LowerTight;

                                        isLCP = true;
                                    }
                                    break;
                                }
                                default:
                                    break;
                            }

                            // Deal with kinematic bodies by incorporating their velocities into the right hand side.
                            for (int i = 0; i < jacRows; ++i)
                            {
                                float velocityOffset = 0;
                                if (bodyAIsKinematic)
                                {
                                    AddConstraintSpaceScalarVelocity(ref velocityOffset, ref JBlocks[i],
                                        in kinematicLinearVelocityA, in kinematicAngularVelocityA, startColumnA);
                                }
                                if (bodyBIsKinematic)
                                {
                                    AddConstraintSpaceScalarVelocity(ref velocityOffset, ref JBlocks[i],
                                        in kinematicLinearVelocityB, in kinematicAngularVelocityB, startColumnB);
                                }

                                if (velocityOffset != 0)
                                {
                                    rhs[startRow + i] -= velocityOffset;
                                }
                            }

                            AddBodyVelocities(startRow, jacRows, bodyAIsStatic || bodyAIsKinematic ? -1 : ixBodyA,
                                bodyBIsStatic || bodyBIsKinematic ? -1 : ixBodyB, ref context);

        }

        /// <summary>
        /// Subtracts Ju_ from one Jacobian's rows of the right-hand side, rhs = v0 - phi_0/dt - JinvM * (dtg + Mu_).
        /// Gravity is already in u_, and JinvM * Mu_ is Ju_. A body index of -1 marks a body that isn't dynamic.
        /// </summary>
        static unsafe void AddBodyVelocities(int startRow, int jacRows, int dynamicBodyA, int dynamicBodyB,
            ref RowBuildContext context)
        {
            const int startColumnA = 0;
            const int startColumnB = 6;

            float* rhsBodyVelocity = stackalloc float[jacRows];
            UnsafeUtility.MemClear(rhsBodyVelocity, sizeof(float) * jacRows);

            if (dynamicBodyA >= 0)
            {
                ComputeRHSGImpulse(context.JacobianRows, context.MotionVelocities[dynamicBodyA], context.MotionDatas[dynamicBodyA],
                    startRow, startColumnA, jacRows, rhsBodyVelocity);
            }

            if (dynamicBodyB >= 0)
            {
                ComputeRHSGImpulse(context.JacobianRows, context.MotionVelocities[dynamicBodyB], context.MotionDatas[dynamicBodyB],
                    startRow, startColumnB, jacRows, rhsBodyVelocity);
            }

            for (int i = 0; i < jacRows; ++i)
            {
                context.RightHandSide[startRow + i] -= rhsBodyVelocity[i];
            }
        }
    }
}
