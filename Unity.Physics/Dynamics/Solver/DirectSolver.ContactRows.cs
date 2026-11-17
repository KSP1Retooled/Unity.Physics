using Unity.Collections;
using Unity.Mathematics;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// The fraction of a contact's penetration that one substep removes. Removing all of it at once launches whatever
        /// rests on the contact, and the bodies fall back deeper the next time.
        /// </summary>
        const float ContactRecovery = 0.2f;

        /// <summary>
        /// Contact penetration corrected through velocity. Deeper penetration uses SolvePushOut without adding
        /// speed.
        /// </summary>
        const float ContactSlop = 0.01f;

        /// <summary>
        /// Contact compliance relative to row inverse mass, small enough to support heavy stacks on light parts
        /// while making forces on redundant face contacts unique.
        /// </summary>
        const float ContactCompliance = 1e-3f;

        /// <summary>
        /// The right-hand side of a contact's normal row in the sparse solve, with its regularisation. A contact that is
        /// still apart may close its gap in this substep and no more; one that is touching pushes out part of its depth.
        /// </summary>
        static float SparseContactRow(float distance, in NativeArray<MotionVelocity> motionVelocities, int bodyA, int bodyB,
            bool bodyAIsFixed, bool bodyBIsFixed, float timestep, out float epsilon)
        {
            float inverseMass = (bodyAIsFixed ? 0.0f : motionVelocities[bodyA].InverseMass)
                + (bodyBIsFixed ? 0.0f : motionVelocities[bodyB].InverseMass);
            epsilon = ContactCompliance * inverseMass;
            return distance > 0.0f ? -distance / timestep : -(ContactRecovery / timestep) * distance;
        }

        /// <summary>
        /// Store normal impulse per contact point and accumulate step-wide normal/friction totals for later substep
        /// bounds.
        /// </summary>
        static unsafe void HandOverContactImpulses(ref JacobianHeader header, float* lambda, float collisionTolerance)
        {
            ref var contactJacobian = ref header.AccessBaseJacobian<ContactJacobian>();
            float sum = 0.0f;
            float pushed = 0.0f;
            int normalRow = 0;
            for (int i = 0; i < contactJacobian.BaseJacobian.NumContacts; ++i)
            {
                ref var point = ref header.AccessAngularJacobian(i);
                if (point.ContactDistance <= collisionTolerance)
                {
                    // The point already holds its warm start, which the events count apart from this.
                    float solved = lambda[normalRow++];
                    point.Jac.Impulse = math.max(point.Jac.Impulse + solved, 0.0f);
                    sum += solved;
                }

                // A point without a row still pushed with its warm start.
                pushed += point.Jac.Impulse;
            }

            contactJacobian.SumImpulsesOverSubsteps += sum;
            contactJacobian.NormalImpulseOverSubsteps += pushed;
            if (contactJacobian.CoefficientOfFriction > 0 && normalRow > 0)
            {
                contactJacobian.FrictionImpulseOverSubsteps += new float3(lambda[normalRow], lambda[normalRow + 1], lambda[normalRow + 2]);
            }
        }
    }
}
