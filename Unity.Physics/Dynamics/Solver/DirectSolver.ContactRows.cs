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
    }
}
