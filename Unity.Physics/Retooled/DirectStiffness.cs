using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// The stiffest joint a direct solve builds: one that would swing the mass it moves through <see cref="Period"/>. A
    /// constraint can raise it with <see cref="Constraint.DirectStiffnessScale"/>.
    /// </summary>
    public static class DirectStiffness
    {
        /// <summary>
        /// The period, in seconds. It is fixed rather than the substep, because a joint's static sag under a load is the
        /// load over its stiffness: a cap that followed the substep would make a craft sag with the square of the substep
        /// and tie its stiffness to how many substeps run.
        /// </summary>
        public const float Period = 0.005f;

        /// <summary>
        /// The cap for an inverse mass, or for a rotation the mean of each body's inverse inertia over its three axes, summed
        /// over the bodies that move.
        /// </summary>
        public static float Maximum(float inverseMass)
        {
            float angularFrequency = Math.Constants.Tau / Period;
            return angularFrequency * angularFrequency / math.max(inverseMass, 1e-9f);
        }
    }
}
