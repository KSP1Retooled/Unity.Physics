using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Defines the maximum joint stiffness for a direct solve. At this stiffness, the effective mass would oscillate
    /// with <see cref="Period"/> as its period. A constraint can raise the maximum using
    /// <see cref="Constraint.DirectStiffnessScale"/> .
    /// </summary>
    public static class DirectStiffness
    {
        /// <summary>
        /// The oscillation period in seconds. This is fixed independently of the substep duration. A joint's static sag
        /// is the load divided by its stiffness. If the stiffness cap depended on substep duration, the sag would vary
        /// with the square of that duration, making the craft's stiffness depend on the number of substeps.
        /// </summary>
        public const float Period = 0.005f;

        /// <summary>
        /// Returns the stiffness cap for the given inverse mass. For rotation, use the sum over moving bodies of each
        /// body's mean inverse inertia across its three axes.
        /// </summary>
        public static float Maximum(float inverseMass)
        {
            float angularFrequency = Math.Constants.Tau / Period;
            return angularFrequency * angularFrequency / math.max(inverseMass, 1e-9f);
        }
    }
}
