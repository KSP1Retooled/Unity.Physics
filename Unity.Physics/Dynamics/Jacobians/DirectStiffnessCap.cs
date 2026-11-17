using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Joint stiffness cap for direct solving, based on the effective mass and Period.
    /// Constraint.DirectStiffnessScale can raise the cap.
    /// </summary>
    internal static class DirectStiffnessCap
    {
        /// <summary>
        /// Oscillation period in seconds used for the stiffness cap, independent of substep duration to keep static
        /// sag consistent.
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
