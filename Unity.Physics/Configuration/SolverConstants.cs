namespace Unity.Physics.Configuration
{
    internal static class SolverConstants
    {
        internal const int SolverIterations = 12;
        internal const int DirectIslandSubsteps = 2;

        // The maximum speed, in m/s, at which contacts separate penetrating bodies. This is the default value of
        // Rigidbody.maxDepenetrationVelocity. PhysX applies it to each body, and the stock game never changes it.
        internal const float MaxDepenetrationVelocity = 10.0f;
    }
}
