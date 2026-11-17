namespace Unity.Physics.Store.Versions
{
    internal static class Serials
    {
        // Use JointTopology.Changed to preserve both the topology bumps and the serial sequence.
        // Serial values feed identity keys and sort orders, so changing the sequence could change results.
        internal static uint Next() => JointTopology.Changed();
    }
}
