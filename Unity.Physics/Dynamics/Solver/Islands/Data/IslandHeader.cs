namespace Unity.Physics.Dynamics.Solver.Islands.Data
{
    /// <summary>Pairs a Jacobian pointer with its island index for indexing and row construction.</summary>
    internal unsafe struct IslandHeader
    {
        public JacobianHeader* Header;
        public int Island;
    }
}
