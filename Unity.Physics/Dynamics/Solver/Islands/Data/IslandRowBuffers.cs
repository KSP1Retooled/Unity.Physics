using Unity.Numerics.Linear.Dense.Primitives;

namespace Unity.Physics.Dynamics.Solver.Islands.Data
{
    /// <summary>Points to the rows that the direct and vessel solvers build for one island.</summary>
    internal unsafe struct IslandRowBuffers
    {
        public float* Jacobian;
        public float* Epsilon;
        public float* Lower;
        public float* Upper;
        public float* RightHandSide;
        public float* Recovery;
        public LCP.MLCPIndexFlag* Flags;
        public int Rows;
    }
}
