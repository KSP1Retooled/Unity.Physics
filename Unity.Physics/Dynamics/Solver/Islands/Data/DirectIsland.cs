namespace Unity.Physics.Dynamics.Solver.Islands.Data
{
    /// <summary>
    /// Describes the Jacobians for one direct island and the rows that the island solvers use for the current substep.
    /// </summary>
    internal struct DirectIsland
    {
        const int SharedSolveParts = 256;
        const int SharedSolveRows = 512;

        public int FirstHeader;

        /// <summary>
        /// The exclusive end index of the joint Jacobians, which precede the contact Jacobians.
        /// </summary>
        public int JointEnd;

        /// <summary>
        /// The exclusive end index of the Jacobians included in the solve. Contact Jacobians after this index are
        /// updated but not solved.
        /// </summary>
        public int SolveEnd;

        public int HeaderEnd;

        /// <summary>
        /// The number of contact pairs included with the joints in the sparse solve.
        /// </summary>
        public int ContactPairs;

        public int FirstRow;
        public int Rows;

        /// <summary>
        /// The number of rows in the joint Jacobians, which remains the same across substeps.
        /// </summary>
        public int JointRows;

        /// <summary>
        /// The range of this island's vessels in the scheduler buffer and their total
        /// number of parts. An island with a vessel is solved even if it has no Jacobian rows.
        /// </summary>
        public int FirstVessel;
        public int VesselCount;
        public int VesselParts;

        /// <summary>The number of the island's bodies that are not parts of its vessels.</summary>
        public int OtherBodies;

        /// <summary>
        /// Whether the island has enough rows or vessel parts to benefit from sharing the solve among lanes.
        /// Smaller islands are solved independently in parallel.
        /// </summary>
        public bool SolvesInTeam => Rows >= SharedSolveRows || VesselParts >= SharedSolveParts;

        /// <summary>
        /// Whether the fused vessel job solves this island for the whole step. The jobs of each
        /// substep then skip it.
        /// </summary>
        public bool Fused;
    }
}
