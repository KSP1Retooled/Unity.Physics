namespace Unity.Physics.Dynamics.Vessels.Structure
{
    /// <summary>
    /// Live constraint between sections. The child side is the tree child or closure part B;
    /// rows apply opposite impulses to the two sides.
    /// </summary>
    internal struct VesselLink
    {
        /// <summary>The tree edge (its child part), or -1 for a closure.</summary>
        public int Edge;

        /// <summary>The closure, or -1 for a tree edge.</summary>
        public int Closure;

        public int ChildPart;
        public int ParentPart;
        public int ChildSection;
        public int ParentSection;

        /// <summary>The section eliminated first, which owns the link's rows, and the other one.</summary>
        public int Lower;

        public int Upper;

        /// <summary>The block coupling the two sections, row Lower and column Upper, in the solver's block array.</summary>
        public int Coupling;

        /// <summary>Where the upper section's diagonal and right-hand side take this link's share: their own, or a chunk's buffer.</summary>
        public int UpperDiagonal;

        public int UpperRhs;
    }
}
