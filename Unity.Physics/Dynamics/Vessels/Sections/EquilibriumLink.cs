using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>
    /// One link of a section's equilibrium system: a locked tree edge or a closure inside the section. Its stiffness and its
    /// joint point are written in the frame of the section's first part, which turns with the section.
    /// </summary>
    internal struct EquilibriumLink
    {
        /// <summary>
        /// The unknown that each side moves with, as its position in the elimination order, or -1 when that side is held
        /// fixed. B is the child of a tree edge, or part B of a closure.
        /// </summary>
        public int A;

        public int B;

        /// <summary>
        /// For a tree edge, the place of its child part in the section's list of parts. For a closure, -1 minus the
        /// closure's place in the section's list of closures.
        /// </summary>
        public int Target;

        /// <summary>The joint point, from the first part's centre, in the first part's frame.</summary>
        public double3 Point;

        /// <summary>U diag(w) U^T for the linear rows and for the angular rows, with w each row's h^2 k + h c.</summary>
        public double3x3 Linear;

        public double3x3 Angular;
    }
}
