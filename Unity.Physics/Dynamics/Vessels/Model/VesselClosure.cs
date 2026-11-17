using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Vessels.Model
{
    /// <summary>
    /// Loop-closing connection with six compliant rows. Omit closures internal to a rigid section.
    /// </summary>
    internal struct VesselClosure
    {
        public int PartA;
        public int PartB;

        /// <summary>The joint point in each part's frame, relative to its centre of mass.</summary>
        public double3 AnchorA;

        public double3 AnchorB;

        /// <summary>The axes in part A's frame, as columns.</summary>
        public double3x3 Frame;

        /// <summary>The rest orientation of part B relative to part A, calculated as conj(A) * B.</summary>
        public double4 Rest;

        public double LinearStiffness;
        public double AngularStiffness;
        public double LinearDamping;
        public double AngularDamping;
    }
}
