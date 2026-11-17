namespace Unity.Numerics.Linear.Solvers.Lowlevel
{
    /// <summary>
    /// Coupled row bounds for block principal pivoting.
    /// Expand problem bounds on [FirstRow, FirstRow + RowCount) by Factor times the nonnegative total source
    /// impulse,
    /// including HeldSourceImpulse. Friction bounds account for impulses already applied earlier in the step.
    /// </summary>
    public struct BoundCoupling
    {
        public int FirstRow;
        public int RowCount;
        public int FirstSourceRow;
        public int SourceRowCount;
        public float Factor;

        /// <summary>
        /// Source impulse already applied by warm starting or earlier substeps, added to newly solved impulses for
        /// coupled bounds.
        /// </summary>
        public float HeldSourceImpulse;
    }
}
