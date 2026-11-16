namespace Unity.Physics.Numerics.Retooled
{
    /// <summary>
    /// Describes a group of rows whose bounds depend on the impulses of other rows. The solver widens the problem
    /// bounds of the rows from FirstRow up to, but not including, FirstRow + RowCount. It widens them by <see
    /// cref="Factor"/> times the total source impulse, or by zero if that total is negative. The total source impulse
    /// is the sum of the impulses that the solve finds for the source rows, plus <see cref="HeldSourceImpulse"/>. For
    /// contact friction, the friction rows are the coupled rows and the normal row is the source. Both the normal
    /// impulse and the friction impulse count over the whole step. The problem bounds of a friction row are therefore
    /// the negative of the impulse that the row already applied earlier in the step. If the total normal impulse is
    /// zero, the total friction impulse for the step is also zero.
    /// </summary>
    public struct BoundCoupling
    {
        public int FirstRow;
        public int RowCount;
        public int FirstSourceRow;
        public int SourceRowCount;
        public float Factor;

        /// <summary>
        /// The source impulse that was already applied before this solve, through warm starting or in an earlier
        /// substep of the same step. It counts toward the coupled bounds together with the impulse that this solve
        /// finds.
        /// </summary>
        public float HeldSourceImpulse;
    }
}
