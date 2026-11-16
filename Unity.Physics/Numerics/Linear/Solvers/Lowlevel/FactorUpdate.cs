namespace Unity.Numerics.Linear.Solvers.Lowlevel
{
    /// <summary>
    /// Picks how a round of <see cref="BlockPrincipalPivoting"/> changes the factor of its free rows when they differ
    /// from the free rows of the last round.
    /// </summary>
    public enum FactorUpdate : byte
    {
        /// <summary>Every round factors the free rows from the start.</summary>
        Refactor,

        /// <summary>
        /// The factor keeps its rows up to the first row that left the free set. The rows after it that stay and the
        /// rows that joined are inserted after them, with faer's insertion of rows and columns at the end.
        /// </summary>
        KeepPrefix,

        /// <summary>
        /// The rows that left the free set are deleted from the factor wherever they are, with faer's deletion of rows
        /// and columns, and the rows that joined are inserted at the end.
        /// </summary>
        Edit,
    }
}
