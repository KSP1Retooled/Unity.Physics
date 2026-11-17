namespace Unity.Numerics.Linear.Solvers.Lowlevel
{
    /// <summary>
    /// Strategy for updating the free-row factor between pivot rounds.
    /// </summary>
    public enum FactorUpdate : byte
    {
        /// <summary>Every round factors the free rows from the start.</summary>
        Refactor,

        /// <summary>
        /// Keep the prefix before the first departing free row, then reinsert remaining and newly free rows.
        /// </summary>
        KeepPrefix,

        /// <summary>
        /// Delete departing free rows and append newly free rows using Cholesky row/column updates.
        /// </summary>
        Edit,
    }
}
