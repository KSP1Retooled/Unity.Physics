namespace Unity.Numerics.Linear.Solvers.Lowlevel
{
    /// <summary>
    /// Accumulate factorization, update and reuse counts across block principal pivoting solves.
    /// Count free-set changes relative to the preceding round, excluding each solve's first round.
    /// </summary>
    public struct BlockPrincipalPivotingStats
    {
        public long Rounds;
        public long CouplingRounds;
        public long Factorizations;
        public long Updates;
        public long Reuses;
        public long AddedRows;
        public long RemovedRows;

        /// <summary>
        /// Total, minimum and maximum free rows across rounds. Ignore empty rounds for the minimum.
        /// </summary>
        public long FreeRows;
        public long MostFreeRows;
        public long FewestFreeRows;

        /// <summary>The time spent factoring or changing the factor, in profiler timestamp units.</summary>
        public long FactorTicks;

        /// <summary>
        /// The time spent on the right-hand side and the solves with the factor, and the time spent checking the rows'
        /// conditions, in profiler timestamp units.
        /// </summary>
        public long SolveTicks;
        public long PivotTicks;

        /// <summary>The part of <see cref="FactorTicks"/> spent on factors from the start, with the copy of W_FF into them.</summary>
        public long FullFactorTicks;
    }
}
