namespace Unity.Numerics.Linear.Solvers.Lowlevel
{
    /// <summary>
    /// Counts what solves by <see cref="BlockPrincipalPivoting"/> did with the factor of their free rows. A caller that
    /// passes the same counts to several solves gets the sums over those solves. A round of pivoting either factored
    /// W_FF from the start, changed the last round's factor, or used that factor as it was. The rows that joined and
    /// left the free set are counted against the round before, so the first round of a solve adds none.
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
        /// The free rows that the rounds solved for, summed over the rounds, and the most and the fewest free rows that
        /// one round had. The fewest leaves out rounds with no free rows.
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
