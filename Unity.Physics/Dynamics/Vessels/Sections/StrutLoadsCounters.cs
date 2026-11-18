namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>Running totals for strut load diagnostics.</summary>
    internal struct StrutLoadsCounters
    {
        /// <summary>The systems factored, the unknowns in them, and the time spent factoring, in profiler ticks.</summary>
        public long Factored;

        public long FactoredUnknowns;
        public long FactorTicks;

        /// <summary>The section solves, and the time spent in them, in profiler ticks.</summary>
        public long Solves;

        public long SolveTicks;

        /// <summary>The blocks off the diagonal of the factors.</summary>
        public long FactoredBlocks;

        /// <summary>The solve's chunks and the columns of its top, summed over the systems factored.</summary>
        public long FactoredChunks;

        public long FactoredTop;

        /// <summary>The time spent in the substitution of the top, which one lane does alone.</summary>
        public long TopTicks;

        /// <summary>How many times the systems were matched to a structure.</summary>
        public long Prepares;

        /// <summary>The factor's time in its two parts: the ordering and structure, and the numbers.</summary>
        public long OrderTicks;

        public long NumericTicks;

        /// <summary>How many kept systems were used again for a new structure.</summary>
        public long Reused;

        /// <summary>Systems that a later structure used again after at least one structure went without them.</summary>
        public long Reactivated;

        public long LoadTicks;

        /// <summary>The portion of LoadTicks spent accumulating substep loads outside a solve.</summary>
        public long AccumulateTicks;
        public long SubstituteTicks;
        public long LinkTicks;
    }
}
