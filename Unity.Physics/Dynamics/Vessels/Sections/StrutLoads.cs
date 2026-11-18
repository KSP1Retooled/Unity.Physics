using Unity.Burst;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>Markers and totals for strut loads.</summary>
    internal static class StrutLoads
    {
        struct CountsKey { }

        /// <summary>The profiler markers around the factoring of the systems and around each section's solve.</summary>
        internal static readonly Unity.Profiling.ProfilerMarker FactorMarker =
            new Unity.Profiling.ProfilerMarker(Unity.Profiling.ProfilerCategory.Physics, "Retooled.Physics.Vessel.StrutFactor");

        internal static readonly Unity.Profiling.ProfilerMarker SolveMarker =
            new Unity.Profiling.ProfilerMarker(Unity.Profiling.ProfilerCategory.Physics, "Retooled.Physics.Vessel.StrutSolve");

        /// <summary>Running totals since the program started, for the cost report.</summary>
        internal struct Totals
        {
            /// <summary>The systems factored, the unknowns in them, and the time spent factoring, in profiler ticks.</summary>
            public long Factored;

            public long FactoredUnknowns;
            public long FactorTicks;

            /// <summary>The section solves, and the time spent in them, in profiler ticks.</summary>
            public long Solves;

            public long SolveTicks;

            /// <summary>The blocks off the diagonal of the factors, and the solves' time in their three parts.</summary>
            public long FactoredBlocks;

            public long LoadTicks;
            public long SubstituteTicks;
            public long LinkTicks;
        }

        internal static readonly SharedStatic<Totals> Counts = SharedStatic<Totals>.GetOrCreate<CountsKey>();
    }
}
