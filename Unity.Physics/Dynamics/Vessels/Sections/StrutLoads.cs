using Unity.Burst;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>Markers and totals for strut loads.</summary>
    internal static class StrutLoads
    {
        struct CountsKey { }

        /// <summary>The profiler markers around the factoring of the systems and around each section's solve.</summary>
        internal static readonly Unity.Profiling.ProfilerMarker FactorMarker =
            new Unity.Profiling.ProfilerMarker(Unity.Profiling.ProfilerCategory.Physics, "Physics.Sections.StrutFactor");

        internal static readonly Unity.Profiling.ProfilerMarker SolveMarker =
            new Unity.Profiling.ProfilerMarker(Unity.Profiling.ProfilerCategory.Physics, "Physics.Sections.StrutSolve");

        internal static readonly Unity.Profiling.ProfilerMarker BookkeepingMarker =
            new Unity.Profiling.ProfilerMarker(Unity.Profiling.ProfilerCategory.Physics, "Physics.Sections.StrutBookkeeping");

        internal static readonly SharedStatic<StrutLoadsCounters> Counts = SharedStatic<StrutLoadsCounters>.GetOrCreate<CountsKey>();
    }
}
