using System;
using Unity.Burst;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>
    /// The switch for the strut loads. With it on, the locked edges and the strut closures inside a rigid section share the
    /// section's loads as they would with every edge live and very stiff. With it off, each locked edge carries its
    /// child's whole subtree and a closure inside a section carries nothing, as before.
    /// RETOOLED_STRUT_LOADS=0 turns it off.
    /// </summary>
    internal static class StrutLoads
    {
        struct OffKey { }

        struct CountsKey { }

        static readonly SharedStatic<bool> Off = SharedStatic<bool>.GetOrCreate<OffKey>();

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

        /// <summary>Whether the sections with closures inside them take their loads from a solve for static equilibrium.</summary>
        public static bool Enabled
        {
            get => !Off.Data;
            set => Off.Data = !value;
        }

        /// <summary>Reads RETOOLED_STRUT_LOADS from the environment once in each domain, on the main thread.</summary>
        public static void Initialize()
        {
            // Burst code reads this class, so the managed flag lives in a class that Burst never sees.
            if (StrutLoadsInitialized.Done)
            {
                return;
            }

            StrutLoadsInitialized.Done = true;
            string value = Environment.GetEnvironmentVariable("RETOOLED_STRUT_LOADS");
            Enabled = !(value == "0" || (value != null && value.Equals("false", StringComparison.OrdinalIgnoreCase)));
        }
    }

    /// <summary>Whether <see cref="StrutLoads.Initialize"/> has read the environment in this domain.</summary>
    static class StrutLoadsInitialized
    {
        public static bool Done;
    }
}
