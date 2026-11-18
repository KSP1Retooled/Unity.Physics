using System.Threading;
using Unity.Burst;

namespace Unity.Physics.Dynamics.Solver.Islands.Data
{
    /// <summary>
    /// Counts of unfused islands, first-substep rows, bodies and shareable solves, plus fused vessel part
    /// count.
    /// </summary>
    internal struct IslandWorkCounters
    {
        public int Islands;
        public int Rows;
        public int Bodies;
        public int TeamIslands;
        public int FusedParts;

        /// <summary>The step that the counts are from, or 0 if no step has recorded them.</summary>
        public long Step;

        /// <summary>
        /// The number of island solves, each one for a whole step, that the fused vessel job has done since the
        /// program started. Tests read it to check that an island went to the fused job.
        /// </summary>
        internal static long FusedIslandSolves => FusedIslandSolveCount.Data;

        struct FusedIslandSolveKey { }

        static readonly SharedStatic<long> FusedIslandSolveCount = SharedStatic<long>.GetOrCreate<FusedIslandSolveKey>();

        /// <summary>
        /// Number of steps using serial unfused substeps, for tests of scheduling choices.
        /// </summary>
        internal static long UnfusedFoldedSteps => UnfusedFoldedStepCount.Data;

        struct UnfusedFoldedStepKey { }

        static readonly SharedStatic<long> UnfusedFoldedStepCount = SharedStatic<long>.GetOrCreate<UnfusedFoldedStepKey>();

        internal static void CountFusedSolve() => Interlocked.Increment(ref FusedIslandSolveCount.Data);

        internal static void CountUnfusedFoldedStep() => Interlocked.Increment(ref UnfusedFoldedStepCount.Data);
    }
}
