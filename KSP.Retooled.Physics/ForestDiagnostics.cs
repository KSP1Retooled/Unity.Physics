using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Islands;

namespace KSP.Retooled.Physics
{
    internal static class ForestDiagnostics
    {
        /// <summary>
        /// Return cumulative forest solves, sparse fallbacks and structure builds for the most recently stepped
        /// world.
        /// Call on the main thread between steps.
        /// </summary>
        internal static unsafe void GetCounts(out long solves, out long fallbacks, out long builds)
        {
            solves = fallbacks = builds = 0;
            if (StepKernel.Current != null)
            {
                StepKernel.Current->IslandSolverCache.GetForestCounts(out solves, out fallbacks, out builds);
            }
        }

        /// <summary>
        /// Returns section counts for the world most recently stepped. See IslandSolverCache.GetSectionCounts.
        /// </summary>
        internal static unsafe void GetSectionCounts(out long sectionSum, out long lockedSum, out long structureBuilds,
            out long lockChanges, out double structureMilliseconds, out double residual, out long lastSections, out long lastLocked)
        {
            sectionSum = lockedSum = structureBuilds = lockChanges = lastSections = lastLocked = 0;
            structureMilliseconds = residual = 0;
            if (StepKernel.Current != null)
            {
                StepKernel.Current->IslandSolverCache.GetSectionCounts(out sectionSum, out lockedSum, out structureBuilds,
                    out lockChanges, out structureMilliseconds, out residual, out lastSections, out lastLocked);
            }
        }

        /// <summary>
        /// Return lock changes for contact pairs, maximum fallback boundary rows and ten-bin deflection histograms.
        /// See IslandSolverCache.DeflectionHistogram for bins.
        /// </summary>
        internal static unsafe void GetLockDiagnostics(out long contactChanges, out long fallbackBoundary, long[] all, long[] locked)
        {
            contactChanges = fallbackBoundary = 0;
            if (StepKernel.Current == null)
            {
                return;
            }

            IslandSolverCache cache = StepKernel.Current->IslandSolverCache;
            contactChanges = cache.ForestCounter(IslandSolverCache.ForestContactLockChanges);
            fallbackBoundary = cache.ForestCounter(IslandSolverCache.ForestFallbackBoundaryMax);
            long* a = stackalloc long[10];
            long* l = stackalloc long[10];
            cache.DeflectionHistogram(ForestSwitches.LockAngle, ForestSwitches.YieldAngle, a, l);
            for (int i = 0; i < 10; i++)
            {
                all[i] = a[i];
                locked[i] = l[i];
            }
        }

        /// <summary>
        /// Cumulative milliseconds in serial forest stages for the most recently stepped world.
        /// </summary>
        internal static unsafe double[] SerialMilliseconds()
        {
            var result = new double[5];
            if (StepKernel.Current != null)
            {
                IslandSolverCache cache = StepKernel.Current->IslandSolverCache;
                for (int i = 0; i < 5; i++)
                {
                    result[i] = cache.ForestMilliseconds(IslandSolverCache.ForestClassifyTime + i);
                }
            }

            return result;
        }

        /// <summary>
        /// Clear cached island forests and lock states for the most recently stepped world.
        /// Call on the main thread between steps.
        /// </summary>
        internal static unsafe void ForgetIslands()
        {
            if (StepKernel.Current != null)
            {
                StepKernel.Current->IslandSolverCache.ForgetForests();
            }
        }

        /// <summary>
        /// The cumulative time in milliseconds spent settling sections after the first solve, measured since the world
        /// most recently stepped was built.
        /// </summary>
        internal static unsafe double SettleMilliseconds => StepKernel.Current != null ? StepKernel.Current->IslandSolverCache.ForestSettleMilliseconds : 0.0;

        /// <summary>
        /// The largest deflection in radians used by lock tests after the last forest solve, averaged over the step.
        /// </summary>
        internal static unsafe double LastDeflection => StepKernel.Current != null ? StepKernel.Current->IslandSolverCache.LastDeflection : 0.0;

        /// <summary>
        /// Returns cumulative solve times in milliseconds since the world most recently stepped was built. See
        /// IslandSolverCache.GetSolveTimes.
        /// </summary>
        internal static unsafe void GetTimes(out double sparse, out double forestStart, out double forestSolve, out double forestFinish,
            out double forestFactor, out double forestBoundary, out double forestDrift)
        {
            sparse = forestStart = forestSolve = forestFinish = forestFactor = forestBoundary = forestDrift = 0;
            if (StepKernel.Current != null)
            {
                StepKernel.Current->IslandSolverCache.GetSolveTimes(out sparse, out forestStart, out forestSolve, out forestFinish,
                    out forestFactor, out forestBoundary);
                forestDrift = StepKernel.Current->IslandSolverCache.ForestDriftMilliseconds;
            }
        }
    }
}
