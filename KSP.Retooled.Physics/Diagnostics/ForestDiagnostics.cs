using Unity.Physics.Dynamics.Solver.Forest;
using KSP.Retooled.Physics.Stepping;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Islands;

namespace KSP.Retooled.Physics.Diagnostics
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
            if (StepEntryPoints.Current != null)
            {
                StepEntryPoints.Current->IslandSolverCache.ForestCounters.GetForestCounts(out solves, out fallbacks, out builds);
            }
        }

        /// <summary>
        /// Returns section counts for the world most recently stepped. See ForestCounters.GetSectionCounts.
        /// </summary>
        internal static unsafe void GetSectionCounts(out long sectionSum, out long lockedSum, out long structureBuilds,
            out long lockChanges, out double structureMilliseconds, out double residual, out long lastSections, out long lastLocked)
        {
            sectionSum = lockedSum = structureBuilds = lockChanges = lastSections = lastLocked = 0;
            structureMilliseconds = residual = 0;
            if (StepEntryPoints.Current != null)
            {
                StepEntryPoints.Current->IslandSolverCache.ForestCounters.GetSectionCounts(out sectionSum, out lockedSum, out structureBuilds,
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
            if (StepEntryPoints.Current == null)
            {
                return;
            }

            IslandSolverCache cache = StepEntryPoints.Current->IslandSolverCache;
            contactChanges = cache.ForestCounters.ForestCounter(ForestCounters.ForestContactLockChanges);
            fallbackBoundary = cache.ForestCounters.ForestCounter(ForestCounters.ForestFallbackBoundaryMax);
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
            if (StepEntryPoints.Current != null)
            {
                IslandSolverCache cache = StepEntryPoints.Current->IslandSolverCache;
                for (int i = 0; i < 5; i++)
                {
                    result[i] = cache.ForestCounters.ForestMilliseconds(ForestCounters.ForestClassifyTime + i);
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
            if (StepEntryPoints.Current != null)
            {
                StepEntryPoints.Current->IslandSolverCache.ForgetForests();
            }
        }

        /// <summary>
        /// The cumulative time in milliseconds spent settling sections after the first solve, measured since the world
        /// most recently stepped was built.
        /// </summary>
        internal static unsafe double SettleMilliseconds => StepEntryPoints.Current != null ? StepEntryPoints.Current->IslandSolverCache.ForestCounters.ForestSettleMilliseconds : 0.0;

        /// <summary>
        /// The most recent deflection in radians that the forest recorded.
        /// </summary>
        internal static unsafe double LastDeflection => StepEntryPoints.Current != null ? StepEntryPoints.Current->IslandSolverCache.ForestCounters.LastDeflection : 0.0;

        /// <summary>
        /// Returns cumulative solve times in milliseconds since the world most recently stepped was built. See
        /// ForestCounters.GetSolveTimes.
        /// </summary>
        internal static unsafe void GetTimes(out double sparse, out double forestStart, out double forestSolve, out double forestFinish,
            out double forestFactor, out double forestBoundary, out double forestDrift)
        {
            sparse = forestStart = forestSolve = forestFinish = forestFactor = forestBoundary = forestDrift = 0;
            if (StepEntryPoints.Current != null)
            {
                StepEntryPoints.Current->IslandSolverCache.ForestCounters.GetSolveTimes(out forestStart, out forestSolve, out forestFinish,
                    out forestFactor, out forestBoundary);
                sparse = StepEntryPoints.Current->IslandSolverCache.SparseCounters.Milliseconds;
                forestDrift = StepEntryPoints.Current->IslandSolverCache.ForestCounters.ForestDriftMilliseconds;
            }
        }
    }
}
