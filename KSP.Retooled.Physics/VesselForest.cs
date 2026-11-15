using System;
using Unity.Burst;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Selects how the direct solver solves an island. When disabled, the sparse solver handles every row. When
    /// enabled, joints that constrain their bodies like part joints form a vessel forest. The section solver eliminates
    /// the forest rows using that structure. All other rows in the island form a boundary block, which is solved
    /// against the forest (see <see cref="ForestSettings"/>). These settings are stored in a SharedStatic so that Burst
    /// code can read them during a step. Setting RETOOLED_FOREST=1 in the environment enables the forest when the
    /// backend attaches, allowing an entire test run to use it.
    ///
    /// <see cref="Locking"/> allows the forest solver to lock a tree edge when its joints carry little load. Locking is
    /// disabled by default; RETOOLED_LOCK=1 enables it when the backend attaches. The two bodies then move as one rigid
    /// section until their load would deflect the joint springs beyond <see cref="YieldAngle"/> or exceed
    /// <see cref="YieldFraction"/> of their break limits. See <see cref="ForestSettings"/>.
    /// </summary>
    public static class VesselForest
    {
        private readonly struct Key { }

        private readonly struct LockKey { }

        private static readonly SharedStatic<ForestSettings> Shared = SharedStatic<ForestSettings>.GetOrCreate<Key>();

        private static readonly SharedStatic<LockFractions> Fractions = SharedStatic<LockFractions>.GetOrCreate<LockKey>();

        private struct LockFractions
        {
            public float Lock;
            public float Yield;

            /// <summary>
            /// The number of steps used to smooth deflection, plus one. Zero selects the default setting.
            /// </summary>
            public int SmoothingPlusOne;

            /// <summary>
            /// The minimum number of steps to hold a lock, plus one. Zero selects the default setting.
            /// </summary>
            public int HoldPlusOne;

            public int Hold
            {
                get => HoldPlusOne - 1;
                set => HoldPlusOne = value + 1;
            }
        }

        private static bool initialized;

        /// <summary>The number of boundary rows above which the solver uses the sparse solver for the island.</summary>
        public const int DefaultBoundaryRowCap = 512;

        /// <summary>The fraction of joint break limits below which smoothed loads permit a pair to lock.</summary>
        public const float DefaultLockFraction = 0.25f;

        /// <summary>
        /// The fraction of joint break limits above which a smoothed load causes a locked pair to yield.
        /// </summary>
        public const float DefaultYieldFraction = 0.5f;

        public static bool Enabled
        {
            get => Shared.Data.Enabled;
            set => Shared.Data.Enabled = value;
        }

        public static int BoundaryRowCap
        {
            get => Shared.Data.BoundaryRowCap > 0 ? Shared.Data.BoundaryRowCap : DefaultBoundaryRowCap;
            set => Shared.Data.BoundaryRowCap = value;
        }

        /// <summary>
        /// Whether the forest solver locks tree edges under low loads. See <see cref="ForestSettings.Locking"/>.
        /// </summary>
        public static bool Locking
        {
            get => Shared.Data.Locking;
            set => Shared.Data.Locking = value;
        }

        /// <summary>The angle in radians used to permit locking. See <see cref="ForestSettings.LockAngle"/>.</summary>
        public static float LockAngle
        {
            get => Shared.Data.LockAngle > 0.0f ? Shared.Data.LockAngle : ForestSettings.DefaultLockAngle;
            set => Shared.Data.LockAngle = value;
        }

        /// <summary>
        /// The angle in radians used to trigger yielding. See <see cref="ForestSettings.YieldAngle"/>.
        /// </summary>
        public static float YieldAngle
        {
            get => Shared.Data.YieldAngle > 0.0f ? Shared.Data.YieldAngle : ForestSettings.DefaultYieldAngle;
            set => Shared.Data.YieldAngle = value;
        }

        /// <summary>
        /// The minimum number of steps to hold a lock. Negative values select the default. See
        /// <see cref="ForestSettings.MinimumHoldSteps"/>.
        /// </summary>
        public static int MinimumHoldSteps
        {
            get => Fractions.Data.Hold >= 0 ? Fractions.Data.Hold : ForestSettings.DefaultMinimumHoldSteps;
            set => Fractions.Data.Hold = value;
        }

        /// <summary>
        /// The number of steps used to smooth deflection. Negative values select the default. See
        /// <see cref="ForestSettings.DeflectionSmoothingSteps"/>.
        /// </summary>
        public static int DeflectionSmoothingSteps
        {
            get => Fractions.Data.SmoothingPlusOne > 0 ? Fractions.Data.SmoothingPlusOne - 1 : ForestSettings.DefaultDeflectionSmoothingSteps;
            set => Fractions.Data.SmoothingPlusOne = value + 1;
        }

        public static float LockFraction
        {
            get => Fractions.Data.Lock > 0.0f ? Fractions.Data.Lock : DefaultLockFraction;
            set => Fractions.Data.Lock = value;
        }

        public static float YieldFraction
        {
            get => Fractions.Data.Yield > 0.0f ? Fractions.Data.Yield : DefaultYieldFraction;
            set => Fractions.Data.Yield = value;
        }

        /// <summary>
        /// Returns cumulative counts for the world most recently stepped, measured since it was built: island solves
        /// using the forest, solves with the forest enabled that fell back to the sparse solver, and forest structures
        /// built. Call on the main thread between steps.
        /// </summary>
        internal static unsafe void GetCounts(out long solves, out long fallbacks, out long builds)
        {
            solves = fallbacks = builds = 0;
            if (StepKernel.Current != null)
            {
                StepKernel.Current->DirectSolverCache.GetForestCounts(out solves, out fallbacks, out builds);
            }
        }

        /// <summary>
        /// Returns section counts for the world most recently stepped. See DirectSolverCache.GetSectionCounts.
        /// </summary>
        internal static unsafe void GetSectionCounts(out long sectionSum, out long lockedSum, out long structureBuilds,
            out long lockChanges, out double structureMilliseconds, out double residual, out long lastSections, out long lastLocked)
        {
            sectionSum = lockedSum = structureBuilds = lockChanges = lastSections = lastLocked = 0;
            structureMilliseconds = residual = 0;
            if (StepKernel.Current != null)
            {
                StepKernel.Current->DirectSolverCache.GetSectionCounts(out sectionSum, out lockedSum, out structureBuilds,
                    out lockChanges, out structureMilliseconds, out residual, out lastSections, out lastLocked);
            }
        }

        /// <summary>
        /// Returns lock diagnostics for the world most recently stepped: changes to locks on pairs involved in
        /// contacts, the largest number of boundary rows in an island that fell back to the sparse solver, and
        /// deflection histograms for all tree pairs and locked tree pairs. Each histogram has 10 bins; see
        /// DirectSolverCache.DeflectionHistogram.
        /// </summary>
        internal static unsafe void GetLockDiagnostics(out long contactChanges, out long fallbackBoundary, long[] all, long[] locked)
        {
            contactChanges = fallbackBoundary = 0;
            if (StepKernel.Current == null)
            {
                return;
            }

            DirectSolverCache cache = StepKernel.Current->DirectSolverCache;
            contactChanges = cache.ForestCounter(DirectSolverCache.ForestContactLockChanges);
            fallbackBoundary = cache.ForestCounter(DirectSolverCache.ForestFallbackBoundaryMax);
            long* a = stackalloc long[10];
            long* l = stackalloc long[10];
            cache.DeflectionHistogram(LockAngle, YieldAngle, a, l);
            for (int i = 0; i < 10; i++)
            {
                all[i] = a[i];
                locked[i] = l[i];
            }
        }

        /// <summary>
        /// Returns the cumulative time in milliseconds spent in serial stages of the forest solver for the world most
        /// recently stepped. Times are measured since the world was built and cover row classification, lock decisions,
        /// sums for sections, sums of body impulses and sums for subtrees.
        /// </summary>
        internal static unsafe double[] SerialMilliseconds()
        {
            var result = new double[5];
            if (StepKernel.Current != null)
            {
                DirectSolverCache cache = StepKernel.Current->DirectSolverCache;
                for (int i = 0; i < 5; i++)
                {
                    result[i] = cache.ForestMilliseconds(DirectSolverCache.ForestClassifyTime + i);
                }
            }

            return result;
        }

        /// <summary>
        /// Clears the cached island forests and their lock states for the world most recently stepped. This prevents
        /// the next scene from reusing lock states from an earlier scene with the same structure. Call on the main
        /// thread between steps.
        /// </summary>
        internal static unsafe void ForgetIslands()
        {
            if (StepKernel.Current != null)
            {
                StepKernel.Current->DirectSolverCache.ForgetForests();
            }
        }

        /// <summary>
        /// The cumulative time in milliseconds spent settling sections after the first solve, measured since the world
        /// most recently stepped was built.
        /// </summary>
        internal static unsafe double SettleMilliseconds => StepKernel.Current != null ? StepKernel.Current->DirectSolverCache.ForestSettleMilliseconds : 0.0;

        /// <summary>
        /// The largest deflection in radians used by lock tests after the last forest solve, averaged over the step.
        /// </summary>
        internal static unsafe double LastDeflection => StepKernel.Current != null ? StepKernel.Current->DirectSolverCache.LastDeflection : 0.0;

        /// <summary>
        /// Returns cumulative solve times in milliseconds since the world most recently stepped was built. See
        /// DirectSolverCache.GetSolveTimes.
        /// </summary>
        internal static unsafe void GetTimes(out double sparse, out double forestStart, out double forestSolve, out double forestFinish,
            out double forestFactor, out double forestBoundary, out double forestDrift)
        {
            sparse = forestStart = forestSolve = forestFinish = forestFactor = forestBoundary = forestDrift = 0;
            if (StepKernel.Current != null)
            {
                StepKernel.Current->DirectSolverCache.GetSolveTimes(out sparse, out forestStart, out forestSolve, out forestFinish,
                    out forestFactor, out forestBoundary);
                forestDrift = StepKernel.Current->DirectSolverCache.ForestDriftMilliseconds;
            }
        }

        /// <summary>Reads the environment once, on the main thread, before the first step.</summary>
        internal static void Initialize()
        {
            if (initialized)
            {
                return;
            }

            initialized = true;
            string value = Environment.GetEnvironmentVariable("RETOOLED_FOREST");
            if (!string.IsNullOrEmpty(value))
            {
                Enabled = value == "1" || value.Equals("true", StringComparison.OrdinalIgnoreCase);
                UnityEngine.Debug.Log($"[VesselForest] RETOOLED_FOREST={value}: forest path {(Enabled ? "on" : "off")}");
            }

            string locking = Environment.GetEnvironmentVariable("RETOOLED_LOCK");
            if (!string.IsNullOrEmpty(locking))
            {
                Locking = locking == "1" || locking.Equals("true", StringComparison.OrdinalIgnoreCase);
                UnityEngine.Debug.Log($"[VesselForest] RETOOLED_LOCK={locking}: locking {(Locking ? "on" : "off")}");
            }
        }
    }
}
