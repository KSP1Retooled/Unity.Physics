using System;
using Unity.Burst;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The switch between the two ways a direct island solves. Off, every row goes to the sparse solver. On, the joints
    /// that hold their bodies as a part joint does form a vessel forest whose rows the section solver eliminates
    /// structurally, and every other row of the island is a boundary block solved against it (see
    /// <see cref="ForestSettings"/>). The step reads it from Burst, so it lives in a SharedStatic. The environment variable
    /// RETOOLED_FOREST=1 turns it on at attach, which is how a whole test run takes the forest path.
    ///
    /// <see cref="Locking"/> (off by default; RETOOLED_LOCK=1 at attach) lets the forest lock a tree edge whose joints
    /// carry little: the pair moves as one rigid section until its load would deflect its springs past
    /// <see cref="YieldAngle"/> or pass <see cref="YieldFraction"/> of its break limits. See <see cref="ForestSettings"/>.
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

            /// <summary>The minimum hold in steps, plus one, so that a zeroed static reads as the default.</summary>
            public int HoldPlusOne;

            public int Hold
            {
                get => HoldPlusOne - 1;
                set => HoldPlusOne = value + 1;
            }
        }

        private static bool initialized;

        /// <summary>The boundary rows above which an island solves the old way.</summary>
        public const int DefaultBoundaryRowCap = 256;

        /// <summary>A joint whose smoothed load is under this fraction of its break limits lets its pair lock.</summary>
        public const float DefaultLockFraction = 0.25f;

        /// <summary>A locked pair yields once a joint's smoothed load passes this fraction of its break limits.</summary>
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

        /// <summary>Whether the forest locks tree edges that carry little, see <see cref="ForestSettings.Locking"/>.</summary>
        public static bool Locking
        {
            get => Shared.Data.Locking;
            set => Shared.Data.Locking = value;
        }

        /// <summary>Radians; see <see cref="ForestSettings.LockAngle"/>.</summary>
        public static float LockAngle
        {
            get => Shared.Data.LockAngle > 0.0f ? Shared.Data.LockAngle : ForestSettings.DefaultLockAngle;
            set => Shared.Data.LockAngle = value;
        }

        /// <summary>Radians; see <see cref="ForestSettings.YieldAngle"/>.</summary>
        public static float YieldAngle
        {
            get => Shared.Data.YieldAngle > 0.0f ? Shared.Data.YieldAngle : ForestSettings.DefaultYieldAngle;
            set => Shared.Data.YieldAngle = value;
        }

        /// <summary>Steps; see <see cref="ForestSettings.MinimumHoldSteps"/>. Negative reads as the default.</summary>
        public static int MinimumHoldSteps
        {
            get => Fractions.Data.Hold >= 0 ? Fractions.Data.Hold : ForestSettings.DefaultMinimumHoldSteps;
            set => Fractions.Data.Hold = value;
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
        /// The last stepped world's forest counts since it was built: island solves on the forest path, forest-enabled
        /// solves that fell back to the sparse solver, and forest structures built. Main thread, between steps.
        /// </summary>
        internal static unsafe void GetCounts(out long solves, out long fallbacks, out long builds)
        {
            solves = fallbacks = builds = 0;
            if (StepKernel.Current != null)
            {
                StepKernel.Current->DirectSolverCache.GetForestCounts(out solves, out fallbacks, out builds);
            }
        }

        /// <summary>The last stepped world's section counts, see DirectSolverCache.GetSectionCounts.</summary>
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

        /// <summary>The last stepped world's milliseconds settling sections after the first solve, since it was built.</summary>
        internal static unsafe double SettleMilliseconds => StepKernel.Current != null ? StepKernel.Current->DirectSolverCache.ForestSettleMilliseconds : 0.0;

        /// <summary>The largest deflection, in radians, the lock tests read after the last forest solve (a step's mean).</summary>
        internal static unsafe double LastDeflection => StepKernel.Current != null ? StepKernel.Current->DirectSolverCache.LastDeflection : 0.0;

        /// <summary>The last stepped world's solve times in ms since it was built, see DirectSolverCache.GetSolveTimes.</summary>
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
