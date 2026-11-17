using System;
using Unity.Burst;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Islands;

namespace Unity.Physics.Configuration
{
    /// <summary>
    /// Control forest solving and optional load-based locking of vessel joints.
    /// RETOOLED_FOREST=1 enables the forest at attach; RETOOLED_LOCK=1 enables locking.
    /// Locked bodies form a rigid section until deflection or load exceeds the configured yield limits.
    /// </summary>
    internal static class ForestSwitches
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
        internal const int DefaultBoundaryRowCap = 512;

        /// <summary>The fraction of joint break limits below which smoothed loads permit a pair to lock.</summary>
        internal const float DefaultLockFraction = 0.25f;

        /// <summary>
        /// The fraction of joint break limits above which a smoothed load causes a locked pair to yield.
        /// </summary>
        internal const float DefaultYieldFraction = 0.5f;

        internal static bool Enabled
        {
            get => Shared.Data.Enabled;
            set => Shared.Data.Enabled = value;
        }

        internal static int BoundaryRowCap
        {
            get => Shared.Data.BoundaryRowCap > 0 ? Shared.Data.BoundaryRowCap : DefaultBoundaryRowCap;
            set => Shared.Data.BoundaryRowCap = value;
        }

        /// <summary>
        /// The limits on a vessel island's boundary block: contact points, self-contact manifolds and pivoting rounds. A
        /// field that is zero takes its default. See <see cref="BoundaryCaps"/>.
        /// </summary>
        internal static BoundaryCaps Caps
        {
            get => Shared.Data.Caps;
            set => Shared.Data.Caps = value;
        }

        /// <summary>
        /// Whether the forest solver locks tree edges under low loads. See <see cref="ForestSettings.Locking"/>.
        /// </summary>
        internal static bool Locking
        {
            get => Shared.Data.Locking;
            set => Shared.Data.Locking = value;
        }

        /// <summary>The angle in radians used to permit locking. See <see cref="ForestSettings.LockAngle"/>.</summary>
        internal static float LockAngle
        {
            get => Shared.Data.LockAngle > 0.0f ? Shared.Data.LockAngle : ForestSettings.DefaultLockAngle;
            set => Shared.Data.LockAngle = value;
        }

        /// <summary>
        /// The angle in radians used to trigger yielding. See <see cref="ForestSettings.YieldAngle"/>.
        /// </summary>
        internal static float YieldAngle
        {
            get => Shared.Data.YieldAngle > 0.0f ? Shared.Data.YieldAngle : ForestSettings.DefaultYieldAngle;
            set => Shared.Data.YieldAngle = value;
        }

        /// <summary>
        /// The minimum number of steps to hold a lock. Negative values select the default. See
        /// <see cref="ForestSettings.MinimumHoldSteps"/>.
        /// </summary>
        internal static int MinimumHoldSteps
        {
            get => Fractions.Data.Hold >= 0 ? Fractions.Data.Hold : ForestSettings.DefaultMinimumHoldSteps;
            set => Fractions.Data.Hold = value;
        }

        /// <summary>
        /// The number of steps used to smooth deflection. Negative values select the default. See
        /// <see cref="ForestSettings.DeflectionSmoothingSteps"/>.
        /// </summary>
        internal static int DeflectionSmoothingSteps
        {
            get => Fractions.Data.SmoothingPlusOne > 0 ? Fractions.Data.SmoothingPlusOne - 1 : ForestSettings.DefaultDeflectionSmoothingSteps;
            set => Fractions.Data.SmoothingPlusOne = value + 1;
        }

        internal static float LockFraction
        {
            get => Fractions.Data.Lock > 0.0f ? Fractions.Data.Lock : DefaultLockFraction;
            set => Fractions.Data.Lock = value;
        }

        internal static float YieldFraction
        {
            get => Fractions.Data.Yield > 0.0f ? Fractions.Data.Yield : DefaultYieldFraction;
            set => Fractions.Data.Yield = value;
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

            // Dormant joint rows stay out of a vessel island's boundary unless RETOOLED_DORMANT_ROWS=0.
            string dormant = Environment.GetEnvironmentVariable("RETOOLED_DORMANT_ROWS");
            if (!string.IsNullOrEmpty(dormant))
            {
                BoundaryCaps caps = Caps;
                caps.KeepDormantRows = dormant == "0" || dormant.Equals("false", StringComparison.OrdinalIgnoreCase);
                Caps = caps;
                UnityEngine.Debug.Log($"[VesselForest] RETOOLED_DORMANT_ROWS={dormant}: dormant rows {(caps.KeepDormantRows ? "kept in the boundary" : "left out")}");
            }
        }
    }
}
