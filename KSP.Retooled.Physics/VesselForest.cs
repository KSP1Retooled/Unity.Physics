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
    /// </summary>
    public static class VesselForest
    {
        private readonly struct Key { }

        private static readonly SharedStatic<ForestSettings> Shared = SharedStatic<ForestSettings>.GetOrCreate<Key>();

        private static bool initialized;

        /// <summary>The boundary rows above which an island solves the old way.</summary>
        public const int DefaultBoundaryRowCap = 256;

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
        }
    }
}
