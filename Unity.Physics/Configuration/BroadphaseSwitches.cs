using Unity.Burst;

namespace Unity.Physics.Configuration
{
    internal static class BroadphaseSwitches
    {
        /// <summary>
        /// Enable dynamic tree reuse. RETOOLED_TREE_REUSE=0 disables it.
        /// Burst reads this switch through SharedStatic. Always refitting bypasses the count and quality limits.
        /// </summary>
        internal static bool TreeReuseEnabled
        {
            get => TreeReuseMode.Data != 1;
            set => TreeReuseMode.Data = value ? 0 : 1;
        }

        /// <summary>
        /// Whether every build refits the dynamic tree when the dynamic bodies are the same, however bad the tree has
        /// become. RETOOLED_TREE_REUSE=always turns it on, to check that a tree of any shape gives the same results.
        /// </summary>
        internal static bool TreeAlwaysRefitted
        {
            get => TreeReuseMode.Data == 2;
            set => TreeReuseMode.Data = value ? 2 : 0;
        }

        private struct TreeReuseModeKey { }

        private static readonly SharedStatic<int> TreeReuseMode = SharedStatic<int>.GetOrCreate<TreeReuseModeKey>();

        /// <summary>
        /// Reuse bounds cached by the previous refit. Tests can disable this for comparison.
        /// </summary>
        internal static bool BoundsCacheEnabled
        {
            get => BoundsCacheOff.Data == 0;
            set => BoundsCacheOff.Data = value ? 0 : 1;
        }

        private struct BoundsCacheOffKey { }

        private static readonly SharedStatic<int> BoundsCacheOff = SharedStatic<int>.GetOrCreate<BoundsCacheOffKey>();

        private struct CheckOnKey { }

        private static readonly SharedStatic<int> SharedCheckOn = SharedStatic<int>.GetOrCreate<CheckOnKey>();

        /// <summary>
        /// Validate cached bounds, filters and responses against recomputed values while still using cached
        /// results.
        /// The bounds cache records mismatches.
        /// </summary>
        internal static bool BoundsCheckEnabled
        {
            get => SharedCheckOn.Data != 0;
            set => SharedCheckOn.Data = value ? 1 : 0;
        }

        private struct ShapesOffKey { }

        private static readonly SharedStatic<int> SharedShapesOff = SharedStatic<int>.GetOrCreate<ShapesOffKey>();

        /// <summary>
        /// Whether the cache keeps the collider geometry that it needs to compute each body's bounds.
        /// Set RETOOLED_SHAPE_BOUNDS=0 to turn this off.
        /// </summary>
        internal static bool ShapeBoundsEnabled
        {
            get => SharedShapesOff.Data == 0;
            set => SharedShapesOff.Data = value ? 0 : 1;
        }

        /// <summary>
        /// Test option leaving all refit work to queries or owner helpers instead of scheduled workers.
        /// </summary>
        internal static bool RefitJobsLeaveTheWork
        {
            get => RefitJobsLeaveTheWorkFlag.Data != 0;
            set => RefitJobsLeaveTheWorkFlag.Data = value ? 1 : 0;
        }

        private struct JobsLeaveTheWorkKey { }

        private static readonly SharedStatic<int> RefitJobsLeaveTheWorkFlag = SharedStatic<int>.GetOrCreate<JobsLeaveTheWorkKey>();
    }
}
