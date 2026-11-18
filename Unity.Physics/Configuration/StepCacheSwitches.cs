using Unity.Burst;

namespace Unity.Physics.Configuration
{
    internal static class StepCacheSwitches
    {
        /// <summary>
        /// Reuse unchanged static body rows. RETOOLED_STATIC_REUSE=0 disables it.
        /// Stored in SharedStatic because Burst reads it.
        /// </summary>
        internal static bool StaticReuseEnabled
        {
            get => StaticReuseOff.Data == 0;
            set => StaticReuseOff.Data = value ? 0 : 1;
        }

        private struct StaticReuseOffKey { }

        private static readonly SharedStatic<int> StaticReuseOff = SharedStatic<int>.GetOrCreate<StaticReuseOffKey>();
    }
}
