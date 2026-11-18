using Unity.Burst;

namespace Unity.Physics.Configuration
{
    internal static class StepCacheSwitches
    {
        internal static bool MemoryChecks
        {
            get => MemoryChecksValue.Data;
            set => MemoryChecksValue.Data = value;
        }

        internal static bool RefillVesselEdges
        {
            get => RefillVesselEdgesValue.Data;
            set => RefillVesselEdgesValue.Data = value;
        }

        internal static bool ClearSolverSlots
        {
            get => ClearSolverSlotsValue.Data;
            set => ClearSolverSlotsValue.Data = value;
        }

        private struct MemoryChecksKey { }
        private struct RefillVesselEdgesKey { }
        private struct ClearSolverSlotsKey { }
        private static readonly SharedStatic<bool> MemoryChecksValue = SharedStatic<bool>.GetOrCreate<MemoryChecksKey>();
        private static readonly SharedStatic<bool> RefillVesselEdgesValue = SharedStatic<bool>.GetOrCreate<RefillVesselEdgesKey>();
        private static readonly SharedStatic<bool> ClearSolverSlotsValue = SharedStatic<bool>.GetOrCreate<ClearSolverSlotsKey>();

        /// <summary>
        /// Reuse unchanged static body rows. RETOOLED_STATIC_REUSE=0 disables it.
        /// Stored in SharedStatic because Burst reads it.
        /// </summary>
        internal static bool StaticReuseEnabled
        {
            get => StaticReuseOff.Data == 0;
            set => StaticReuseOff.Data = value ? 0 : 1;
        }

        /// <summary>Reuse a serial island sort when its input is unchanged. RETOOLED_PAIR_SORT_REUSE=0 disables it.</summary>
        internal static bool PairSortReuseEnabled
        {
            get => PairSortReuseOff.Data == 0;
            set => PairSortReuseOff.Data = value ? 0 : 1;
        }

        private struct StaticReuseOffKey { }

        private struct PairSortReuseOffKey { }

        private static readonly SharedStatic<int> PairSortReuseOff = SharedStatic<int>.GetOrCreate<PairSortReuseOffKey>();

        private static readonly SharedStatic<int> StaticReuseOff = SharedStatic<int>.GetOrCreate<StaticReuseOffKey>();
    }
}
