using Unity.Burst;

namespace Unity.Physics.Configuration
{
    /// <summary>
    /// Control persistent vessels compiled from part joints between moving bodies.
    /// Vessels replace those joints' solver constraints and enable forest solving.
    /// Enabled by default. Set RETOOLED_VESSEL=0 or false to disable vessels at the first world build.
    /// </summary>
    internal static class VesselSwitches
    {
        private readonly struct Key { }

        private readonly struct MinimumPartsKey { }

        private static readonly SharedStatic<bool> Shared = SharedStatic<bool>.GetOrCreate<Key>();

        private static readonly SharedStatic<int> MinimumPartsShared = SharedStatic<int>.GetOrCreate<MinimumPartsKey>();

        /// <summary>
        /// Default minimum vessel size: two parts, so any eligible joined craft can use the vessel solver.
        /// </summary>
        internal const int DefaultMinimumParts = 2;

        /// <summary>
        /// Steps to retain a removed edge or closure as a tombstone; see CompileVesselsJob.
        /// </summary>
        internal const int TombstoneSteps = 10;

        /// <summary>
        /// Minimum parts needed to compile a vessel. Smaller crafts keep Unity Physics joints.
        /// Values below one restore the default. RETOOLED_VESSEL_MIN_PARTS sets the starting value.
        /// Changes take effect on the next vessel compilation.
        /// </summary>
        internal static int MinimumParts
        {
            get => MinimumPartsShared.Data > 0 ? MinimumPartsShared.Data : DefaultMinimumParts;
            set => MinimumPartsShared.Data = value;
        }

        internal static bool Enabled
        {
            get => Shared.Data;
            set => Shared.Data = value;
        }

        private struct StrutExperimentKey { }

        private static readonly SharedStatic<int> StrutExperimentShared = SharedStatic<int>.GetOrCreate<StrutExperimentKey>();

        /// <summary>Use 1 to skip strut solves, or 2 to skip only the deferred solve at the end of a step.</summary>
        internal static int StrutExperiment
        {
            get => StrutExperimentShared.Data;
            set => StrutExperimentShared.Data = value;
        }
    }
}
