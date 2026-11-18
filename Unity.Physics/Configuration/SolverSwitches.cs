using Unity.Burst;

namespace Unity.Physics.Configuration
{
    internal static class SolverSwitches
    {
        /// <summary>
        /// Solve vessel-only islands in one fused job across all substeps, including their contact/joint boundary.
        /// Set RETOOLED_VESSEL_FUSED=0 to disable it at the first world build.
        /// </summary>
        internal static bool VesselFused
        {
            get => !VesselFusedOff.Data;
            set => VesselFusedOff.Data = !value;
        }

        private struct VesselFusedKey { }

        /// <summary>
        /// Allow fused vessel islands with contact or joint boundary rows. Benchmarks may disable this.
        /// </summary>
        internal static bool FusedContacts
        {
            get => !FusedContactsOff.Data;
            set => FusedContactsOff.Data = !value;
        }

        private struct FusedContactsKey { }

        private static readonly SharedStatic<bool> FusedContactsOff = SharedStatic<bool>.GetOrCreate<FusedContactsKey>();

        private static readonly SharedStatic<bool> VesselFusedOff = SharedStatic<bool>.GetOrCreate<VesselFusedKey>();

        /// <summary>
        /// Minimum vessel part count for fused solving. By default every eligible island is fused.
        /// RETOOLED_VESSEL_FUSED_PARTS sets the starting value.
        /// </summary>
        internal static int FusedVesselParts
        {
            get => FusedVesselPartsValue.Data > 0 ? FusedVesselPartsValue.Data : DefaultFusedVesselParts;
            set => FusedVesselPartsValue.Data = value;
        }

        /// <summary>The value of <see cref="FusedVesselParts"/> until something sets it.</summary>
        internal const int DefaultFusedVesselParts = 1;

        private struct FusedVesselPartsKey { }

        private static readonly SharedStatic<int> FusedVesselPartsValue = SharedStatic<int>.GetOrCreate<FusedVesselPartsKey>();

        /// <summary>
        /// Select chained or serial unfused substeps. Auto uses previous-step work counts.
        /// Set RETOOLED_SUBSTEP_FOLD=0 to force Chain or 1 to force Fold.
        /// </summary>
        internal enum SubstepFolding
        {
            Auto = 0,
            Fold = 1,
            Chain = 2,
        }

        internal static SubstepFolding UnfusedFolding
        {
            get => (SubstepFolding)UnfusedFoldingValue.Data;
            set => UnfusedFoldingValue.Data = (int)value;
        }

        private struct UnfusedFoldingKey { }

        private static readonly SharedStatic<int> UnfusedFoldingValue = SharedStatic<int>.GetOrCreate<UnfusedFoldingKey>();

        /// <summary>
        /// Maximum prior-step pair count for serial island sorting. Set to -1 to force the parallel chain.
        /// Stored in SharedStatic for Burst.
        /// </summary>
        internal static int SerialIslandPairLimit
        {
            get => DefaultSerialIslandPairLimit + SerialIslandPairLimitValue.Data;
            set => SerialIslandPairLimitValue.Data = value - DefaultSerialIslandPairLimit;
        }

        /// <summary>
        /// The value of <see cref="SerialIslandPairLimit"/> until something sets it. One thread sorts this many pairs
        /// in a few tens of microseconds, about as long as the parallel chain spends waiting between its jobs.
        /// </summary>
        internal const int DefaultSerialIslandPairLimit = 4096;

        private struct SerialIslandPairLimitKey { }

        // Store the difference from the default so that a zero value selects the default.
        private static readonly SharedStatic<int> SerialIslandPairLimitValue = SharedStatic<int>.GetOrCreate<SerialIslandPairLimitKey>();
    }
}
