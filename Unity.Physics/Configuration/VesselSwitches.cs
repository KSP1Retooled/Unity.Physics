using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Memory;
using Unity.Physics.Dynamics.Vessels.Detection;

namespace Unity.Physics.Configuration
{
    /// <summary>
    /// Control persistent vessels compiled from part joints between moving bodies.
    /// Vessels replace those joints' solver constraints and enable forest solving.
    /// Enabled by default; RETOOLED_VESSEL=0 or false disables them at first attach.
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
        public const int DefaultMinimumParts = 2;

        /// <summary>
        /// Steps to retain a removed edge or closure as a tombstone; see CompileVesselsJob.
        /// </summary>
        public const int TombstoneSteps = 10;

        /// <summary>
        /// Minimum parts needed to compile a vessel. Smaller crafts keep Unity Physics joints.
        /// Values below one restore the default. Read RETOOLED_VESSEL_MIN_PARTS at first attach.
        /// Changes take effect on the next vessel compilation.
        /// </summary>
        public static int MinimumParts
        {
            get => MinimumPartsShared.Data > 0 ? MinimumPartsShared.Data : DefaultMinimumParts;
            set => MinimumPartsShared.Data = value;
        }

        private static bool initialized;

        public static bool Enabled
        {
            get => Shared.Data;
            set => Shared.Data = value;
        }

        /// <summary>Reads the environment once, on the main thread, before the first step.</summary>
        internal static void Initialize()
        {
            if (initialized)
            {
                return;
            }

            initialized = true;
            string minimum = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_MIN_PARTS");
            if (int.TryParse(minimum, out int parts))
            {
                MinimumParts = parts;
                UnityEngine.Debug.Log($"[VesselPipeline] RETOOLED_VESSEL_MIN_PARTS={minimum}: vessels have at least {MinimumParts} parts");
            }

            // RETOOLED_STRUT_LOADS=0 brings back the tree method for the loads inside a locked section.
            StrutLoads.Initialize();
            if (!StrutLoads.Enabled)
            {
                UnityEngine.Debug.Log("[VesselPipeline] RETOOLED_STRUT_LOADS=0: each locked edge carries its whole subtree, and closures inside a section carry nothing");
            }

            string fused = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_FUSED");
            Unity.Physics.Solver.VesselFused = fused != "0";
            string fold = Environment.GetEnvironmentVariable("RETOOLED_SUBSTEP_FOLD");
            Unity.Physics.Solver.UnfusedFolding = fold == "1" ? Unity.Physics.Solver.SubstepFolding.Fold
                : fold == "0" ? Unity.Physics.Solver.SubstepFolding.Chain : Unity.Physics.Solver.SubstepFolding.Auto;
            string fusedParts = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_FUSED_PARTS");
            if (int.TryParse(fusedParts, out int fusedMinimum) && fusedMinimum > 0)
            {
                Unity.Physics.Solver.FusedVesselParts = fusedMinimum;
                UnityEngine.Debug.Log($"[VesselPipeline] RETOOLED_VESSEL_FUSED_PARTS={fusedParts}: the fused job takes vessel islands of {fusedMinimum} parts or more");
            }

            string value = Environment.GetEnvironmentVariable("RETOOLED_VESSEL");
            if (string.IsNullOrEmpty(value))
            {
                Enabled = true;
                return;
            }

            Enabled = !(value == "0" || value.Equals("false", StringComparison.OrdinalIgnoreCase));
            UnityEngine.Debug.Log($"[VesselPipeline] RETOOLED_VESSEL={value}: vessels {(Enabled ? "on" : "off")}");
        }
    }
}
