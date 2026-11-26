using System;
using Unity.Physics.Dynamics.Solver.Forest;

namespace Unity.Physics.Configuration
{
    internal static class EnvironmentSwitches
    {
        private static bool initialized;
        private static string boundsCache;
        private static string boundsCheck;
        private static string shapeBounds;
        private static string treeReuse;
        private static string staticReuse;

        /// <summary>Read environment settings once on the main thread before the first world state is built.</summary>
        internal static void Read()
        {
            if (initialized)
            {
                return;
            }

            initialized = true;
            FlightRecorderSwitches.ReadEnvironment(Environment.GetEnvironmentVariable("RETOOLED_FLIGHT_RECORDER"));
            ReadForest();
            ReadVessels();
            StepCacheSwitches.PairSortReuseEnabled = Environment.GetEnvironmentVariable("RETOOLED_PAIR_SORT_REUSE") != "0";
            boundsCache = Environment.GetEnvironmentVariable("RETOOLED_BOUNDS_CACHE");
            boundsCheck = Environment.GetEnvironmentVariable("RETOOLED_BOUNDS_CHECK");
            shapeBounds = Environment.GetEnvironmentVariable("RETOOLED_SHAPE_BOUNDS");
            treeReuse = Environment.GetEnvironmentVariable("RETOOLED_TREE_REUSE");
            staticReuse = Environment.GetEnvironmentVariable("RETOOLED_STATIC_REUSE");
            StepCacheSwitches.MemoryChecks = Environment.GetEnvironmentVariable("RETOOLED_MEMORY_CHECKS") == "1";
            StepCacheSwitches.RefillVesselEdges = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_EDGE_CACHE") == "0";
            StepCacheSwitches.ClearSolverSlots = Environment.GetEnvironmentVariable("RETOOLED_SOLVER_SLOT_CACHE") == "0";
        }

        /// <summary>
        /// Restore the cache defaults for each new world.
        /// </summary>
        internal static void ApplyToWorld()
        {
            if (!initialized)
            {
                Read();
            }

            ApplyBoundsCache();
            ApplyTreeReuse();
            ApplyStaticReuse();
        }

        private static void ReadForest()
        {
            string value = Environment.GetEnvironmentVariable("RETOOLED_FOREST");
            if (!string.IsNullOrEmpty(value))
            {
                ForestSwitches.Enabled = value == "1" || value.Equals("true", StringComparison.OrdinalIgnoreCase);
                UnityEngine.Debug.Log($"[ForestSwitches] RETOOLED_FOREST={value}: forest path {(ForestSwitches.Enabled ? "on" : "off")}");
            }

            string locking = Environment.GetEnvironmentVariable("RETOOLED_LOCK");
            if (!string.IsNullOrEmpty(locking))
            {
                ForestSwitches.Locking = locking == "1" || locking.Equals("true", StringComparison.OrdinalIgnoreCase);
                UnityEngine.Debug.Log($"[ForestSwitches] RETOOLED_LOCK={locking}: locking {(ForestSwitches.Locking ? "on" : "off")}");
            }

            // Dormant joint rows stay out of a vessel island's boundary unless RETOOLED_DORMANT_ROWS=0.
            string dormant = Environment.GetEnvironmentVariable("RETOOLED_DORMANT_ROWS");
            if (!string.IsNullOrEmpty(dormant))
            {
                BoundaryCaps caps = ForestSwitches.Caps;
                caps.KeepDormantRows = dormant == "0" || dormant.Equals("false", StringComparison.OrdinalIgnoreCase);
                ForestSwitches.Caps = caps;
                UnityEngine.Debug.Log($"[ForestSwitches] RETOOLED_DORMANT_ROWS={dormant}: dormant rows {(caps.KeepDormantRows ? "kept in the boundary" : "left out")}");
            }
        }

        private static void ReadVessels()
        {
            string minimum = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_MIN_PARTS");
            if (int.TryParse(minimum, out int parts))
            {
                VesselSwitches.MinimumParts = parts;
                UnityEngine.Debug.Log($"[VesselSwitches] RETOOLED_VESSEL_MIN_PARTS={minimum}: vessels have at least {VesselSwitches.MinimumParts} parts");
            }

            VesselSwitches.StrutExperiment = int.TryParse(Environment.GetEnvironmentVariable("STRUT_EXPERIMENT"), out int experiment)
                ? experiment : 0;

            string fold = Environment.GetEnvironmentVariable("RETOOLED_SUBSTEP_FOLD");
            SolverSwitches.UnfusedFolding = fold == "1" ? SolverSwitches.SubstepFolding.Fold
                : fold == "0" ? SolverSwitches.SubstepFolding.Chain : SolverSwitches.SubstepFolding.Auto;
            string fusedParts = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_FUSED_PARTS");
            if (int.TryParse(fusedParts, out int fusedMinimum) && fusedMinimum > 0)
            {
                SolverSwitches.FusedVesselParts = fusedMinimum;
                UnityEngine.Debug.Log($"[VesselSwitches] RETOOLED_VESSEL_FUSED_PARTS={fusedParts}: the fused job takes vessel islands of {fusedMinimum} parts or more");
            }

            string value = Environment.GetEnvironmentVariable("RETOOLED_VESSEL");
            if (string.IsNullOrEmpty(value))
            {
                VesselSwitches.Enabled = true;
                return;
            }

            VesselSwitches.Enabled = !IsOff(value);
            UnityEngine.Debug.Log($"[VesselSwitches] RETOOLED_VESSEL={value}: vessels {(VesselSwitches.Enabled ? "on" : "off")}");
        }

        private static void ApplyBoundsCache()
        {
            BroadphaseSwitches.BoundsCacheEnabled = boundsCache != "0";
            BroadphaseSwitches.BoundsCheckEnabled = boundsCheck == "1";
            BroadphaseSwitches.ShapeBoundsEnabled = shapeBounds != "0";
            if (boundsCache == "0" || boundsCheck == "1" || shapeBounds == "0")
            {
                UnityEngine.Debug.Log($"[StoreStepper] RETOOLED_BOUNDS_CACHE={boundsCache}, RETOOLED_BOUNDS_CHECK={boundsCheck}, RETOOLED_SHAPE_BOUNDS={shapeBounds}: bounds cache {(BroadphaseSwitches.BoundsCacheEnabled ? "on" : "off")}, check {(BroadphaseSwitches.BoundsCheckEnabled ? "on" : "off")}, shape bounds {(BroadphaseSwitches.ShapeBoundsEnabled ? "on" : "off")}");
            }
        }

        private static void ApplyTreeReuse()
        {
            BroadphaseSwitches.TreeReuseEnabled = treeReuse != "0";
            if (treeReuse == "always")
            {
                BroadphaseSwitches.TreeAlwaysRefitted = true;
            }

            if (treeReuse == "0" || treeReuse == "always")
            {
                UnityEngine.Debug.Log($"[StoreStepper] RETOOLED_TREE_REUSE={treeReuse}: {(treeReuse == "0" ? "every build builds the dynamic tree from scratch" : "every build refits the dynamic tree when the dynamic bodies are the same")}");
            }
        }

        private static void ApplyStaticReuse()
        {
            StepCacheSwitches.StaticReuseEnabled = staticReuse != "0";
            if (staticReuse == "0")
            {
                UnityEngine.Debug.Log("[StoreStepper] RETOOLED_STATIC_REUSE=0: every build writes the static bodies' rows from the store");
            }
        }

        private static bool IsOff(string value)
        {
            return value == "0" || string.Equals(value, "false", StringComparison.OrdinalIgnoreCase);
        }
    }
}
