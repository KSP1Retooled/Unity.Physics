using System;
using Unity.Physics.Stepping.Recording;

namespace Unity.Physics.Configuration
{
    /// <summary>Controls the backend flight recorder. Set these properties from the main thread.</summary>
    public static class FlightRecorderSwitches
    {
        private static bool enabled;
        private static bool setAtRuntime;

        public static bool Enabled
        {
            get => enabled;
            set
            {
                setAtRuntime = true;
                enabled = value;
                if (!value)
                {
                    FlightRecorder.Clear();
                }
            }
        }

        public static int WindowSteps { get; set; } = 300;
        public static int PostTriggerSteps { get; set; } = 25;
        public static float LinearVelocityThreshold { get; set; } = 200f;
        public static float AngularVelocityThreshold { get; set; } = 200f;
        public static int MaxFiles { get; set; } = 4;
        public static string OutputDirectory { get; set; }

        /// <summary>Queues a save of the most recently stepped world and returns its path, or null.</summary>
        public static string Save() => SaveWithReason("manual");

        public static string SaveWithReason(string reason) => FlightRecorder.SaveLatest(reason ?? "manual");

        /// <summary>Reports the current window and the outcome of the most recent save.</summary>
        public static string GetStatus() => FlightRecorder.Status(Enabled);

        internal static void ReadEnvironment(string value)
        {
            if (!setAtRuntime)
            {
                enabled = value == "1" || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
            }
        }
    }
}
