using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>Reports the physics backend selected for this run in the project's DynamicsManager settings.</summary>
    public static class BackendInfo
    {
        /// <summary>
        /// Defines the integration id of the Unity Physics backend, "RTPH". Setting m_CurrentBackendId to this value
        /// selects that backend.
        /// </summary>
        public const uint RetooledId = 0x52545048;

        public static bool IsRetooled => UnityEngine.Physics.GetCurrentIntegrationInfo().id == RetooledId;

        /// <summary>
        /// Logs one line that identifies the selected backend, so that readers can tell which backend produced the log.
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void LogSelection()
        {
            IntegrationInfo current = UnityEngine.Physics.GetCurrentIntegrationInfo();
            Debug.Log($"[RetooledPhysics] backend {current.name} {current.id:X8}, fallback {current.isFallback}");
        }
    }
}
