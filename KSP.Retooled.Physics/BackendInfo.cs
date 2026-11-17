using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>Reports the physics backend selected for this run in the project's DynamicsManager settings.</summary>
    public static class BackendInfo
    {
        /// <summary>
        /// Unity Physics backend ID ("RTPH"). Set m_CurrentBackendId to select it.
        /// </summary>
        public const uint RetooledId = 0x52545048;

        public static bool IsRetooled => UnityEngine.Physics.GetCurrentIntegrationInfo().id == RetooledId;

        /// <summary>
        /// Log the selected backend at startup.
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void LogSelection()
        {
            IntegrationInfo current = UnityEngine.Physics.GetCurrentIntegrationInfo();
            Debug.Log($"[RetooledPhysics] backend {current.name} {current.id:X8}, fallback {current.isFallback}");
        }
    }
}
