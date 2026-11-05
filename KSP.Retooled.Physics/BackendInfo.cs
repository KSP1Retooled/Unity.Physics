using UnityEngine;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>Which physics backend this run selected in the project's DynamicsManager settings.</summary>
    public static class BackendInfo
    {
        /// <summary>The integration id of the Unity Physics backend, "RTPH", the value of m_CurrentBackendId that selects it.</summary>
        public const uint RetooledId = 0x52545048;

        public static bool IsRetooled => Physics.GetCurrentIntegrationInfo().id == RetooledId;

        /// <summary>The one line that tells which backend a log came from.</summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void LogSelection()
        {
            IntegrationInfo current = Physics.GetCurrentIntegrationInfo();
            Debug.Log($"[RetooledPhysics] backend {current.name} {current.id:X8}, fallback {current.isFallback}");
        }
    }
}
