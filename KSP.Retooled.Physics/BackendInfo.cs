namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>Which physics backend this run selected in the project's DynamicsManager settings.</summary>
    public static class BackendInfo
    {
        /// <summary>The integration id of the Unity Physics backend, "RTPH", the value of m_CurrentBackendId that selects it.</summary>
        public const uint RetooledId = 0x52545048;

        public static bool IsRetooled => UnityEngine.Physics.GetCurrentIntegrationInfo().id == RetooledId;
    }
}
