namespace Unity.Physics.Base.Threading
{
    /// <summary>
    /// Count published/helped stages and first-helper arrival times across teams for handoff benchmarks.
    /// </summary>
    internal struct WorkTeamCounters
    {
        public long Stages;
        public long HelpedStages;
        public long ArrivalTicks;

        internal static readonly Unity.Burst.SharedStatic<WorkTeamCounters> Shared =
            Unity.Burst.SharedStatic<WorkTeamCounters>.GetOrCreate<WorkTeamCounters>();

        public static WorkTeamCounters Current => Shared.Data;

        public static void Reset()
        {
            Shared.Data = default;
        }
    }
}
