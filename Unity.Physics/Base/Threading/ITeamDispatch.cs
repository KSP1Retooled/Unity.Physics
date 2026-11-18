namespace Unity.Physics.Base.Threading
{
    /// <summary>Runs one task after a team lane claims it.</summary>
    internal interface ITeamDispatch
    {
        void Run(in WorkTeam.Stage stage, int task);
    }
}
