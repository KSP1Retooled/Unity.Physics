using Unity.Physics.Base.Threading;
using static Unity.Physics.Dynamics.Solver.Islands.Data.IslandTaskKind;

namespace Unity.Physics.Dynamics.Solver.Vessels
{
    /// <summary>Runs the tasks that the direct and vessel solvers publish to their teams.</summary>
    internal unsafe struct VesselTaskDispatch : ITeamDispatch
    {
        public void Run(in WorkTeam.Stage stage, int task)
        {
            switch (stage.Kind)
            {
                case ForestVesselKind:
                    VesselIslandSolver.ForestVesselTask(stage.Context, task);
                    break;
            }
        }
    }
}
