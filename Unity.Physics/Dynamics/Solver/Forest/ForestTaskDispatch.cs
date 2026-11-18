using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Vessels.Sections;
using static Unity.Physics.Dynamics.Solver.Islands.Data.IslandTaskKind;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    /// <summary>Runs the tasks that the forest solver publishes to its team.</summary>
    internal unsafe struct ForestTaskDispatch : ITeamDispatch
    {
        public void Run(in WorkTeam.Stage stage, int task)
        {
            switch (stage.Kind)
            {
                case ForestFactorKind:
                    ForestSolver.ForestFactorTask(stage.Context, task);
                    break;
                case ForestForwardKind:
                    ((VesselContext*)stage.Context)->ForwardChunk(task);
                    break;
                case ForestSubstituteKind:
                    ((VesselContext*)stage.Context)->SubstituteChunk(task);
                    break;
                case ForestSectionsKind:
                    ForestSolver.ForestSectionTask(stage.Context, task);
                    break;
                case ForestClassifyKind:
                    ForestSolver.ForestClassifyTask(stage.Context, task);
                    break;
                case ForestBoundaryKind:
                    ForestSolver.ForestBoundaryTask(stage.Context, task);
                    break;
            }
        }
    }
}
