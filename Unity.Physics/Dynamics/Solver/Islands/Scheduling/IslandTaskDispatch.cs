using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Dynamics.Solver.Vessels;
using Unity.Physics.Dynamics.Vessels.Sections;
using static Unity.Physics.Dynamics.Solver.Islands.Data.IslandTaskKind;

namespace Unity.Physics.Dynamics.Solver.Islands.Scheduling
{
    /// <summary>Runs any stage of an island solve when a team helper claims a task.</summary>
    internal unsafe struct IslandTaskDispatch : ITeamDispatch
    {
        public void Run(in WorkTeam.Stage stage, int task)
        {
            if (stage.Kind == PredictDriftKind)
            {
                VesselIslandSolver.PredictDriftTask(stage.Context, task);
            }
            else if (stage.Kind == ApplyImpulsesKind)
            {
                VesselIslandSolver.ApplyImpulsesTask(stage.Context, task);
            }
            else if (stage.Kind == SetupBodiesKind)
            {
                VesselIslandSolver.SetupBodiesTask(stage.Context, task);
            }
            else if (stage.Kind == SummarizeJacobiansKind)
            {
                VesselIslandSolver.SummarizeJacobiansTask(stage.Context, task);
            }
            else if (stage.Kind == FillBlocksKind)
            {
                VesselIslandSolver.FillBlocksTask(stage.Context, task);
            }
            else if (stage.Kind == IterativePhaseKind)
            {
                IslandSolveScheduler.IterativePhaseTask(stage.Context, task);
            }
            else if (stage.Kind == ForestFactorKind)
            {
                ForestSolver.ForestFactorTask(stage.Context, task);
            }
            else if (stage.Kind == ForestForwardKind)
            {
                ((VesselContext*)stage.Context)->ForwardChunk(task);
            }
            else if (stage.Kind == ForestSubstituteKind)
            {
                ((VesselContext*)stage.Context)->SubstituteChunk(task);
            }
            else if (stage.Kind == ForestSectionsKind)
            {
                ForestSolver.ForestSectionTask(stage.Context, task);
            }
            else if (stage.Kind == ForestClassifyKind)
            {
                ForestSolver.ForestClassifyTask(stage.Context, task);
            }
            else if (stage.Kind == ForestVesselKind)
            {
                VesselIslandSolver.ForestVesselTask(stage.Context, task);
            }
            else if (stage.Kind == ForestBoundaryKind)
            {
                ForestSolver.ForestBoundaryTask(stage.Context, task);
            }
            else if (stage.Kind == StrutLoadsKind)
            {
                VesselIslandSolver.StrutLoadTask(stage.Context, task);
            }
            else if (stage.Kind == StrutForwardKind)
            {
                VesselIslandSolver.StrutForwardTask(stage.Context, task);
            }
            else if (stage.Kind == StrutBackKind)
            {
                VesselIslandSolver.StrutBackTask(stage.Context, task);
            }
            else if (stage.Kind == StrutLinksKind)
            {
                VesselIslandSolver.StrutLinkTask(stage.Context, task);
            }
            else
            {
                ((SparseConstraintSolver*)stage.Context)->RunTask(stage.Kind, task, (double*)stage.Vector);
            }
        }
    }
}
