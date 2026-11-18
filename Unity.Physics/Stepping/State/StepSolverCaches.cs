using Unity.Physics.Dynamics.Simulation.Islands;
using Unity.Physics.Dynamics.Solver.Islands;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the direct solver and island sorting caches for the lifetime of the stepper.
    /// The solver and island scheduler decide when to reuse their entries.
    /// </summary>
    internal unsafe struct StepSolverCaches
    {
        internal IslandSolverCache directSolverCache;
        internal SolverIslandCache solverIslandCache;

        internal void InitializeDirectSolver()
        {
            directSolverCache = IslandSolverCache.Create();
        }

        internal void Initialize()
        {
            solverIslandCache = SolverIslandCache.Create();
        }

        internal void DisposeDirectSolver()
        {
            directSolverCache.Dispose();
        }

        internal void Dispose()
        {
            solverIslandCache.Dispose();
        }
    }
}
