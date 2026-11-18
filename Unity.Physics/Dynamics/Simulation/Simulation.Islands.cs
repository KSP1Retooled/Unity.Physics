using Unity.Jobs;

namespace Unity.Physics
{
    public partial struct Simulation
    {
        /// <summary>
        /// Handle completing when motions are final, possibly before solver cleanup and event export.
        /// Consumers of events must still wait for FinalSimulationJobHandle.
        /// </summary>
        internal JobHandle MotionsSolvedHandle => StepContext.HasMotionsSolved ? StepContext.MotionsSolved : FinalSimulationJobHandle;

        /// <summary>
        /// Group the filtered pairs into islands.
        /// Remove unsolved pairs before calling so that they do not connect islands or occupy phases.
        /// </summary>
        internal JobHandle ScheduleIslandJobs(SimulationStepInput input, JobHandle inputDeps, bool multiThreaded = true)
        {
            if (input.World.NumDynamicBodies == 0)
            {
                return inputDeps;
            }

            JobHandle handle = DispatchPairSequencer.ScheduleSortPairsByIsland(input, StepContext.PairsToSortByIsland,
                StepContext.PhasedDispatchPairs, StepContext.SolverSchedulerInfo, inputDeps, out JobHandle disposed, multiThreaded);
            StepContext.IslandsSorted = true;

            m_StepHandles.FinalDisposeHandle = JobHandle.CombineDependencies(m_StepHandles.FinalDisposeHandle,
                StepContext.PairsToSortByIsland.Dispose(handle), disposed);
            m_StepHandles.FinalExecutionHandle = handle;
            return handle;
        }
    }
}
