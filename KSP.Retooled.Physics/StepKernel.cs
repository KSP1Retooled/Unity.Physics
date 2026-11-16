using Unity.Burst;
using Unity.Jobs;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Provides Burst entry points for <see cref="StepState"/>. As with the other kernels, these methods run in Mono
    /// until asynchronous Burst compilation finishes. Each step runs the physics system group. One group serves every
    /// physics scene, so its systems use <see cref="Current"/> to find the world for the current step.
    /// </summary>
    [BurstCompile]
    internal static unsafe class StepKernel
    {
        private readonly struct CurrentKey { }

        private static readonly SharedStatic<nint> CurrentState = SharedStatic<nint>.GetOrCreate<CurrentKey>();

        internal static StepState* Current
        {
            get => (StepState*)CurrentState.Data;
            set => CurrentState.Data = (nint)value;
        }

        [BurstCompile]
        internal static void ApplyForces(float deltaTime)
        {
            NoteRunInMono();
            Current->ApplyForces(deltaTime);
        }

        [BurstCompile]
        internal static void BuildWorld(float deltaTime, JobHandle* dependency)
        {
            *dependency = Current->BuildWorld(deltaTime, *dependency);
        }

        [BurstCompile]
        internal static void SortIslands(SimulationSingleton* singleton, SimulationStepInput* input, JobHandle* dependency)
        {
            *dependency = Current->SortIslands(*singleton, ref *input, *dependency);
        }

        [BurstCompile]
        internal static void ReadContacts(SimulationSingleton* singleton, float deltaTime, JobHandle* dependency)
        {
            *dependency = Current->ReadContacts(*singleton, deltaTime, *dependency);
        }

        [BurstCompile]
        internal static void PrepareSolve(SimulationSingleton* singleton, SimulationStepInput* input, JobHandle* dependency)
        {
            *dependency = Current->PrepareSolve(*singleton, ref *input, *dependency);
        }

        [BurstCompile]
        internal static void FinishStep(Simulation* simulation, float deltaTime)
        {
            Current->preparedSimulation = null;
            Current->FinishStep(*simulation, deltaTime);
        }

        /// <summary>
        /// Finishes the step whose solve PrepareSolve prepared, while that solve may still be running, and returns 1.
        /// Returns 0 and does nothing if no solve has been prepared since the last step finished.
        /// </summary>
        [BurstCompile]
        internal static int FinishPreparedStep(float deltaTime)
        {
            Simulation* simulation = Current->preparedSimulation;
            if (simulation == null)
            {
                return 0;
            }

            Current->preparedSimulation = null;
            Current->preparedInputs.Complete();
            Current->FinishStep(*simulation, deltaTime);
            return 1;
        }

        [BurstCompile]
        internal static void BuildForQueries(StepState* state)
        {
            NoteRunInMono();
            state->BuildForQueries();
        }

        /// <summary>Burst omits this method, so it runs only in Mono.</summary>
        [BurstDiscard]
        private static void NoteRunInMono()
        {
            BackendHost.NoteManagedStep();
        }
    }
}
