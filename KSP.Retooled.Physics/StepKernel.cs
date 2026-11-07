using Unity.Burst;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Burst's way into <see cref="StepState"/>. As with the other kernels, these run in Mono until the async compile
    /// delivers them.
    /// </summary>
    [BurstCompile]
    internal static unsafe class StepKernel
    {
        [BurstCompile]
        internal static void Step(StepState* state, float deltaTime)
        {
            NoteRunInMono();
            state->Step(deltaTime);
        }

        [BurstCompile]
        internal static void BuildForQueries(StepState* state)
        {
            NoteRunInMono();
            state->BuildForQueries();
        }

        /// <summary>Burst drops this, so it only runs in the Mono form.</summary>
        [BurstDiscard]
        private static void NoteRunInMono()
        {
            BackendHost.NoteManagedStep();
        }
    }
}
