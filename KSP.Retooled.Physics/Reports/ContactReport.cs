using Unity.Profiling;
using Unity.Jobs;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Reports
{
    internal static unsafe partial class ContactReport
    {
        private static readonly ProfilerMarker StepPublishContactsMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Backend.PublishContacts");

        internal static JobHandle Schedule(StoreStepper* state, JobHandle dependency)
        {
            using (StepPublishContactsMarker.Auto())
            {
                return new PublishContactsJob
                {
                    World = state->Store,
                    Globals = state->Globals,
                    Records = state->Manifolds,
                    Points = state->ManifoldPoints,
                    Touching = state->Touching,
                    RemovedTriggerPairs = state->RemovedTriggerPairs,
                    Step = state->NextContactStep(),
                }.Schedule(dependency);
            }
        }
    }
}
