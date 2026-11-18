using Unity.Profiling;
using Unity.Jobs;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Reports
{
    internal static unsafe partial class ContactReport
    {
        private static readonly ProfilerMarker StepPublishContactsMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Backend.PublishContacts");

        internal static void Publish(StoreStepper* state, JobHandle contactsJoined)
        {
            using (StepPublishContactsMarker.Auto())
            {
                // Sort the manifolds after the sleep contact job has read them.
                contactsJoined.Complete();
                new PublishContactsJob
                {
                    World = state->Store,
                    Globals = state->Globals,
                    Records = state->Manifolds,
                    Points = state->ManifoldPoints,
                    Touching = state->Touching,
                    RemovedTriggerPairs = state->RemovedTriggerPairs,
                    Step = state->NextContactStep(),
                }.Run();
            }
        }
    }
}
