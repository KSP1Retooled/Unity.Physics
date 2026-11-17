using KSP.Retooled.Physics.Diagnostics;
using Unity.Jobs;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Reports
{
    internal static unsafe partial class ContactReport
    {
        internal static void Publish(StepState* state, JobHandle contactsJoined)
        {
            using (BackendMarkers.StepPublishContacts.Auto())
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
