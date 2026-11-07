using Unity.Profiling;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// A span for each way into the backend, and for each stage of a step. Native callers leave no samples of their own, so
    /// without these the time reads as self time of whichever Physics call got there first. Burst compiles these like any
    /// static marker, which takes a name it can read at compile time.
    /// </summary>
    internal static class BackendMarkers
    {
        internal static readonly ProfilerMarker Step = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step");
        internal static readonly ProfilerMarker BuildQueryWorld = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BuildQueryWorld");
        internal static readonly ProfilerMarker RefreshColliders = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.RefreshColliders");
        internal static readonly ProfilerMarker BuildMeshCollider = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BuildMeshCollider");
        internal static readonly ProfilerMarker Query = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Query");
        internal static readonly ProfilerMarker BodyCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BodyCommand");
        internal static readonly ProfilerMarker ShapeCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ShapeCommand");
        internal static readonly ProfilerMarker WorldCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand");
        internal static readonly ProfilerMarker JointCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.JointCommand");
        internal static readonly ProfilerMarker CookMesh = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.CookMesh");
        internal static readonly ProfilerMarker ValidateMesh = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ValidateMesh");
        internal static readonly ProfilerMarker DestroyWorld = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.DestroyWorld");

        internal static readonly ProfilerMarker StepApplyForces = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ApplyForces");
        internal static readonly ProfilerMarker StepSchedule = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Schedule");
        internal static readonly ProfilerMarker StepWaitForContacts = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForContacts");
        internal static readonly ProfilerMarker StepScheduleSolve = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ScheduleSolve");
        internal static readonly ProfilerMarker StepWaitForSolve = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForSolve");
        internal static readonly ProfilerMarker StepWriteBack = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WriteBack");
        internal static readonly ProfilerMarker StepReadJointForces = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadJointForces");
        internal static readonly ProfilerMarker StepReadImpulses = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadImpulses");
        internal static readonly ProfilerMarker StepPublishContacts = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.PublishContacts");
        internal static readonly ProfilerMarker StepSleep = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Sleep");
        internal static readonly ProfilerMarker StepPublishQueryWorld = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.PublishQueryWorld");
    }
}
