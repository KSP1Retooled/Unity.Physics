using Unity.Profiling;

namespace KSP.Retooled.Physics
{
    public static unsafe partial class BackendHost
    {
        private static readonly ProfilerMarker StepMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Backend.Step");
        private static readonly ProfilerMarker BuildQueryWorldMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Backend.BuildQueryWorld");
        private static readonly ProfilerMarker DestroyWorldMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Backend.DestroyWorld");
    }
}
