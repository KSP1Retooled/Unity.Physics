using Unity.Profiling;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// A span for each way into the backend. Native callers leave no samples of their own, so without these the time reads as
    /// self time of whichever Physics call got there first. Burst compiles these like any static marker.
    /// </summary>
    internal static class BackendMarkers
    {
        internal static readonly ProfilerMarker Step = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.Step");
        internal static readonly ProfilerMarker BuildQueryWorld = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.BuildQueryWorld");
        internal static readonly ProfilerMarker RefreshColliders = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.RefreshColliders");
        internal static readonly ProfilerMarker BuildMeshCollider = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.BuildMeshCollider");
        internal static readonly ProfilerMarker Query = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.Query");
        internal static readonly ProfilerMarker BodyCommand = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.BodyCommand");
        internal static readonly ProfilerMarker ShapeCommand = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.ShapeCommand");
        internal static readonly ProfilerMarker WorldCommand = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.WorldCommand");
        internal static readonly ProfilerMarker JointCommand = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.JointCommand");
        internal static readonly ProfilerMarker CookMesh = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.CookMesh");
        internal static readonly ProfilerMarker ValidateMesh = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.ValidateMesh");
        internal static readonly ProfilerMarker DestroyWorld = new ProfilerMarker(ProfilerCategory.Physics, "RetooledPhysics.DestroyWorld");
    }
}
