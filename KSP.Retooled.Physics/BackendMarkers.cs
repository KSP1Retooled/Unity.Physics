using Unity.Profiling;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Profiler markers for backend entry points and physics step stages.
    /// </summary>
    internal static class BackendMarkers
    {
        internal static readonly ProfilerMarker Step = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step");
        internal static readonly ProfilerMarker BuildQueryWorld = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BuildQueryWorld");
        internal static readonly ProfilerMarker RefreshColliders = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.RefreshColliders");
        internal static readonly ProfilerMarker Query = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Query");
        internal static readonly ProfilerMarker BodyCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BodyCommand");
        internal static readonly ProfilerMarker ShapeCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ShapeCommand");
        internal static readonly ProfilerMarker WorldCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand");
        internal static readonly ProfilerMarker JointCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.JointCommand");
        internal static readonly ProfilerMarker VehicleCommand = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.VehicleCommand");
        internal static readonly ProfilerMarker CookMesh = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.CookMesh");
        internal static readonly ProfilerMarker ValidateMesh = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ValidateMesh");
        internal static readonly ProfilerMarker DestroyWorld = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.DestroyWorld");

        // Time expensive commands individually. For simple field reads and writes, recording a marker costs too
        // much.
        internal static readonly ProfilerMarker CreateBody = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand.CreateBody");
        internal static readonly ProfilerMarker DestroyBody = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand.DestroyBody");
        internal static readonly ProfilerMarker CreateJoint = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand.CreateJoint");
        internal static readonly ProfilerMarker DestroyJoint = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand.DestroyJoint");
        internal static readonly ProfilerMarker ForgetBodyJoints = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand.ForgetBodyJoints");
        internal static readonly ProfilerMarker CountBodies = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand.CountBodies");
        internal static readonly ProfilerMarker ShapesByLayerMask = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.WorldCommand.ShapesByLayerMask");
        internal static readonly ProfilerMarker AttachShape = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BodyCommand.AttachShape");
        internal static readonly ProfilerMarker DetachShape = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BodyCommand.DetachShape");
        internal static readonly ProfilerMarker CreateShape = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ShapeCommand.CreateShape");
        internal static readonly ProfilerMarker DestroyShape = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ShapeCommand.DestroyShape");
        internal static readonly ProfilerMarker SetGeometry = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ShapeCommand.SetGeometry");
        internal static readonly ProfilerMarker CreateHeightField = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ShapeCommand.CreateHeightField");
        internal static readonly ProfilerMarker UpdateHeightField = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.ShapeCommand.UpdateHeightField");
        internal static readonly ProfilerMarker AdoptBody = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BodyEntities.Adopt");
        internal static readonly ProfilerMarker ReleaseBody = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BodyEntities.Release");
        internal static readonly ProfilerMarker HandOverRow = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.BodyEntities.HandOver");
        internal static readonly ProfilerMarker StepApplyForces = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ApplyForces");
        internal static readonly ProfilerMarker StepSchedule = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Schedule");
        internal static readonly ProfilerMarker BuildPartition = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.Partition");
        internal static readonly ProfilerMarker BuildJoints = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.Joints");
        internal static readonly ProfilerMarker BuildReset = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.Reset");
        internal static readonly ProfilerMarker BuildScheduleBodies = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.ScheduleBodies");
        internal static readonly ProfilerMarker BuildScheduleBroadphase = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.ScheduleBroadphase");
        internal static readonly ProfilerMarker ShiftStaticTree = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.ShiftStaticTree");
        internal static readonly ProfilerMarker StepWaitForSolve = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForSolve");
        internal static readonly ProfilerMarker StepWriteBack = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WriteBack");
        internal static readonly ProfilerMarker StepReadJointForces = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadJointForces");
        internal static readonly ProfilerMarker StepWaitForTree = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForTree");
        internal static readonly ProfilerMarker StepReadImpulses = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadImpulses");
        internal static readonly ProfilerMarker StepPublishContacts = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.PublishContacts");
        internal static readonly ProfilerMarker StepSleep = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Sleep");
        internal static readonly ProfilerMarker StepPublishQueryWorld = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.PublishQueryWorld");
    }
}
