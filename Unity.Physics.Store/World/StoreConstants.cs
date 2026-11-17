namespace Unity.Physics.Store.World
{
    internal static class StoreConstants
    {
        // The geometry types and flags of the plugin API.
        internal const uint GeometrySphere = 1;
        internal const uint GeometryCapsule = 3;
        internal const uint GeometryBox = 4;
        internal const uint GeometryConvexMesh = 5;
        internal const uint GeometryTriangleMesh = 6;
        internal const uint GeometryHeightField = 7;
        internal const uint ShapeFlagSceneQuery = 1 << 0;
        internal const uint ShapeFlagTrigger = 1 << 1;

        internal const uint QueryUseDynamicBodies = 1 << 0;
        internal const uint QueryUseStaticBodies = 1 << 1;
        internal const uint QueryUseTriggerShapes = 1 << 2;
        internal const uint QueryAllowBackfaces = 1 << 4;
        internal const uint QueryAllowMultipleMeshHits = 1 << 5;

        // QueryWorldBuild in Store.h.
        internal const int QueryWorldFailed = 0;
        internal const int QueryWorldBuilt = 1;
        internal const int QueryWorldDeferred = 2;

        internal const uint CombineAverage = 0;
        internal const uint CombineMultiply = 1;
        internal const uint CombineMinimum = 2;
        internal const uint CombineMaximum = 3;

        // Physics::ConstraintAxis.
        internal const int AxisAngularX = 0;
        internal const int AxisAngularY = 1;
        internal const int AxisAngularZ = 2;
        internal const int AxisAngularYZ = 3;
        internal const int AxisLinearX = 4;
        internal const int AxisDistance = 7;
        internal const int AxisSlerp = 8;
        internal const int JointAxisCount = 9;

        // Physics::JointDofLock.
        internal const uint DofLocked = 0;
        internal const uint DofLimited = 1;
        internal const uint DofFree = 2;

        // Physics::JointType.
        internal const uint JointTypeFixed = 0;
        internal const uint JointTypePrismatic = 1;
        internal const uint JointTypeHinge = 2;
        internal const uint JointTypeBallAndSocket = 3;
        internal const uint JointTypeDistance = 5;
        internal const uint JointTypeDof6 = 6;

        // Physics::JointFlags.
        internal const uint JointFlagCollisionEnabled = 1 << 2;

        internal const uint TriggerEventEnter = 1;
        internal const uint TriggerEventExit = 2;

        /// <summary>
        /// Trigger removal event type. Zero EngineObjectIds clear Unity's cached pair without sending a message.
        /// </summary>
        internal const uint TriggerEventShapeRemoved = 0;
    }
}
