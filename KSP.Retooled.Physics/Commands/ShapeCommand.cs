namespace KSP.Retooled.Physics.Commands
{
    /// <summary>
    /// Shape command IDs shared with Shape.cpp. Attach verifies the count; Mesh.cpp handles mesh commands.
    /// </summary>
    internal enum ShapeCommand : uint
    {
        Invalid = 0,
        CreateShape,
        DestroyShape,
        GetActorDescriptor,
        GetUserData,
        GetFlags,
        SetFlags,
        GetFilterData,
        SetFilterData,
        SetModifiableContacts,
        SetSupportedMessages,
        SetMaterial,
        GetLocalPose,
        GetPose,
        SetPose,
        SetContactOffset,
        GetWorldBounds,
        GetBoundsAtPose,
        ShouldIgnoreCollision,
        IgnoreCollision,
        CleanupIgnoredColliders,
        CookCollisionMesh,
        CookCollisionMeshStream,
        CreateCollisionMeshFromStream,
        CreateHeightField,
        UpdateHeightFieldRegion,
        DestroyHeightField,
        DestroyCollisionMesh,
        ExtractCollisionMeshData,
        GetGeometryType,
        SetGeometry,
        GetGeometry,
        RemapTriangleIndexToSourceIndex,
        Count,
    }
}
