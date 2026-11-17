namespace KSP.Retooled.Physics.Commands
{
    /// <summary>
    /// World command IDs shared with World.cpp. Attach verifies the count.
    /// Commands required before attach stay native.
    /// </summary>
    internal enum WorldCommand : uint
    {
        Invalid = 0,
        CreateWorld,
        DestroyWorld,
        SetGravity,
        GetGravity,
        SetBounceThreshold,
        GetBounceThreshold,
        GetBasicStats,
        SetScratchBufferChunkCount,
        Simulate,
        GetLastSimulationStepResults,
        ReleaseLastSimulationStepData,
        GetLastSimulationStats,
        GetUserData,
        CreateBody,
        AddActorsToScene,
        DestroyBody,
        CreateArticulation,
        DestroyArticulation,
        CreateJoint,
        DestroyJoint,
        GetArticulationCount,
        GetArticulations,
        GetAllShapesByLayerMask,
        Count,
    }
}
