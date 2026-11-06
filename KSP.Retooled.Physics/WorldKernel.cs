using System;
using AOT;
using Unity.Burst;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>The plugin API's world commands, by the id World.cpp passes. Attach checks the count against the plugin's.</summary>
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

    [BurstCompile]
    internal static unsafe class WorldKernel
    {
        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(NativeStore.CommandFunction))]
        internal static void Run(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals)
        {
        }
    }
}
