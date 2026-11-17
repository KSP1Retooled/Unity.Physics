using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Memory;

namespace Unity.Physics.Dynamics.Vessels.Detection
{
    internal static partial class VesselLoads
    {
        /// <summary>See <see cref="VesselBuild.ScheduleClearLoads"/>.</summary>
        [BurstCompile]
        internal unsafe struct ClearVesselLoadsJob : IJob
        {
            public NativeArray<VesselEdgeLoad> EdgeLoads;
            public NativeArray<VesselEdgeLoad> ClosureLoads;
            public NativeArray<bool> EdgeLocked;

            public void Execute()
            {
                UnsafeUtility.MemClear(EdgeLoads.GetUnsafePtr(), (long)EdgeLoads.Length * sizeof(VesselEdgeLoad));
                UnsafeUtility.MemClear(ClosureLoads.GetUnsafePtr(), (long)ClosureLoads.Length * sizeof(VesselEdgeLoad));
                UnsafeUtility.MemClear(EdgeLocked.GetUnsafePtr(), EdgeLocked.Length);
            }
        }
    }
}
