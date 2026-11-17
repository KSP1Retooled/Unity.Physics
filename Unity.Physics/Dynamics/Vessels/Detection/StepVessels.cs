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
    internal static class StepVessels
    {
        /// <summary>
        /// Chooses the vessels that take part in the step, as <see cref="VesselBuild.ChooseStepVessels"/> describes, and
        /// fills each part's built body. The main thread runs this job.
        /// </summary>
        [BurstCompile]
        internal unsafe struct ChooseStepVesselsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [ReadOnly] public NativeArray<int> BuiltIndices;
            public int DynamicCount;
            [ReadOnly] public NativeArray<VesselRecord> Vessels;
            [ReadOnly] public NativeArray<int> PartStoreBodies;
            [WriteOnly] public NativeArray<byte> Active;
            [WriteOnly] public NativeArray<int> PartBodies;
            public NativeList<VesselRecord> StepVessels;

            public void Execute()
            {
                const BodyFlags held = BodyFlags.Sleeping | BodyFlags.Static | BodyFlags.Kinematic;
                StepVessels.Clear();
                for (int v = 0; v < Vessels.Length; v++)
                {
                    VesselRecord vessel = Vessels[v];
                    bool takesPart = true;
                    for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount && takesPart; part++)
                    {
                        int storeBody = PartStoreBodies[part];
                        int built = BuiltIndices[storeBody];
                        takesPart = built >= 0 && built < DynamicCount && (World->bodies[storeBody].flags & held) == 0;
                    }

                    Active[v] = takesPart ? (byte)1 : (byte)0;
                    if (takesPart)
                    {
                        StepVessels.Add(vessel);
                    }
                }

                for (int part = 0; part < PartStoreBodies.Length; part++)
                {
                    PartBodies[part] = BuiltIndices[PartStoreBodies[part]];
                }
            }
        }
    }
}
