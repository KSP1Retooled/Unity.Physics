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
using Unity.Physics.Dynamics.Joint.Grouping;

namespace Unity.Physics.Dynamics.Vessels.Detection
{
    internal static partial class VesselLoads
    {
        /// <summary>
        /// List joints and groups outside active vessels for impulse-event load processing. VesselLoadsJob handles the
        /// rest.
        /// </summary>
        [BurstCompile]
        internal struct ListOtherJointsJob : IJob
        {
            [ReadOnly]
            public NativeArray<JointUnit> Units;

            [ReadOnly]
            public NativeArray<JointGroup> Groups;

            [ReadOnly]
            public NativeArray<JointGroupMember> Members;

            [ReadOnly]
            public NativeArray<int> UnitVessels;

            [ReadOnly]
            public NativeArray<byte> Active;

            public NativeList<int> OtherJoints;
            public NativeList<int> OtherGroups;

            public void Execute()
            {
                OtherJoints.Clear();
                OtherGroups.Clear();
                for (int u = 0; u < Units.Length; u++)
                {
                    if (UnitVessels[u] >= 0 && Active[UnitVessels[u]] != 0)
                    {
                        continue;
                    }

                    JointUnit unit = Units[u];
                    if (unit.Group < 0)
                    {
                        OtherJoints.Add(unit.Joint);
                        continue;
                    }

                    OtherGroups.Add(unit.Group);
                    JointGroup group = Groups[unit.Group];
                    for (int i = 0; i < group.MemberCount; i++)
                    {
                        OtherJoints.Add(Members[group.FirstMember + i].Joint);
                    }
                }
            }
        }
    }
}
