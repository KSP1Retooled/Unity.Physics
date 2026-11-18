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
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Dynamics.Bodies;
using JointLoadJobs = Unity.Physics.Dynamics.Joint.Loads.JointLoads;

namespace Unity.Physics.Dynamics.Vessels.Detection
{
    internal static partial class VesselLoads
    {
        /// <summary>
        /// Convert vessel loads to joint loads, distribute group loads and settle break flags.
        /// Loads act on the leading joint's body0. Read solved motion poses so that this job can run alongside
        /// write-back.
        /// The impulse stream contributes the loads of the limits that ran as boundary rows.
        /// </summary>
        [BurstCompile]
        internal unsafe struct VesselLoadsJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            /// <summary>
            /// Joint load states indexed by the world list. Other joint jobs concurrently write disjoint entries.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public JointLoadState* States;

            [ReadOnly]
            public NativeArray<JointUnit> Units;

            [ReadOnly]
            public NativeArray<JointGroup> Groups;

            [ReadOnly]
            public NativeArray<JointGroupMember> Members;

            /// <summary>The solved motions, which give the poses of the bodies before they are written back.</summary>
            [ReadOnly]
            public NativeArray<MotionData> MotionDatas;

            /// <summary>The index of each store body in the built world, which is its motion's index for a moving body.</summary>
            [ReadOnly]
            public NativeArray<int> BuiltIndices;

            public float DeltaTime;

            /// <summary>
            /// For each part, the joint unit for its edge, or -1 for a vessel root. Closure units are stored separately in
            /// <see cref="ClosureUnits"/> .
            /// </summary>
            [ReadOnly]
            public NativeArray<int> PartUnits;

            [ReadOnly]
            public NativeArray<int> ClosureUnits;

            [ReadOnly]
            public NativeArray<byte> PartIsBody0;

            [ReadOnly]
            public NativeArray<int> UnitVessels;

            [ReadOnly]
            public NativeArray<byte> Active;

            [ReadOnly]
            public NativeArray<VesselEdgeLoad> EdgeLoads;

            [ReadOnly]
            public NativeArray<VesselEdgeLoad> ClosureLoads;

            // The jobs for the other joints write their loads at the same time. No joint is in both sets. This job uses the
            // loads only to share the load of a group among its members.
            [NativeDisableParallelForRestriction]
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<JointLoad> JointLoads;

            [ReadOnly]
            public NativeArray<JointLoad> GroupLoads;

            // As with the loads, the jobs for the other joints write the flags of their own joints at the same time.
            [NativeDisableParallelForRestriction]
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<byte> Overloaded;

            /// <summary>Processes part indices first, followed by closure indices.</summary>
            public void Execute(int index)
            {
                int parts = PartUnits.Length;
                int unit;
                VesselEdgeLoad load;
                float sign;
                if (index < parts)
                {
                    unit = PartUnits[index];
                    if (unit < 0)
                    {
                        return;
                    }

                    // The load acts on the child. If the child is body1, reverse the sign to obtain the load on body0.
                    load = EdgeLoads[index];
                    sign = PartIsBody0[index] != 0 ? 1.0f : -1.0f;
                }
                else
                {
                    unit = ClosureUnits[index - parts];
                    if (unit < 0)
                    {
                        return;
                    }

                    // The load acts on part B, which is body1. Reverse the sign to obtain the load on body0.
                    load = ClosureLoads[index - parts];
                    sign = -1.0f;
                }

                if (Active[UnitVessels[unit]] == 0)
                {
                    return;
                }

                // Active vessel bodies remain awake until later sleep jobs, so no per-joint wake check is needed.
                JointUnit jointUnit = Units[unit];
                JointLoad jointLoad = jointUnit.Group < 0 ? JointLoads[jointUnit.Joint] : GroupLoads[jointUnit.Group];
                jointLoad.Force += sign * (float3)load.Force;
                jointLoad.Torque += sign * (float3)load.Torque;
                jointLoad.IncludesAngularDrive = true;
                if (jointUnit.Group < 0)
                {
                    bool overloaded = JointLoadJobs.SettleJointLoadsJob.SettleAwake(World->joints[jointUnit.Joint], States + jointUnit.Joint, jointLoad, DeltaTime);
                    Overloaded[jointUnit.Joint] = overloaded ? (byte)1 : (byte)0;
                    return;
                }

                JointGroup group = Groups[jointUnit.Group];
                for (int i = 0; i < group.MemberCount; i++)
                {
                    JointLoads[Members[group.FirstMember + i].Joint] = default;
                }

                quaternion frame = LeaderFrame(World->joints[group.Leader], MotionDatas, BuiltIndices);
                JointLoadJobs.ShareGroupLoadsJob.Share(group, frame, jointLoad, Members, JointLoads);
                for (int i = 0; i < group.MemberCount; i++)
                {
                    int joint = Members[group.FirstMember + i].Joint;
                    bool overloaded = JointLoadJobs.SettleJointLoadsJob.SettleAwake(World->joints[joint], States + joint, JointLoads[joint], DeltaTime);
                    Overloaded[joint] = overloaded ? (byte)1 : (byte)0;
                }
            }

            /// <summary>
            /// Return the leader body0 rotation from solved motion, matching its pose after write-back.
            /// </summary>
            internal static quaternion LeaderFrame(JointData* leader, NativeArray<MotionData> motionDatas, NativeArray<int> builtIndices)
            {
                if (leader->body0 == null)
                {
                    return leader->rotation0;
                }

                BodyKinematics.SolvedPose(motionDatas[builtIndices[(int)leader->body0->index]], out _, out quaternion rotation);
                return math.mul(rotation, leader->rotation0);
            }
        }
    }
}
