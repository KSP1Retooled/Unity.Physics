using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Joint.Grouping;

namespace Unity.Physics.Dynamics.Joint.Loads
{
    internal static unsafe partial class JointLoads
    {
        /// <summary>
        /// Distribute group forces equally and torques by member stiffness, including forces from offset anchors.
        /// </summary>
        [BurstCompile]
        internal unsafe struct ShareGroupLoadsJob : IJobParallelForDefer
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [ReadOnly]
            public NativeArray<JointGroup> Groups;

            [ReadOnly]
            public NativeArray<JointGroupMember> Members;

            [ReadOnly]
            public NativeArray<JointLoad> GroupLoads;

            [NativeDisableParallelForRestriction]
            public NativeArray<JointLoad> JointLoads;

            /// <summary>When <see cref="Listed"/> is true, the job handles only the groups in this list.</summary>
            [ReadOnly]
            public NativeArray<int> ListedGroups;

            public bool Listed;

            public void Execute(int item)
            {
                int index = Listed ? ListedGroups[item] : item;
                JointGroup group = Groups[index];
                quaternion frame = JointConstraints.WorldRotation0(World->joints[group.Leader]);
                Share(group, frame, GroupLoads[index], Members, JointLoads);
            }

            /// <summary>
            /// Add a group member's load share in the leader's body0 joint frame.
            /// </summary>
            internal static void Share(in JointGroup group, quaternion frame, in JointLoad carried, NativeArray<JointGroupMember> members,
                NativeArray<JointLoad> jointLoads)
            {
                float3 share = math.rotate(math.conjugate(frame), carried.Force - carried.LimitForce) / group.MemberCount;
                float3 torque = math.rotate(math.conjugate(frame), carried.Torque - carried.LimitTorque);
                float3 turn = RotationForTorque(group, torque);
                // Members of a group share the same stop. Its reaction acts directly at the stop,
                // without going through the drive springs or a couple at the offset anchors.
                float3 limitForce = carried.LimitForce / group.MemberCount;
                float3 limitTorque = carried.LimitTorque / group.MemberCount;
                for (int i = 0; i < group.MemberCount; i++)
                {
                    JointGroupMember member = members[group.FirstMember + i];
                    JointLoad load = jointLoads[member.Joint];
                    load.Force += math.rotate(frame, share + group.LinearRatio * math.cross(turn, member.Offset));
                    load.Torque += math.rotate(frame, member.AngularStiffness * turn);
                    load.Force += limitForce;
                    load.Torque += limitTorque;
                    load.LimitForce += limitForce;
                    load.LimitTorque += limitTorque;
                    load.IncludesAngularDrive = true;
                    jointLoads[member.Joint] = load;
                }
            }

            private static float3 RotationForTorque(in JointGroup group, float3 torque)
            {
                return math.select(torque / group.Stiffness, float3.zero, group.Stiffness <= 0.0f);
            }
        }
    }
}
