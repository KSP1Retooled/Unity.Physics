using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Physics.Math;
using SolverJoint = Unity.Physics.Joint;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Memory;

namespace Unity.Physics.Dynamics.Joint.Grouping
{
    internal struct JointGroupMember
    {
        public int Joint;

        /// <summary>
        /// The offset from the group's anchor to the member's anchor on body 0, expressed in the joint frame.
        /// </summary>
        public float3 Offset;

        public float3 AngularStiffness;
    }
}
