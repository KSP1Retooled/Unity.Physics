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
    /// <summary>
    /// Joints between the same bodies represented as one unit, with per-axis stiffness relative to the direct
    /// solver cap.
    /// </summary>
    internal struct JointGroup
    {
        /// <summary>
        /// The entity version used for a built joint whose entity index identifies a group rather than a store joint.
        /// </summary>
        internal const int EntityVersion = 2;

        public int Leader;
        public int FirstMember;
        public int MemberCount;

        /// <summary>The ratio of the linear stiffness cap to the angular stiffness cap.</summary>
        public float LinearRatio;

        public float3 Stiffness;
    }
}
