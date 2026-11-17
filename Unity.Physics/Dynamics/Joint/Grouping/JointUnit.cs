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
using Unity.Physics.Dynamics.Joint.Translation;

namespace Unity.Physics.Dynamics.Joint.Grouping
{
    /// <summary>
    /// The input for one work item in <see cref="JointTranslation.BuildJointsJob"/> : a single store joint, or the group led by that
    /// joint.
    /// </summary>
    internal struct JointUnit
    {
        public int Joint;

        /// <summary>The group that the joint leads, or -1 if the joint is built alone.</summary>
        public int Group;
    }
}
