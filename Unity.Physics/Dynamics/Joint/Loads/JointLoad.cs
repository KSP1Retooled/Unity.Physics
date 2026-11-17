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

namespace Unity.Physics.Dynamics.Joint.Loads
{
    /// <summary>
    /// Force and torque collected this step for a single joint or a merged joint group.
    /// </summary>
    internal struct JointLoad
    {
        public float3 Force;
        public float3 Torque;
    }
}
