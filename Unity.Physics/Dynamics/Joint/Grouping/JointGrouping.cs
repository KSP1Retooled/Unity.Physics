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
    internal static partial class JointGrouping
    {
        /// <summary>
        /// Compute joint output offsets from per-unit counts and resize the world's joint array.
        /// </summary>
        [BurstCompile]
        internal struct LayOutJointsJob : IJob
        {
            public NativeList<int> Written;
            public NativeList<SolverJoint> Output;

            public void Execute()
            {
                int next = 0;
                for (int i = 0; i < Written.Length; i++)
                {
                    int count = Written[i];
                    Written[i] = next;
                    next += count;
                }

                Output.ResizeUninitialized(next);
            }
        }

        /// <summary>
        /// Copies the joints written for each unit by <see cref="JointTranslation.BuildJointsJob"/> to the range assigned by
        /// <see cref="LayOutJointsJob"/> .
        /// </summary>
        [BurstCompile]
        internal unsafe struct CopyJointsJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeStream.Reader Input;

            [ReadOnly]
            public NativeArray<int> FirstJoints;

            [NativeDisableParallelForRestriction]
            public NativeArray<SolverJoint> Output;

            public void Execute(int index)
            {
                NativeStream.Reader input = Input;
                int count = input.BeginForEachIndex(index);
                SolverJoint* output = (SolverJoint*)Output.GetUnsafePtr() + FirstJoints[index];
                for (int j = 0; j < count; j++)
                {
                    output[j] = input.Read<SolverJoint>();
                }

                input.EndForEachIndex();
            }
        }
    }
}
