using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Bodies;

namespace Unity.Physics.Dynamics.Sleep
{
    internal static partial class FallAsleep
    {
        /// <summary>
        /// Apply the PhysX island sleep rule after solving: all connected bodies must have zero wake counters.
        /// Static and kinematic bodies do not connect islands. Defer sleeping until contact reporting finishes.
        /// </summary>
        [BurstCompile]
        internal unsafe struct SleepJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            public NativeList<int> Parents;

            public NativeList<int> Falling;

            /// <summary>
            /// Has a non-zero value once a solved body's counter reaches zero during this step. This job clears the value
            /// for the next step.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public int* RunOut;

            public void Execute()
            {
                Falling.Clear();
                if (*RunOut == 0)
                {
                    return;
                }

                *RunOut = 0;
                int count = (int)World->bodyCount;
                NativeArray<int> parents = Parents.AsArray();
                for (uint i = 0; i < World->jointCount; i++)
                {
                    JointData* joint = World->joints[i];
                    if (joint->body0 != null && joint->body1 != null && (joint->state & JointConstraints.Inactive) == 0
                        && Solved(StoreMemory.Row(joint->body0)) && Solved(StoreMemory.Row(joint->body1)))
                    {
                        Union(parents, (int)joint->body0->index, (int)joint->body1->index);
                    }
                }

                var islandAwake = new NativeArray<bool>(count, Allocator.Temp);
                for (int i = 0; i < count; i++)
                {
                    BodyData* body = World->bodies + i;
                    if (Solved(body) && (body->flags & BodyFlags.Sleeping) == 0)
                    {
                        islandAwake[Find(parents, i)] |= body->sleep->WakeCounter > 0.0f;
                    }
                }

                for (int i = 0; i < count; i++)
                {
                    BodyData* body = World->bodies + i;
                    if (Solved(body) && (body->flags & BodyFlags.Sleeping) == 0 && !islandAwake[Find(parents, i)])
                    {
                        Falling.Add(i);
                    }
                }
            }

            internal static bool Solved(BodyData* body)
            {
                const BodyFlags immobile = BodyFlags.Static | BodyFlags.Kinematic | BodyFlags.ArticulationLink;
                return (body->flags & immobile) == 0 && (body->flags & BodyFlags.Simulated) != 0;
            }

            internal static int Find(NativeArray<int> parents, int index)
            {
                while (parents[index] != index)
                {
                    parents[index] = parents[parents[index]];
                    index = parents[index];
                }

                return index;
            }

            internal static void Union(NativeArray<int> parents, int a, int b)
            {
                parents[Find(parents, a)] = Find(parents, b);
            }
        }
    }
}
