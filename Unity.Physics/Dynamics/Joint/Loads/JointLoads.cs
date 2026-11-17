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
        internal static bool IsAsleep(Body* body)
        {
            const BodyFlags resting = BodyFlags.Static | BodyFlags.Kinematic | BodyFlags.Sleeping;
            return body == null || (StoreMemory.Row(body)->flags & resting) != 0;
        }

        internal static bool IsAsleep(JointData* joint)
        {
            return IsAsleep(joint->body0) && IsAsleep(joint->body1);
        }

        /// <summary>
        /// Initialize joint loads to zero, preserving previous loads for unsolved resting bodies.
        /// If Listed is true, process only Joints.
        /// </summary>
        [BurstCompile]
        internal unsafe struct SeedJointLoadsJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            /// <summary>The load state of each joint, at the joint's index in the world's list. See <see cref="JointLoadStates"/>.</summary>
            [NativeDisableUnsafePtrRestriction]
            public JointLoadState* States;

            [NativeDisableParallelForRestriction]
            public NativeArray<JointLoad> Loads;

            [ReadOnly]
            public NativeArray<int> Joints;

            public bool Listed;

            public void Execute(int item)
            {
                int index = Listed ? Joints[item] : item;
                bool asleep = JointLoads.IsAsleep(World->joints[index]);
                Loads[index] = new JointLoad
                {
                    Force = asleep ? States[index].Force : float3.zero,
                    Torque = asleep ? States[index].Torque : float3.zero,
                };
            }
        }

        /// <summary>
        /// Convert impulse events to world-space forces and torques in place, one stream bucket per work item.
        /// </summary>
        [BurstCompile]
        internal unsafe struct RotateJointImpulsesJob : IJobParallelForDefer
        {
            public NativeStream.Reader Events;

            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [ReadOnly]
            public NativeArray<JointGroup> Groups;

            public float DeltaTime;

            public void Execute(int bucket)
            {
                float scale = 1.0f / DeltaTime;
                NativeStream.Reader reader = Events;
                int count = reader.BeginForEachIndex(bucket);
                for (int i = 0; i < count; i++)
                {
                    ImpulseEventData* impulse = (ImpulseEventData*)reader.ReadUnsafePtr(sizeof(ImpulseEventData));
                    bool grouped = impulse->JointEntity.Version == JointGroup.EntityVersion;
                    JointData* joint = World->joints[grouped ? Groups[impulse->JointEntity.Index].Leader : impulse->JointEntity.Index];
                    quaternion frame = impulse->Type == ConstraintType.Linear ? JointConstraints.WorldRotation0(joint) : JointConstraints.WorldRotation1(joint);
                    impulse->Impulse = math.rotate(frame, impulse->Impulse) * scale;
                }

                reader.EndForEachIndex();
            }
        }

        /// <summary>
        /// Sum converted impulse events in solver order. Negate body A's load to report the owner's load.
        /// </summary>
        [BurstCompile]
        internal struct SumJointImpulsesJob : IJob
        {
            public NativeStream.Reader Events;

            public NativeArray<JointLoad> JointLoads;

            public NativeArray<JointLoad> GroupLoads;

            public void Execute()
            {
                for (int bucket = 0; bucket < Events.ForEachCount; bucket++)
                {
                    int count = Events.BeginForEachIndex(bucket);
                    for (int i = 0; i < count; i++)
                    {
                        ImpulseEventData impulse = Events.Read<ImpulseEventData>();
                        bool grouped = impulse.JointEntity.Version == JointGroup.EntityVersion;
                        NativeArray<JointLoad> loads = grouped ? GroupLoads : JointLoads;
                        JointLoad load = loads[impulse.JointEntity.Index];
                        if (impulse.Type == ConstraintType.Linear)
                        {
                            load.Force -= impulse.Impulse;
                        }
                        else
                        {
                            load.Torque -= impulse.Impulse;
                        }

                        loads[impulse.JointEntity.Index] = load;
                    }

                    Events.EndForEachIndex();
                }
            }
        }
    }
}
