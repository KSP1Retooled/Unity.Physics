using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// What a store joint, or a group the step built as one joint, carried this step, gathered before it is written back.
    /// </summary>
    internal struct JointLoad
    {
        public float3 Force;
        public float3 Torque;

        /// <summary>Whether the joint's smoothed load passed its break limits, for <see cref="BreakJointsJob"/>.</summary>
        public bool Overloaded;
    }

    /// <summary>
    /// The jobs that read what the joints' constraints applied from the step's impulse events, which arrive once per
    /// constraint with the impulse of the whole step, then break the joints over their limits. A constraint's impulse acts
    /// on body A, the connected body, and Unity reports what acts on the joint's owner. Drives count for neither, as under
    /// PhysX, so they raise no events. A joint whose bodies sleep isn't solved, and it keeps reporting the last forces it
    /// had, as under PhysX. A broken joint stops constraining from the next step and waits in the world's break list for
    /// whatever publishes the step's results.
    ///
    /// Most of the work is turning each event into the world, which reads the joint's bodies and runs in parallel. A joint's
    /// events are still summed one after another in the order the solve wrote them, so the forces come out the same however
    /// the work is spread.
    /// </summary>
    internal static unsafe class JointForceJobs
    {
        internal static bool IsAsleep(NativeStore.Body* body)
        {
            const NativeStore.BodyFlags resting = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Sleeping;
            return body == null || (StoreOps.Hot(body)->flags & resting) != 0;
        }

        internal static bool IsAsleep(NativeStore.JointData* joint)
        {
            return IsAsleep(joint->body0) && IsAsleep(joint->body1);
        }
    }

    /// <summary>Starts each joint's load from nothing, or from what it last carried if it isn't solved.</summary>
    [BurstCompile]
    internal unsafe struct SeedJointLoadsJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        public NativeArray<JointLoad> Loads;

        public void Execute(int index)
        {
            NativeStore.JointData* joint = World->joints[index];
            bool asleep = JointForceJobs.IsAsleep(joint);
            Loads[index] = new JointLoad
            {
                Force = asleep ? joint->force : float3.zero,
                Torque = asleep ? joint->torque : float3.zero,
            };
        }
    }

    /// <summary>
    /// Turns each impulse event, one bucket of the event stream per index, into the force or torque it means in the world,
    /// written over the event's impulse. Nothing reads the events after <see cref="SumJointImpulsesJob"/>.
    /// </summary>
    [BurstCompile]
    internal unsafe struct RotateJointImpulsesJob : IJobParallelForDefer
    {
        public NativeStream.Reader Events;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

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
                NativeStore.JointData* joint = World->joints[grouped ? Groups[impulse->JointEntity.Index].Leader : impulse->JointEntity.Index];
                quaternion frame = impulse->Type == ConstraintType.Linear ? JointConstraints.WorldRotation0(joint) : JointConstraints.WorldRotation1(joint);
                impulse->Impulse = math.rotate(frame, impulse->Impulse) * scale;
            }

            reader.EndForEachIndex();
        }
    }

    /// <summary>
    /// Adds the turned events into the loads of their joints and groups, in the order the solve wrote them. What acts on
    /// the joint's owner is the opposite of what acts on body A.
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

    /// <summary>
    /// Splits what a group's joint carried among its members as their own joints would have carried it at rest. The group
    /// turns by its torque over its stiffness; each member holds its own share of that turn as torque, and its linear rows
    /// hold the turn at their offset as force. Every member takes an equal share of the force. A joint is a member of one
    /// group at most, so the groups split their loads in parallel.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ShareGroupLoadsJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<JointGroup> Groups;

        [ReadOnly]
        public NativeArray<JointGroupMember> Members;

        [ReadOnly]
        public NativeArray<JointLoad> GroupLoads;

        [NativeDisableParallelForRestriction]
        public NativeArray<JointLoad> JointLoads;

        public void Execute(int index)
        {
            JointGroup group = Groups[index];
            JointLoad carried = GroupLoads[index];
            NativeStore.JointData* leader = World->joints[group.Leader];
            quaternion frame = JointConstraints.WorldRotation0(leader);
            float3 share = math.rotate(math.conjugate(frame), carried.Force) / group.MemberCount;
            float3 turn = math.select(math.rotate(math.conjugate(frame), carried.Torque) / group.Stiffness, float3.zero, group.Stiffness <= 0.0f);
            for (int i = 0; i < group.MemberCount; i++)
            {
                JointGroupMember member = Members[group.FirstMember + i];
                JointLoad load = JointLoads[member.Joint];
                load.Force += math.rotate(frame, share + group.LinearRatio * math.cross(turn, member.Offset));
                load.Torque += math.rotate(frame, member.AngularStiffness * turn);
                JointLoads[member.Joint] = load;
            }
        }
    }

    /// <summary>
    /// Gives each joint its load and follows it with the load that counts against the joint's break limits, marking the
    /// joints that pass them.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SettleJointLoadsJob : IJobParallelFor
    {
        /// <summary>
        /// How long, in seconds, a load must last before it counts in full against a joint's break force. The direct solve
        /// holds a craft rigidly, so a part landing on the ground stops everything above it within one step, and the
        /// joints carry that whole stop at once. PhysX spreads the same stop over several steps, so it never breaks them.
        /// </summary>
        private const float BreakingTimeConstant = 0.1f;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        public NativeArray<JointLoad> Loads;

        public float DeltaTime;

        public void Execute(int index)
        {
            NativeStore.JointData* joint = World->joints[index];
            JointLoad load = Loads[index];
            joint->force = load.Force;
            joint->torque = load.Torque;
            if (JointForceJobs.IsAsleep(joint))
            {
                return;
            }

            float follow = DeltaTime / (BreakingTimeConstant + DeltaTime);
            joint->breakingForce += (math.length(joint->force) - joint->breakingForce) * follow;
            joint->breakingTorque += (math.length(joint->torque) - joint->breakingTorque) * follow;
            bool overloaded = joint->breakingForce > joint->breakForce || joint->breakingTorque > joint->breakTorque;
            load.Overloaded = (joint->state & JointConstraints.Inactive) == 0 && overloaded;
            Loads[index] = load;
        }
    }

    /// <summary>Breaks the overloaded joints, listing them in the world's break list in store order.</summary>
    [BurstCompile]
    internal unsafe struct BreakJointsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        [ReadOnly]
        public NativeArray<JointLoad> Loads;

        public uint EngineObjectIdOffset;

        public void Execute()
        {
            World->jointBreakCount = 0;
            for (int i = 0; i < Loads.Length; i++)
            {
                if (Loads[i].Overloaded)
                {
                    Break(World->joints[i]);
                }
            }
        }

        private void Break(NativeStore.JointData* joint)
        {
            joint->state |= NativeStore.JointState.Broken;
            if (World->jointBreakCount == World->jointBreakCapacity)
            {
                World->jointBreakCapacity = StoreOps.Grown(World->jointBreakCapacity);
                World->jointBreaks = (NativeStore.JointBreak*)StoreOps.Reallocate(Globals, World->jointBreaks, World->jointBreakCapacity * (ulong)sizeof(NativeStore.JointBreak));
            }

            World->jointBreaks[World->jointBreakCount++] = new NativeStore.JointBreak
            {
                engineObjectId = joint->userData != null ? *(ulong*)((byte*)joint->userData + EngineObjectIdOffset) : 0,
                linearMagnitude = joint->breakingForce,
                angularMagnitude = joint->breakingTorque,
            };
        }
    }
}
