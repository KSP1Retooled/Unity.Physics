using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The force and torque on a joint during this step, collected before they are written back to the store. The load
    /// may be for a single store joint or for a group that was built as one joint.
    /// </summary>
    internal struct JointLoad
    {
        public float3 Force;
        public float3 Torque;
    }

    /// <summary>
    /// Jobs that calculate joint loads from the impulse events for this step, then break joints whose loads exceed
    /// their limits. Each constraint produces one event containing its impulse for the whole step. That impulse acts on
    /// body A, the connected body. Unity reports the load on the joint's owner instead. As in PhysX, drives contribute
    /// neither to reported loads nor to breaking, so they produce no events. Joints whose bodies are asleep are not
    /// solved and continue to report their previous forces, as in PhysX. A broken joint stops constraining its bodies
    /// from the next step. Its break record remains in the world's break list until the results of the step are
    /// published.
    ///
    /// Most of the work involves converting each event into world space using the joint's bodies. That conversion runs
    /// in parallel. The converted events for a joint are then summed in the order that the solver wrote them. This
    /// gives the same forces regardless of how the parallel work is scheduled.
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

    /// <summary>
    /// Initializes each joint's load to zero, or to its previous load if both bodies are held in place and the joint is
    /// not solved. When <see cref="Listed"/> is true, the job handles only the joints in <see cref="Joints"/>.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SeedJointLoadsJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [NativeDisableParallelForRestriction]
        public NativeArray<JointLoad> Loads;

        [ReadOnly]
        public NativeArray<int> Joints;

        public bool Listed;

        public void Execute(int item)
        {
            int index = Listed ? Joints[item] : item;
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
    /// Converts each impulse event into a force or torque in world space, processing one bucket of the event stream per
    /// work item. The result replaces the impulse in the event. No code reads the events after
    /// <see cref="SumJointImpulsesJob"/> .
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
    /// Adds the converted events to the loads for their joints and groups, in the order that the solver wrote them. The
    /// load on the joint's owner has the opposite sign to the load on body A.
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
    /// Distributes a group's load among its members according to their resistance to displacement at rest. Dividing the
    /// group's torque by its stiffness gives an equivalent rotation. Each member contributes torque according to its
    /// angular stiffness. Its linear constraints also contribute force because its anchor is offset from the group's
    /// mean anchor. The group's net force is divided equally among the members. Each joint belongs to at most one
    /// group, so the loads for different groups can be distributed in parallel.
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

        /// <summary>When <see cref="Listed"/> is true, the job handles only the groups in this list.</summary>
        [ReadOnly]
        public NativeArray<int> ListedGroups;

        public bool Listed;

        public void Execute(int item)
        {
            int index = Listed ? ListedGroups[item] : item;
            Share(World, Groups[index], GroupLoads[index], Members, JointLoads);
        }

        /// <summary>Adds each member's share of the group's load to the member's load.</summary>
        internal static void Share(NativeStore.WorldData* world, in JointGroup group, in JointLoad carried, NativeArray<JointGroupMember> members,
            NativeArray<JointLoad> jointLoads)
        {
            NativeStore.JointData* leader = world->joints[group.Leader];
            quaternion frame = JointConstraints.WorldRotation0(leader);
            float3 share = math.rotate(math.conjugate(frame), carried.Force) / group.MemberCount;
            float3 turn = math.select(math.rotate(math.conjugate(frame), carried.Torque) / group.Stiffness, float3.zero, group.Stiffness <= 0.0f);
            for (int i = 0; i < group.MemberCount; i++)
            {
                JointGroupMember member = members[group.FirstMember + i];
                JointLoad load = jointLoads[member.Joint];
                load.Force += math.rotate(frame, share + group.LinearRatio * math.cross(turn, member.Offset));
                load.Torque += math.rotate(frame, member.AngularStiffness * turn);
                jointLoads[member.Joint] = load;
            }
        }
    }

    /// <summary>
    /// Writes each joint's load to the store, then smooths the force and torque magnitudes used for breaking. Marks
    /// active joints whose smoothed loads exceed their break limits in <see cref="Overloaded"/> .
    /// </summary>
    [BurstCompile]
    internal unsafe struct SettleJointLoadsJob : IJobParallelFor
    {
        /// <summary>
        /// The time constant, in seconds, for smoothing both force and torque before comparing them with the break
        /// limits. A constant load approaches its full contribution over successive steps. The direct solver holds a
        /// craft rigidly, so one part landing can stop all of the parts above it within a single step. Their joints
        /// carry the force of that stop at once. PhysX spreads the stop over several steps, which reduces the peak load
        /// and makes breakage less likely.
        /// </summary>
        private const float BreakingTimeConstant = 0.1f;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<JointLoad> Loads;

        /// <summary>For each joint, 1 if the joint is active and its smoothed load exceeds its break limits, or 0.</summary>
        [NativeDisableParallelForRestriction]
        public NativeArray<byte> Overloaded;

        /// <summary>When <see cref="Listed"/> is true, the job handles only the joints in this list.</summary>
        [ReadOnly]
        public NativeArray<int> Joints;

        public bool Listed;

        public float DeltaTime;

        public void Execute(int item)
        {
            int index = Listed ? Joints[item] : item;
            Overloaded[index] = Settle(World->joints[index], Loads[index], DeltaTime) ? (byte)1 : (byte)0;
        }

        /// <summary>
        /// Writes one joint's load to the store and smooths it, as described for this job. Returns whether the joint is
        /// active and its smoothed load exceeds its break limits.
        /// </summary>
        internal static bool Settle(NativeStore.JointData* joint, in JointLoad load, float deltaTime)
        {
            if (JointForceJobs.IsAsleep(joint))
            {
                joint->force = load.Force;
                joint->torque = load.Torque;
                return false;
            }

            return SettleAwake(joint, load, deltaTime);
        }

        /// <summary>
        /// Settles the load of a joint that the caller knows has at least one moving body that is awake, as
        /// <see cref="Settle"/> does, and returns the same result. The caller saves the reads of the joint's two
        /// bodies, which are often not in the cache.
        /// </summary>
        internal static bool SettleAwake(NativeStore.JointData* joint, in JointLoad load, float deltaTime)
        {
            joint->force = load.Force;
            joint->torque = load.Torque;
            float follow = deltaTime / (BreakingTimeConstant + deltaTime);
            joint->breakingForce += (math.length(joint->force) - joint->breakingForce) * follow;
            joint->breakingTorque += (math.length(joint->torque) - joint->breakingTorque) * follow;
            bool overloaded = joint->breakingForce > joint->breakForce || joint->breakingTorque > joint->breakTorque;
            return (joint->state & JointConstraints.Inactive) == 0 && overloaded;
        }
    }

    /// <summary>
    /// Breaks overloaded joints and adds their break records to the world's break list in store order. A joint rarely
    /// breaks, so the job reads the flags eight at a time and skips each group of eight that is all zero.
    /// </summary>
    [BurstCompile]
    internal unsafe struct BreakJointsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        /// <summary>For each joint, 1 if the joint breaks in this step, or 0.</summary>
        [ReadOnly]
        public NativeArray<byte> Overloaded;

        public uint EngineObjectIdOffset;

        public void Execute()
        {
            World->jointBreakCount = 0;
            byte* flags = (byte*)Overloaded.GetUnsafeReadOnlyPtr();
            int count = Overloaded.Length;
            int i = 0;
            for (; i + sizeof(ulong) <= count; i += sizeof(ulong))
            {
                if (UnsafeUtility.ReadArrayElement<ulong>(flags + i, 0) == 0)
                {
                    continue;
                }

                for (int j = i; j < i + sizeof(ulong); j++)
                {
                    BreakIfOverloaded(flags, j);
                }
            }

            for (; i < count; i++)
            {
                BreakIfOverloaded(flags, i);
            }
        }

        private void BreakIfOverloaded(byte* flags, int joint)
        {
            if (flags[joint] != 0)
            {
                Break(World->joints[joint]);
            }
        }

        private void Break(NativeStore.JointData* joint)
        {
            joint->state |= NativeStore.JointState.Broken;
            JointTopology.Changed();
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
