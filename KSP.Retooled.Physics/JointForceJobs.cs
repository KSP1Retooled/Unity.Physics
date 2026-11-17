using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Force and torque collected this step for a single joint or a merged joint group.
    /// </summary>
    internal struct JointLoad
    {
        public float3 Force;
        public float3 Torque;
    }

    /// <summary>
    /// Current joint loads and smoothed magnitudes, stored densely alongside the world's joint list.
    /// Copy loads back to native records when StepState is disposed. Commands update cached break limits and
    /// inactivity.
    /// </summary>
    internal struct JointLoadState
    {
        // The force and torque that the joint applied during the last step, as in NativeStore.JointData.
        public float3 Force;

        // The smoothed magnitude of the force, used to decide whether the joint breaks.
        public float BreakingForce;
        public float3 Torque;
        public float BreakingTorque;
        public float BreakForce;
        public float BreakTorque;

        // The joint's serial number. A command that reads the array checks that the slot belongs to its joint.
        public uint Serial;

        // 1 when the joint is broken or has lost a body (JointConstraints.Inactive), otherwise 0.
        public uint Inactive;

        internal static unsafe JointLoadState Of(NativeStore.JointData* joint)
        {
            return new JointLoadState
            {
                Force = joint->force,
                BreakingForce = joint->breakingForce,
                Torque = joint->torque,
                BreakingTorque = joint->breakingTorque,
                BreakForce = joint->breakForce,
                BreakTorque = joint->breakTorque,
                Serial = joint->serial,
                Inactive = (joint->state & JointConstraints.Inactive) != 0 ? 1u : 0u,
            };
        }
    }

    /// <summary>
    /// Map worlds to their dense joint load arrays. Main-thread commands keep slots aligned with the joint list.
    /// </summary>
    internal static unsafe class JointLoadStates
    {
        private const int Capacity = 32;

        private readonly struct Key { }

        private struct Table
        {
            public int Count;
            public fixed long Worlds[Capacity];
            public fixed long States[Capacity];
        }

        private static readonly SharedStatic<Table> Shared = SharedStatic<Table>.GetOrCreate<Key>();

        /// <summary>
        /// Fills a world's array from its joint records and adds it to the table. Until now, the records held the
        /// loads.
        /// </summary>
        internal static void Register(NativeStore.WorldData* world, UnsafeList<JointLoadState>* states)
        {
            ref Table table = ref Shared.Data;
            if (table.Count == Capacity)
            {
                throw new System.InvalidOperationException($"more than {Capacity} physics worlds have joint loads");
            }

            states->Resize((int)world->jointCount);
            for (uint i = 0; i < world->jointCount; i++)
            {
                (*states)[(int)i] = JointLoadState.Of(world->joints[i]);
            }

            table.Worlds[table.Count] = (long)world;
            table.States[table.Count] = (long)states;
            table.Count++;
        }

        /// <summary>
        /// Copy dense loads back to native joint records and unregister the world's array.
        /// </summary>
        internal static void Unregister(NativeStore.WorldData* world)
        {
            ref Table table = ref Shared.Data;
            for (int e = 0; e < table.Count; e++)
            {
                if (table.Worlds[e] != (long)world)
                {
                    continue;
                }

                var states = (UnsafeList<JointLoadState>*)table.States[e];
                for (uint i = 0; i < world->jointCount && i < (uint)states->Length; i++)
                {
                    NativeStore.JointData* joint = world->joints[i];
                    JointLoadState state = (*states)[(int)i];
                    if (state.Serial == joint->serial)
                    {
                        joint->force = state.Force;
                        joint->torque = state.Torque;
                        joint->breakingForce = state.BreakingForce;
                        joint->breakingTorque = state.BreakingTorque;
                    }
                }

                table.Count--;
                table.Worlds[e] = table.Worlds[table.Count];
                table.States[e] = table.States[table.Count];
                return;
            }
        }

        /// <summary>Returns the array of a world, or null if the world has no step state.</summary>
        internal static UnsafeList<JointLoadState>* Of(NativeStore.WorldData* world)
        {
            ref Table table = ref Shared.Data;
            for (int e = 0; e < table.Count; e++)
            {
                if (table.Worlds[e] == (long)world)
                {
                    return (UnsafeList<JointLoadState>*)table.States[e];
                }
            }

            return null;
        }

        /// <summary>
        /// Return the joint's load slot, or null when its world has no StepState.
        /// </summary>
        internal static JointLoadState* Find(NativeStore.JointData* joint)
        {
            ref Table table = ref Shared.Data;
            for (int e = 0; e < table.Count; e++)
            {
                var world = (NativeStore.WorldData*)table.Worlds[e];
                var states = (UnsafeList<JointLoadState>*)table.States[e];
                uint index = joint->index;
                if (index < world->jointCount && world->joints[index] == joint && index < (uint)states->Length
                    && (*states)[(int)index].Serial == joint->serial)
                {
                    return states->Ptr + index;
                }
            }

            return null;
        }

        /// <summary>Gives a joint that was just added to the end of its world's list a slot at the end of the array.</summary>
        internal static void Created(NativeStore.WorldData* world, NativeStore.JointData* joint)
        {
            UnsafeList<JointLoadState>* states = Of(world);
            if (states == null)
            {
                return;
            }

            states->Resize((int)world->jointCount);
            (*states)[(int)joint->index] = JointLoadState.Of(joint);
        }

        /// <summary>
        /// Move the last load slot into a removed joint's slot. Call before changing the world's joint list.
        /// </summary>
        internal static void Destroying(NativeStore.WorldData* world, NativeStore.JointData* joint)
        {
            UnsafeList<JointLoadState>* states = Of(world);
            if (states == null)
            {
                return;
            }

            int last = (int)world->jointCount - 1;
            (*states)[(int)joint->index] = (*states)[last];
            states->Resize(last);
        }

        /// <summary>Copies a joint's break limits and whether it is inactive from its record into its slot.</summary>
        internal static void Refresh(NativeStore.JointData* joint)
        {
            JointLoadState* state = Find(joint);
            if (state != null)
            {
                Refresh(state, joint);
            }
        }

        internal static void Refresh(JointLoadState* state, NativeStore.JointData* joint)
        {
            state->BreakForce = joint->breakForce;
            state->BreakTorque = joint->breakTorque;
            state->Inactive = (joint->state & JointConstraints.Inactive) != 0 ? 1u : 0u;
        }
    }

    /// <summary>
    /// Compute joint loads from impulse events and break joints above their limits.
    /// Convert events in parallel, then sum in solver order for deterministic results.
    /// Exclude drives and retain sleeping joints' previous loads. Broken joints stop constraining on the next step.
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
    /// Initialize joint loads to zero, preserving previous loads for unsolved resting bodies.
    /// If Listed is true, process only Joints.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SeedJointLoadsJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

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
            bool asleep = JointForceJobs.IsAsleep(World->joints[index]);
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

    /// <summary>
    /// Distribute group forces equally and torques by member stiffness, including forces from offset anchors.
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
    /// Store joint loads, smooth force/torque magnitudes and mark active overloaded joints.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SettleJointLoadsJob : IJobParallelFor
    {
        /// <summary>
        /// Smoothing time constant in seconds for break-test force and torque.
        /// Smoothing reduces one-step load spikes when the direct solver stops a rigid craft.
        /// </summary>
        private const float BreakingTimeConstant = 0.1f;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        /// <summary>The load state of each joint, at the joint's index in the world's list. See <see cref="JointLoadStates"/>.</summary>
        [NativeDisableUnsafePtrRestriction]
        public JointLoadState* States;

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
            Overloaded[index] = Settle(World->joints[index], States + index, Loads[index], DeltaTime) ? (byte)1 : (byte)0;
        }

        /// <summary>
        /// Store and smooth one joint's load. Return whether the active joint exceeds a break limit.
        /// </summary>
        internal static bool Settle(NativeStore.JointData* joint, JointLoadState* state, in JointLoad load, float deltaTime)
        {
            if (JointForceJobs.IsAsleep(joint))
            {
                state->Force = load.Force;
                state->Torque = load.Torque;
                return false;
            }

            return SettleAwake(state, load, deltaTime);
        }

        /// <summary>
        /// Settle a joint load when the caller already knows that at least one moving body is awake.
        /// </summary>
        internal static bool SettleAwake(JointLoadState* state, in JointLoad load, float deltaTime)
        {
            state->Force = load.Force;
            state->Torque = load.Torque;
            float follow = deltaTime / (BreakingTimeConstant + deltaTime);
            state->BreakingForce += (math.length(state->Force) - state->BreakingForce) * follow;
            state->BreakingTorque += (math.length(state->Torque) - state->BreakingTorque) * follow;
            bool overloaded = state->BreakingForce > state->BreakForce || state->BreakingTorque > state->BreakTorque;
            return state->Inactive == 0 && overloaded;
        }
    }

    /// <summary>
    /// Break overloaded joints and emit break records in store order.
    /// </summary>
    [BurstCompile]
    internal unsafe struct BreakJointsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        /// <summary>The load state of each joint, at the joint's index in the world's list. See <see cref="JointLoadStates"/>.</summary>
        [NativeDisableUnsafePtrRestriction]
        public JointLoadState* States;

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
                Break(World->joints[joint], States + joint);
            }
        }

        private void Break(NativeStore.JointData* joint, JointLoadState* state)
        {
            joint->state |= NativeStore.JointState.Broken;
            state->Inactive = 1;
            JointTopology.Changed();
            if (World->jointBreakCount == World->jointBreakCapacity)
            {
                World->jointBreakCapacity = StoreOps.Grown(World->jointBreakCapacity);
                World->jointBreaks = (NativeStore.JointBreak*)StoreOps.Reallocate(Globals, World->jointBreaks, World->jointBreakCapacity * (ulong)sizeof(NativeStore.JointBreak));
            }

            World->jointBreaks[World->jointBreakCount++] = new NativeStore.JointBreak
            {
                engineObjectId = joint->userData != null ? *(ulong*)((byte*)joint->userData + EngineObjectIdOffset) : 0,
                linearMagnitude = state->BreakingForce,
                angularMagnitude = state->BreakingTorque,
            };
        }
    }
}
