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
    /// The values that the jobs for joint loads read and write for one joint in every step. While a world has a
    /// <see cref="StepState"/>, these values sit in a dense array that <see cref="JointLoadStates"/> keeps beside the
    /// world's list of joints, at the joint's index in that list. The jobs then don't need to reach the joint records,
    /// which are large and spread through memory. The array holds the only current copy of the loads and of the
    /// smoothed magnitudes. The records get them back when the world's step state is disposed. The break limits and
    /// whether the joint is inactive are copies of the record's values, which the joint commands keep up to date.
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
    /// Finds the dense array of <see cref="JointLoadState"/> for each world that has a <see cref="StepState"/>. The
    /// step state registers its array when it is made and removes it when it is disposed. The joint commands use this
    /// table to keep each world's array in the same order as the world's list of joints: a new joint gets a slot at
    /// the end, and a destroyed joint's slot takes the last slot, as in the list. The commands that change a break
    /// limit or a joint's bodies copy the new values into the joint's slot, and GetForces reads the loads from it.
    /// Commands and registrations arrive on the main thread, outside a step.
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
        /// Writes the loads in a world's array back to its joint records, and takes the array out of the table. From
        /// now on, the records hold the loads again.
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
        /// Returns the slot of a joint, or null if the joint's world has no step state. The command doesn't know the
        /// joint's world, so the method looks for the world whose list holds the joint at the joint's index.
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
        /// Moves the last slot into the slot of a joint that is about to be destroyed, as
        /// <see cref="JointKernel.Destroy"/> does with the world's list. Call this before the list changes.
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
            JointGroup group = Groups[index];
            quaternion frame = JointConstraints.WorldRotation0(World->joints[group.Leader]);
            Share(group, frame, GroupLoads[index], Members, JointLoads);
        }

        /// <summary>
        /// Adds each member's share of the group's load to the member's load. The frame is the world rotation of the
        /// leading joint's frame on body0, which <see cref="JointConstraints.WorldRotation0"/> gives once the poses
        /// are written back.
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
    /// Writes each joint's load to its load state, then smooths the force and torque magnitudes used for breaking. Marks
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
        /// Writes one joint's load to its load state and smooths it, as described for this job. Returns whether the
        /// joint is active and its smoothed load exceeds its break limits.
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
        /// Settles the load of a joint that the caller knows has at least one moving body that is awake, as
        /// <see cref="Settle"/> does, and returns the same result. The caller saves the reads of the joint's record
        /// and of its two bodies, which are often not in the cache.
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
