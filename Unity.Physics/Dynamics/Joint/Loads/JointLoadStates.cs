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

namespace Unity.Physics.Dynamics.Joint.Loads
{
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
        internal static void Register(WorldData* world, UnsafeList<JointLoadState>* states)
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
        internal static void Unregister(WorldData* world)
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
                    JointData* joint = world->joints[i];
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
        internal static UnsafeList<JointLoadState>* Of(WorldData* world)
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
        internal static JointLoadState* Find(JointData* joint)
        {
            ref Table table = ref Shared.Data;
            for (int e = 0; e < table.Count; e++)
            {
                var world = (WorldData*)table.Worlds[e];
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
        internal static void Created(WorldData* world, JointData* joint)
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
        internal static void Destroying(WorldData* world, JointData* joint)
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
        internal static void Refresh(JointData* joint)
        {
            JointLoadState* state = Find(joint);
            if (state != null)
            {
                Refresh(state, joint);
            }
        }

        internal static void Refresh(JointLoadState* state, JointData* joint)
        {
            state->BreakForce = joint->breakForce;
            state->BreakTorque = joint->breakTorque;
            state->Inactive = (joint->state & JointConstraints.Inactive) != 0 ? 1u : 0u;
        }
    }
}
