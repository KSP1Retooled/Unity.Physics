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
    internal static partial class JointBreaking
    {
        /// <summary>
        /// Break overloaded joints and emit break records in store order.
        /// </summary>
        [BurstCompile]
        internal unsafe struct BreakJointsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [NativeDisableUnsafePtrRestriction]
            public StoreGlobals* Globals;

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

            private void Break(JointData* joint, JointLoadState* state)
            {
                joint->state |= JointState.Broken;
                state->Inactive = 1;
                JointTopology.Changed();
                if (World->jointBreakCount == World->jointBreakCapacity)
                {
                    World->jointBreakCapacity = StoreMemory.Grown(World->jointBreakCapacity);
                    World->jointBreaks = (JointBreak*)StoreMemory.Reallocate(Globals, World->jointBreaks, World->jointBreakCapacity * (ulong)sizeof(JointBreak));
                }

                World->jointBreaks[World->jointBreakCount++] = new JointBreak
                {
                    engineObjectId = joint->userData != null ? *(ulong*)((byte*)joint->userData + EngineObjectIdOffset) : 0,
                    linearMagnitude = state->BreakingForce,
                    angularMagnitude = state->BreakingTorque,
                };
            }
        }
    }
}
