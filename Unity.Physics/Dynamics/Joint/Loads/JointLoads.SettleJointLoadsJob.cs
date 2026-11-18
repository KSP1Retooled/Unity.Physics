using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;

namespace Unity.Physics.Dynamics.Joint.Loads
{
    internal static unsafe partial class JointLoads
    {
        /// <summary>
        /// Store joint loads, smooth force/torque magnitudes and mark active overloaded joints.
        /// </summary>
        [BurstCompile]
        internal unsafe struct SettleJointLoadsJob : IJobParallelFor, IJobParallelForDefer
        {
            /// <summary>
            /// Smoothing time constant in seconds for break-test force and torque.
            /// Smoothing reduces one-step load spikes when the direct solver stops a rigid craft.
            /// </summary>
            private const float BreakingTimeConstant = 0.1f;

            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

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
            internal static bool Settle(JointData* joint, JointLoadState* state, in JointLoad load, float deltaTime)
            {
                if (JointLoads.IsAsleep(joint))
                {
                    if (JointGrouping.GroupJointsJob.Immobile(joint->body0) && JointGrouping.GroupJointsJob.Immobile(joint->body1))
                    {
                        // The joint cannot apply a load while both ends are held. Drop the break history
                        // so that a load from before packing cannot break the joint after unpacking.
                        state->Force = float3.zero;
                        state->Torque = float3.zero;
                        state->BreakingForce = 0.0f;
                        state->BreakingTorque = 0.0f;
                    }
                    else
                    {
                        state->Force = load.Force;
                        state->Torque = load.Torque;
                    }

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
    }
}
