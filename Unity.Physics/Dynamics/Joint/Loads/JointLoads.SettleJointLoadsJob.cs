using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Joint.Translation;
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
                        state->FullForce = float3.zero;
                        state->FullTorque = float3.zero;
                        state->SmoothedFullForce = 0.0f;
                        state->SmoothedFullTorque = 0.0f;
                    }

                    return false;
                }

                return SettleAwake(joint, state, load, deltaTime);
            }

            /// <summary>
            /// Settle a joint load when the caller already knows that at least one moving body is awake.
            /// </summary>
            internal static bool SettleAwake(JointData* joint, JointLoadState* state, in JointLoad load, float deltaTime)
            {
                state->FullForce = load.Force;
                state->FullTorque = load.Torque;
                if (joint->type == StoreConstants.JointTypeDof6)
                {
                    state->Force = ConstraintLoad(joint, StoreConstants.AxisLinearX, load.Force, load.LimitForce);
                    state->Torque = ConstraintLoad(joint, StoreConstants.AxisAngularX, load.Torque, load.LimitTorque);
                }
                else
                {
                    state->Force = load.Force;
                    state->Torque = load.Torque;
                }

                float follow = deltaTime / (BreakingTimeConstant + deltaTime);
                float force = math.length(load.Force);
                float torque = math.length(load.Torque);
                state->SmoothedFullForce += (force - state->SmoothedFullForce) * follow;
                state->SmoothedFullTorque += (torque - state->SmoothedFullTorque) * follow;
                state->BreakingForce += (math.length(state->Force) - state->BreakingForce) * follow;
                state->BreakingTorque += (math.length(state->Torque) - state->BreakingTorque) * follow;
                bool overloaded = state->BreakingForce > state->BreakForce || state->BreakingTorque > state->BreakTorque;
                return state->Inactive == 0 && overloaded;
            }

            /// <summary>
            /// Keep the measured limit reaction and the part of the remaining load that acts along locked axes.
            /// For mixed merged or vessel joints, projection estimates the split of the combined load.
            /// </summary>
            private static float3 ConstraintLoad(JointData* joint, int firstAxis, float3 total, float3 limit)
            {
                JointAxis* axes = JointData.AxisOf(joint, firstAxis);
                bool3 locked = new bool3(axes[0].dofLock == StoreConstants.DofLocked,
                    axes[1].dofLock == StoreConstants.DofLocked, axes[2].dofLock == StoreConstants.DofLocked);
                if (math.all(locked))
                {
                    return total;
                }

                if (!math.any(locked))
                {
                    return limit;
                }

                quaternion frame = JointConstraints.WorldRotation0(joint);
                float3 local = math.rotate(math.conjugate(frame), total - limit);
                return limit + math.rotate(frame, math.select(float3.zero, local, locked));
            }
        }
    }
}
