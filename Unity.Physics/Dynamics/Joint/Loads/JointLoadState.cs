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
    /// Current joint loads and smoothed magnitudes, stored densely alongside the world's joint list.
    /// Copy loads back to native records when StoreStepper is disposed. Commands update cached break limits and
    /// inactivity.
    /// </summary>
    internal struct JointLoadState
    {
        // The force and torque that the joint applied during the last step, as in JointData.
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

        internal static unsafe JointLoadState Of(JointData* joint)
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
}
