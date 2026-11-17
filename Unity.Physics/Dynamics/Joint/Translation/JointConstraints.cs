using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Physics.Math;
using SolverJoint = Unity.Physics.Joint;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Memory;

namespace Unity.Physics.Dynamics.Joint.Translation
{
    /// <summary>Calculates effective masses and the state of the two joint frames for the joint jobs.</summary>
    internal static unsafe class JointConstraints
    {
        internal const JointState Inactive = JointState.Broken | JointState.BodyDestroyed;

        // Above this threshold, a drive's force limit has no practical effect, and calculating an impulse from it may
        // exceed the range of a float. Stock joints use 1e20; Unity's default is float.MaxValue.
        internal const float UnlimitedForce = 1e15f;

        private static float InverseMass(Body* body)
        {
            if (body == null)
            {
                return 0.0f;
            }

            BodyData* hot = StoreMemory.Row(body);
            return (hot->flags & (BodyFlags.Kinematic | BodyFlags.Static)) != 0 ? 0.0f : (*hot).Mass()->InverseMass;
        }

        private static float InverseInertia(Body* body, float3 worldAxis)
        {
            if (body == null)
            {
                return 0.0f;
            }

            BodyData* hot = StoreMemory.Row(body);
            if ((hot->flags & (BodyFlags.Kinematic | BodyFlags.Static)) != 0)
            {
                return 0.0f;
            }

            float3 axis = math.rotate(math.conjugate(math.mul((*hot).Pose()->Rotation, (*hot).Mass()->Transform.rot)), worldAxis);
            return math.csum(axis * axis * (*hot).Mass()->InverseInertia);
        }

        internal static float LinearMass(JointData* joint)
        {
            return 1.0f / math.max(InverseMass(joint->body0) + InverseMass(joint->body1), 1e-9f);
        }

        /// <summary>The effective inertia of the two bodies for rotation about one joint axis.</summary>
        internal static float AngularMass(JointData* joint, int axis)
        {
            float3 local = float3.zero;
            local[axis] = 1.0f;
            float3 world = math.rotate(WorldRotation0(joint), local);
            return 1.0f / math.max(InverseInertia(joint->body0, world) + InverseInertia(joint->body1, world), 1e-9f);
        }

        /// <summary>
        /// Return a drive for the whole rotation, or null when separate motors are required.
        /// An active slerp drive takes precedence. Use motors when force limits must be enforced.
        /// </summary>
        internal static JointAxis* WholeRotationDrive(JointData* joint)
        {
            for (int i = 0; i < 3; i++)
            {
                if (JointData.AxisOf(joint, StoreConstants.AxisAngularX + i)->dofLock == StoreConstants.DofLocked)
                {
                    return null;
                }
            }

            JointAxis* slerp = JointData.AxisOf(joint, StoreConstants.AxisSlerp);
            if (slerp->driveSpring > 0.0f || slerp->driveDamper > 0.0f)
            {
                return slerp;
            }

            JointAxis* twist = JointData.AxisOf(joint, StoreConstants.AxisAngularX);
            JointAxis* swing = JointData.AxisOf(joint, StoreConstants.AxisAngularYZ);
            bool sameDrive = twist->driveSpring == swing->driveSpring && twist->driveDamper == swing->driveDamper;
            bool noForceLimit = twist->driveMaxForce >= UnlimitedForce && swing->driveMaxForce >= UnlimitedForce;
            return sameDrive && noForceLimit && twist->driveSpring > 0.0f ? twist : null;
        }

        internal static quaternion WorldRotation0(JointData* joint)
        {
            return joint->body0 != null ? math.mul((*StoreMemory.Row(joint->body0)).Pose()->Rotation, joint->rotation0) : joint->rotation0;
        }

        internal static quaternion WorldRotation1(JointData* joint)
        {
            return joint->body1 != null ? math.mul((*StoreMemory.Row(joint->body1)).Pose()->Rotation, joint->rotation1) : joint->rotation1;
        }

        private static float3 WorldAnchor(Body* body, float3 local)
        {
            if (body == null)
            {
                return local;
            }

            BodyData* hot = StoreMemory.Row(body);
            return (*hot).Pose()->Position + math.rotate((*hot).Pose()->Rotation, local);
        }

        /// <summary>
        /// The position of frame 1's origin relative to frame 0's origin, expressed along frame 0's axes. Linear drives
        /// and limits measure this position.
        /// </summary>
        internal static float3 RelativePosition(JointData* joint)
        {
            float3 difference = WorldAnchor(joint->body1, joint->position1) - WorldAnchor(joint->body0, joint->position0);
            return math.rotate(math.conjugate(WorldRotation0(joint)), difference);
        }

        /// <summary>Body 1's angular velocity relative to body 0, in world space.</summary>
        internal static float3 RelativeAngularVelocity(JointData* joint)
        {
            float3 first = joint->body0 != null ? BodyKinematics.WorldAngularVelocity(*StoreMemory.Row(joint->body0)) : float3.zero;
            float3 second = joint->body1 != null ? BodyKinematics.WorldAngularVelocity(*StoreMemory.Row(joint->body1)) : float3.zero;
            return second - first;
        }

        /// <summary>
        /// Target angular velocity of frame 1 relative to frame 0. Negate Unity's target to match PhysX;
        /// other drive targets keep their signs.
        /// </summary>
        internal static float3 TargetAngularVelocity(JointData* joint)
        {
            return -joint->targetAngularVelocity;
        }

        /// <summary>
        /// Expresses a rotation as its axis multiplied by its angle. Each angular drive uses the component along its
        /// own axis.
        /// </summary>
        internal static float3 AngleAxis(quaternion rotation)
        {
            if (rotation.value.w < 0.0f)
            {
                rotation.value = -rotation.value;
            }

            float sine = math.length(rotation.value.xyz);
            return sine < 1e-6f ? rotation.value.xyz * 2.0f : rotation.value.xyz / sine * (2.0f * math.atan2(sine, rotation.value.w));
        }
    }
}
