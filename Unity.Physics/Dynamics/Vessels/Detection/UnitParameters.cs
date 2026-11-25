using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Memory;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Joint.Grouping;

namespace Unity.Physics.Dynamics.Vessels.Detection
{
    /// <summary>
    /// Vessel edge or closure parameters: body-local anchors and frames, angular stiffness/damping and break
    /// limits.
    /// Represent joint groups at their mean anchor.
    /// </summary>
    internal unsafe struct UnitParameters
    {
        public float3 Anchor0;
        public float3 Anchor1;
        public quaternion Rotation0;
        public quaternion Rotation1;

        /// <summary>
        /// Spring stiffness and damping for each angular axis of the joint frame: twist, followed by the two swing
        /// axes.
        /// </summary>
        public double3 Stiffness;

        public double3 Damping;
        public double BreakForce;
        public double BreakTorque;

        /// <summary>
        /// Squared offset threshold in square metres above which rigid linear constraints prevent rotation about an
        /// axis.
        /// </summary>
        private const float SpreadSquared = 1e-8f;

        public static UnitParameters Of(WorldData* world, JointUnit unit, NativeArray<JointGroup> groups,
            NativeArray<JointGroupMember> members, ServoAxis servo = ServoAxis.None)
        {
            var parameters = new UnitParameters();
            JointData* leader = world->joints[unit.Joint];
            parameters.Rotation0 = leader->rotation0;
            parameters.Rotation1 = leader->rotation1;
            int count = unit.Group < 0 ? 1 : groups[unit.Group].MemberCount;
            int firstMember = unit.Group < 0 ? 0 : groups[unit.Group].FirstMember;
            for (int i = 0; i < count; i++)
            {
                JointData* joint = unit.Group < 0 ? leader : world->joints[members[firstMember + i].Joint];
                parameters.Anchor0 += joint->position0;
                parameters.Anchor1 += joint->position1;
                parameters.BreakForce += joint->breakForce;
                parameters.BreakTorque += joint->breakTorque;
            }

            parameters.Anchor0 /= count;
            parameters.Anchor1 /= count;

            if (servo != ServoAxis.None)
            {
                parameters.Stiffness = double.PositiveInfinity;
                if (servo == ServoAxis.AngularX)
                {
                    parameters.Stiffness.x = 0.0;
                }

                return parameters;
            }

            // Treat locked axes as rigid. For a driven axis, sum the spring stiffness and damping from the members'
            // drives.
            bool3 hard = false;
            for (int i = 0; i < count; i++)
            {
                JointData* joint = unit.Group < 0 ? leader : world->joints[members[firstMember + i].Joint];
                JointAxis* wholeRotation = JointConstraints.WholeRotationDrive(joint);
                for (int axis = 0; axis < 3; axis++)
                {
                    JointAxis* drive = wholeRotation != null ? wholeRotation : JointTranslation.BuildJointsJob.AngularDrive(joint, axis);
                    if (JointData.AxisOf(joint, StoreConstants.AxisAngularX + axis)->dofLock == StoreConstants.DofLocked)
                    {
                        hard[axis] = true;
                    }
                    else if (JointTranslation.BuildJointsJob.HasAxisDrive(drive))
                    {
                        parameters.Stiffness[axis] += drive->driveSpring;
                        parameters.Damping[axis] += drive->driveDamper;
                    }
                }
            }

            // Rigid linear constraints prevent rotation unless every anchor offset lies along the axis within
            // the spread threshold.
            if (count > 1)
            {
                quaternion toJoint = math.conjugate(leader->rotation0);
                float3 spread = float3.zero;
                for (int i = 0; i < count; i++)
                {
                    float3 offset = math.rotate(toJoint, world->joints[members[firstMember + i].Joint]->position0 - parameters.Anchor0);
                    spread += math.lengthsq(offset) - offset * offset;
                }

                hard |= spread > SpreadSquared;
            }

            for (int axis = 0; axis < 3; axis++)
            {
                if (hard[axis])
                {
                    parameters.Stiffness[axis] = double.PositiveInfinity;
                }
            }

            return parameters;
        }

        /// <summary>
        /// Converts a point from the body's local frame to its motion frame, whose origin is the centre of mass and
        /// whose axes are the principal inertia axes.
        /// </summary>
        public static double3 InMotionFrame(BodyData* body, float3 point)
        {
            RigidTransform bodyFromMotion = body->mass->Transform;
            return math.rotate(math.conjugate(bodyFromMotion.rot), point - bodyFromMotion.pos);
        }

        /// <summary>Converts a rotation from the body's local frame to its motion frame.</summary>
        public static quaternion InMotionFrame(BodyData* body, quaternion rotation)
        {
            return math.mul(math.conjugate(body->mass->Transform.rot), rotation);
        }

        /// <summary>
        /// Rotation from part B's motion frame to part A's at joint rest, when the two joint frames align in world
        /// space.
        /// </summary>
        public static double4 Rest(BodyData* bodyA, quaternion frameA, BodyData* bodyB, quaternion frameB)
        {
            quaternion rest = math.mul(math.mul(InMotionFrame(bodyA, frameA), math.conjugate(frameB)), bodyB->mass->Transform.rot);
            return math.normalize(rest).value;
        }

        /// <summary>
        /// Returns true if two motion frames are the same bit for bit, so that values computed from one hold for the
        /// other.
        /// </summary>
        public static bool Same(RigidTransform a, RigidTransform b)
        {
            return UnsafeUtility.MemCmp(&a, &b, sizeof(RigidTransform)) == 0;
        }

        /// <summary>
        /// Return the highest joint settings stamp in a unit, changing whenever any member's settings change.
        /// </summary>
        public static uint Stamp(WorldData* world, JointUnit unit, NativeArray<JointGroup> groups,
            NativeArray<JointGroupMember> members)
        {
            if (unit.Group < 0)
            {
                return world->joints[unit.Joint]->settingsStamp;
            }

            // The leader is one of the group's members.
            JointGroup group = groups[unit.Group];
            uint stamp = 0;
            for (int i = 0; i < group.MemberCount; i++)
            {
                stamp = math.max(stamp, world->joints[members[group.FirstMember + i].Joint]->settingsStamp);
            }

            return stamp;
        }

        public static double3x3 Frame(BodyData* body, quaternion frame)
        {
            return new double3x3(new float3x3(InMotionFrame(body, frame)));
        }
    }
}
