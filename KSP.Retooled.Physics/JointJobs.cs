using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using static Unity.Physics.Math;
using PhysicsJoint = Unity.Physics.Joint;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// A sleeping body counts as immovable in the step, so one joined to an awake body wakes with it, as a PhysX island does.
    /// A pass wakes one joint's distance, so it repeats until a chain of bodies has settled.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WakeJointedBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        public void Execute()
        {
            while (WakeThroughJoints(World))
            {
            }
        }

        /// <summary>One pass over the joints. Waking spreads along a chain of them, so the caller repeats until nothing wakes.</summary>
        internal static bool WakeThroughJoints(NativeStore.WorldData* world)
        {
            bool woke = false;
            for (uint i = 0; i < world->jointCount; i++)
            {
                NativeStore.JointData* joint = world->joints[i];
                if (joint->body0 == null || joint->body1 == null || (joint->state & JointConstraints.Inactive) != 0)
                {
                    continue;
                }

                NativeStore.BodyData* first = StoreOps.Hot(joint->body0);
                NativeStore.BodyData* second = StoreOps.Hot(joint->body1);
                woke |= WakeFrom(first, second);
                woke |= WakeFrom(second, first);
            }

            return woke;
        }

        private static bool WakeFrom(NativeStore.BodyData* awake, NativeStore.BodyData* sleeping)
        {
            const NativeStore.BodyFlags immobile = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic;
            bool otherMoves = (awake->flags & (immobile | NativeStore.BodyFlags.Sleeping)) == 0;
            if (!otherMoves || (sleeping->flags & NativeStore.BodyFlags.Sleeping) == 0 || (sleeping->flags & immobile) != 0)
            {
                return false;
            }

            JointKernel.WakeRow(sleeping);
            return true;
        }
    }

    /// <summary>
    /// Turns each store joint, or group of them that <see cref="GroupJointsJob"/> found, into the Unity Physics joints that
    /// express it. One of ours can take several: a Unity Physics joint holds three constraints, and all of them share one
    /// pair of frames. Each unit writes its own run of the stream, which <see cref="GatherJointsJob"/> lays out in order.
    /// </summary>
    /// <remarks>
    /// Body A of a built joint is the store's body1, the connected body, and body B is body0, the joint's owner, because the
    /// limits Unity sends measure frame 1 relative to frame 0 and a Unity Physics constraint measures A relative to B.
    /// </remarks>
    [BurstCompile]
    internal unsafe struct BuildJointsJob : IJobParallelForDefer
    {
        // The solver hands out a constraint's impulse only as an event past a threshold, so a threshold of zero reports them all.
        private static readonly float3 ReportEveryImpulse = float3.zero;

        // Unity Physics derives a damper's strength from a spring, so a pure damper gets one too weak to matter next to it.
        private const float PureDamperSpringRatio = 1e-3f;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<int> BuiltIndices;

        [ReadOnly]
        public NativeArray<JointUnit> Units;

        public NativeStream.Writer Output;

        // A unit writes only its own group and that group's members.
        [NativeDisableParallelForRestriction]
        public NativeArray<JointGroup> Groups;

        [NativeDisableParallelForRestriction]
        public NativeArray<JointGroupMember> Members;

        public int DynamicCount;
        public int AnchorIndex;
        public float DeltaTime;

        /// <summary>Each unit's place in the vessel forest, see <see cref="GroupJointsJob"/>.</summary>
        [ReadOnly]
        public NativeArray<JointForestRole> Roles;

        private PhysicsJoint pending;
        private JointForestRole role;
        private int pendingCount;
        private bool emitted;

        public void Execute(int index)
        {
            Output.BeginForEachIndex(index);
            JointUnit unit = Units[index];
            role = Roles[index];
            if (unit.Group < 0)
            {
                Append(World->joints[unit.Joint]);
            }
            else
            {
                AppendGroup(unit.Group);
            }

            Output.EndForEachIndex();
        }

        internal static int BodyKey(NativeStore.Body* body)
        {
            return body == null ? -1 : (int)body->index;
        }

        internal static bool SameFrames(NativeStore.JointData* first, NativeStore.JointData* second)
        {
            return math.all(first->rotation0.value == second->rotation0.value) && math.all(first->rotation1.value == second->rotation1.value)
                && first->flags == second->flags;
        }

        /// <summary>
        /// Whether the joint holds its anchors together and does nothing to their rotation but hold it too: each axis
        /// locked, or held by a spring towards where it is, or free. Joints like that between the same two bodies add up to
        /// one. A stock part joint locks its twist, because Unity locks an axis whose two limits are equal, and holds its
        /// swing with drives.
        /// </summary>
        internal static bool Mergeable(NativeStore.JointData* joint)
        {
            if ((joint->state & JointConstraints.Inactive) != 0 || joint->type != NativeStore.JointTypeDof6)
            {
                return false;
            }

            int lockedAngles = 0;
            for (int i = 0; i < 3; i++)
            {
                if (NativeStore.AxisOf(joint, NativeStore.AxisLinearX + i)->dofLock != NativeStore.DofLocked && !HeldByDrive(joint, i))
                {
                    return false;
                }

                if (NativeStore.AxisOf(joint, NativeStore.AxisAngularX + i)->dofLock == NativeStore.DofLocked)
                {
                    lockedAngles++;
                }
            }

            if (lockedAngles == 3)
            {
                return true;
            }

            bool atRest = joint->targetRotation.value.Equals(quaternion.identity.value) && math.all(joint->targetAngularVelocity == float3.zero);
            if (!atRest || HasNearAngularLimit(joint))
            {
                return false;
            }

            if (JointConstraints.WholeRotationDrive(joint) != null)
            {
                return true;
            }

            for (int i = 0; i < 3; i++)
            {
                NativeStore.JointAxis* drive = AngularDrive(joint, i);
                bool locked = NativeStore.AxisOf(joint, NativeStore.AxisAngularX + i)->dofLock == NativeStore.DofLocked;
                if (!locked && HasAxisDrive(drive) && (drive->driveSpring <= 0.0f || drive->driveMaxForce < JointConstraints.UnlimitedForce))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>The drive on one rotation axis: the twist drive about x, and the swing drive about y and z.</summary>
        private static NativeStore.JointAxis* AngularDrive(NativeStore.JointData* joint, int axis)
        {
            return NativeStore.AxisOf(joint, axis == 0 ? NativeStore.AxisAngularX : NativeStore.AxisAngularYZ);
        }

        /// <summary>
        /// Whether a limit <see cref="AddAngularDof"/> would build comes within <see cref="UnreachedAngle"/> of the joint's
        /// rest. Stock part joints limit their swing near half a turn, which a joint holding its anchors never reaches.
        /// </summary>
        private static bool HasNearAngularLimit(NativeStore.JointData* joint)
        {
            NativeStore.JointAxis* twist = NativeStore.AxisOf(joint, NativeStore.AxisAngularX);
            NativeStore.JointAxis* swing = NativeStore.AxisOf(joint, NativeStore.AxisAngularYZ);
            bool nearTwist = twist->dofLock == NativeStore.DofLimited && (twist->min > -UnreachedAngle || twist->max < UnreachedAngle);
            bool nearY = NativeStore.AxisOf(joint, NativeStore.AxisAngularY)->dofLock == NativeStore.DofLimited && swing->min < UnreachedAngle;
            bool nearZ = NativeStore.AxisOf(joint, NativeStore.AxisAngularZ)->dofLock == NativeStore.DofLimited && swing->max < UnreachedAngle;
            return nearTwist || nearY || nearZ;
        }

        /// <summary>An angle in radians that a merged group, holding its rotation, never turns through.</summary>
        private const float UnreachedAngle = 0.5f;

        private static bool HasAxisDrive(NativeStore.JointAxis* drive)
        {
            return drive->driveMaxForce > 0.0f && (drive->driveSpring > 0.0f || drive->driveDamper > 0.0f);
        }

        /// <summary>
        /// One joint at the members' mean anchor, as stiff as all of them together. The direct solve caps each row's
        /// stiffness, so the merged rows ask for a multiple of that cap: the linear rows for one cap per member, and each
        /// rotation for every member's own hold on it plus what their linear rows resist through their distance from the
        /// mean, which for small turns is a stiffness of the linear cap times that distance squared.
        /// </summary>
        private void AppendGroup(int index)
        {
            JointGroup group = Groups[index];
            NativeStore.JointData* leader = World->joints[group.Leader];
            float3 anchor0 = float3.zero;
            float3 anchor1 = float3.zero;
            for (int i = 0; i < group.MemberCount; i++)
            {
                NativeStore.JointData* member = World->joints[Members[group.FirstMember + i].Joint];
                anchor0 += member->position0;
                anchor1 += member->position1;
            }

            anchor0 /= group.MemberCount;
            anchor1 /= group.MemberCount;

            float inverseMass = MovingInverseMass(leader->body0) + MovingInverseMass(leader->body1);
            float inverseInertia = MovingInverseInertia(leader->body0) + MovingInverseInertia(leader->body1);
            float angularCap = DirectStiffness.Maximum(inverseInertia);
            group.LinearRatio = inverseInertia / math.max(inverseMass, 1e-9f);
            group.Stiffness = float3.zero;
            quaternion toJoint = math.conjugate(leader->rotation0);
            for (int i = 0; i < group.MemberCount; i++)
            {
                JointGroupMember member = Members[group.FirstMember + i];
                NativeStore.JointData* joint = World->joints[member.Joint];
                member.Offset = math.rotate(toJoint, joint->position0 - anchor0);
                member.AngularStiffness = AngularStiffness(joint, angularCap);
                group.Stiffness += member.AngularStiffness + group.LinearRatio * (math.lengthsq(member.Offset) - member.Offset * member.Offset);
                Members[group.FirstMember + i] = member;
            }

            Groups[index] = group;

            if (!Begin(leader, anchor0, anchor1, new Entity { Index = index, Version = JointGroup.EntityVersion }))
            {
                return;
            }

            Constraint linear = Constraint.BallAndSocket(ReportEveryImpulse);
            linear.DirectStiffnessScale = group.MemberCount;
            Add(linear);
            for (int axis = 0; axis < 3; axis++)
            {
                if (group.Stiffness[axis] > 0.0f)
                {
                    Constraint angular = Constraint.Twist(axis, new FloatRange(0.0f, 0.0f), ReportEveryImpulse);
                    angular.DirectStiffnessScale = group.Stiffness[axis];
                    Add(angular);
                }
            }

            Flush();
        }

        /// <summary>
        /// A member's hold on each rotation, in multiples of the cap. A spring is compared with the cap directly, as the
        /// stiffness Unity asked for.
        /// </summary>
        private static float3 AngularStiffness(NativeStore.JointData* joint, float angularCap)
        {
            NativeStore.JointAxis* wholeRotation = JointConstraints.WholeRotationDrive(joint);
            float3 stiffness = float3.zero;
            for (int i = 0; i < 3; i++)
            {
                NativeStore.JointAxis* drive = wholeRotation != null ? wholeRotation : AngularDrive(joint, i);
                if (NativeStore.AxisOf(joint, NativeStore.AxisAngularX + i)->dofLock == NativeStore.DofLocked)
                {
                    stiffness[i] = 1.0f;
                }
                else if (HasAxisDrive(drive))
                {
                    stiffness[i] = math.min(drive->driveSpring / angularCap, 1.0f);
                }
            }

            return stiffness;
        }

        /// <summary>The inverse mass the step gives a body, which holds a sleeping one in place.</summary>
        private static float MovingInverseMass(NativeStore.Body* body)
        {
            return HeldInPlace(body) ? 0.0f : StoreOps.Hot(body)->mass->InverseMass;
        }

        /// <summary>The mean of a body's inverse inertia over its axes, which the cap on a rotation uses.</summary>
        private static float MovingInverseInertia(NativeStore.Body* body)
        {
            return HeldInPlace(body) ? 0.0f : math.csum(StoreOps.Hot(body)->mass->InverseInertia) / 3.0f;
        }

        private void Append(NativeStore.JointData* joint)
        {
            if ((joint->state & JointConstraints.Inactive) != 0)
            {
                return;
            }

            if (!Begin(joint, joint->position0, joint->position1, new Entity { Index = (int)joint->index, Version = 1 }))
            {
                return;
            }

            switch (joint->type)
            {
                case NativeStore.JointTypeFixed:
                    Add(Constraint.BallAndSocket(ReportEveryImpulse));
                    AddLockedAngles(new bool3(true), Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio, ReportEveryImpulse);
                    break;
                case NativeStore.JointTypeHinge:
                    AddHinge(joint);
                    break;
                case NativeStore.JointTypeBallAndSocket:
                    Add(Constraint.BallAndSocket(ReportEveryImpulse));
                    break;
                case NativeStore.JointTypeDistance:
                    AddDistance(joint);
                    break;
                case NativeStore.JointTypeDof6:
                    AddLinearDof(joint);
                    AddAngularDof(joint);
                    AddLinearDrives(joint);
                    AddAngularDrives(joint);
                    break;
            }

            // A joint with nothing to constrain still takes part in deciding whether its two bodies collide.
            if (!emitted && pendingCount == 0)
            {
                pending.Constraints.Length = 0;
                Output.Write(pending);
            }

            Flush();
        }

        /// <summary>Starts the joint's constraints, and returns false when it gets none.</summary>
        private bool Begin(NativeStore.JointData* joint, float3 anchor0, float3 anchor1, Entity entity)
        {
            int indexA = BuiltIndex(joint->body1);
            int indexB = BuiltIndex(joint->body0);
            if (indexA < 0 || indexB < 0 || (indexA >= DynamicCount && indexB >= DynamicCount))
            {
                return false;
            }

            pending = new PhysicsJoint
            {
                BodyPair = new BodyIndexPair { BodyIndexA = indexA, BodyIndexB = indexB },
                AFromJoint = new MTransform(joint->rotation1, anchor1),
                BFromJoint = new MTransform(joint->rotation0, anchor0),
                Entity = entity,
                EnableCollision = (joint->flags & NativeStore.JointFlagCollisionEnabled) != 0,
                ForestRole = role,
            };
            pendingCount = 0;
            emitted = false;

            // Nothing on either side can move, and an island's solver may be the direct one, which refuses such a joint.
            if (HeldInPlace(joint->body0) && HeldInPlace(joint->body1))
            {
                pending.Constraints.Length = 0;
                Output.Write(pending);
                return false;
            }

            return true;
        }

        private static bool HeldInPlace(NativeStore.Body* body)
        {
            const NativeStore.BodyFlags held = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Sleeping;
            return body == null || (StoreOps.Hot(body)->flags & held) != 0;
        }

        private int BuiltIndex(NativeStore.Body* body)
        {
            return body == null ? AnchorIndex : BuiltIndices[(int)body->index];
        }

        private void Add(Constraint constraint)
        {
            if (pendingCount == 3)
            {
                Flush();
            }

            ((Constraint*)UnsafeUtility.AddressOf(ref pending.Constraints))[pendingCount++] = constraint;
        }

        private void Flush()
        {
            if (pendingCount > 0)
            {
                pending.Constraints.Length = (byte)pendingCount;
                Output.Write(pending);
                pendingCount = 0;
                emitted = true;
            }
        }

        private void AddHinge(NativeStore.JointData* joint)
        {
            Add(Constraint.BallAndSocket(ReportEveryImpulse));
            AddLockedAngles(new bool3(false, true, true), Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio, ReportEveryImpulse);

            if ((joint->state & NativeStore.JointState.HingeLimitEnabled) != 0)
            {
                NativeStore.JointAxis* axis = NativeStore.AxisOf(joint, NativeStore.AxisAngularX);
                Add(Constraint.Twist(0, new FloatRange(axis->min, axis->max), ReportEveryImpulse));
            }

            if (joint->hingeMotorMaxForce > 0.0f && !MotorCoasts(joint))
            {
                Add(Constraint.AngularVelocityMotor(joint->hingeMotorTargetVelocity, MaxImpulse(joint->hingeMotorMaxForce),
                    Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio));
            }
        }

        /// <summary>A free-spinning motor adds nothing to a hinge already turning faster than its target in the same direction.</summary>
        private bool MotorCoasts(NativeStore.JointData* joint)
        {
            if ((joint->state & NativeStore.JointState.HingeMotorFreeSpin) == 0)
            {
                return false;
            }

            float speed = math.dot(JointConstraints.RelativeAngularVelocity(joint), math.rotate(JointConstraints.WorldRotation0(joint), new float3(1.0f, 0.0f, 0.0f)));
            float target = joint->hingeMotorTargetVelocity;
            return target >= 0.0f ? speed >= target : speed <= target;
        }

        private void AddDistance(NativeStore.JointData* joint)
        {
            NativeStore.JointAxis* axis = NativeStore.AxisOf(joint, NativeStore.AxisDistance);
            Softness(axis->limitSpring, axis->limitDamping, JointConstraints.LinearMass(joint), false, out float frequency, out float damping);
            Add(Constraint.LimitedDistance(new FloatRange(axis->min, axis->max), ReportEveryImpulse, frequency, damping));
        }

        /// <summary>
        /// A drive too strong for any force to move, with nothing to aim at, is a lock. Stock part joints hold their position
        /// this way, and only a lock reports the force that breaks them.
        /// </summary>
        private static bool HeldByDrive(NativeStore.JointData* joint, int index)
        {
            NativeStore.JointAxis* axis = NativeStore.AxisOf(joint, NativeStore.AxisLinearX + index);
            return axis->driveSpring >= JointConstraints.UnlimitedForce && axis->driveMaxForce >= JointConstraints.UnlimitedForce
                && joint->targetPosition[index] == 0.0f && joint->targetLinearVelocity[index] == 0.0f;
        }

        private void AddLinearDof(NativeStore.JointData* joint)
        {
            bool3 locked = default;
            bool3 limited = default;
            for (int i = 0; i < 3; i++)
            {
                uint dofLock = NativeStore.AxisOf(joint, NativeStore.AxisLinearX + i)->dofLock;
                bool held = HeldByDrive(joint, i);
                locked[i] = dofLock == NativeStore.DofLocked || held;
                limited[i] = dofLock == NativeStore.DofLimited && !held;
            }

            if (math.any(locked))
            {
                Add(LinearAxes(locked, 0.0f, 0.0f, Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio));
            }

            if (math.any(limited))
            {
                // Unity's linear limit is one distance over the limited axes together, which is what these constraints measure.
                NativeStore.JointAxis* axis = NativeStore.AxisOf(joint, NativeStore.AxisDistance);
                Softness(axis->limitSpring, axis->limitDamping, JointConstraints.LinearMass(joint), false, out float frequency, out float damping);
                bool signed = limited.x ^ limited.y ^ limited.z && !math.all(limited);
                Add(LinearAxes(limited, signed ? -axis->max : axis->min, axis->max, frequency, damping));
            }
        }

        /// <summary>The package's linear constraints fix which axes they hold, and a joint's locks and limits can pick any of them.</summary>
        private static Constraint LinearAxes(bool3 axes, float min, float max, float frequency, float damping)
        {
            return new Constraint
            {
                ConstrainedAxes = axes,
                Type = ConstraintType.Linear,
                Min = min,
                Max = max,
                SpringFrequency = frequency,
                DampingRatio = damping,
                MaxImpulse = ReportEveryImpulse,
            };
        }

        private void AddAngularDof(NativeStore.JointData* joint)
        {
            bool3 locked = default;
            for (int i = 0; i < 3; i++)
            {
                locked[i] = NativeStore.AxisOf(joint, NativeStore.AxisAngularX + i)->dofLock == NativeStore.DofLocked;
            }

            if (math.any(locked))
            {
                AddLockedAngles(locked, Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio, ReportEveryImpulse);
            }

            NativeStore.JointAxis* twist = NativeStore.AxisOf(joint, NativeStore.AxisAngularX);
            if (twist->dofLock == NativeStore.DofLimited && (twist->min > -math.PI || twist->max < math.PI))
            {
                Softness(twist->limitSpring, twist->limitDamping, JointConstraints.AngularMass(joint, 0), false, out float frequency, out float damping);
                Add(Constraint.Twist(0, new FloatRange(twist->min, twist->max), ReportEveryImpulse, frequency, damping));
            }

            // The swing limit is the pair (angle about y, angle about z), in min and max.
            NativeStore.JointAxis* swing = NativeStore.AxisOf(joint, NativeStore.AxisAngularYZ);
            bool limitedY = NativeStore.AxisOf(joint, NativeStore.AxisAngularY)->dofLock == NativeStore.DofLimited && swing->min < math.PI;
            bool limitedZ = NativeStore.AxisOf(joint, NativeStore.AxisAngularZ)->dofLock == NativeStore.DofLimited && swing->max < math.PI;
            if (limitedY || limitedZ)
            {
                Softness(swing->limitSpring, swing->limitDamping, JointConstraints.AngularMass(joint, 1), false, out float frequency, out float damping);
                if (limitedY && limitedZ && swing->min == swing->max)
                {
                    Add(Constraint.Cone(0, new FloatRange(0.0f, swing->min), ReportEveryImpulse, frequency, damping));
                }
                else
                {
                    // Unity Physics has no ellipse, so unequal swing limits become a box inside the angles Unity sent.
                    if (limitedY)
                    {
                        Add(Constraint.Twist(1, new FloatRange(-swing->min, swing->min), ReportEveryImpulse, frequency, damping));
                    }

                    if (limitedZ)
                    {
                        Add(Constraint.Twist(2, new FloatRange(-swing->max, swing->max), ReportEveryImpulse, frequency, damping));
                    }
                }
            }
        }

        /// <summary>
        /// Holds the given rotation axes at zero with one 1D constraint each. The solver only accumulates a true torque for
        /// these: AngularLimit3DJacobian adds its error, not its impulse, and AngularLimit2DJacobian adds along an axis that
        /// follows the error, which sums to nothing meaningful. A joint's torque and its break test need that number.
        /// </summary>
        private void AddLockedAngles(bool3 locked, float frequency, float damping, float3 impulseThreshold)
        {
            for (int i = 0; i < 3; i++)
            {
                if (locked[i])
                {
                    Add(Constraint.Twist(i, new FloatRange(0.0f, 0.0f), impulseThreshold, frequency, damping));
                }
            }
        }

        private void AddLinearDrives(NativeStore.JointData* joint)
        {
            float mass = JointConstraints.LinearMass(joint);
            float3 current = JointConstraints.RelativePosition(joint);
            for (int i = 0; i < 3; i++)
            {
                NativeStore.JointAxis* axis = NativeStore.AxisOf(joint, NativeStore.AxisLinearX + i);
                if (axis->dofLock == NativeStore.DofLocked || HeldByDrive(joint, i) || axis->driveMaxForce <= 0.0f || (axis->driveSpring <= 0.0f && axis->driveDamper <= 0.0f))
                {
                    continue;
                }

                // F = k(x* - x) + c(v* - v) = k(x* + (c/k)v* - x) - cv, which is the one spring and damper a motor offers. A pure
                // damper keeps its spring negligible by aiming it from where the joint is now.
                float spring = math.max(axis->driveSpring, axis->driveDamper * PureDamperSpringRatio);
                float velocityOffset = axis->driveDamper / spring * joint->targetLinearVelocity[i];
                float target = (axis->driveSpring > 0.0f ? joint->targetPosition[i] : current[i]) + velocityOffset;

                Softness(spring, axis->driveDamper, mass, axis->driveUsesAcceleration != 0, out float frequency, out float damping);
                Add(OnAxis(i, Constraint.MotorPlanar(target, MaxImpulse(axis), frequency, damping)));
            }
        }

        private void AddAngularDrives(NativeStore.JointData* joint)
        {
            NativeStore.JointAxis* twist = NativeStore.AxisOf(joint, NativeStore.AxisAngularX);
            NativeStore.JointAxis* swing = NativeStore.AxisOf(joint, NativeStore.AxisAngularYZ);
            NativeStore.JointAxis* wholeRotation = JointConstraints.WholeRotationDrive(joint);
            if (wholeRotation != null)
            {
                AddSoftRotation(joint, wholeRotation);
            }
            else
            {
                AddAxisDrive(joint, 0, twist);
                AddAxisDrive(joint, 1, swing);
                AddAxisDrive(joint, 2, swing);
            }
        }

        /// <summary>
        /// A spring and damper that hold the whole rotation at its target, solved the way a locked joint is, unlike three
        /// separate motors. It is a drive, so its constraints report nothing.
        /// </summary>
        private void AddSoftRotation(NativeStore.JointData* joint, NativeStore.JointAxis* drive)
        {
            Flush();

            float spring = math.max(drive->driveSpring, drive->driveDamper * PureDamperSpringRatio);
            float3 velocityOffset = drive->driveDamper / spring * JointConstraints.TargetAngularVelocity(joint);
            quaternion target = math.mul(joint->targetRotation, quaternion.AxisAngle(math.normalizesafe(velocityOffset, new float3(1.0f, 0.0f, 0.0f)), math.length(velocityOffset)));

            pending.AFromJoint = new MTransform(math.mul(joint->rotation1, math.conjugate(target)), joint->position1);
            float mass = JointConstraints.AngularMass(joint, 1);
            Softness(spring, drive->driveDamper, mass, drive->driveUsesAcceleration != 0, out float frequency, out float damping);
            AddLockedAngles(new bool3(true), frequency, damping, Constraint.DefaultMaxImpulse);
            Flush();

            pending.AFromJoint = new MTransform(joint->rotation1, joint->position1);
        }

        private void AddAxisDrive(NativeStore.JointData* joint, int index, NativeStore.JointAxis* drive)
        {
            if (NativeStore.AxisOf(joint, NativeStore.AxisAngularX + index)->dofLock == NativeStore.DofLocked || drive->driveMaxForce <= 0.0f)
            {
                return;
            }

            float mass = JointConstraints.AngularMass(joint, index);
            if (drive->driveSpring > 0.0f)
            {
                float3 angles = JointConstraints.AngleAxis(joint->targetRotation);
                float offset = drive->driveDamper / drive->driveSpring * JointConstraints.TargetAngularVelocity(joint)[index];
                Softness(drive->driveSpring, drive->driveDamper, mass, drive->driveUsesAcceleration != 0, out float frequency, out float damping);
                Add(OnAxis(index, Constraint.MotorTwist(angles[index] + offset, MaxImpulse(drive), frequency, damping)));
            }
            else if (drive->driveDamper > 0.0f)
            {
                Softness(drive->driveDamper * PureDamperSpringRatio, drive->driveDamper, mass, drive->driveUsesAcceleration != 0, out float frequency, out float damping);
                Add(OnAxis(index, Constraint.AngularVelocityMotor(JointConstraints.TargetAngularVelocity(joint)[index], MaxImpulse(drive), frequency, damping)));
            }
        }

        /// <summary>
        /// Moves one of the package's motors, which all act along the joint's x axis, onto another. The motor's jacobian takes
        /// its axis from the constraint, and its impulse limit stays in the first slot whichever axis that is.
        /// </summary>
        private static Constraint OnAxis(int index, Constraint motor)
        {
            float target = motor.Target.x;
            motor.Target = float3.zero;
            motor.Target[index] = target;
            motor.ConstrainedAxes = default;
            motor.ConstrainedAxes[index] = true;
            return motor;
        }

        private float MaxImpulse(NativeStore.JointAxis* drive)
        {
            return drive->driveMaxForce >= JointConstraints.UnlimitedForce ? float.PositiveInfinity : MaxImpulse(drive->driveMaxForce);
        }

        /// <summary>The limit for the whole step, which the simulation divides among the substeps of a direct island.</summary>
        private float MaxImpulse(float force)
        {
            return force * DeltaTime;
        }

        /// <summary>
        /// A spring in N/m (or N m/rad) and a damper as the frequency and damping ratio Unity Physics asks for. A constraint
        /// has no mass of its own, so these hold for the mass given, which each step recomputes. A drive whose force is an
        /// acceleration behaves the same on any mass, which a mass of one expresses.
        /// </summary>
        private static void Softness(float spring, float damper, float mass, bool acceleration, out float frequency, out float dampingRatio)
        {
            frequency = Constraint.DefaultSpringFrequency;
            dampingRatio = Constraint.DefaultDampingRatio;
            if (spring <= 0.0f && damper <= 0.0f)
            {
                return;
            }

            float effectiveMass = acceleration ? 1.0f : mass;
            spring = math.max(spring, damper * PureDamperSpringRatio);
            float soft = math.sqrt(spring / effectiveMass) / (2.0f * math.PI);
            if (soft < Constraint.DefaultSpringFrequency)
            {
                frequency = soft;
                dampingRatio = damper / (2.0f * math.sqrt(spring * effectiveMass));
            }
        }
    }

    /// <summary>
    /// Lists what <see cref="BuildJointsJob"/> builds, one unit for each store joint outside a group and one for each
    /// group, in the store's order, which the built joints keep.
    /// </summary>
    [BurstCompile]
    internal unsafe struct GroupJointsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        public NativeList<JointGroup> Groups;
        public NativeList<JointGroupMember> Members;
        public NativeList<JointUnit> Units;

        /// <summary>Each unit's place in the vessel forest, or None for every unit when the forest is off.</summary>
        public NativeList<JointForestRole> Roles;

        public bool Forest;

        public void Execute()
        {
            Units.Clear();
            NativeArray<int> groupOf = FindGroups();
            for (int i = 0; i < World->jointCount; i++)
            {
                int group = groupOf[i];
                if (group < 0 || Groups[group].Leader == i)
                {
                    Units.Add(new JointUnit { Joint = i, Group = group });
                }
            }

            FindForest();
        }

        /// <summary>
        /// The vessel forest's spanning trees, by Kruskal over the units that hold their bodies as a part joint does
        /// (<see cref="BuildJointsJob.Mergeable"/>) in the store's order, which puts part joints before the struts added
        /// after them: a unit that joins two trees is a tree edge, and one whose bodies are already in one tree closes a
        /// loop. A unit held by something that doesn't move only touches one body, so it is an edge that joins nothing.
        /// Everything else is no part of the forest. Linear in the joints, like the grouping beside it.
        /// </summary>
        private void FindForest()
        {
            Roles.ResizeUninitialized(Units.Length);
            if (!Forest)
            {
                for (int u = 0; u < Units.Length; u++)
                {
                    Roles[u] = JointForestRole.None;
                }

                return;
            }

            int bodyCount = (int)World->bodyCount;
            var root = new NativeArray<int>(bodyCount, Allocator.Temp);
            for (int i = 0; i < bodyCount; i++)
            {
                root[i] = i;
            }

            for (int u = 0; u < Units.Length; u++)
            {
                NativeStore.JointData* joint = World->joints[Units[u].Joint];
                if (!BuildJointsJob.Mergeable(joint))
                {
                    Roles[u] = JointForestRole.None;
                    continue;
                }

                if (Immobile(joint->body0) || Immobile(joint->body1))
                {
                    Roles[u] = JointForestRole.Tree;
                    continue;
                }

                int a = Find(root, (int)joint->body0->index), b = Find(root, (int)joint->body1->index);
                if (a == b)
                {
                    Roles[u] = JointForestRole.Closure;
                    continue;
                }

                root[a] = b;
                Roles[u] = JointForestRole.Tree;
            }
        }

        private static bool Immobile(NativeStore.Body* body)
        {
            const NativeStore.BodyFlags immobile = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic;
            return body == null || (StoreOps.Hot(body)->flags & immobile) != 0;
        }

        private static int Find(NativeArray<int> root, int v)
        {
            while (root[v] != v)
            {
                root[v] = root[root[v]];
                v = root[v];
            }

            return v;
        }

        /// <summary>
        /// Gathers the joints that hold the same two bodies rigidly together, which the step solves as one: stock holds a
        /// stack node of size 2 or more with three joints between the same parts. Three joints carry 18 rows where 6 do, and
        /// those redundant rows make up most of a large craft's direct solve. A joint joins the first such joint found
        /// between its bodies when the two share their frames' orientation, and a group needs two joints.
        /// </summary>
        private NativeArray<int> FindGroups()
        {
            Groups.Clear();
            Members.Clear();
            int count = (int)World->jointCount;
            var groupOf = new NativeArray<int>(count, Allocator.Temp);
            var leaderOf = new NativeArray<int>(count, Allocator.Temp);
            var memberCount = new NativeArray<int>(count, Allocator.Temp);
            var leaders = new NativeHashMap<int2, int>(count, Allocator.Temp);
            for (int i = 0; i < count; i++)
            {
                groupOf[i] = -1;
                leaderOf[i] = -1;
                NativeStore.JointData* joint = World->joints[i];
                if (!BuildJointsJob.Mergeable(joint))
                {
                    continue;
                }

                var bodies = new int2(BuildJointsJob.BodyKey(joint->body0), BuildJointsJob.BodyKey(joint->body1));
                if (!leaders.TryGetValue(bodies, out int leader))
                {
                    leader = i;
                    leaders.Add(bodies, leader);
                }
                else if (!BuildJointsJob.SameFrames(World->joints[leader], joint))
                {
                    continue;
                }

                leaderOf[i] = leader;
                memberCount[leader]++;
            }

            // A leader comes before its other members, so its group exists by the time they are reached.
            for (int i = 0; i < count; i++)
            {
                int leader = leaderOf[i];
                if (leader < 0 || memberCount[leader] < 2)
                {
                    continue;
                }

                if (leader == i)
                {
                    groupOf[i] = Groups.Length;
                    Groups.Add(new JointGroup { Leader = i, FirstMember = Members.Length });
                    Members.ResizeUninitialized(Members.Length + memberCount[i]);
                }

                int group = groupOf[leader];
                groupOf[i] = group;
                JointGroup joined = Groups[group];
                Members[joined.FirstMember + joined.MemberCount] = new JointGroupMember { Joint = i };
                joined.MemberCount++;
                Groups[group] = joined;
            }

            return groupOf;
        }
    }

    /// <summary>Copies what <see cref="BuildJointsJob"/> wrote into the world's joints, unit by unit.</summary>
    [BurstCompile]
    internal struct GatherJointsJob : IJob
    {
        public NativeStream.Reader Input;
        public NativeList<PhysicsJoint> Output;

        public void Execute()
        {
            Output.ResizeUninitialized(Input.Count());
            int next = 0;
            for (int i = 0; i < Input.ForEachCount; i++)
            {
                int count = Input.BeginForEachIndex(i);
                for (int j = 0; j < count; j++)
                {
                    Output[next++] = Input.Read<PhysicsJoint>();
                }

                Input.EndForEachIndex();
            }
        }
    }

    /// <summary>What one lane of <see cref="BuildJointsJob"/> builds: a store joint alone, or the group it leads.</summary>
    internal struct JointUnit
    {
        public int Joint;

        /// <summary>The group the joint leads, or -1 for a joint built alone.</summary>
        public int Group;
    }

    /// <summary>
    /// Store joints between the same two bodies that the step builds as one, see <see cref="BuildJointsJob"/>. Stiffness is
    /// in multiples of the direct solve's cap on a rotation, per axis of the joint frame.
    /// </summary>
    internal struct JointGroup
    {
        /// <summary>The version of a built joint's entity whose index is a group, not a store joint.</summary>
        internal const int EntityVersion = 2;

        public int Leader;
        public int FirstMember;
        public int MemberCount;

        /// <summary>The cap on a linear row over the cap on a rotation.</summary>
        public float LinearRatio;

        public float3 Stiffness;
    }

    internal struct JointGroupMember
    {
        public int Joint;

        /// <summary>The member's anchor on body 0 from the group's, in the joint frame.</summary>
        public float3 Offset;

        public float3 AngularStiffness;
    }

    /// <summary>What the joint jobs ask of a store joint: its masses and the state of its two frames.</summary>
    internal static unsafe class JointConstraints
    {
        internal const NativeStore.JointState Inactive = NativeStore.JointState.Broken | NativeStore.JointState.BodyDestroyed;

        // Past this a drive's force limit can't matter, and a step's impulse couldn't be derived from it in a float. Stock joints
        // use 1e20 and Unity's default is float.MaxValue.
        internal const float UnlimitedForce = 1e15f;

        private static float InverseMass(NativeStore.Body* body)
        {
            if (body == null)
            {
                return 0.0f;
            }

            NativeStore.BodyData* hot = StoreOps.Hot(body);
            return (hot->flags & (NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Static)) != 0 ? 0.0f : hot->mass->InverseMass;
        }

        private static float InverseInertia(NativeStore.Body* body, float3 worldAxis)
        {
            if (body == null)
            {
                return 0.0f;
            }

            NativeStore.BodyData* hot = StoreOps.Hot(body);
            if ((hot->flags & (NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Static)) != 0)
            {
                return 0.0f;
            }

            float3 axis = math.rotate(math.conjugate(math.mul(hot->pose->Rotation, hot->mass->Transform.rot)), worldAxis);
            return math.csum(axis * axis * hot->mass->InverseInertia);
        }

        internal static float LinearMass(NativeStore.JointData* joint)
        {
            return 1.0f / math.max(InverseMass(joint->body0) + InverseMass(joint->body1), 1e-9f);
        }

        /// <summary>The inertia the joint moves about one of its axes, shared by its two bodies.</summary>
        internal static float AngularMass(NativeStore.JointData* joint, int axis)
        {
            float3 local = float3.zero;
            local[axis] = 1.0f;
            float3 world = math.rotate(WorldRotation0(joint), local);
            return 1.0f / math.max(InverseInertia(joint->body0, world) + InverseInertia(joint->body1, world), 1e-9f);
        }

        /// <summary>
        /// The drive that holds the whole rotation in one constraint set, or null when each axis needs a motor of its own.
        /// Unity leaves the other drives in place when it switches to slerp, so a slerp drive with any strength outranks them.
        /// The soft constraints can't honour a force limit, so one that matters takes the motors.
        /// </summary>
        internal static NativeStore.JointAxis* WholeRotationDrive(NativeStore.JointData* joint)
        {
            for (int i = 0; i < 3; i++)
            {
                if (NativeStore.AxisOf(joint, NativeStore.AxisAngularX + i)->dofLock == NativeStore.DofLocked)
                {
                    return null;
                }
            }

            NativeStore.JointAxis* slerp = NativeStore.AxisOf(joint, NativeStore.AxisSlerp);
            if (slerp->driveSpring > 0.0f || slerp->driveDamper > 0.0f)
            {
                return slerp;
            }

            NativeStore.JointAxis* twist = NativeStore.AxisOf(joint, NativeStore.AxisAngularX);
            NativeStore.JointAxis* swing = NativeStore.AxisOf(joint, NativeStore.AxisAngularYZ);
            bool sameDrive = twist->driveSpring == swing->driveSpring && twist->driveDamper == swing->driveDamper;
            bool noForceLimit = twist->driveMaxForce >= UnlimitedForce && swing->driveMaxForce >= UnlimitedForce;
            return sameDrive && noForceLimit && twist->driveSpring > 0.0f ? twist : null;
        }

        internal static quaternion WorldRotation0(NativeStore.JointData* joint)
        {
            return joint->body0 != null ? math.mul(StoreOps.Hot(joint->body0)->pose->Rotation, joint->rotation0) : joint->rotation0;
        }

        internal static quaternion WorldRotation1(NativeStore.JointData* joint)
        {
            return joint->body1 != null ? math.mul(StoreOps.Hot(joint->body1)->pose->Rotation, joint->rotation1) : joint->rotation1;
        }

        private static float3 WorldAnchor(NativeStore.Body* body, float3 local)
        {
            if (body == null)
            {
                return local;
            }

            NativeStore.BodyData* hot = StoreOps.Hot(body);
            return hot->pose->Position + math.rotate(hot->pose->Rotation, local);
        }

        /// <summary>Frame 1's origin in frame 0's axes, the coordinate a linear drive and its limits measure.</summary>
        internal static float3 RelativePosition(NativeStore.JointData* joint)
        {
            float3 difference = WorldAnchor(joint->body1, joint->position1) - WorldAnchor(joint->body0, joint->position0);
            return math.rotate(math.conjugate(WorldRotation0(joint)), difference);
        }

        /// <summary>Body 1's angular velocity relative to body 0, in world space.</summary>
        internal static float3 RelativeAngularVelocity(NativeStore.JointData* joint)
        {
            float3 first = joint->body0 != null ? StoreOps.WorldAngularVelocity(*StoreOps.Hot(joint->body0)) : float3.zero;
            float3 second = joint->body1 != null ? StoreOps.WorldAngularVelocity(*StoreOps.Hot(joint->body1)) : float3.zero;
            return second - first;
        }

        /// <summary>
        /// Frame 1's target angular velocity relative to frame 0. PhysX drives with the opposite sign of the value Unity
        /// sends for this one target, unlike the position, rotation and linear velocity.
        /// </summary>
        internal static float3 TargetAngularVelocity(NativeStore.JointData* joint)
        {
            return -joint->targetAngularVelocity;
        }

        /// <summary>A rotation as axis times angle, the shape in which each drive axis reads its share.</summary>
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
