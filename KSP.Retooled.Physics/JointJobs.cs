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
    /// Sleeping bodies are treated as immovable during the step. Before the step, wake sleeping dynamic bodies that are
    /// connected to awake dynamic bodies by active joints, as PhysX does for an island. Repeat the pass until no more
    /// bodies wake. Depending on the order of joints, one pass can propagate waking across several joints in a chain.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WakeJointedBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        public void Execute()
        {
            // Only a sleeping body can be woken.
            bool anyAsleep = false;
            for (uint i = 0; i < World->bodyCount && !anyAsleep; i++)
            {
                anyAsleep = (World->bodies[i].flags & NativeStore.BodyFlags.Sleeping) != 0;
            }

            while (anyAsleep && WakeThroughJoints(World))
            {
            }
        }

        /// <summary>
        /// Makes one pass over the joints to wake connected bodies. The caller repeats the pass until no more bodies
        /// wake.
        /// </summary>
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
    /// Builds Unity Physics joints for each store joint or group found by <see cref="GroupJointsJob"/> . A single store
    /// joint may require several Unity Physics joints: each Unity Physics joint can contain three constraints, all
    /// using the same pair of frames. Each work item writes the joints for one unit to a separate section of the stream
    /// and records their count. <see cref="LayOutJointsJob"/> and <see cref="CopyJointsJob"/> then place those sections
    /// in order.
    /// </summary>
    /// <remarks>
    /// Body A of a built joint is the store's body1, the connected body. Body B is body0, the joint's owner. Unity
    /// sends limits for frame 1 relative to frame 0, while Unity Physics measures A relative to B, so this ordering
    /// matches the two conventions.
    /// </remarks>
    [BurstCompile]
    internal unsafe struct BuildJointsJob : IJobParallelForDefer
    {
        // The solver reports constraint impulses as events when they exceed a threshold. A zero threshold reports all
        // impulses.
        private static readonly float3 ReportEveryImpulse = float3.zero;

        // Unity Physics calculates damping strength from spring strength. To represent a pure damper, use a spring
        // whose effect is negligible compared with the damping.
        private const float PureDamperSpringRatio = 1e-3f;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<int> BuiltIndices;

        [ReadOnly]
        public NativeArray<JointUnit> Units;

        public NativeStream.Writer Output;

        /// <summary>The number of joints written for each unit.</summary>
        public NativeArray<int> Written;

        // Each work item writes only to its unit's group and the members of that group.
        [NativeDisableParallelForRestriction]
        public NativeArray<JointGroup> Groups;

        [NativeDisableParallelForRestriction]
        public NativeArray<JointGroupMember> Members;

        public int DynamicCount;
        public int AnchorIndex;
        public float DeltaTime;

        /// <summary>The role of each unit in the vessel forest. See <see cref="GroupJointsJob"/> .</summary>
        [ReadOnly]
        public NativeArray<JointForestRole> Roles;

        /// <summary>
        /// Whether locking is enabled for pairs in the forest. For each forest unit between two moving bodies,
        /// calculate whether the pair may lock or remain locked. The load used for breaking must be below
        /// <see cref="LockFraction"/> of the break limits to lock, or below <see cref="YieldFraction"/> to remain
        /// locked. A joint with all three angular axes locked permits both regardless of load, until it breaks. Loads
        /// change every step, so these permissions are calculated here rather than when the groups are built.
        /// </summary>
        public bool Locking;
        public float LockFraction;
        public float YieldFraction;

        /// <summary>
        /// Whether vessels are enabled for this step. The vessel solver handles joints in active vessels, so no Unity
        /// Physics constraints are built for those joints. A joint without constraints may still be needed to disable
        /// collisions. See <see cref="AppendVesselUnit"/> .
        /// </summary>
        public bool Vessels;

        /// <summary>For each unit, the vessel that it belongs to, or -1.</summary>
        [ReadOnly]
        public NativeArray<int> UnitVessels;

        /// <summary>For each vessel, 1 when it takes part in this step.</summary>
        [ReadOnly]
        public NativeArray<byte> VesselActive;

        private PhysicsJoint pending;
        private JointForestRole role;
        private JointForestLock forestLock;
        private int pendingCount;
        private bool emitted;
        private int written;

        public void Execute(int index)
        {
            Output.BeginForEachIndex(index);
            written = 0;
            JointUnit unit = Units[index];
            role = Roles[index];
            forestLock = JointForestLock.None;
            if (Locking && role != JointForestRole.None)
            {
                NativeStore.JointData* lead = World->joints[unit.Joint];
                if (!GroupJointsJob.Immobile(lead->body0) && !GroupJointsJob.Immobile(lead->body1))
                {
                    forestLock = UnitLock(unit);
                }
            }
            if (Vessels && UnitVessels[index] >= 0 && VesselActive[UnitVessels[index]] != 0)
            {
                AppendVesselUnit(unit);
            }
            else if (unit.Group < 0)
            {
                Append(World->joints[unit.Joint]);
            }
            else
            {
                AppendGroup(unit.Group);
            }

            Output.EndForEachIndex();
            Written[index] = written;
        }

        private void Write(in PhysicsJoint joint)
        {
            Output.Write(joint);
            written++;
        }

        /// <summary>Returns the locking permissions shared by all joints in a unit.</summary>
        private JointForestLock UnitLock(JointUnit unit)
        {
            if (unit.Group < 0)
            {
                return JointLock(World->joints[unit.Joint]);
            }

            JointGroup group = Groups[unit.Group];
            JointForestLock allowed = JointForestLock.Lockable | JointForestLock.Holdable;
            for (int i = 0; i < group.MemberCount; i++)
            {
                allowed &= JointLock(World->joints[Members[group.FirstMember + i].Joint]);
            }

            return allowed;
        }

        private JointForestLock JointLock(NativeStore.JointData* joint)
        {
            int lockedAngles = 0;
            for (int i = 0; i < 3; i++)
            {
                lockedAngles += NativeStore.AxisOf(joint, NativeStore.AxisAngularX + i)->dofLock == NativeStore.DofLocked ? 1 : 0;
            }

            if (lockedAngles == 3)
            {
                return JointForestLock.Lockable | JointForestLock.Holdable;
            }

            float fraction = math.max(
                joint->breakForce > 0.0f ? joint->breakingForce / joint->breakForce : 0.0f,
                joint->breakTorque > 0.0f ? joint->breakingTorque / joint->breakTorque : 0.0f);
            return (fraction < LockFraction ? JointForestLock.Lockable : JointForestLock.None)
                | (fraction < YieldFraction ? JointForestLock.Holdable : JointForestLock.None);
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
        /// Returns whether the joint can be merged with other joints between the same two bodies. Its anchors must be
        /// held together. Each angular axis may be locked, held at rest by a spring, or free; nearby angular limits and
        /// nonzero rotation or angular velocity targets prevent merging unless all angular axes are locked. Eligible
        /// joints can be represented together as one joint. A stock part joint locks its twist because Unity locks an
        /// axis when both limits are equal. Drives resist its swing.
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

        /// <summary>
        /// The drive for an angular axis: the twist drive for x, or the shared swing drive for y and z.
        /// </summary>
        internal static NativeStore.JointAxis* AngularDrive(NativeStore.JointData* joint, int axis)
        {
            return NativeStore.AxisOf(joint, axis == 0 ? NativeStore.AxisAngularX : NativeStore.AxisAngularYZ);
        }

        /// <summary>
        /// Returns whether a limit that <see cref="AddAngularDof"/> would build is within <see cref="UnreachedAngle"/>
        /// of the joint's rest orientation. Stock part joints have swing limits near half a turn. A joint that holds
        /// its anchors together is not expected to reach those limits.
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

        /// <summary>
        /// The angular distance from rest, in radians, that a merged group is assumed not to reach while resisting
        /// rotation.
        /// </summary>
        private const float UnreachedAngle = 0.5f;

        internal static bool HasAxisDrive(NativeStore.JointAxis* drive)
        {
            return drive->driveMaxForce > 0.0f && (drive->driveSpring > 0.0f || drive->driveDamper > 0.0f);
        }

        /// <summary>
        /// Builds one joint at the mean of the members' anchors, with their combined stiffness. The direct solver caps
        /// the stiffness of each row, so the merged rows use multiples of that cap. Each linear row uses one cap per
        /// member. For each angular row, add the members' angular stiffness and the resistance to rotation from their
        /// offset linear constraints. For small rotations, that additional stiffness is the linear cap times the
        /// squared perpendicular distance from the axis through the mean anchor.
        /// </summary>
        private void AppendGroup(int index)
        {
            JointGroup group = PrepareGroup(index, out float3 anchor0, out float3 anchor1);
            NativeStore.JointData* leader = World->joints[group.Leader];
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
        /// Processes one unit of a vessel that participates in this step. The vessel solver handles its constraints, so
        /// no Unity Physics joint is built unless collisions between its bodies must be disabled. In that case, build a
        /// joint with no constraints, which Unity Physics uses to disable the collisions. For a group, calculate its
        /// stiffness and the offsets of its members even when no joint is built. <see cref="ShareGroupLoadsJob"/> needs
        /// those values to distribute the group's load after the step.
        /// </summary>
        private void AppendVesselUnit(JointUnit unit)
        {
            NativeStore.JointData* leader = World->joints[unit.Joint];
            float3 anchor0 = leader->position0;
            float3 anchor1 = leader->position1;
            Entity entity = new Entity { Index = (int)leader->index, Version = 1 };
            if (unit.Group >= 0)
            {
                PrepareGroup(unit.Group, out anchor0, out anchor1);
                entity = new Entity { Index = unit.Group, Version = JointGroup.EntityVersion };
            }

            if ((leader->flags & NativeStore.JointFlagCollisionEnabled) != 0)
            {
                return;
            }

            int indexA = BuiltIndex(leader->body1);
            int indexB = BuiltIndex(leader->body0);
            pending = new PhysicsJoint
            {
                BodyPair = new BodyIndexPair { BodyIndexA = indexA, BodyIndexB = indexB },
                AFromJoint = new MTransform(leader->rotation1, anchor1),
                BFromJoint = new MTransform(leader->rotation0, anchor0),
                Entity = entity,
                EnableCollision = false,
            };
            pending.Constraints.Length = 0;
            Write(pending);
        }

        /// <summary>
        /// Calculates the mean anchors and angular stiffness for a group, and the offset and angular stiffness for each
        /// member. Angular stiffness is expressed as a multiple of the direct solver's cap. Stores the results in the
        /// group and member records.
        /// </summary>
        private JointGroup PrepareGroup(int index, out float3 anchor0, out float3 anchor1)
        {
            JointGroup group = Groups[index];
            NativeStore.JointData* leader = World->joints[group.Leader];
            anchor0 = float3.zero;
            anchor1 = float3.zero;
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
            return group;
        }

        /// <summary>
        /// The angular stiffness of a member for each axis, expressed as a multiple of the cap. Compare the spring
        /// stiffness supplied by Unity directly with the cap, limiting the result to one.
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

        /// <summary>
        /// The inverse mass used for a body during the step. Bodies held in place, including sleeping bodies, have zero
        /// inverse mass.
        /// </summary>
        private static float MovingInverseMass(NativeStore.Body* body)
        {
            return HeldInPlace(body) ? 0.0f : StoreOps.Hot(body)->mass->InverseMass;
        }

        /// <summary>
        /// The mean inverse inertia across a body's axes, used to calculate the angular stiffness cap. Returns zero for
        /// bodies held in place.
        /// </summary>
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

            // Unity Physics uses a joint to decide whether its bodies can collide even when the joint has no
            // constraints.
            if (!emitted && pendingCount == 0)
            {
                pending.Constraints.Length = 0;
                Write(pending);
            }

            Flush();
        }

        /// <summary>
        /// Initializes the joint before adding constraints. Returns false if the joint cannot be built or both bodies
        /// are held in place. In the latter case, writes a joint with no constraints to preserve its collision setting.
        /// </summary>
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
                ForestLock = forestLock,
            };
            pendingCount = 0;
            emitted = false;

            // Both bodies are held in place. Emit no constraints because the island may use the direct solver, which
            // rejects constraints between two immovable bodies.
            if (HeldInPlace(joint->body0) && HeldInPlace(joint->body1))
            {
                pending.Constraints.Length = 0;
                Write(pending);
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
                Write(pending);
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

        /// <summary>
        /// Returns whether a free-spinning hinge motor should apply no impulse because the hinge is already at or
        /// beyond its target speed in the target direction.
        /// </summary>
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
        /// Treats a linear drive as a lock when both its spring and force limit meet the unlimited threshold and its
        /// position and velocity targets are zero. Stock part joints hold their anchors together this way. Representing
        /// such a drive as a lock also reports the force needed for breaking; drive constraints do not report that
        /// force.
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
                // Unity limits the combined distance along the limited axes. These constraints measure that distance.
                NativeStore.JointAxis* axis = NativeStore.AxisOf(joint, NativeStore.AxisDistance);
                Softness(axis->limitSpring, axis->limitDamping, JointConstraints.LinearMass(joint), false, out float frequency, out float damping);
                bool signed = limited.x ^ limited.y ^ limited.z && !math.all(limited);
                Add(LinearAxes(limited, signed ? -axis->max : axis->min, axis->max, frequency, damping));
            }
        }

        /// <summary>
        /// Builds a linear constraint for any selection of axes. The package's constraint helpers use fixed selections,
        /// but a joint may lock or limit any combination.
        /// </summary>
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

            // The shared swing limit stores the angle about y in min and the angle about z in max.
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
                    // Unity Physics has no elliptical swing constraint. Use separate angular limits for unequal swing
                    // angles, forming a box from the angles that Unity supplied.
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
        /// Constrains each selected angular axis to zero with a separate 1D constraint. Only these constraints
        /// accumulate a usable torque. AngularLimit3DJacobian accumulates its error instead of its impulse.
        /// AngularLimit2DJacobian accumulates along an axis that changes with the error, so the sum does not represent
        /// a meaningful torque. The reported joint torque and the break test require a usable torque.
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

                // F = k(x* - x) + c(v* - v) = k(x* + (c/k)v* - x) - cv. This expresses the drive using the single
                // spring and damper supported by a motor, with a position offset for the target velocity. For a pure
                // damper, use the current joint position as the spring target before adding that offset, so the weak
                // spring has a negligible effect.
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
        /// Builds a spring and damper to hold the whole rotation at its target. Uses the same constraints as a locked
        /// joint rather than three separate motors. These are drive constraints, so they produce no impulse events.
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
        /// Changes a motor to act along the selected joint axis. The package's motors are initially configured for x.
        /// The motor Jacobian uses the axis selected by the constraint, while the impulse limit remains in the first
        /// slot regardless of the axis.
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

        /// <summary>
        /// The impulse limit for the whole step. The simulation divides it among the substeps of an island that uses
        /// the direct solver.
        /// </summary>
        private float MaxImpulse(float force)
        {
            return force * DeltaTime;
        }

        /// <summary>
        /// Converts spring stiffness in N/m (or N m/rad) and damping into the frequency and damping ratio required by
        /// Unity Physics. These values depend on the supplied effective mass, which is recalculated each step, because
        /// a constraint has no mass of its own. For an acceleration drive, use an effective mass of one so that the
        /// response is independent of body mass.
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
    /// Lists the units for <see cref="BuildJointsJob"/> : one unit for each ungrouped store joint and one for each
    /// group. Units follow store order, and the built joints retain that order.
    /// </summary>
    [BurstCompile]
    internal unsafe struct GroupJointsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        public NativeList<JointGroup> Groups;
        public NativeList<JointGroupMember> Members;
        public NativeList<JointUnit> Units;

        /// <summary>
        /// The role of each unit in the vessel forest, or None for all units when the forest is disabled.
        /// </summary>
        public NativeList<JointForestRole> Roles;

        /// <summary>Resized here for <see cref="BuildJointsJob.Written"/> ; values are written by that job.</summary>
        public NativeList<int> Written;

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
            Written.ResizeUninitialized(Units.Length);
        }

        /// <summary>
        /// Builds spanning trees for the vessel forest using Kruskal's algorithm. Processes mergeable units (
        /// <see cref="BuildJointsJob.Mergeable"/> ) in store order, so part joints precede struts that were added
        /// later. A unit that connects two trees becomes a tree edge. A unit between bodies already in the same tree
        /// becomes a closure. A unit connected to an immobile body is marked as a tree edge without joining trees.
        /// Other units are excluded from the forest. This takes approximately linear time in the number of joints, as
        /// does grouping.
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

        internal static bool Immobile(NativeStore.Body* body)
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
        /// Groups eligible joints between the same two bodies so that they can be solved as one joint. In stock KSP, a
        /// stack node of size 2 or greater uses three joints between the same parts. Those joints require 18 rows where
        /// 6 would suffice. Redundant rows account for most of the work in a large craft's direct solve. A joint is
        /// grouped with the first eligible joint found between its bodies if their frame orientations and flags match.
        /// A group must contain at least two joints.
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

            // The leader appears before the other members in store order, so its group has been created before those
            // members are processed.
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

    /// <summary>
    /// Calculates the start index for each unit's joints from the counts written by <see cref="BuildJointsJob"/> .
    /// Resizes the world's joint array to hold all of them.
    /// </summary>
    [BurstCompile]
    internal struct LayOutJointsJob : IJob
    {
        public NativeList<int> Written;
        public NativeList<PhysicsJoint> Output;

        public void Execute()
        {
            int next = 0;
            for (int i = 0; i < Written.Length; i++)
            {
                int count = Written[i];
                Written[i] = next;
                next += count;
            }

            Output.ResizeUninitialized(next);
        }
    }

    /// <summary>
    /// Copies the joints written for each unit by <see cref="BuildJointsJob"/> to the range assigned by
    /// <see cref="LayOutJointsJob"/> .
    /// </summary>
    [BurstCompile]
    internal unsafe struct CopyJointsJob : IJobParallelForDefer
    {
        [ReadOnly]
        public NativeStream.Reader Input;

        [ReadOnly]
        public NativeArray<int> FirstJoints;

        [NativeDisableParallelForRestriction]
        public NativeArray<PhysicsJoint> Output;

        public void Execute(int index)
        {
            NativeStream.Reader input = Input;
            int count = input.BeginForEachIndex(index);
            PhysicsJoint* output = (PhysicsJoint*)Output.GetUnsafePtr() + FirstJoints[index];
            for (int j = 0; j < count; j++)
            {
                output[j] = input.Read<PhysicsJoint>();
            }

            input.EndForEachIndex();
        }
    }

    /// <summary>
    /// The input for one work item in <see cref="BuildJointsJob"/> : a single store joint, or the group led by that
    /// joint.
    /// </summary>
    internal struct JointUnit
    {
        public int Joint;

        /// <summary>The group that the joint leads, or -1 if the joint is built alone.</summary>
        public int Group;
    }

    /// <summary>
    /// Store joints between the same two bodies that are built as one joint. See <see cref="BuildJointsJob"/> .
    /// Stiffness is stored for each axis of the joint frame, as a multiple of the direct solver's angular stiffness
    /// cap.
    /// </summary>
    internal struct JointGroup
    {
        /// <summary>
        /// The entity version used for a built joint whose entity index identifies a group rather than a store joint.
        /// </summary>
        internal const int EntityVersion = 2;

        public int Leader;
        public int FirstMember;
        public int MemberCount;

        /// <summary>The ratio of the linear stiffness cap to the angular stiffness cap.</summary>
        public float LinearRatio;

        public float3 Stiffness;
    }

    internal struct JointGroupMember
    {
        public int Joint;

        /// <summary>
        /// The offset from the group's anchor to the member's anchor on body 0, expressed in the joint frame.
        /// </summary>
        public float3 Offset;

        public float3 AngularStiffness;
    }

    /// <summary>Calculates effective masses and the state of the two joint frames for the joint jobs.</summary>
    internal static unsafe class JointConstraints
    {
        internal const NativeStore.JointState Inactive = NativeStore.JointState.Broken | NativeStore.JointState.BodyDestroyed;

        // Above this threshold, a drive's force limit has no practical effect, and calculating an impulse from it may
        // exceed the range of a float. Stock joints use 1e20; Unity's default is float.MaxValue.
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

        /// <summary>The effective inertia of the two bodies for rotation about one joint axis.</summary>
        internal static float AngularMass(NativeStore.JointData* joint, int axis)
        {
            float3 local = float3.zero;
            local[axis] = 1.0f;
            float3 world = math.rotate(WorldRotation0(joint), local);
            return 1.0f / math.max(InverseInertia(joint->body0, world) + InverseInertia(joint->body1, world), 1e-9f);
        }

        /// <summary>
        /// Returns a drive that can constrain the whole rotation together, or null if separate motors are needed for
        /// each axis. Unity leaves the other drives configured when switching to slerp, so a slerp drive with any
        /// strength takes precedence. Soft constraints cannot enforce a force limit. If the force limit matters, use
        /// motors instead.
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

        /// <summary>
        /// The position of frame 1's origin relative to frame 0's origin, expressed along frame 0's axes. Linear drives
        /// and limits measure this position.
        /// </summary>
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
        /// The target angular velocity of frame 1 relative to frame 0. PhysX uses the opposite sign to the value that
        /// Unity sends for this target. Position, rotation and linear velocity targets do not need this sign change.
        /// </summary>
        internal static float3 TargetAngularVelocity(NativeStore.JointData* joint)
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
