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
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Dynamics.Vessels.Detection;
using Unity.Physics.Collision.Filter.Groups;

namespace Unity.Physics.Dynamics.Joint.Translation
{
    internal static partial class JointTranslation
    {
        /// <summary>
        /// Build Unity Physics joints for store joints and groups, writing each unit to a separate stream section.
        /// Body A is body1 (connected); body B is body0 (owner), matching Unity's relative frame convention.
        /// </summary>
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
            public WorldData* World;

            /// <summary>
            /// Joint load states indexed by the world's joint list. Locking reads their smoothed loads.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public JointLoadState* States;

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

            /// <summary>The role of each unit in the vessel forest. See <see cref="JointGrouping.GroupJointsJob"/> .</summary>
            [ReadOnly]
            public NativeArray<JointForestRole> Roles;

            /// <summary>
            /// Enable load-based locking for forest pairs. Lock below LockFraction and remain locked below
            /// YieldFraction.
            /// Fully locked angular joints permit both until they break.
            /// </summary>
            public bool Locking;
            public float LockFraction;
            public float YieldFraction;

            /// <summary>
            /// Whether vessels are enabled for this step. The vessel solver handles joints in active vessels, so no Unity
            /// Physics joints are built for those joints.
            /// </summary>
            public bool Vessels;

            /// <summary>For each unit, the vessel that it belongs to, or -1.</summary>
            [ReadOnly]
            public NativeArray<int> UnitVessels;

            /// <summary>For each vessel, 1 when it takes part in this step.</summary>
            [ReadOnly]
            public NativeArray<byte> VesselActive;

            private SolverJoint pending;
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

                // The vessel solver handles the joints of active vessels, and DropIgnoredPairsJob stops their bodies from
                // colliding, so build nothing for them. FillVesselEdgesJob and FillVesselClosuresJob prepare their groups.
                if (Vessels && UnitVessels[index] >= 0 && VesselActive[UnitVessels[index]] != 0)
                {
                    Output.EndForEachIndex();
                    Written[index] = 0;
                    return;
                }

                role = Roles[index];
                forestLock = JointForestLock.None;
                if (Locking && role != JointForestRole.None)
                {
                    JointData* lead = World->joints[unit.Joint];
                    if (!JointGrouping.GroupJointsJob.Immobile(lead->body0) && !JointGrouping.GroupJointsJob.Immobile(lead->body1))
                    {
                        forestLock = UnitLock(unit);
                    }
                }
                if (unit.Group < 0)
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

            private void Write(in SolverJoint joint)
            {
                Output.Write(joint);
                written++;
            }

            /// <summary>Returns the locking permissions shared by all joints in a unit.</summary>
            private JointForestLock UnitLock(JointUnit unit)
            {
                if (unit.Group < 0)
                {
                    return JointLock(World->joints[unit.Joint], States + unit.Joint);
                }

                JointGroup group = Groups[unit.Group];
                JointForestLock allowed = JointForestLock.Lockable | JointForestLock.Holdable;
                for (int i = 0; i < group.MemberCount; i++)
                {
                    int joint = Members[group.FirstMember + i].Joint;
                    allowed &= JointLock(World->joints[joint], States + joint);
                }

                return allowed;
            }

            private JointForestLock JointLock(JointData* joint, JointLoadState* state)
            {
                int lockedAngles = 0;
                for (int i = 0; i < 3; i++)
                {
                    lockedAngles += JointData.AxisOf(joint, StoreConstants.AxisAngularX + i)->dofLock == StoreConstants.DofLocked ? 1 : 0;
                }

                if (lockedAngles == 3)
                {
                    return JointForestLock.Lockable | JointForestLock.Holdable;
                }

                float fraction = math.max(
                    state->BreakForce > 0.0f ? state->BreakingForce / state->BreakForce : 0.0f,
                    state->BreakTorque > 0.0f ? state->BreakingTorque / state->BreakTorque : 0.0f);
                return (fraction < LockFraction ? JointForestLock.Lockable : JointForestLock.None)
                    | (fraction < YieldFraction ? JointForestLock.Holdable : JointForestLock.None);
            }

            internal static int BodyKey(Body* body)
            {
                return body == null ? -1 : (int)body->index;
            }

            internal static bool SameFrames(JointData* first, JointData* second)
            {
                return math.all(first->rotation0.value == second->rotation0.value) && math.all(first->rotation1.value == second->rotation1.value)
                    && first->flags == second->flags;
            }

            /// <summary>
            /// Return whether joints between the same bodies can be merged.
            /// Require held anchors, compatible angular constraints and no conflicting targets or nearby limits.
            /// </summary>
            internal static bool Mergeable(JointData* joint)
            {
                if ((joint->state & JointConstraints.Inactive) != 0 || joint->type != StoreConstants.JointTypeDof6)
                {
                    return false;
                }

                int lockedAngles = 0;
                for (int i = 0; i < 3; i++)
                {
                    if (JointData.AxisOf(joint, StoreConstants.AxisLinearX + i)->dofLock != StoreConstants.DofLocked && !HeldByDrive(joint, i))
                    {
                        return false;
                    }

                    if (JointData.AxisOf(joint, StoreConstants.AxisAngularX + i)->dofLock == StoreConstants.DofLocked)
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
                    JointAxis* drive = AngularDrive(joint, i);
                    bool locked = JointData.AxisOf(joint, StoreConstants.AxisAngularX + i)->dofLock == StoreConstants.DofLocked;
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
            internal static JointAxis* AngularDrive(JointData* joint, int axis)
            {
                return JointData.AxisOf(joint, axis == 0 ? StoreConstants.AxisAngularX : StoreConstants.AxisAngularYZ);
            }

            /// <summary>
            /// Return whether an angular limit lies within UnreachedAngle of the rest orientation.
            /// </summary>
            private static bool HasNearAngularLimit(JointData* joint)
            {
                JointAxis* twist = JointData.AxisOf(joint, StoreConstants.AxisAngularX);
                JointAxis* swing = JointData.AxisOf(joint, StoreConstants.AxisAngularYZ);
                bool nearTwist = twist->dofLock == StoreConstants.DofLimited && (twist->min > -UnreachedAngle || twist->max < UnreachedAngle);
                bool nearY = JointData.AxisOf(joint, StoreConstants.AxisAngularY)->dofLock == StoreConstants.DofLimited && swing->min < UnreachedAngle;
                bool nearZ = JointData.AxisOf(joint, StoreConstants.AxisAngularZ)->dofLock == StoreConstants.DofLimited && swing->max < UnreachedAngle;
                return nearTwist || nearY || nearZ;
            }

            /// <summary>
            /// The angular distance from rest, in radians, that a merged group is assumed not to reach while resisting
            /// rotation.
            /// </summary>
            private const float UnreachedAngle = 0.5f;

            internal static bool HasAxisDrive(JointAxis* drive)
            {
                return drive->driveMaxForce > 0.0f && (drive->driveSpring > 0.0f || drive->driveDamper > 0.0f);
            }

            /// <summary>
            /// Build a group joint at the mean anchor with combined angular stiffness.
            /// Include rotational resistance from offset linear constraints.
            /// </summary>
            private void AppendGroup(int index)
            {
                JointGroup group = PrepareGroup(index, out float3 anchor0, out float3 anchor1);
                JointData* leader = World->joints[group.Leader];
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
            /// Prepare stiffness and member offsets for a vessel group so that ShareGroupLoadsJob can distribute its
            /// load.
            /// </summary>
            internal static void PrepareVesselGroup(WorldData* world, JointUnit unit, NativeArray<JointGroup> groups,
                NativeArray<JointGroupMember> members)
            {
                if (unit.Group >= 0)
                {
                    PrepareGroup(world, groups, members, unit.Group, out _, out _);
                }
            }

            private JointGroup PrepareGroup(int index, out float3 anchor0, out float3 anchor1)
            {
                return PrepareGroup(World, Groups, Members, index, out anchor0, out anchor1);
            }

            /// <summary>
            /// Compute group mean anchors, angular stiffness and member offsets.
            /// Express angular stiffness as multiples of the direct solver cap.
            /// </summary>
            private static JointGroup PrepareGroup(WorldData* world, NativeArray<JointGroup> groups, NativeArray<JointGroupMember> members,
                int index, out float3 anchor0, out float3 anchor1)
            {
                JointGroup group = groups[index];
                JointData* leader = world->joints[group.Leader];
                anchor0 = float3.zero;
                anchor1 = float3.zero;
                for (int i = 0; i < group.MemberCount; i++)
                {
                    JointData* member = world->joints[members[group.FirstMember + i].Joint];
                    anchor0 += member->position0;
                    anchor1 += member->position1;
                }

                anchor0 /= group.MemberCount;
                anchor1 /= group.MemberCount;

                float inverseMass = MovingInverseMass(leader->body0) + MovingInverseMass(leader->body1);
                float inverseInertia = MovingInverseInertia(leader->body0) + MovingInverseInertia(leader->body1);
                float angularCap = DirectStiffnessCap.Maximum(inverseInertia);
                group.LinearRatio = inverseInertia / math.max(inverseMass, 1e-9f);
                group.Stiffness = float3.zero;
                quaternion toJoint = math.conjugate(leader->rotation0);
                for (int i = 0; i < group.MemberCount; i++)
                {
                    JointGroupMember member = members[group.FirstMember + i];
                    JointData* joint = world->joints[member.Joint];
                    member.Offset = math.rotate(toJoint, joint->position0 - anchor0);
                    member.AngularStiffness = AngularStiffness(joint, angularCap);
                    group.Stiffness += member.AngularStiffness + group.LinearRatio * (math.lengthsq(member.Offset) - member.Offset * member.Offset);
                    members[group.FirstMember + i] = member;
                }

                groups[index] = group;
                return group;
            }

            /// <summary>
            /// Angular stiffness per axis as a fraction of the direct solver cap, limited to one.
            /// </summary>
            private static float3 AngularStiffness(JointData* joint, float angularCap)
            {
                JointAxis* wholeRotation = JointConstraints.WholeRotationDrive(joint);
                float3 stiffness = float3.zero;
                for (int i = 0; i < 3; i++)
                {
                    JointAxis* drive = wholeRotation != null ? wholeRotation : AngularDrive(joint, i);
                    if (JointData.AxisOf(joint, StoreConstants.AxisAngularX + i)->dofLock == StoreConstants.DofLocked)
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
            private static float MovingInverseMass(Body* body)
            {
                return HeldInPlace(body) ? 0.0f : (*StoreMemory.Row(body)).mass->InverseMass;
            }

            /// <summary>
            /// The mean inverse inertia across a body's axes, used to calculate the angular stiffness cap. Returns zero for
            /// bodies held in place.
            /// </summary>
            private static float MovingInverseInertia(Body* body)
            {
                return HeldInPlace(body) ? 0.0f : math.csum((*StoreMemory.Row(body)).mass->InverseInertia) / 3.0f;
            }

            private void Append(JointData* joint)
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
                    case StoreConstants.JointTypeFixed:
                        Add(Constraint.BallAndSocket(ReportEveryImpulse));
                        AddLockedAngles(new bool3(true), Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio, ReportEveryImpulse);
                        break;
                    case StoreConstants.JointTypeHinge:
                        AddHinge(joint);
                        break;
                    case StoreConstants.JointTypeBallAndSocket:
                        Add(Constraint.BallAndSocket(ReportEveryImpulse));
                        break;
                    case StoreConstants.JointTypeDistance:
                        AddDistance(joint);
                        break;
                    case StoreConstants.JointTypeDof6:
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
            /// Initialize joint construction. Return false if unavailable or both bodies are immovable;
            /// still emit an empty joint in the latter case to preserve collision settings.
            /// </summary>
            private bool Begin(JointData* joint, float3 anchor0, float3 anchor1, Entity entity)
            {
                int indexA = BuiltIndex(joint->body1);
                int indexB = BuiltIndex(joint->body0);
                if (indexA < 0 || indexB < 0 || (indexA >= DynamicCount && indexB >= DynamicCount))
                {
                    return false;
                }

                pending = new SolverJoint
                {
                    BodyPair = new BodyIndexPair { BodyIndexA = indexA, BodyIndexB = indexB },
                    AFromJoint = new MTransform(joint->rotation1, anchor1),
                    BFromJoint = new MTransform(joint->rotation0, anchor0),
                    Entity = entity,
                    EnableCollision = (joint->flags & StoreConstants.JointFlagCollisionEnabled) != 0,
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

            private static bool HeldInPlace(Body* body)
            {
                const BodyFlags held = BodyFlags.Static | BodyFlags.Kinematic | BodyFlags.Sleeping;
                return body == null || (StoreMemory.Row(body)->flags & held) != 0;
            }

            private int BuiltIndex(Body* body)
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

            private void AddHinge(JointData* joint)
            {
                Add(Constraint.BallAndSocket(ReportEveryImpulse));
                AddLockedAngles(new bool3(false, true, true), Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio, ReportEveryImpulse);

                if ((joint->state & JointState.HingeLimitEnabled) != 0)
                {
                    JointAxis* axis = JointData.AxisOf(joint, StoreConstants.AxisAngularX);
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
            private bool MotorCoasts(JointData* joint)
            {
                if ((joint->state & JointState.HingeMotorFreeSpin) == 0)
                {
                    return false;
                }

                float speed = math.dot(JointConstraints.RelativeAngularVelocity(joint), math.rotate(JointConstraints.WorldRotation0(joint), new float3(1.0f, 0.0f, 0.0f)));
                float target = joint->hingeMotorTargetVelocity;
                return target >= 0.0f ? speed >= target : speed <= target;
            }

            private void AddDistance(JointData* joint)
            {
                JointAxis* axis = JointData.AxisOf(joint, StoreConstants.AxisDistance);
                Softness(axis->limitSpring, axis->limitDamping, JointConstraints.LinearMass(joint), false, out float frequency, out float damping);
                Add(Constraint.LimitedDistance(new FloatRange(axis->min, axis->max), ReportEveryImpulse, frequency, damping));
            }

            /// <summary>
            /// Treat an unlimited linear drive with zero targets as a lock so that its force contributes to joint
            /// breaking.
            /// </summary>
            private static bool HeldByDrive(JointData* joint, int index)
            {
                JointAxis* axis = JointData.AxisOf(joint, StoreConstants.AxisLinearX + index);
                return axis->driveSpring >= JointConstraints.UnlimitedForce && axis->driveMaxForce >= JointConstraints.UnlimitedForce
                    && joint->targetPosition[index] == 0.0f && joint->targetLinearVelocity[index] == 0.0f;
            }

            private void AddLinearDof(JointData* joint)
            {
                bool3 locked = default;
                bool3 limited = default;
                for (int i = 0; i < 3; i++)
                {
                    uint dofLock = JointData.AxisOf(joint, StoreConstants.AxisLinearX + i)->dofLock;
                    bool held = HeldByDrive(joint, i);
                    locked[i] = dofLock == StoreConstants.DofLocked || held;
                    limited[i] = dofLock == StoreConstants.DofLimited && !held;
                }

                if (math.any(locked))
                {
                    Add(LinearAxes(locked, 0.0f, 0.0f, Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio));
                }

                if (math.any(limited))
                {
                    // Unity limits the combined distance along the limited axes. These constraints measure that distance.
                    JointAxis* axis = JointData.AxisOf(joint, StoreConstants.AxisDistance);
                    Softness(axis->limitSpring, axis->limitDamping, JointConstraints.LinearMass(joint), false, out float frequency, out float damping);
                    bool signed = limited.x ^ limited.y ^ limited.z && !math.all(limited);
                    Add(LinearAxes(limited, signed ? -axis->max : axis->min, axis->max, frequency, damping));
                }
            }

            /// <summary>
            /// Build a linear constraint for any locked or limited axis combination.
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

            private void AddAngularDof(JointData* joint)
            {
                bool3 locked = default;
                for (int i = 0; i < 3; i++)
                {
                    locked[i] = JointData.AxisOf(joint, StoreConstants.AxisAngularX + i)->dofLock == StoreConstants.DofLocked;
                }

                if (math.any(locked))
                {
                    AddLockedAngles(locked, Constraint.DefaultSpringFrequency, Constraint.DefaultDampingRatio, ReportEveryImpulse);
                }

                JointAxis* twist = JointData.AxisOf(joint, StoreConstants.AxisAngularX);
                if (twist->dofLock == StoreConstants.DofLimited && (twist->min > -math.PI || twist->max < math.PI))
                {
                    Softness(twist->limitSpring, twist->limitDamping, JointConstraints.AngularMass(joint, 0), false, out float frequency, out float damping);
                    Add(Constraint.Twist(0, new FloatRange(twist->min, twist->max), ReportEveryImpulse, frequency, damping));
                }

                // The shared swing limit stores the angle about y in min and the angle about z in max.
                JointAxis* swing = JointData.AxisOf(joint, StoreConstants.AxisAngularYZ);
                bool limitedY = JointData.AxisOf(joint, StoreConstants.AxisAngularY)->dofLock == StoreConstants.DofLimited && swing->min < math.PI;
                bool limitedZ = JointData.AxisOf(joint, StoreConstants.AxisAngularZ)->dofLock == StoreConstants.DofLimited && swing->max < math.PI;
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
            /// Use separate 1D constraints for locked angular axes so that accumulated impulses give usable torques for
            /// reporting and breaking.
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

            private void AddLinearDrives(JointData* joint)
            {
                float mass = JointConstraints.LinearMass(joint);
                float3 current = JointConstraints.RelativePosition(joint);
                for (int i = 0; i < 3; i++)
                {
                    JointAxis* axis = JointData.AxisOf(joint, StoreConstants.AxisLinearX + i);
                    if (axis->dofLock == StoreConstants.DofLocked || HeldByDrive(joint, i) || axis->driveMaxForce <= 0.0f || (axis->driveSpring <= 0.0f && axis->driveDamper <= 0.0f))
                    {
                        continue;
                    }

                    // Express target velocity as a spring target offset: F = k(x* + (c/k)v* - x) - cv.
                    // For a pure damper, target the current position before applying the offset.
                    float spring = math.max(axis->driveSpring, axis->driveDamper * PureDamperSpringRatio);
                    float velocityOffset = axis->driveDamper / spring * joint->targetLinearVelocity[i];
                    float target = (axis->driveSpring > 0.0f ? joint->targetPosition[i] : current[i]) + velocityOffset;

                    Softness(spring, axis->driveDamper, mass, axis->driveUsesAcceleration != 0, out float frequency, out float damping);
                    Add(OnAxis(i, Constraint.MotorPlanar(target, MaxImpulse(axis), frequency, damping)));
                }
            }

            private void AddAngularDrives(JointData* joint)
            {
                JointAxis* twist = JointData.AxisOf(joint, StoreConstants.AxisAngularX);
                JointAxis* swing = JointData.AxisOf(joint, StoreConstants.AxisAngularYZ);
                JointAxis* wholeRotation = JointConstraints.WholeRotationDrive(joint);
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
            /// Build whole-rotation spring and damper constraints at the target. Drives emit no impulse events.
            /// </summary>
            private void AddSoftRotation(JointData* joint, JointAxis* drive)
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

            private void AddAxisDrive(JointData* joint, int index, JointAxis* drive)
            {
                if (JointData.AxisOf(joint, StoreConstants.AxisAngularX + index)->dofLock == StoreConstants.DofLocked || drive->driveMaxForce <= 0.0f)
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
            /// Select the motor's joint axis. Keep the impulse limit in the first slot, as required by the motor
            /// Jacobian.
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

            private float MaxImpulse(JointAxis* drive)
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
            /// Convert stiffness and damping to frequency and damping ratio using this step's effective mass.
            /// Use unit mass for acceleration drives.
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
    }
}
