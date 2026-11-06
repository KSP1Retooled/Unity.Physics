using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using static Unity.Physics.Math;
using PhysicsJoint = Unity.Physics.Joint;

namespace KSP.Retooled.PhysicsBackend
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
            bool woke = true;
            while (woke)
            {
                woke = false;
                for (uint i = 0; i < World->jointCount; i++)
                {
                    NativeStore.JointData* joint = World->joints[i];
                    if (joint->body0 == null || joint->body1 == null || (joint->state & JointConstraints.Inactive) != 0)
                    {
                        continue;
                    }

                    NativeStore.BodyData* first = StoreOps.Hot(joint->body0);
                    NativeStore.BodyData* second = StoreOps.Hot(joint->body1);
                    woke |= WakeFrom(first, second);
                    woke |= WakeFrom(second, first);
                }
            }
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
    /// Turns each store joint into the Unity Physics joints that express it. One of ours can take several: a Unity Physics
    /// joint holds three constraints, and all of them share one pair of frames.
    /// </summary>
    /// <remarks>
    /// Body A of a built joint is the store's body1, the connected body, and body B is body0, the joint's owner, because the
    /// limits Unity sends measure frame 1 relative to frame 0 and a Unity Physics constraint measures A relative to B.
    /// </remarks>
    [BurstCompile]
    internal unsafe struct BuildJointsJob : IJob
    {
        // The solver hands out a constraint's impulse only as an event past a threshold, so a threshold of zero reports them all.
        private static readonly float3 ReportEveryImpulse = float3.zero;

        // Unity Physics derives a damper's strength from a spring, so a pure damper gets one too weak to matter next to it.
        private const float PureDamperSpringRatio = 1e-3f;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<int> BuiltIndices;

        public NativeList<PhysicsJoint> Output;

        public int DynamicCount;
        public int AnchorIndex;
        public float DeltaTime;

        private PhysicsJoint pending;
        private int pendingCount;
        private bool emitted;

        public void Execute()
        {
            Output.Clear();
            for (uint i = 0; i < World->jointCount; i++)
            {
                Append(World->joints[i]);
            }
        }

        private void Append(NativeStore.JointData* joint)
        {
            if ((joint->state & JointConstraints.Inactive) != 0)
            {
                return;
            }

            int indexA = BuiltIndex(joint->body1);
            int indexB = BuiltIndex(joint->body0);
            if (indexA < 0 || indexB < 0 || (indexA >= DynamicCount && indexB >= DynamicCount))
            {
                return;
            }

            pending = new PhysicsJoint
            {
                BodyPair = new BodyIndexPair { BodyIndexA = indexA, BodyIndexB = indexB },
                AFromJoint = new MTransform(joint->rotation1, joint->position1),
                BFromJoint = new MTransform(joint->rotation0, joint->position0),
                Entity = new Entity { Index = (int)joint->index, Version = 1 },
                EnableCollision = (joint->flags & NativeStore.JointFlagCollisionEnabled) != 0,
            };
            pendingCount = 0;
            emitted = false;

            // A joint picks its solver here: SolverType on what this emits, with EnableDirectSolver on the built world and
            // DirectSolverSettings on the step input. Everything is iterative for now.
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
                Output.Add(pending);
            }

            Flush();
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
                Output.Add(pending);
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
                Add(AngularVelocityMotor(0, joint->hingeMotorTargetVelocity, joint->hingeMotorMaxForce * DeltaTime,
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

        private void AddLinearDof(NativeStore.JointData* joint)
        {
            bool3 locked = default;
            bool3 limited = default;
            for (int i = 0; i < 3; i++)
            {
                uint dofLock = NativeStore.AxisOf(joint, NativeStore.AxisLinearX + i)->dofLock;
                locked[i] = dofLock == NativeStore.DofLocked;
                limited[i] = dofLock == NativeStore.DofLimited;
            }

            if (math.any(locked))
            {
                Add(new Constraint
                {
                    ConstrainedAxes = locked,
                    Type = ConstraintType.Linear,
                    SpringFrequency = Constraint.DefaultSpringFrequency,
                    DampingRatio = Constraint.DefaultDampingRatio,
                    MaxImpulse = ReportEveryImpulse,
                });
            }

            if (math.any(limited))
            {
                // Unity's linear limit is one distance over the limited axes together, which is what these constraints measure.
                NativeStore.JointAxis* axis = NativeStore.AxisOf(joint, NativeStore.AxisDistance);
                Softness(axis->limitSpring, axis->limitDamping, JointConstraints.LinearMass(joint), false, out float frequency, out float damping);
                bool signed = limited.x ^ limited.y ^ limited.z && !math.all(limited);
                Add(new Constraint
                {
                    ConstrainedAxes = limited,
                    Type = ConstraintType.Linear,
                    Min = signed ? -axis->max : axis->min,
                    Max = axis->max,
                    SpringFrequency = frequency,
                    DampingRatio = damping,
                    MaxImpulse = ReportEveryImpulse,
                });
            }
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
                if (axis->dofLock == NativeStore.DofLocked || (axis->driveSpring <= 0.0f && axis->driveDamper <= 0.0f))
                {
                    continue;
                }

                // F = k(x* - x) + c(v* - v) = k(x* + (c/k)v* - x) - cv, which is the one spring and damper a motor offers. A pure
                // damper keeps its spring negligible by aiming it from where the joint is now.
                float spring = math.max(axis->driveSpring, axis->driveDamper * PureDamperSpringRatio);
                float velocityOffset = axis->driveDamper / spring * joint->targetLinearVelocity[i];
                float target = (axis->driveSpring > 0.0f ? joint->targetPosition[i] : current[i]) + velocityOffset;

                Softness(spring, axis->driveDamper, mass, axis->driveUsesAcceleration != 0, out float frequency, out float damping);
                float3 targets = float3.zero;
                targets[i] = target;
                bool3 axes = default;
                axes[i] = true;
                Add(new Constraint
                {
                    ConstrainedAxes = axes,
                    Type = ConstraintType.PositionMotor,
                    SpringFrequency = frequency,
                    DampingRatio = damping,
                    MaxImpulse = new float3(MaxImpulse(axis), 0.0f, 0.0f),
                    Target = targets,
                });
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
            if (NativeStore.AxisOf(joint, NativeStore.AxisAngularX + index)->dofLock == NativeStore.DofLocked)
            {
                return;
            }

            float mass = JointConstraints.AngularMass(joint, index);
            if (drive->driveSpring > 0.0f)
            {
                float3 angles = JointConstraints.AngleAxis(joint->targetRotation);
                float offset = drive->driveDamper / drive->driveSpring * JointConstraints.TargetAngularVelocity(joint)[index];
                Softness(drive->driveSpring, drive->driveDamper, mass, drive->driveUsesAcceleration != 0, out float frequency, out float damping);

                float3 targets = float3.zero;
                targets[index] = angles[index] + offset;
                bool3 axes = default;
                axes[index] = true;
                Add(new Constraint
                {
                    ConstrainedAxes = axes,
                    Type = ConstraintType.RotationMotor,
                    Min = float.NegativeInfinity,
                    Max = float.PositiveInfinity,
                    SpringFrequency = frequency,
                    DampingRatio = damping,
                    MaxImpulse = new float3(MaxImpulse(drive), 0.0f, 0.0f),
                    Target = targets,
                });
            }
            else if (drive->driveDamper > 0.0f)
            {
                Softness(drive->driveDamper * PureDamperSpringRatio, drive->driveDamper, mass, drive->driveUsesAcceleration != 0, out float frequency, out float damping);
                Add(AngularVelocityMotor(index, JointConstraints.TargetAngularVelocity(joint)[index], MaxImpulse(drive), frequency, damping));
            }
        }

        private static Constraint AngularVelocityMotor(int index, float target, float maxImpulse, float frequency, float damping)
        {
            float3 targets = float3.zero;
            targets[index] = target;
            bool3 axes = default;
            axes[index] = true;
            return new Constraint
            {
                ConstrainedAxes = axes,
                Type = ConstraintType.AngularVelocityMotor,
                SpringFrequency = frequency,
                DampingRatio = damping,
                MaxImpulse = new float3(maxImpulse, 0.0f, 0.0f),
                Target = targets,
            };
        }

        /// <summary>A motor's impulse accumulates over the whole step, substeps included, and starts over at the next.</summary>
        private float MaxImpulse(NativeStore.JointAxis* drive)
        {
            return drive->driveMaxForce >= JointConstraints.UnlimitedForce ? float.PositiveInfinity : drive->driveMaxForce * DeltaTime;
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
    /// Reads what the joints' constraints applied from the step's impulse events, which arrive once per constraint with the
    /// impulse of the whole step, then breaks the joints over their limits. A constraint's impulse acts on body A, the
    /// connected body, and Unity reports what acts on the joint's owner. Drives count for neither, as under PhysX, so they
    /// raise no events. A broken joint stops constraining from the next step and waits in the world's break list for
    /// whatever publishes the step's results.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ReadJointForcesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        public ImpulseEvents Events;
        public float DeltaTime;
        public uint EngineObjectIdOffset;

        public void Execute()
        {
            for (uint i = 0; i < World->jointCount; i++)
            {
                World->joints[i]->force = float3.zero;
                World->joints[i]->torque = float3.zero;
            }

            float scale = 1.0f / DeltaTime;
            foreach (ImpulseEvent impulse in Events)
            {
                NativeStore.JointData* joint = World->joints[impulse.JointEntity.Index];
                if (impulse.Type == ConstraintType.Linear)
                {
                    joint->force -= math.rotate(JointConstraints.WorldRotation0(joint), impulse.Impulse) * scale;
                }
                else
                {
                    joint->torque -= math.rotate(JointConstraints.WorldRotation1(joint), impulse.Impulse) * scale;
                }
            }

            World->jointBreakCount = 0;
            for (uint i = 0; i < World->jointCount; i++)
            {
                NativeStore.JointData* joint = World->joints[i];
                float force = math.length(joint->force);
                float torque = math.length(joint->torque);
                if ((joint->state & JointConstraints.Inactive) == 0 && (force > joint->breakForce || torque > joint->breakTorque))
                {
                    Break(joint, force, torque);
                }
            }
        }

        private void Break(NativeStore.JointData* joint, float force, float torque)
        {
            joint->state |= NativeStore.JointState.Broken;
            if (World->jointBreakCount == World->jointBreakCapacity)
            {
                World->jointBreakCapacity = StoreOps.Grown(World->jointBreakCapacity);
                World->jointBreaks = (NativeStore.JointBreak*)StoreOps.Reallocate(Globals, World->jointBreaks, World->jointBreakCapacity * (ulong)sizeof(NativeStore.JointBreak));
            }

            World->jointBreaks[World->jointBreakCount++] = new NativeStore.JointBreak
            {
                engineObjectId = joint->userData != null ? *(ulong*)((byte*)joint->userData + EngineObjectIdOffset) : 0,
                linearMagnitude = force,
                angularMagnitude = torque,
            };
        }
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
            return (hot->flags & (NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Static)) != 0 ? 0.0f : hot->inverseMass;
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

            float3 axis = math.rotate(math.conjugate(math.mul(hot->rotation, hot->inertiaRotation)), worldAxis);
            return math.csum(axis * axis * hot->inverseInertia);
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
            return joint->body0 != null ? math.mul(StoreOps.Hot(joint->body0)->rotation, joint->rotation0) : joint->rotation0;
        }

        internal static quaternion WorldRotation1(NativeStore.JointData* joint)
        {
            return joint->body1 != null ? math.mul(StoreOps.Hot(joint->body1)->rotation, joint->rotation1) : joint->rotation1;
        }

        private static float3 WorldAnchor(NativeStore.Body* body, float3 local)
        {
            if (body == null)
            {
                return local;
            }

            NativeStore.BodyData* hot = StoreOps.Hot(body);
            return hot->position + math.rotate(hot->rotation, local);
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
            float3 first = joint->body0 != null ? StoreOps.Hot(joint->body0)->angularVelocity : float3.zero;
            float3 second = joint->body1 != null ? StoreOps.Hot(joint->body1)->angularVelocity : float3.zero;
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
