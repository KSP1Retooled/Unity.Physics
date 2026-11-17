using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Dynamics.Joint.Translation;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Partition simulated bodies into dynamic and static lists and assign built indices to store rows.
    /// Run on the main thread because scheduling needs the counts.
    /// </summary>
    [BurstCompile]
    internal unsafe struct PartitionBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        public NativeList<int> DynamicIndices;
        public NativeList<int> StaticIndices;
        public NativeList<int> BuiltIndices;

        /// <summary>The flags of a dynamic body that receives its pose during write-back instead of from its motion.</summary>
        public BodyFlags PosedByWriteBack;

        /// <summary>Set to 1 if no dynamic body has a flag in <see cref="PosedByWriteBack"/>, and to 0 otherwise.</summary>
        public NativeReference<int> MotionsMatchStore;

        public void Execute()
        {
            int count = (int)World->bodyCount;
            DynamicIndices.Clear();
            StaticIndices.Clear();
            BuiltIndices.ResizeUninitialized(count);
            int matches = 1;
            for (int i = 0; i < count; i++)
            {
                BodyFlags flags = World->bodies[i].flags;
                BuiltIndices[i] = -1;
                if ((flags & BodyFlags.Simulated) == 0)
                {
                    continue;
                }

                if ((flags & BodyFlags.Static) != 0)
                {
                    BuiltIndices[i] = StaticIndices.Length;
                    StaticIndices.Add(i);
                }
                else
                {
                    BuiltIndices[i] = DynamicIndices.Length;
                    DynamicIndices.Add(i);
                    if ((flags & PosedByWriteBack) != 0)
                    {
                        matches = 0;
                    }
                }
            }

            int dynamicCount = DynamicIndices.Length;
            for (int s = 0; s < StaticIndices.Length; s++)
            {
                BuiltIndices[StaticIndices[s]] += dynamicCount;
            }

            MotionsMatchStore.Value = matches;
        }
    }

    /// <summary>
    /// Apply forces, gravity, damping and velocity limits before solving, matching PhysX semi-implicit integration.
    /// Pass zero gravity to Unity Physics to avoid applying it twice.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ApplyForcesJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public BodyData* Bodies;

        public float3 Gravity;
        public float DeltaTime;

        public void Execute(int index)
        {
            NoteManagedRun();

            ref BodyData body = ref Bodies[index];
            const BodyFlags immobile = BodyFlags.Static | BodyFlags.Kinematic;
            if ((body.flags & immobile) != 0 || (body.flags & BodyFlags.Simulated) == 0)
            {
                ClearAccumulators(ref body);
                return;
            }

            if ((body.flags & BodyFlags.Sleeping) != 0)
            {
                return;
            }

            float3 linear = body.Velocity()->Linear;
            linear += (body.Forces()->Force * body.Mass()->InverseMass + body.Forces()->Acceleration) * DeltaTime;
            linear += body.Forces()->Impulse * body.Mass()->InverseMass + body.Forces()->VelocityChange;
            if ((body.flags & BodyFlags.Gravity) != 0)
            {
                linear += Gravity * DeltaTime;
            }

            quaternion inertiaFromWorld = math.conjugate(math.mul(body.Pose()->Rotation, body.Mass()->Transform.rot));
            float3 torqueImpulse = body.Forces()->Torque * DeltaTime + body.Forces()->AngularImpulse;
            float3 acceleration = body.Forces()->AngularAcceleration * DeltaTime + body.Forces()->AngularVelocityChange;
            float3 angular = body.Velocity()->Angular + math.rotate(inertiaFromWorld, acceleration);
            angular += math.rotate(inertiaFromWorld, torqueImpulse) * body.Mass()->InverseInertia;

            linear *= math.max(1.0f - body.Damping()->Linear * DeltaTime, 0.0f);
            angular *= math.max(1.0f - body.Damping()->Angular * DeltaTime, 0.0f);

            body.Velocity()->Linear = ClampLength(linear, body.Limits()->MaxLinearVelocity);
            body.Velocity()->Angular = ClampLength(angular, body.Limits()->MaxAngularVelocity);
            ClearAccumulators(ref body);
        }

        /// <summary>Burst omits this method, so the counter increases only when the job runs in Mono.</summary>
        [BurstDiscard]
        private static void NoteManagedRun()
        {
            BackendHost.NoteManagedJobRun();
        }

        internal static float3 ClampLength(float3 value, float limit)
        {
            float squared = math.lengthsq(value);
            return squared > limit * limit ? value * (limit / math.sqrt(squared)) : value;
        }

        private static void ClearAccumulators(ref BodyData body)
        {
            body.Forces()->Force = float3.zero;
            body.Forces()->Torque = float3.zero;
            body.Forces()->Acceleration = float3.zero;
            body.Forces()->AngularAcceleration = float3.zero;
            body.Forces()->Impulse = float3.zero;
            body.Forces()->AngularImpulse = float3.zero;
            body.Forces()->VelocityChange = float3.zero;
            body.Forces()->AngularVelocityChange = float3.zero;
        }
    }

    /// <summary>
    /// Copy store bodies and motion data into the built world, with dynamic bodies first and the world anchor last.
    /// Treat sleeping and kinematic bodies as infinite mass. Use zero kinematic velocities when DeltaTime is zero.
    /// </summary>
    [BurstCompile]
    internal unsafe struct BuildBodiesJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public BodyData* Store;

        [ReadOnly]
        public NativeArray<int> DynamicIndices;

        [ReadOnly]
        public NativeArray<int> StaticIndices;

        public NativeArray<RigidBody> Bodies;
        public NativeArray<MotionData> MotionDatas;
        public NativeArray<MotionVelocity> MotionVelocities;

        public float DeltaTime;

        /// <summary>
        /// Optional per-body bounds cache flags. Set when CompareBodies is true and the overwritten body's pose and
        /// collider match exactly.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        public byte* Same;

        /// <summary>Whether the bodies that this job overwrites are the bodies that the bounds cache holds.</summary>
        public bool CompareBodies;

        /// <summary>
        /// Read serials and expansion factors from their original sources instead of cached row fields.
        /// Used by tests to verify identical built worlds.
        /// </summary>
        public bool ReadSources;

        /// <summary>
        /// Reuse retained static rows and the anchor, writing only dynamic bodies when true.
        /// </summary>
        public bool KeepStatics;

        /// <summary>
        /// Number of bodies to write. Include statics and the anchor unless KeepStatics is true.
        /// </summary>
        public int Count => KeepStatics ? DynamicIndices.Length : DynamicIndices.Length + StaticIndices.Length + 1;

        public void Execute(int builtIndex)
        {
            if (builtIndex < DynamicIndices.Length)
            {
                int index = DynamicIndices[builtIndex];
                ref BodyData body = ref Store[index];
                RigidBody built = RigidBodyOf(ref body, index, ReadSources);
                if (Same != null)
                {
                    // This job writes the body's memory anyway, so reading it first costs little.
                    Same[builtIndex] = CompareBodies && DynamicBoundsCache.View.Matches(Bodies[builtIndex], built) ? (byte)1 : (byte)0;
                }

                Bodies[builtIndex] = built;
                WriteMotion(builtIndex, ref body);
            }
            else if (builtIndex < DynamicIndices.Length + StaticIndices.Length)
            {
                int index = StaticIndices[builtIndex - DynamicIndices.Length];
                Bodies[builtIndex] = RigidBodyOf(ref Store[index], index, ReadSources);
            }
            else
            {
                Bodies[builtIndex] = RigidBody.Zero;
            }
        }

        /// <summary>Builds the Unity Physics body for a store row. See <see cref="EntityVersion"/> for its entity.</summary>
        private static RigidBody RigidBodyOf(ref BodyData body, int index, bool readSources)
        {
            return new RigidBody
            {
                Collider = body.Collider()->Value,
                WorldFromBody = WorldFromBody(ref body),
                Entity = new Entity { Index = index, Version = EntityVersion(ref body, index, readSources) },
                Scale = 1.0f,
            };
        }

        /// <summary>
        /// Return a stable, nonzero body identity from its serial number, independent of its store row.
        /// Ownerless test rows use negative identities derived from their indices. Read the cached serial unless
        /// readSources is true.
        /// </summary>
        private static int EntityVersion(ref BodyData body, int index, bool readSources)
        {
            uint serial = body.serial;
            if (readSources)
            {
                serial = body.owner != null ? body.owner->serial : 0u;
            }

            return serial != 0 ? (int)serial : -(index + 1);
        }

        internal static RigidTransform WorldFromBody(ref BodyData body)
        {
            RigidTransform worldFromBody = new RigidTransform(body.Pose()->Rotation, body.Pose()->Position);
            ShapeData* posed = ColliderBuilder.PosedShape((BodyData*)UnsafeUtility.AddressOf(ref body));
            if (posed != null)
            {
                worldFromBody = math.mul(worldFromBody, new RigidTransform(posed->localRotation, posed->localPosition));
            }

            return worldFromBody;
        }

        private void WriteMotion(int motionIndex, ref BodyData body)
        {
            RigidTransform bodyFromMotion = body.Mass()->Transform;
            RigidTransform worldFromMotion = math.mul(new RigidTransform(body.Pose()->Rotation, body.Pose()->Position), bodyFromMotion);

            bool kinematic = (body.flags & BodyFlags.Kinematic) != 0;
            bool sleeping = (body.flags & BodyFlags.Sleeping) != 0;
            bool heldInPlace = kinematic || sleeping;

            float3 linear = body.Velocity()->Linear;
            float3 angular = body.Velocity()->Angular;
            if (kinematic)
            {
                linear = float3.zero;
                angular = float3.zero;
                if ((body.flags & BodyFlags.HasKinematicTarget) != 0 && DeltaTime > 0.0f)
                {
                    // Motion velocities describe movement of the centre of mass. Rotating around the body origin also
                    // moves the centre of mass.
                    float3 targetCenter = body.Target()->Position + math.rotate(body.Target()->Rotation, body.Mass()->Transform.pos);
                    linear = (targetCenter - worldFromMotion.pos) / DeltaTime;
                    quaternion turn = math.normalize(math.mul(body.Target()->Rotation, math.conjugate(body.Pose()->Rotation)));
                    angular = math.rotate(math.conjugate(worldFromMotion.rot), JointConstraints.AngleAxis(turn) / DeltaTime);
                }
            }

            // The row holds a copy of the collider's expansion factor, which is 0 when the body has no collider. Adding
            // 0 to the length gives the same bits as not adding it, because the length is never -0.
            float colliderExpansion = body.colliderExpansion;
            if (ReadSources)
            {
                colliderExpansion = body.Collider()->Value.IsCreated ? body.Collider()->Value.Value.MassProperties.AngularExpansionFactor : 0.0f;
            }

            // The build doesn't write the result back into the body's PhysicsMass, because no code in this package reads it
            // there.
            float expansion = math.length(body.Mass()->Transform.pos) + colliderExpansion;

            MotionDatas[motionIndex] = new MotionData
            {
                WorldFromMotion = worldFromMotion,
                BodyFromMotion = bodyFromMotion,
                LinearDamping = 0.0f,
                AngularDamping = 0.0f,
            };

            MotionVelocities[motionIndex] = new MotionVelocity
            {
                LinearVelocity = linear,
                AngularVelocity = angular,
                InverseInertia = heldInPlace ? float3.zero : body.Mass()->InverseInertia,
                InverseMass = heldInPlace ? 0.0f : body.Mass()->InverseMass,
                AngularExpansionFactor = expansion,
                GravityFactor = 0.0f,
            };
        }
    }

    /// <summary>
    /// Updates the poses of the built static bodies from the store when nothing else in the world has changed.
    /// </summary>
    [BurstCompile]
    internal unsafe struct PoseStaticsJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public BodyData* Store;

        [ReadOnly]
        public NativeArray<int> StaticIndices;

        /// <summary>The static bodies in the built world, in the order of <see cref="StaticIndices"/> .</summary>
        public NativeArray<RigidBody> StaticBodies;

        public void Execute(int i)
        {
            RigidBody body = StaticBodies[i];
            body.WorldFromBody = BuildBodiesJob.WorldFromBody(ref Store[StaticIndices[i]]);
            StaticBodies[i] = body;
        }
    }

    /// <summary>
    /// Writes the solved body poses and velocities back to the store. Linear velocities are in world space; angular
    /// velocities are in the motion frame.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WriteBackJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public BodyData* Store;

        [ReadOnly]
        public NativeArray<int> DynamicIndices;

        [ReadOnly]
        public NativeArray<MotionData> MotionDatas;

        [ReadOnly]
        public NativeArray<MotionVelocity> MotionVelocities;

        public void Execute(int index)
        {
            ref BodyData body = ref Store[DynamicIndices[index]];
            if ((body.flags & BodyFlags.Sleeping) != 0)
            {
                return;
            }

            if ((body.flags & BodyFlags.Kinematic) != 0)
            {
                if ((body.flags & BodyFlags.HasKinematicTarget) != 0)
                {
                    body.Pose()->Position = body.Target()->Position;
                    BodyKinematics.SetRotation(body, body.Target()->Rotation);
                    body.flags &= ~BodyFlags.HasKinematicTarget;
                }

                return;
            }

            MotionVelocity velocity = MotionVelocities[index];
            BodyKinematics.SolvedPose(MotionDatas[index], out body.Pose()->Position, out body.Pose()->Rotation);
            body.Velocity()->Linear = ApplyForcesJob.ClampLength(velocity.LinearVelocity, body.Limits()->MaxLinearVelocity);
            body.Velocity()->Angular = ApplyForcesJob.ClampLength(velocity.AngularVelocity, body.Limits()->MaxAngularVelocity);
        }

    }
}
