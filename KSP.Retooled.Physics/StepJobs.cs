using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Integrates what PhysX integrates before its solver: collected forces, gravity, damping and the velocity limits.
    /// The step passes zero gravity to Unity Physics because of this, so a body's velocity already holds the whole
    /// step's acceleration when contacts are solved, as under PhysX, and positions move by the semi-implicit
    /// <c>v * dt</c> that PhysX gives.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ApplyForcesJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Bodies;

        public float3 Gravity;
        public float DeltaTime;

        public void Execute(int index)
        {
            ref NativeStore.BodyData body = ref Bodies[index];
            const NativeStore.BodyFlags immobile = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic;
            if ((body.flags & immobile) != 0 || (body.flags & NativeStore.BodyFlags.Simulated) == 0)
            {
                ClearAccumulators(ref body);
                return;
            }

            if ((body.flags & NativeStore.BodyFlags.Sleeping) != 0)
            {
                return;
            }

            float3 linear = body.linearVelocity;
            linear += (body.force * body.inverseMass + body.acceleration) * DeltaTime;
            linear += body.impulse * body.inverseMass + body.velocityChange;
            if ((body.flags & NativeStore.BodyFlags.Gravity) != 0)
            {
                linear += Gravity * DeltaTime;
            }

            quaternion worldFromInertia = math.mul(body.rotation, body.inertiaRotation);
            float3 torqueImpulse = body.torque * DeltaTime + body.angularImpulse;
            float3 angular = body.angularVelocity + body.angularAcceleration * DeltaTime + body.angularVelocityChange;
            angular += math.rotate(worldFromInertia, math.rotate(math.conjugate(worldFromInertia), torqueImpulse) * body.inverseInertia);

            linear *= math.max(1.0f - body.linearDamping * DeltaTime, 0.0f);
            angular *= math.max(1.0f - body.angularDamping * DeltaTime, 0.0f);

            body.linearVelocity = ClampLength(linear, body.maxLinearVelocity);
            body.angularVelocity = ClampLength(angular, body.maxAngularVelocity);
            ClearAccumulators(ref body);
        }

        internal static float3 ClampLength(float3 value, float limit)
        {
            float squared = math.lengthsq(value);
            return squared > limit * limit ? value * (limit / math.sqrt(squared)) : value;
        }

        private static void ClearAccumulators(ref NativeStore.BodyData body)
        {
            body.force = float3.zero;
            body.torque = float3.zero;
            body.acceleration = float3.zero;
            body.angularAcceleration = float3.zero;
            body.impulse = float3.zero;
            body.angularImpulse = float3.zero;
            body.velocityChange = float3.zero;
            body.angularVelocityChange = float3.zero;
        }
    }

    /// <summary>
    /// Fills the built world's bodies and motions from the store. Dynamic bodies come first, as Unity Physics requires;
    /// kinematic and sleeping bodies are among them with infinite mass, and the last static is the world's anchor body.
    /// With a <see cref="DeltaTime"/> of zero the kinematic velocities stay zero, which is what a query world wants.
    /// </summary>
    [BurstCompile]
    internal unsafe struct BuildBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [ReadOnly]
        public NativeArray<int> DynamicIndices;

        [ReadOnly]
        public NativeArray<int> StaticIndices;

        public NativeArray<RigidBody> Bodies;
        public NativeArray<MotionData> MotionDatas;
        public NativeArray<MotionVelocity> MotionVelocities;

        public float DeltaTime;

        public void Execute()
        {
            for (int i = 0; i < DynamicIndices.Length; i++)
            {
                int index = DynamicIndices[i];
                ref NativeStore.BodyData body = ref Store[index];
                Bodies[i] = RigidBodyOf(ref body, index);
                WriteMotion(i, ref body);
            }

            for (int i = 0; i < StaticIndices.Length; i++)
            {
                int index = StaticIndices[i];
                Bodies[DynamicIndices.Length + i] = RigidBodyOf(ref Store[index], index);
            }

            Bodies[DynamicIndices.Length + StaticIndices.Length] = RigidBody.Zero;
        }

        private static RigidBody RigidBodyOf(ref NativeStore.BodyData body, int index)
        {
            return new RigidBody
            {
                Collider = ColliderBuilder.FromPointer(body.collider),
                WorldFromBody = new RigidTransform(body.rotation, body.position),
                Entity = new Entity { Index = index, Version = 1 },
                Scale = 1.0f,
            };
        }

        private void WriteMotion(int motionIndex, ref NativeStore.BodyData body)
        {
            RigidTransform bodyFromMotion = new RigidTransform(body.inertiaRotation, body.localCenterOfMass);
            RigidTransform worldFromMotion = math.mul(new RigidTransform(body.rotation, body.position), bodyFromMotion);

            bool kinematic = (body.flags & NativeStore.BodyFlags.Kinematic) != 0;
            bool sleeping = (body.flags & NativeStore.BodyFlags.Sleeping) != 0;
            bool heldInPlace = kinematic || sleeping;

            float3 linear = body.linearVelocity;
            float3 angular = math.rotate(math.conjugate(worldFromMotion.rot), body.angularVelocity);
            if (kinematic)
            {
                linear = float3.zero;
                angular = float3.zero;
                if ((body.flags & NativeStore.BodyFlags.HasKinematicTarget) != 0 && DeltaTime > 0.0f)
                {
                    linear = (body.targetPosition - body.position) / DeltaTime;
                    angular = math.rotate(math.conjugate(worldFromMotion.rot), AngularVelocityTo(body.rotation, body.targetRotation));
                }
            }

            float expansion = math.length(body.localCenterOfMass);
            if (body.collider != null)
            {
                expansion += ColliderBuilder.FromPointer(body.collider).Value.MassProperties.AngularExpansionFactor;
            }

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
                InverseInertia = heldInPlace ? float3.zero : body.inverseInertia,
                InverseMass = heldInPlace ? 0.0f : body.inverseMass,
                AngularExpansionFactor = expansion,
                GravityFactor = 0.0f,
            };
        }

        private float3 AngularVelocityTo(quaternion from, quaternion to)
        {
            quaternion delta = math.normalize(math.mul(to, math.conjugate(from)));
            if (delta.value.w < 0.0f)
            {
                delta.value = -delta.value;
            }

            float sine = math.length(delta.value.xyz);
            if (sine < 1e-6f)
            {
                return delta.value.xyz * 2.0f / DeltaTime;
            }

            return delta.value.xyz / sine * (2.0f * math.atan2(sine, delta.value.w)) / DeltaTime;
        }
    }

    /// <summary>Publishes the solved motions to the store as body poses and world-space velocities.</summary>
    [BurstCompile]
    internal unsafe struct WriteBackJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [ReadOnly]
        public NativeArray<int> DynamicIndices;

        [ReadOnly]
        public NativeArray<MotionData> MotionDatas;

        [ReadOnly]
        public NativeArray<MotionVelocity> MotionVelocities;

        public void Execute(int index)
        {
            ref NativeStore.BodyData body = ref Store[DynamicIndices[index]];
            if ((body.flags & NativeStore.BodyFlags.Sleeping) != 0)
            {
                return;
            }

            if ((body.flags & NativeStore.BodyFlags.Kinematic) != 0)
            {
                if ((body.flags & NativeStore.BodyFlags.HasKinematicTarget) != 0)
                {
                    body.position = body.targetPosition;
                    body.rotation = body.targetRotation;
                    body.flags &= ~NativeStore.BodyFlags.HasKinematicTarget;
                }

                return;
            }

            MotionData motion = MotionDatas[index];
            MotionVelocity velocity = MotionVelocities[index];
            RigidTransform worldFromBody = math.mul(motion.WorldFromMotion, math.inverse(motion.BodyFromMotion));

            body.position = worldFromBody.pos;
            body.rotation = math.normalize(worldFromBody.rot);
            body.linearVelocity = ApplyForcesJob.ClampLength(velocity.LinearVelocity, body.maxLinearVelocity);
            body.angularVelocity = ApplyForcesJob.ClampLength(math.rotate(motion.WorldFromMotion.rot, velocity.AngularVelocity), body.maxAngularVelocity);
        }
    }
}
