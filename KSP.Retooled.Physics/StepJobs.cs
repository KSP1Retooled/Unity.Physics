using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
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
            NoteManagedRun();

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

            float3 linear = body.velocity->Linear;
            linear += (body.forces->Force * body.mass->InverseMass + body.forces->Acceleration) * DeltaTime;
            linear += body.forces->Impulse * body.mass->InverseMass + body.forces->VelocityChange;
            if ((body.flags & NativeStore.BodyFlags.Gravity) != 0)
            {
                linear += Gravity * DeltaTime;
            }

            quaternion inertiaFromWorld = math.conjugate(math.mul(body.pose->Rotation, body.mass->Transform.rot));
            float3 torqueImpulse = body.forces->Torque * DeltaTime + body.forces->AngularImpulse;
            float3 acceleration = body.forces->AngularAcceleration * DeltaTime + body.forces->AngularVelocityChange;
            float3 angular = body.velocity->Angular + math.rotate(inertiaFromWorld, acceleration);
            angular += math.rotate(inertiaFromWorld, torqueImpulse) * body.mass->InverseInertia;

            linear *= math.max(1.0f - body.damping->Linear * DeltaTime, 0.0f);
            angular *= math.max(1.0f - body.damping->Angular * DeltaTime, 0.0f);

            body.velocity->Linear = ClampLength(linear, body.limits->MaxLinearVelocity);
            body.velocity->Angular = ClampLength(angular, body.limits->MaxAngularVelocity);
            ClearAccumulators(ref body);
        }

        /// <summary>Burst drops this, so the count only moves while a step runs the job's Mono code.</summary>
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

        private static void ClearAccumulators(ref NativeStore.BodyData body)
        {
            body.forces->Force = float3.zero;
            body.forces->Torque = float3.zero;
            body.forces->Acceleration = float3.zero;
            body.forces->AngularAcceleration = float3.zero;
            body.forces->Impulse = float3.zero;
            body.forces->AngularImpulse = float3.zero;
            body.forces->VelocityChange = float3.zero;
            body.forces->AngularVelocityChange = float3.zero;
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
            RigidTransform worldFromBody = new RigidTransform(body.pose->Rotation, body.pose->Position);
            NativeStore.ShapeData* posed = ColliderBuilder.PosedShape((NativeStore.BodyData*)UnsafeUtility.AddressOf(ref body));
            if (posed != null)
            {
                worldFromBody = math.mul(worldFromBody, new RigidTransform(posed->localRotation, posed->localPosition));
            }

            return new RigidBody
            {
                Collider = body.collider->Value,
                WorldFromBody = worldFromBody,
                Entity = new Entity { Index = index, Version = 1 },
                Scale = 1.0f,
            };
        }

        private void WriteMotion(int motionIndex, ref NativeStore.BodyData body)
        {
            RigidTransform bodyFromMotion = body.mass->Transform;
            RigidTransform worldFromMotion = math.mul(new RigidTransform(body.pose->Rotation, body.pose->Position), bodyFromMotion);

            bool kinematic = (body.flags & NativeStore.BodyFlags.Kinematic) != 0;
            bool sleeping = (body.flags & NativeStore.BodyFlags.Sleeping) != 0;
            bool heldInPlace = kinematic || sleeping;

            float3 linear = body.velocity->Linear;
            float3 angular = body.velocity->Angular;
            if (kinematic)
            {
                linear = float3.zero;
                angular = float3.zero;
                if ((body.flags & NativeStore.BodyFlags.HasKinematicTarget) != 0 && DeltaTime > 0.0f)
                {
                    // Motion velocities are those of the centre of mass, and a rotation about the body origin moves it.
                    float3 targetCenter = body.target->Position + math.rotate(body.target->Rotation, body.mass->Transform.pos);
                    linear = (targetCenter - worldFromMotion.pos) / DeltaTime;
                    quaternion turn = math.normalize(math.mul(body.target->Rotation, math.conjugate(body.pose->Rotation)));
                    angular = math.rotate(math.conjugate(worldFromMotion.rot), JointConstraints.AngleAxis(turn) / DeltaTime);
                }
            }

            float expansion = math.length(body.mass->Transform.pos);
            if (body.collider->Value.IsCreated)
            {
                expansion += body.collider->Value.Value.MassProperties.AngularExpansionFactor;
            }

            body.mass->AngularExpansionFactor = expansion;

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
                InverseInertia = heldInPlace ? float3.zero : body.mass->InverseInertia,
                InverseMass = heldInPlace ? 0.0f : body.mass->InverseMass,
                AngularExpansionFactor = expansion,
                GravityFactor = 0.0f,
            };
        }
    }

    /// <summary>Copies what a query reads about bodies and shapes out of the store, in the built world's body order.</summary>
    [BurstCompile]
    internal unsafe struct SnapshotQueryDataJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [ReadOnly]
        public NativeArray<RigidBody> Bodies;

        public NativeArray<QueryBody> QueryBodies;
        public NativeArray<QueryShape> QueryShapes;
        public uint EngineObjectIdOffset;

        public void Execute()
        {
            int nextShape = 0;
            for (int i = 0; i < Bodies.Length; i++)
            {
                Entity entity = Bodies[i].Entity;
                if (entity.Version == 0)
                {
                    QueryBodies[i] = default;
                    continue;
                }

                NativeStore.BodyData* body = Store + entity.Index;
                QueryBodies[i] = new QueryBody
                {
                    Owner = body->owner,
                    FirstShape = nextShape,
                    SingleShape = ColliderBuilder.FirstSupported(body),
                    IsStatic = (body->flags & NativeStore.BodyFlags.Static) != 0 ? 1 : 0,
                };

                for (int s = 0; s < body->shapeCount; s++)
                {
                    NativeStore.ShapeData* shape = body->shapes[s];
                    shape->owner->querySlot = (uint)nextShape + 1;
                    QueryShapes[nextShape++] = new QueryShape
                    {
                        EngineObjectId = shape->userData != null ? *(ulong*)((byte*)shape->userData + EngineObjectIdOffset) : 0,
                        UserData = shape->userData,
                        Handle = shape->owner,
                        Flags = shape->flags,
                        GeometryType = shape->geometryType,
                    };
                }
            }
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
                    body.pose->Position = body.target->Position;
                    StoreOps.SetRotation(body, body.target->Rotation);
                    body.flags &= ~NativeStore.BodyFlags.HasKinematicTarget;
                }

                return;
            }

            MotionData motion = MotionDatas[index];
            MotionVelocity velocity = MotionVelocities[index];
            RigidTransform worldFromBody = math.mul(motion.WorldFromMotion, math.inverse(motion.BodyFromMotion));

            body.pose->Position = worldFromBody.pos;
            body.pose->Rotation = math.normalize(worldFromBody.rot);
            body.velocity->Linear = ApplyForcesJob.ClampLength(velocity.LinearVelocity, body.limits->MaxLinearVelocity);
            body.velocity->Angular = ApplyForcesJob.ClampLength(velocity.AngularVelocity, body.limits->MaxAngularVelocity);
        }
    }
}
