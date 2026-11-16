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
    /// Updates velocities before the solver runs, as PhysX does, by applying accumulated forces, gravity, damping and
    /// velocity limits. The backend therefore passes zero gravity to Unity Physics. When the solver resolves contacts,
    /// the velocities already include the acceleration for the whole step. Positions are then updated by <c> v * dt
    /// </c> , using the semi-implicit integration that PhysX uses.
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
    /// Copies bodies and motion data from the store into the built world. Dynamic bodies come first, as Unity Physics
    /// requires. Kinematic and sleeping bodies are included among them with infinite mass. The last static body is the
    /// anchor for the world. When <see cref="DeltaTime"/> is zero, kinematic velocities remain zero, as required for a
    /// world built for queries.
    /// </summary>
    [BurstCompile]
    internal unsafe struct BuildBodiesJob : IJobParallelFor
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

        /// <summary>The number of built bodies, including the anchor.</summary>
        public int Count => DynamicIndices.Length + StaticIndices.Length + 1;

        public void Execute(int builtIndex)
        {
            if (builtIndex < DynamicIndices.Length)
            {
                int index = DynamicIndices[builtIndex];
                ref NativeStore.BodyData body = ref Store[index];
                Bodies[builtIndex] = RigidBodyOf(ref body, index);
                WriteMotion(builtIndex, ref body);
            }
            else if (builtIndex < DynamicIndices.Length + StaticIndices.Length)
            {
                int index = StaticIndices[builtIndex - DynamicIndices.Length];
                Bodies[builtIndex] = RigidBodyOf(ref Store[index], index);
            }
            else
            {
                Bodies[builtIndex] = RigidBody.Zero;
            }
        }

        /// <summary>
        /// Builds the Unity Physics body for a store row. The entity's index is the row, and its version is the body's
        /// serial number. The store gives a destroyed body's row to the next body that it makes, but it never gives two
        /// bodies the same serial number. So Unity Physics can tell two bodies apart by the whole entity, even when they
        /// had the same row: the island cache uses this to drop the lock states of a body that no longer exists. The
        /// version is never zero, because the code that reads these bodies treats a version of zero as a body that is
        /// not in the store. A row that a test builds by hand has no owner, and its version is 1.
        /// </summary>
        private static RigidBody RigidBodyOf(ref NativeStore.BodyData body, int index)
        {
            uint serial = body.owner != null ? body.owner->serial : 0u;
            return new RigidBody
            {
                Collider = body.collider->Value,
                WorldFromBody = WorldFromBody(ref body),
                Entity = new Entity { Index = index, Version = serial != 0 ? (int)serial : 1 },
                Scale = 1.0f,
            };
        }

        internal static RigidTransform WorldFromBody(ref NativeStore.BodyData body)
        {
            RigidTransform worldFromBody = new RigidTransform(body.pose->Rotation, body.pose->Position);
            NativeStore.ShapeData* posed = ColliderBuilder.PosedShape((NativeStore.BodyData*)UnsafeUtility.AddressOf(ref body));
            if (posed != null)
            {
                worldFromBody = math.mul(worldFromBody, new RigidTransform(posed->localRotation, posed->localPosition));
            }

            return worldFromBody;
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
                    // Motion velocities describe movement of the centre of mass. Rotating around the body origin also
                    // moves the centre of mass.
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

    /// <summary>
    /// Updates the poses of the built static bodies from the store when nothing else in the world has changed.
    /// </summary>
    [BurstCompile]
    internal unsafe struct PoseStaticsJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

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
    /// Calculates the first index for each built body in the list of query shapes, then resizes the list to hold all
    /// shapes.
    /// </summary>
    [BurstCompile]
    internal unsafe struct FindFirstShapesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [ReadOnly]
        public NativeArray<int> DynamicIndices;

        [ReadOnly]
        public NativeArray<int> StaticIndices;

        public NativeArray<int> FirstShapes;

        public NativeList<QueryShape> QueryShapes;

        public void Execute()
        {
            int count = 0;
            for (int i = 0; i < FirstShapes.Length; i++)
            {
                FirstShapes[i] = count;
                int row = StoreRow(DynamicIndices, StaticIndices, i);
                if (row >= 0)
                {
                    count += (int)Store[row].shapeCount;
                }
            }

            QueryShapes.ResizeUninitialized(count);
        }

        /// <summary>
        /// The store index of a built body, or -1 for the anchor. <see cref="BuildBodiesJob"/> builds the moving bodies
        /// first, then the static bodies, then the anchor. Reading the indices instead of the built bodies lets the
        /// snapshot run while the dynamic tree is rebuilt, which writes the built bodies.
        /// </summary>
        internal static int StoreRow(NativeArray<int> dynamicIndices, NativeArray<int> staticIndices, int builtIndex)
        {
            if (builtIndex < dynamicIndices.Length)
            {
                return dynamicIndices[builtIndex];
            }

            builtIndex -= dynamicIndices.Length;
            return builtIndex < staticIndices.Length ? staticIndices[builtIndex] : -1;
        }
    }

    /// <summary>
    /// Copies the body and shape data that queries need from the store. The data follows the order of bodies in the
    /// built world, with one body per work item. <see cref="FirstShapes"/> contains the index where each body's shapes
    /// begin.
    /// </summary>
    [BurstCompile]
    internal unsafe struct SnapshotQueryDataJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [ReadOnly]
        public NativeArray<int> DynamicIndices;

        [ReadOnly]
        public NativeArray<int> StaticIndices;

        [ReadOnly]
        public NativeArray<int> FirstShapes;

        public NativeArray<QueryBody> QueryBodies;

        [NativeDisableParallelForRestriction]
        public NativeArray<QueryShape> QueryShapes;

        public uint EngineObjectIdOffset;

        public void Execute(int i)
        {
            int row = FindFirstShapesJob.StoreRow(DynamicIndices, StaticIndices, i);
            if (row < 0)
            {
                QueryBodies[i] = default;
                return;
            }

            int nextShape = FirstShapes[i];
            NativeStore.BodyData* body = Store + row;
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

    /// <summary>
    /// Writes the solved body poses and velocities back to the store. Linear velocities are in world space; angular
    /// velocities are in the motion frame.
    /// </summary>
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
