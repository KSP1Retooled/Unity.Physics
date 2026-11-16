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
    /// Updates every WheelCollider for one step, as PhysX does in its vehicle update before simulation. For each wheel,
    /// the job casts a ray down the suspension from the body's pose at the start of the step. It applies equal and
    /// opposite impulses to the vehicle body and any dynamic body beneath the wheel, along the ground normal. This job
    /// uses the built world after velocities and the broadphase have been prepared, and before the solver runs. All
    /// wheels are updated in one job because several wheels may push the same body.
    /// </summary>
    /// <remarks>
    /// The suspension force is <c>max(sprungMass * g + spring * jounce, 0) + damper * jounceSpeed</c>. Jounce is
    /// measured upward from the rest position that Unity sets through the target position. KSP's VehiclePhysics assumes
    /// this formula when it selects a target position to cancel the weight supported by the suspension. It also uses
    /// the reported hit force to recover the damping contribution. As in PhysX, only the spring contribution is clamped
    /// to zero. An extending damper pulls the body down, which helps keep a landing aircraft on its wheels. The game
    /// handles tyre friction and sets the stiffness of every friction curve to zero.
    /// <para>
    /// The suspension cannot compress beyond its stop. As in PhysX, a constraint between the body and the world
    /// prevents further compression. The constraint is added during the step in which the suspension would reach the
    /// stop, so that the solver stops the body there. If the constraint only corrected the body after it passed the
    /// stop, the resulting push would leave the body moving upward. The reported hit force includes the spring and
    /// damper contributions but excludes the force from this constraint. The game uses that reported force to calculate
    /// tyre grip.
    /// </para>
    /// </remarks>
    [BurstCompile]
    internal unsafe struct WheelJob : IJob
    {
        // This value is added to the speed along the wheel when calculating the slip angle. It keeps the angle finite
        // when the wheel is barely rolling.
        private const float SlipSpeedPadding = 1.0f;

        // This value is added to the speed of the wheel surface when calculating rolling slip, for the same reason.
        private const float RollSpeedPadding = 0.1f;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* Store;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        [ReadOnly]
        public CollisionWorld Collision;

        [ReadOnly]
        public NativeArray<MotionData> MotionDatas;

        public NativeArray<MotionVelocity> MotionVelocities;

        [ReadOnly]
        public NativeArray<int> BuiltIndices;

        /// <summary>
        /// Stores the joints for this step. Constraints that prevent suspensions from compressing past their stops are
        /// added here for this step only.
        /// </summary>
        public NativeList<PhysicsJoint> Joints;

        /// <summary>
        /// Identifies the static body at the origin of the built world. Joints use this body to represent an attachment
        /// to the world.
        /// </summary>
        public int AnchorIndex;

        public float DeltaTime;

        public void Execute()
        {
            float gravity = math.length(Store->gravity);
            for (uint v = 0; v < Store->vehicleCount; v++)
            {
                NativeStore.VehicleData* vehicle = Store->vehicles[v];
                if (vehicle != null && vehicle->body != null)
                {
                    Drive(vehicle, gravity);
                }
            }
        }

        private void Drive(NativeStore.VehicleData* vehicle, float gravity)
        {
            int row = (int)vehicle->body->index;
            NativeStore.BodyData* body = Store->bodies + row;
            int built = row < BuiltIndices.Length ? BuiltIndices[row] : -1;
            if (built < 0 || (body->flags & NativeStore.BodyFlags.Sleeping) != 0)
            {
                return;
            }

            RigidTransform pose = Collision.Bodies[built].WorldFromBody;
            float3 centreOfMass = math.transform(pose, body->mass->Transform.pos);
            float3 up = math.rotate(pose.rot, math.up());
            bool pushed = built < MotionVelocities.Length && MotionVelocities[built].InverseMass > 0.0f;

            NativeStore.WheelSlot* wheels = NativeStore.WheelsOf(vehicle);
            for (uint i = 0; i < vehicle->wheelCount; i++)
            {
                NativeStore.WheelSlot* wheel = wheels + i;
                if (wheel->enabled == 0)
                {
                    continue;
                }

                DriveWheel(vehicle, wheel, row, built, pose, centreOfMass, up, gravity, pushed);

                // A body with a driven wheel stays awake, as in PhysX. The game relies on this by applying a small
                // motor torque.
                if (wheel->motorTorque != 0.0f)
                {
                    body->sleep->WakeCounter = math.max(body->sleep->WakeCounter, StoreOps.WakeCounterReset);
                }
            }
        }

        private void DriveWheel(NativeStore.VehicleData* vehicle, NativeStore.WheelSlot* wheel, int row, int built, RigidTransform pose,
            float3 centreOfMass, float3 up, float gravity, bool pushed)
        {
            ref NativeStore.WheelSuspension suspension = ref wheel->suspension;
            float radius = wheel->wheel.radius;
            float3 top = centreOfMass + math.rotate(pose.rot, wheel->centreOffset) + up * suspension.maxCompression;
            float length = suspension.maxCompression + suspension.maxDroop + radius;

            // The wheel has no shape of its own, so its collision group comes from a lookup by its collider's id.
            ulong engineId = wheel->userData != null ? *(ulong*)((byte*)wheel->userData + Globals->engineObjectIdOffset) : 0;
            uint groupRecord = GroupTable.RecordOfCollider(Globals, engineId);
            var collector = new WheelRayCollector(Collision.Bodies, Store->bodies, Globals, row, wheel->collidesWith, wheel->belongsTo,
                wheel->filterIds >> 16, groupRecord, -up);
            Collision.CastRay(new RaycastInput
            {
                Start = top,
                End = top - up * length,
                Filter = new CollisionFilter { BelongsTo = uint.MaxValue, CollidesWith = wheel->collidesWith },
            }, ref collector);

            Spin(wheel);

            if (collector.Shape == null)
            {
                wheel->jounce = -suspension.maxDroop;
                wheel->grounded = 0;
                wheel->hit = default;
                wheel->hitShape = null;
                return;
            }

            RaycastHit hit = collector.Closest;
            float reach = suspension.maxCompression - (hit.Fraction * length - radius);
            float jounce = math.clamp(reach, -suspension.maxDroop, suspension.maxCompression);
            // Damping is calculated from the speed at which the body approaches the ground at the start of the step,
            // measured along the ground normal. Motion across the ground therefore does not contribute, regardless of
            // the body's tilt. Calculating the rate from the difference in jounce between steps would also count the
            // solver's corrections to joint positions as suspension travel. That can cause bouncing when a light part
            // with a wheel is attached to a heavy body.
            float3 normal = hit.SurfaceNormal;
            float3 own = vehicle->startLinear + math.cross(vehicle->startAngular, hit.Position - centreOfMass);
            float3 contactVelocity = own - VelocityAt(hit.RigidBodyIndex, hit.Position);
            float jounceSpeed = -math.dot(contactVelocity, normal) / math.max(math.dot(up, normal), 0.1f);
            bool bottomingOut = reach + math.max(jounceSpeed, 0.0f) * DeltaTime > suspension.maxCompression;
            if (reach > suspension.maxCompression)
            {
                // The spring cannot travel beyond the stop. Limit the compression speed used for damping to the travel
                // that remained before this step.
                jounceSpeed = math.min(jounceSpeed, (suspension.maxCompression - wheel->jounce) / DeltaTime);
            }

            float force = math.max(suspension.sprungMass * gravity + suspension.springStrength * jounce, 0.0f)
                + suspension.springDamperRate * jounceSpeed;
            wheel->jounce = jounce;
            wheel->grounded = 1;
            wheel->hitShape = collector.Shape->owner;

            float3 heading = math.rotate(pose.rot, math.rotate(quaternion.RotateY(wheel->steerAngle), new float3(0.0f, 0.0f, 1.0f)));
            float3 forward = math.normalizesafe(heading - normal * math.dot(heading, normal), heading);
            float3 sideways = math.cross(normal, forward);

            float along = math.dot(contactVelocity, forward);
            float across = math.dot(contactVelocity, sideways);
            float roll = wheel->rotationSpeed * radius;

            wheel->hit = new NativeStore.WheelQueryResult
            {
                point = hit.Position,
                normal = normal,
                forwardDir = forward,
                sidewaysDir = sideways,
                force = force,
                forwardSlip = (roll - along) / math.max(math.abs(along), math.abs(roll) + RollSpeedPadding),
                sidewaysSlip = math.atan(across / (math.abs(along) + SlipSpeedPadding)),
                suspensionCompression = jounce,
                steerAngle = wheel->steerAngle,
                userData = collector.Shape->userData,
            };

            float3 impulse = normal * (force * DeltaTime);
            if (pushed)
            {
                float3 point = centreOfMass + math.rotate(pose.rot, wheel->forceOffset);
                Push(built, point, impulse);
                if (bottomingOut)
                {
                    HoldAtFullCompression(built, pose, top, up, normal, reach - suspension.maxCompression);
                }
            }

            if (hit.RigidBodyIndex < MotionVelocities.Length && MotionVelocities[hit.RigidBodyIndex].InverseMass > 0.0f)
            {
                Push(hit.RigidBodyIndex, hit.Position, -impulse);
            }
        }

        /// <summary>
        /// Adds a one-sided constraint between the body and the world to prevent compression past the suspension stop.
        /// At full compression, the top of the suspension is one wheel radius above the ground, measured along the
        /// suspension axis. <paramref name="excess"/> is the distance that the body has moved past this position. A
        /// negative value means that this much travel remains.
        /// </summary>
        /// <remarks>
        /// The constraint acts along the ground normal, as a contact does when a wheel reaches full compression. If it
        /// acted along a tilted suspension instead, motion across the ground would move the body toward or away from
        /// the world anchor at a fraction of its forward speed. An aircraft landing nose up at 100 m/s would be thrown
        /// off the runway. An aircraft landing nose down would sink past the stop.
        /// </remarks>
        private void HoldAtFullCompression(int built, RigidTransform pose, float3 top, float3 up, float3 normal, float excess)
        {
            float3 side = math.normalize(math.cross(normal, math.abs(normal.y) < 0.9f ? new float3(0.0f, 1.0f, 0.0f) : new float3(1.0f, 0.0f, 0.0f)));
            var alongNormal = new float3x3(normal, side, math.cross(normal, side));

            var joint = new PhysicsJoint
            {
                BodyPair = new BodyIndexPair { BodyIndexA = built, BodyIndexB = AnchorIndex },
                AFromJoint = new MTransform(float3x3.identity, math.transform(math.inverse(pose), top)),
                BFromJoint = new MTransform(alongNormal, top + up * excess),
                Entity = Entity.Null,
            };
            ((Constraint*)UnsafeUtility.AddressOf(ref joint.Constraints))[0] = Constraint.Planar(0, new FloatRange(0.0f, float.MaxValue));
            joint.Constraints.Length = 1;
            Joints.Add(joint);
        }

        /// <summary>
        /// Updates wheel rotation using motor torque, brake torque, and damping. Rotation is not coupled to the ground
        /// because the game disables tyre friction and calculates wheel speeds itself.
        /// </summary>
        private void Spin(NativeStore.WheelSlot* wheel)
        {
            float inertia = wheel->wheel.momentOfInertia > 0.0f ? wheel->wheel.momentOfInertia : 1.0f;
            float speed = wheel->rotationSpeed + (wheel->motorTorque - wheel->wheel.dampingRate * wheel->rotationSpeed) / inertia * DeltaTime;
            float braked = math.abs(wheel->brakeTorque) / inertia * DeltaTime;
            speed = math.sign(speed) * math.max(math.abs(speed) - braked, 0.0f);
            wheel->rotationSpeed = speed;
            wheel->rotationAngle = math.fmod(wheel->rotationAngle + speed * DeltaTime, 2.0f * math.PI);
        }

        /// <summary>
        /// Returns the velocity at a point on a body in the built world. Static bodies have zero velocity and are
        /// stored after the moving bodies.
        /// </summary>
        private float3 VelocityAt(int built, float3 point)
        {
            if (built >= MotionVelocities.Length)
            {
                return float3.zero;
            }

            MotionVelocity motion = MotionVelocities[built];
            RigidTransform worldFromMotion = MotionDatas[built].WorldFromMotion;
            float3 angular = math.rotate(worldFromMotion.rot, motion.AngularVelocity);
            return motion.LinearVelocity + math.cross(angular, point - worldFromMotion.pos);
        }

        private void Push(int built, float3 point, float3 impulse)
        {
            MotionVelocity motion = MotionVelocities[built];
            RigidTransform worldFromMotion = MotionDatas[built].WorldFromMotion;
            float3 torque = math.cross(point - worldFromMotion.pos, impulse);
            motion.LinearVelocity += impulse * motion.InverseMass;
            motion.AngularVelocity += math.rotate(math.conjugate(worldFromMotion.rot), torque) * motion.InverseInertia;
            MotionVelocities[built] = motion;
        }
    }

    /// <summary>
    /// Collects the nearest hit that can support a wheel. It excludes the wheel's own body, triggers, shapes that are
    /// hidden from queries, and layers that do not collide with the wheel. It also excludes pairs disabled by
    /// Physics.IgnoreCollision, shapes that the wheel's collision group stops, and hits on the back of a polygon.
    /// </summary>
    internal unsafe struct WheelRayCollector : ICollector<RaycastHit>
    {
        internal RaycastHit Closest;
        internal NativeStore.ShapeData* Shape;

        private readonly NativeArray<RigidBody> bodies;
        private readonly NativeStore.BodyData* rows;
        private readonly NativeStore.StoreGlobals* globals;
        private readonly int ownRow;
        private readonly uint collidesWith;
        private readonly uint belongsTo;
        private readonly uint ignoreId;
        private readonly uint groupRecord;
        private readonly float3 direction;

        internal WheelRayCollector(NativeArray<RigidBody> bodies, NativeStore.BodyData* rows, NativeStore.StoreGlobals* globals, int ownRow,
            uint collidesWith, uint belongsTo, uint ignoreId, uint groupRecord, float3 direction)
        {
            this.bodies = bodies;
            this.rows = rows;
            this.globals = globals;
            this.ownRow = ownRow;
            this.collidesWith = collidesWith;
            this.belongsTo = belongsTo;
            this.ignoreId = ignoreId;
            this.groupRecord = groupRecord;
            this.direction = direction;
            Closest = default;
            Shape = null;
            MaxFraction = 1.0f;
            NumHits = 0;
        }

        public bool EarlyOutOnFirstHit => false;

        public float MaxFraction { get; private set; }

        public int NumHits { get; private set; }

        public bool AddHit(RaycastHit hit)
        {
            RigidBody body = bodies[hit.RigidBodyIndex];
            if (body.Entity.Version == 0 || body.Entity.Index == ownRow)
            {
                return false;
            }

            NativeStore.BodyData* row = rows + body.Entity.Index;
            NativeStore.ShapeData* shape = row->shapes[ColliderBuilder.ShapeIndexOf(body.Collider, hit.ColliderKey, ColliderBuilder.FirstSupported(row))];
            bool hidden = (shape->flags & NativeStore.ShapeFlagSceneQuery) == 0 || (shape->flags & NativeStore.ShapeFlagTrigger) != 0;
            if (hidden || (shape->belongsTo & collidesWith) == 0 || shape->collisionDisabled != 0)
            {
                return false;
            }

            // Unity sends a wheel a filter that collides with every layer, and PhysX applies the layer collision matrix
            // to the wheel's raycast itself. Unity does send each shape the layers that the matrix lets it collide with,
            // and it sends them again when the matrix changes. So the wheel skips a shape whose layers leave out the
            // wheel's own layer. A shape whose filter has no layers collides with every layer, see ColliderBuilder.
            if (belongsTo != 0 && shape->collidesWith != 0 && (shape->collidesWith & belongsTo) == 0)
            {
                return false;
            }

            if (IgnoreTable.Contains(globals, ignoreId, shape->ignorePairId) || GroupTable.Ignores(globals, groupRecord, shape))
            {
                return false;
            }

            if (QueryCollectors.IsBackface((Collider*)body.Collider.GetUnsafePtr(), hit.ColliderKey, body.WorldFromBody, direction))
            {
                return false;
            }

            Closest = hit;
            Shape = shape;
            MaxFraction = hit.Fraction;
            NumHits = 1;
            return true;
        }
    }
}
