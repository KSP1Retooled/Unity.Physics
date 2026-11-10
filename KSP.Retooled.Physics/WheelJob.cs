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
    /// Drives every WheelCollider for one step, as PhysX's vehicle update does before it simulates: each wheel casts down
    /// its suspension from the body's pose at the start of the step and pushes the body, and whatever dynamic body it stands
    /// on, apart along the ground's normal. It runs on the built world, after the bodies' velocities and the broadphase, and
    /// before anything solves. One job for all of them, because a wheel can push a body another wheel pushes too.
    /// </summary>
    /// <remarks>
    /// The suspension force is <c>max(sprungMass * g + spring * jounce, 0) + damper * jounceSpeed</c>, with the jounce
    /// measured up from the rest position Unity sets through the target position. That is the force KSP's VehiclePhysics
    /// assumes when it picks a target position to cancel the sprung weight and reads its damping back out of the hit's
    /// force. As in PhysX only the spring stops at zero: an extending damper pulls the body down, which keeps a landing
    /// aircraft on its wheels. Tyre friction is left to the game, which sets every curve's stiffness to zero.
    ///
    /// A suspension stops at its full compression, and what more the body asks of it is held by a limit the solver keeps,
    /// between the body and the world, as PhysX holds it. The limit joins the step the suspension would bottom out in, at
    /// the stop itself, so the solver stops the body there; a limit that only pushes the body back out once it is past
    /// leaves it rising at the speed it was pushed with. The hit's force leaves the limit out, so a wheel that bottoms out
    /// reports no more than the spring gives, which is what the game's tyres grip with.
    /// </remarks>
    [BurstCompile]
    internal unsafe struct WheelJob : IJob
    {
        // A slip angle's speed along the wheel is padded by this, so a wheel that is barely rolling gets a finite angle.
        private const float SlipSpeedPadding = 1.0f;

        // The same for the wheel's own surface speed in the rolling slip.
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

        /// <summary>The step's joints, which the limits of bottomed out suspensions join for this step only.</summary>
        public NativeList<PhysicsJoint> Joints;

        /// <summary>The built world's static body at the origin that stands for the world in a joint.</summary>
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

                // A driven wheel keeps its body awake, as in PhysX, which the game relies on with a token motor torque.
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

            var collector = new WheelRayCollector(Collision.Bodies, Store->bodies, Globals, row, wheel->collidesWith, wheel->filterIds >> 16, -up);
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
            // The damper works on how fast the body closes on the ground at the start of the step, along the ground's normal,
            // so that rolling across it counts for nothing however the body leans. A rate taken from the jounce two steps apart
            // also counts the solver pulling joints back into place as suspension travel, which on a light wheel part hanging
            // off a heavy body sets the suspension bouncing.
            float3 normal = hit.SurfaceNormal;
            float3 own = vehicle->startLinear + math.cross(vehicle->startAngular, hit.Position - centreOfMass);
            float3 contactVelocity = own - VelocityAt(hit.RigidBodyIndex, hit.Position);
            float jounceSpeed = -math.dot(contactVelocity, normal) / math.max(math.dot(up, normal), 0.1f);
            bool bottomingOut = reach + math.max(jounceSpeed, 0.0f) * DeltaTime > suspension.maxCompression;
            if (reach > suspension.maxCompression)
            {
                // Past the stop the spring travels no further, so the damper only sees what travel was left of it.
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
        /// Keeps the top of the suspension at least a wheel's radius above the ground along the suspension, which is where
        /// it sits at full compression, with a one-sided limit against the world. <paramref name="excess"/> is how far past
        /// that the body already is, and less than zero while it still has that much travel left.
        /// </summary>
        /// <remarks>
        /// The limit works along the ground's normal. A bottomed-out wheel bears on the ground like any contact, and along a
        /// leaning suspension a body rolling over the ground would close on the world anchor or leave it at a share of its
        /// speed: an aircraft touching down nose up at 100 m/s would be thrown off the runway, and one nose down would sink
        /// through the stop.
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
        /// Turns the wheel under its motor, brake and damping. Nothing couples the spin to the ground, because the game
        /// turns tyre friction off and works out its wheels' speeds itself.
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

        /// <summary>How fast a point of a built body moves. Static bodies, which come after the moving ones, stand still.</summary>
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
    /// The nearest hit a wheel stands on: not its own body, nor a trigger, a shape queries can't see, a layer the wheel
    /// doesn't collide with, a pair Physics.IgnoreCollision switched off, or the back of a polygon.
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
        private readonly uint ignoreId;
        private readonly float3 direction;

        internal WheelRayCollector(NativeArray<RigidBody> bodies, NativeStore.BodyData* rows, NativeStore.StoreGlobals* globals, int ownRow,
            uint collidesWith, uint ignoreId, float3 direction)
        {
            this.bodies = bodies;
            this.rows = rows;
            this.globals = globals;
            this.ownRow = ownRow;
            this.collidesWith = collidesWith;
            this.ignoreId = ignoreId;
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

            if (IgnoreTable.Contains(globals, ignoreId, shape->ignorePairId))
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
