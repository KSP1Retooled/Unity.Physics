using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Physics.Math;
using SolverJoint = Unity.Physics.Joint;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Vehicles;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Collectors;

namespace Unity.Physics.Dynamics.Vehicles
{
    internal static unsafe partial class WheelSolver
    {
        /// <summary>
        /// Update WheelColliders before solving, applying suspension impulses to the vehicle and supporting dynamic
        /// bodies.
        /// Run in one job because wheels may push the same body.
        /// Use max(sprungMass*g + spring*jounce, 0) + damper*jounceSpeed; clamp only the spring contribution.
        /// Add a ground-normal constraint at full compression. Report spring/damper force without the stop constraint
        /// force.
        /// </summary>
        [BurstCompile]
        internal unsafe struct WheelJob : IJob
        {
            // This value is added to the speed along the wheel when calculating the slip angle. It keeps the angle finite
            // when the wheel is barely rolling.
            private const float SlipSpeedPadding = 1.0f;

            // This value is added to the speed of the wheel surface when calculating rolling slip, for the same reason.
            private const float RollSpeedPadding = 0.1f;

            [NativeDisableUnsafePtrRestriction]
            public WorldData* Store;

            [NativeDisableUnsafePtrRestriction]
            public StoreGlobals* Globals;

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
            public NativeList<SolverJoint> Joints;

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
                    VehicleData* vehicle = Store->vehicles[v];
                    if (vehicle != null && vehicle->body != null)
                    {
                        Drive(vehicle, gravity);
                    }
                }
            }

            private void Drive(VehicleData* vehicle, float gravity)
            {
                int row = (int)vehicle->body->index;
                BodyData* body = Store->bodies + row;
                int built = row < BuiltIndices.Length ? BuiltIndices[row] : -1;
                if (built < 0 || (body->flags & BodyFlags.Sleeping) != 0)
                {
                    return;
                }

                RigidTransform pose = Collision.Bodies[built].WorldFromBody;
                float3 centreOfMass = math.transform(pose, (*body).Mass()->Transform.pos);
                float3 up = math.rotate(pose.rot, math.up());
                bool pushed = built < MotionVelocities.Length && MotionVelocities[built].InverseMass > 0.0f;

                WheelSlot* wheels = VehicleData.WheelsOf(vehicle);
                for (uint i = 0; i < vehicle->wheelCount; i++)
                {
                    WheelSlot* wheel = wheels + i;
                    if (wheel->enabled == 0)
                    {
                        continue;
                    }

                    DriveWheel(vehicle, wheel, row, built, pose, centreOfMass, up, gravity, pushed);

                    // A body with a driven wheel stays awake, as in PhysX. The game relies on this by applying a small
                    // motor torque.
                    if (wheel->motorTorque != 0.0f)
                    {
                        (*body).Sleep()->WakeCounter = math.max((*body).Sleep()->WakeCounter, StoreMemory.WakeCounterReset);
                    }
                }
            }

            private void DriveWheel(VehicleData* vehicle, WheelSlot* wheel, int row, int built, RigidTransform pose,
                float3 centreOfMass, float3 up, float gravity, bool pushed)
            {
                ref WheelSuspension suspension = ref wheel->suspension;
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
                // Measure pre-step ground-normal velocity for damping; jounce differences would include solver
                // corrections.
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

                wheel->hit = new WheelQueryResult
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
            /// Add a one-sided ground-normal constraint preventing further suspension compression.
            /// excess is travel past the stop; a negative value is remaining travel.
            /// </summary>
            private void HoldAtFullCompression(int built, RigidTransform pose, float3 top, float3 up, float3 normal, float excess)
            {
                float3 side = math.normalize(math.cross(normal, math.abs(normal.y) < 0.9f ? new float3(0.0f, 1.0f, 0.0f) : new float3(1.0f, 0.0f, 0.0f)));
                var alongNormal = new float3x3(normal, side, math.cross(normal, side));

                var joint = new SolverJoint
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
            private void Spin(WheelSlot* wheel)
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
    }
}
