using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Dynamics.Joint.Translation;
using System.Threading;

namespace Unity.Physics.Dynamics.Bodies
{
    internal static unsafe partial class ExternalForces
    {
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
                Interlocked.Increment(ref managedJobRuns);
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
    }
}
