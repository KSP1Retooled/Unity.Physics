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
                if ((body.flags & BodyFlags.ArticulationLink) != 0)
                {
                    return;
                }

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

                float3 linear = body.velocity->Linear;
                linear += (body.forces->Force * body.mass->InverseMass + body.forces->Acceleration) * DeltaTime;
                linear += body.forces->Impulse * body.mass->InverseMass + body.forces->VelocityChange;
                if ((body.flags & BodyFlags.Gravity) != 0)
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
                Interlocked.Increment(ref managedJobRuns);
            }

            internal static float3 ClampLength(float3 value, float limit)
            {
                float squared = math.lengthsq(value);
                return squared > limit * limit ? value * (limit / math.sqrt(squared)) : value;
            }

            private static void ClearAccumulators(ref BodyData body)
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
    }
}
