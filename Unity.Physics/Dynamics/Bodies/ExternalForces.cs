using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using System.Threading;

namespace Unity.Physics.Dynamics.Bodies
{
    internal static unsafe partial class ExternalForces
    {
        // Physics::ForceMode.
        private const uint ForceModeForce = 0;
        private const uint ForceModeImpulse = 1;
        private const uint ForceModeVelocityChange = 2;
        private const uint ForceModeAcceleration = 5;

        private const BodyFlags Immobile = BodyFlags.Kinematic | BodyFlags.Static;

        internal static void AddForce(BodyData* hot, CommandArgs* a)
        {
            if ((hot->flags & Immobile) != 0)
            {
                return;
            }

            switch (a->mode)
            {
                case ForceModeForce:
                    hot->forces->Force += a->vector;
                    break;
                case ForceModeImpulse:
                    hot->forces->Impulse += a->vector;
                    break;
                case ForceModeVelocityChange:
                    hot->forces->VelocityChange += a->vector;
                    break;
                case ForceModeAcceleration:
                    hot->forces->Acceleration += a->vector;
                    break;
            }

            BodyKinematics.WakeIfAsked(hot, a);
        }

        internal static void AddTorque(BodyData* hot, CommandArgs* a)
        {
            if ((hot->flags & Immobile) != 0)
            {
                return;
            }

            switch (a->mode)
            {
                case ForceModeForce:
                    hot->forces->Torque += a->vector;
                    break;
                case ForceModeImpulse:
                    hot->forces->AngularImpulse += a->vector;
                    break;
                case ForceModeVelocityChange:
                    hot->forces->AngularVelocityChange += a->vector;
                    break;
                case ForceModeAcceleration:
                    hot->forces->AngularAcceleration += a->vector;
                    break;
            }

            BodyKinematics.WakeIfAsked(hot, a);
        }

        internal static float3 AccumulatedForce(BodyData* hot, float deltaTime)
        {
            float perStep = deltaTime > 0.0f ? 1.0f / deltaTime : 0.0f;
            return hot->forces->Force + hot->forces->Acceleration * hot->bodyMass->Mass + hot->forces->Impulse * perStep + hot->forces->VelocityChange * hot->bodyMass->Mass * perStep;
        }

        /// <summary>
        /// Apply the principal moments and axes of inertia to an angular acceleration or velocity change.
        /// </summary>
        private static float3 ApplyInertia(BodyData* hot, float3 angular)
        {
            quaternion rotation = math.mul(hot->pose->Rotation, hot->mass->Transform.rot);
            return math.rotate(rotation, math.rotate(math.conjugate(rotation), angular) * hot->bodyMass->Inertia);
        }

        internal static float3 AccumulatedTorque(BodyData* hot, float deltaTime)
        {
            float perStep = deltaTime > 0.0f ? 1.0f / deltaTime : 0.0f;
            return hot->forces->Torque
                + ApplyInertia(hot, hot->forces->AngularAcceleration)
                + hot->forces->AngularImpulse * perStep
                + ApplyInertia(hot, hot->forces->AngularVelocityChange) * perStep;
        }

        private static long managedJobRuns;

        internal static long ManagedJobRuns => Interlocked.Read(ref managedJobRuns);
    }
}
