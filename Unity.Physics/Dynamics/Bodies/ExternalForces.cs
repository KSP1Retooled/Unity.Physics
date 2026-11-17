using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;

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
                    (*hot).Forces()->Force += a->vector;
                    break;
                case ForceModeImpulse:
                    (*hot).Forces()->Impulse += a->vector;
                    break;
                case ForceModeVelocityChange:
                    (*hot).Forces()->VelocityChange += a->vector;
                    break;
                case ForceModeAcceleration:
                    (*hot).Forces()->Acceleration += a->vector;
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
                    (*hot).Forces()->Torque += a->vector;
                    break;
                case ForceModeImpulse:
                    (*hot).Forces()->AngularImpulse += a->vector;
                    break;
                case ForceModeVelocityChange:
                    (*hot).Forces()->AngularVelocityChange += a->vector;
                    break;
                case ForceModeAcceleration:
                    (*hot).Forces()->AngularAcceleration += a->vector;
                    break;
            }

            BodyKinematics.WakeIfAsked(hot, a);
        }

        internal static float3 AccumulatedForce(BodyData* hot, float deltaTime)
        {
            float perStep = deltaTime > 0.0f ? 1.0f / deltaTime : 0.0f;
            return (*hot).Forces()->Force + (*hot).Forces()->Acceleration * (*hot).BodyMass()->Mass + (*hot).Forces()->Impulse * perStep + (*hot).Forces()->VelocityChange * (*hot).BodyMass()->Mass * perStep;
        }

        /// <summary>
        /// Apply the principal moments and axes of inertia to an angular acceleration or velocity change.
        /// </summary>
        private static float3 ApplyInertia(BodyData* hot, float3 angular)
        {
            quaternion rotation = math.mul((*hot).Pose()->Rotation, (*hot).Mass()->Transform.rot);
            return math.rotate(rotation, math.rotate(math.conjugate(rotation), angular) * (*hot).BodyMass()->Inertia);
        }

        internal static float3 AccumulatedTorque(BodyData* hot, float deltaTime)
        {
            float perStep = deltaTime > 0.0f ? 1.0f / deltaTime : 0.0f;
            return (*hot).Forces()->Torque
                + ApplyInertia(hot, (*hot).Forces()->AngularAcceleration)
                + (*hot).Forces()->AngularImpulse * perStep
                + ApplyInertia(hot, (*hot).Forces()->AngularVelocityChange) * perStep;
        }
    }
}
