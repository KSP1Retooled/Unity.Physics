using Unity.Mathematics;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;

namespace Unity.Physics.Dynamics.Bodies
{
    internal static unsafe class BodyKinematics
    {
        /// <summary>
        /// Returns a body's angular velocity in world space. PhysicsVelocity stores angular velocity relative to the
        /// principal axes of inertia, but Rigidbody callers use world space.
        /// </summary>
        internal static float3 WorldAngularVelocity(in BodyData body)
        {
            return math.rotate(math.mul(body.Pose()->Rotation, body.Mass()->Transform.rot), body.Velocity()->Angular);
        }

        internal static void SetWorldAngularVelocity(in BodyData body, float3 angular)
        {
            quaternion worldFromInertia = math.mul(body.Pose()->Rotation, body.Mass()->Transform.rot);
            body.Velocity()->Angular = math.rotate(math.conjugate(worldFromInertia), angular);
        }

        /// <summary>
        /// Changes a body's rotation while preserving its angular velocity in world space. The angular velocity stored
        /// relative to the principal axes is adjusted to match.
        /// </summary>
        internal static void SetRotation(in BodyData body, quaternion rotation)
        {
            float3 angular = WorldAngularVelocity(body);
            body.Pose()->Rotation = rotation;
            SetWorldAngularVelocity(body, angular);
        }

        /// <summary>
        /// Changes the orientation of the principal axes of inertia while preserving angular velocity in world space,
        /// as <see cref="SetRotation"/> does for the body's rotation.
        /// </summary>
        internal static void SetInertiaRotation(in BodyData body, quaternion rotation)
        {
            float3 angular = WorldAngularVelocity(body);
            body.Mass()->Transform.rot = rotation;
            SetWorldAngularVelocity(body, angular);
        }

        internal static void Wake(BodyData* hot)
        {
            hot->flags &= ~BodyFlags.Sleeping;
            (*hot).Sleep()->WakeCounter = math.max((*hot).Sleep()->WakeCounter, StoreMemory.WakeCounterReset);
        }

        internal static void WakeIfAsked(BodyData* hot, CommandArgs* a)
        {
            if (a->enabled != 0)
            {
                Wake(hot);
            }
        }

        internal static float3 VelocityAt(BodyData* hot, float3 worldPoint)
        {
            float3 worldCenter = (*hot).Pose()->Position + math.rotate((*hot).Pose()->Rotation, (*hot).Mass()->Transform.pos);
            return (*hot).Velocity()->Linear + math.cross(WorldAngularVelocity(*hot), worldPoint - worldCenter);
        }

        /// <summary>
        /// Compute the exact awake-body pose that write-back will store, for jobs reading solved motion
        /// concurrently.
        /// </summary>
        internal static void SolvedPose(in MotionData motion, out float3 position, out quaternion rotation)
        {
            RigidTransform worldFromBody = math.mul(motion.WorldFromMotion, math.inverse(motion.BodyFromMotion));
            position = worldFromBody.pos;
            rotation = math.normalize(worldFromBody.rot);
        }
    }
}
