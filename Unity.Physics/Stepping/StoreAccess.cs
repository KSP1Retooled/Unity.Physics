using Unity.Physics.Store.Bodies;
using Unity.Transforms;

namespace Unity.Physics.Stepping
{
    internal static unsafe class StoreAccess
    {
        internal static LocalTransform* Pose(this in BodyData body) => (LocalTransform*)body.pose;
        internal static void SetPosePointer(this ref BodyData body, LocalTransform* pointer) => body.pose = pointer;

        internal static PhysicsVelocity* Velocity(this in BodyData body) => (PhysicsVelocity*)body.velocity;
        internal static void SetVelocityPointer(this ref BodyData body, PhysicsVelocity* pointer) => body.velocity = pointer;

        internal static PhysicsMass* Mass(this in BodyData body) => (PhysicsMass*)body.mass;
        internal static void SetMassPointer(this ref BodyData body, PhysicsMass* pointer) => body.mass = pointer;

        internal static BodyMass* BodyMass(this in BodyData body) => (BodyMass*)body.bodyMass;
        internal static void SetBodyMassPointer(this ref BodyData body, BodyMass* pointer) => body.bodyMass = pointer;

        internal static PhysicsDamping* Damping(this in BodyData body) => (PhysicsDamping*)body.damping;
        internal static void SetDampingPointer(this ref BodyData body, PhysicsDamping* pointer) => body.damping = pointer;

        internal static BodyLimits* Limits(this in BodyData body) => (BodyLimits*)body.limits;
        internal static void SetLimitsPointer(this ref BodyData body, BodyLimits* pointer) => body.limits = pointer;

        internal static BodySleep* Sleep(this in BodyData body) => (BodySleep*)body.sleep;
        internal static void SetSleepPointer(this ref BodyData body, BodySleep* pointer) => body.sleep = pointer;

        internal static BodyForces* Forces(this in BodyData body) => (BodyForces*)body.forces;
        internal static void SetForcesPointer(this ref BodyData body, BodyForces* pointer) => body.forces = pointer;

        internal static BodyTarget* Target(this in BodyData body) => (BodyTarget*)body.target;
        internal static void SetTargetPointer(this ref BodyData body, BodyTarget* pointer) => body.target = pointer;

        internal static PhysicsCollider* Collider(this in BodyData body) => (PhysicsCollider*)body.collider;
        internal static void SetColliderPointer(this ref BodyData body, PhysicsCollider* pointer) => body.collider = pointer;
    }
}
