using System.Runtime.InteropServices;
using Unity.Physics;
using Unity.Physics.Store.Bodies;
using Unity.Transforms;

namespace KSP.Retooled.Physics.Entities
{
    /// <summary>Stores a body's components while they are not on an entity.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct BodyHome
    {
        public LocalTransform pose;
        public PhysicsVelocity velocity;
        public PhysicsMass mass;
        public BodyMass bodyMass;
        public PhysicsDamping damping;
        public BodyLimits limits;
        public BodySleep sleep;
        public BodyForces forces;
        public BodyTarget target;
        public PhysicsCollider collider;

        internal static BodyHome* Of(Body* body) => (BodyHome*)&body->home;
    }
}
