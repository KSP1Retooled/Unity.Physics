using System.Runtime.InteropServices;
using Unity.Transforms;

namespace Unity.Physics.Store.Bodies
{
    /// <summary>Stores a body's components while they are not on an entity.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyHome
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
    }
}
