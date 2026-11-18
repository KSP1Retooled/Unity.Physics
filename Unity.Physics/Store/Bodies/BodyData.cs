using System.Runtime.InteropServices;
using Unity.Mathematics;
using Unity.Physics.Store.Shapes;
using Unity.Transforms;

namespace Unity.Physics.Store.Bodies
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct BodyData
    {
        public Body* owner;
        public ShapeData** shapes;
        public BodyFlags flags;
        public uint shapeCount;

        /// <summary>
        /// Pointer to the current body pose. The plugin reads it directly only where Body.cpp permits.
        /// </summary>
        public LocalTransform* pose;

        // The plugin uses only the fields above. The fields below can change without rebuilding the plugin.
        public PhysicsVelocity* velocity;
        public PhysicsMass* mass;
        public BodyMass* bodyMass;
        public PhysicsDamping* damping;
        public BodyLimits* limits;
        public BodySleep* sleep;
        public BodyForces* forces;
        public BodyTarget* target;

        /// <summary>
        /// Stores the body's Unity.Physics collider. A null blob means that the body has no collider or that its
        /// collider needs to be rebuilt.
        /// </summary>
        public PhysicsCollider* collider;

        // Holds the two integers of the entity that owns these components.
        public int2 entity;
        public int orderVersion;

        /// <summary>
        /// Counts the joint ends that point to this body. When the count is zero, the body can be destroyed without
        /// searching the joints.
        /// </summary>
        public int jointEnds;

        /// <summary>
        /// Cached owner serial, or zero for an unowned row. Copied with the row when it moves.
        /// </summary>
        public uint serial;

        /// <summary>
        /// Cached collider AngularExpansionFactor, or zero without a collider.
        /// Updated together with the collider by ColliderBuilder.SetCollider.
        /// </summary>
        public float colliderExpansion;
    }
}
