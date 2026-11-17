using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Store.World;

namespace Unity.Physics.Collision.Colliders.Building
{
    /// <summary>
    /// Stores a sphere, box or capsule on the stack for Unity Physics to use in casts and distance calculations.
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    internal unsafe struct PrimitiveCollider
    {
        [FieldOffset(0)]
        private SphereCollider sphere;

        [FieldOffset(0)]
        private BoxCollider box;

        [FieldOffset(0)]
        private CapsuleCollider capsule;

        /// <summary>
        /// Build primitive geometry at the origin and return a pointer within this struct, or null for
        /// non-primitives.
        /// The pointer is valid only while this struct remains valid.
        /// </summary>
        internal Collider* Initialize(uint geometryType, float3 extents, CollisionFilter filter)
        {
            switch (geometryType)
            {
                case StoreConstants.GeometrySphere:
                    sphere.Initialize(ColliderBuilder.SphereOf(extents, float3.zero), filter, Material.Default);
                    return (Collider*)UnsafeUtility.AddressOf(ref sphere);
                case StoreConstants.GeometryBox:
                    box.Initialize(ColliderBuilder.BoxOf(extents, float3.zero, quaternion.identity), filter, Material.Default);
                    return (Collider*)UnsafeUtility.AddressOf(ref box);
                case StoreConstants.GeometryCapsule:
                    capsule.Initialize(ColliderBuilder.CapsuleOf(extents, float3.zero, quaternion.identity), filter, Material.Default);
                    return (Collider*)UnsafeUtility.AddressOf(ref capsule);
                default:
                    return null;
            }
        }
    }
}
