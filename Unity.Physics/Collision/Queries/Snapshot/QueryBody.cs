using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Store.Bodies;

namespace Unity.Physics.Collision.Queries.Snapshot
{
    /// <summary>Describes a body in the built world for use by queries.</summary>
    internal unsafe struct QueryBody
    {
        // This pointer is used for comparisons or passed to Unity. Never dereference it: the body record may have been
        // destroyed by the time a worker reads this snapshot.
        [NativeDisableUnsafePtrRestriction]
        public Body* Owner;

        public int FirstShape;
        public int SingleShape;     // Index of the shape for a noncompound collider, relative to FirstShape.
        public int IsStatic;
        private readonly int reserved0;
    }
}
