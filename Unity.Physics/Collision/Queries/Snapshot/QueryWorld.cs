using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Collision.Queries.Snapshot
{
    /// <summary>
    /// Published collision world and body/shape snapshot for queries on any thread, protected by the world's query
    /// lock.
    /// Workers must not follow snapshot pointers into the mutable store.
    /// </summary>
    internal unsafe struct QueryWorld
    {
        public CollisionWorld Collision;

        // One entry for each body in Collision.
        [NativeDisableUnsafePtrRestriction]
        public QueryBody* Bodies;

        // Stores each body's shapes in store order, starting at QueryBody.FirstShape.
        [NativeDisableUnsafePtrRestriction]
        public QueryShape* Shapes;

        public int ShapeCount;
    }
}
