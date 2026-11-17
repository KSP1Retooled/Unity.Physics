using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Collision.Queries.Snapshot
{
    /// <summary>Describes a shape for use by queries.</summary>
    internal unsafe struct QueryShape
    {
        public ulong EngineObjectId;

        // Passed to Unity without dereferencing it, like QueryBody.Owner.
        [NativeDisableUnsafePtrRestriction]
        public void* UserData;

        [NativeDisableUnsafePtrRestriction]
        public void* Handle;

        public uint Flags;
        public uint GeometryType;
    }
}
