using Unity.Mathematics;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Queries
{
    internal static unsafe class QueryHitFormat
    {
        // RaycastHit.triangleIndex interprets this value as -1.
        internal const uint NoFace = uint.MaxValue;

        /// <summary>
        /// Convert a hit to the plugin format and restore the polygon normal for backface hits.
        /// Report face zero for triangle meshes and no face for other geometry; source triangle mapping is not
        /// retained.
        /// </summary>
        internal static RayCastHit Record(float3 position, float3 normal, float distance, QueryShape* shape, bool backface)
        {
            return new RayCastHit
            {
                point = position,
                normal = backface ? -normal : normal,
                faceId = shape->GeometryType == StoreConstants.GeometryTriangleMesh ? 0u : NoFace,
                distance = distance,
                uv = float2.zero,
                engineObjectId = shape->EngineObjectId,
            };
        }
    }
}
