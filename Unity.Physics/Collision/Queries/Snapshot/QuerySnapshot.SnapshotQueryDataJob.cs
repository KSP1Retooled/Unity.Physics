using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;

namespace Unity.Physics.Collision.Queries.Snapshot
{
    internal static partial class QuerySnapshot
    {
        /// <summary>
        /// Copies the body and shape data that queries need from the store. The data follows the order of bodies in the
        /// built world, with one body per work item. <see cref="FirstShapes"/> contains the index where each body's shapes
        /// begin.
        /// </summary>
        [BurstCompile]
        internal unsafe struct SnapshotQueryDataJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public BodyData* Store;

            [ReadOnly]
            public NativeArray<int> DynamicIndices;

            [ReadOnly]
            public NativeArray<int> StaticIndices;

            [ReadOnly]
            public NativeArray<int> FirstShapes;

            public NativeArray<QueryBody> QueryBodies;

            [NativeDisableParallelForRestriction]
            public NativeArray<QueryShape> QueryShapes;

            public uint EngineObjectIdOffset;

            public void Execute(int i)
            {
                int row = FindFirstShapesJob.StoreRow(DynamicIndices, StaticIndices, i);
                if (row < 0)
                {
                    QueryBodies[i] = default;
                    return;
                }

                int nextShape = FirstShapes[i];
                BodyData* body = Store + row;
                QueryBodies[i] = new QueryBody
                {
                    Owner = body->owner,
                    FirstShape = nextShape,
                    SingleShape = ColliderBuilder.FirstSupported(body),
                    IsStatic = (body->flags & BodyFlags.Static) != 0 ? 1 : 0,
                };

                for (int s = 0; s < body->shapeCount; s++)
                {
                    ShapeData* shape = body->shapes[s];
                    shape->owner->querySlot = (uint)nextShape + 1;
                    QueryShapes[nextShape++] = new QueryShape
                    {
                        EngineObjectId = shape->userData != null ? *(ulong*)((byte*)shape->userData + EngineObjectIdOffset) : 0,
                        UserData = shape->userData,
                        Handle = shape->owner,
                        Flags = shape->flags,
                        GeometryType = shape->geometryType,
                    };
                }
            }
        }
    }
}
