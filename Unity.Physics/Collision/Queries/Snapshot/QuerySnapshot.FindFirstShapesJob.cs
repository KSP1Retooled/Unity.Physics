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
        /// Calculates the first index for each built body in the list of query shapes, then resizes the list to hold all
        /// shapes.
        /// </summary>
        [BurstCompile]
        internal unsafe struct FindFirstShapesJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public BodyData* Store;

            [ReadOnly]
            public NativeArray<int> DynamicIndices;

            [ReadOnly]
            public NativeArray<int> StaticIndices;

            public NativeArray<int> FirstShapes;

            public NativeList<QueryShape> QueryShapes;

            public void Execute()
            {
                int count = 0;
                for (int i = 0; i < FirstShapes.Length; i++)
                {
                    FirstShapes[i] = count;
                    int row = StoreRow(DynamicIndices, StaticIndices, i);
                    if (row >= 0)
                    {
                        count += (int)Store[row].shapeCount;
                    }
                }

                QueryShapes.ResizeUninitialized(count);
            }

            /// <summary>
            /// Map a built index to its store row, or -1 for the anchor. Dynamic bodies precede statics and the anchor.
            /// </summary>
            internal static int StoreRow(NativeArray<int> dynamicIndices, NativeArray<int> staticIndices, int builtIndex)
            {
                if (builtIndex < dynamicIndices.Length)
                {
                    return dynamicIndices[builtIndex];
                }

                builtIndex -= dynamicIndices.Length;
                return builtIndex < staticIndices.Length ? staticIndices[builtIndex] : -1;
            }
        }
    }
}
