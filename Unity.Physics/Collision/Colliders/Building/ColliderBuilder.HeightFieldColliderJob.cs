using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using StoreMesh = Unity.Physics.Store.Shapes.Mesh;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Profiling;

namespace Unity.Physics.Collision.Colliders.Building
{
    internal static unsafe partial class ColliderBuilder
    {
        /// <summary>
        /// Build a height field collider with the grid orientation from ShapeFromGrid.
        /// Use a triangle mesh when cells contain holes, since Unity Physics terrain cannot represent them.
        /// Use triangle collisions to match PhysX.
        /// </summary>
        [BurstCompile]
        private struct HeightFieldColliderJob : IJob
        {
            // Raise one corner slightly to avoid division by zero when scaling an all-zero height field.
            private const float FlatFieldRise = 1.0f / 1024.0f;

            [NativeDisableUnsafePtrRestriction]
            public HeightField* Field;

            public float3 Scale;
            public CollisionFilter Filter;
            public Material Material;

            [NativeDisableUnsafePtrRestriction]
            public void** Result;

            public void Execute()
            {
                int size = (int)Field->resolution;
                var heights = new NativeArray<float>(size * size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int j = 0; j < size; j++)
                {
                    for (int i = 0; i < size; i++)
                    {
                        heights[j * size + i] = Field->heights[(size - 1 - i) * size + j];
                    }
                }

                BlobAssetReference<Collider> collider;
                if (Field->holeCount != 0)
                {
                    collider = CreateMesh(heights, size);
                }
                else
                {
                    if (Field->lowest == 0 && Field->highest == 0)
                    {
                        heights[0] = FlatFieldRise;
                    }

                    collider = TerrainCollider.Create(heights, new int2(size), Scale, TerrainCollider.CollisionMethod.Triangles, Filter, Material);
                }

                *Result = collider.GetUnsafePtr();
            }

            private BlobAssetReference<Collider> CreateMesh(NativeArray<float> heights, int size)
            {
                var vertices = new NativeArray<float3>(size * size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int j = 0; j < size; j++)
                {
                    for (int i = 0; i < size; i++)
                    {
                        vertices[j * size + i] = new float3(i, heights[j * size + i], j) * Scale;
                    }
                }

                int cells = size - 1;
                var triangles = new NativeList<int3>(2 * cells * cells, Allocator.Temp);
                for (int j = 0; j < cells; j++)
                {
                    for (int i = 0; i < cells; i++)
                    {
                        if (Field->cells[(cells - 1 - i) * cells + j] != 0)
                        {
                            int corner = j * size + i;
                            triangles.Add(new int3(corner, corner + size, corner + 1));
                            triangles.Add(new int3(corner + 1, corner + size, corner + size + 1));
                        }
                    }
                }

                return MeshCollider.Create(vertices, triangles.AsArray(), Filter, Material);
            }
        }
    }
}
