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
        /// Build a mesh collider with the shape's scale and pose baked into its vertices.
        /// </summary>
        [BurstCompile]
        private struct MeshColliderJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public MeshData* Mesh;

            public float3 Scale;
            public float3 Center;
            public quaternion Orientation;
            public CollisionFilter Filter;
            public Material Material;

            [NativeDisableUnsafePtrRestriction]
            public void** Result;

            public void Execute()
            {
                int vertexCount = (int)Mesh->vertexCount;
                var vertices = new NativeArray<float3>(vertexCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                float3* source = MeshData.VerticesOf(Mesh);
                for (int i = 0; i < vertexCount; i++)
                {
                    vertices[i] = Center + math.rotate(Orientation, source[i] * Scale);
                }

                BlobAssetReference<Collider> collider;
                if (Mesh->convex != 0)
                {
                    collider = CreateConvex(vertices, Filter, Material);
                }
                else
                {
                    // A scale that mirrors the mesh reverses triangle orientation, so reverse the winding as well.
                    bool mirrored = Scale.x * Scale.y * Scale.z < 0.0f;
                    int triangleCount = (int)Mesh->triangleCount;
                    var triangles = new NativeArray<int3>(triangleCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    int3* sourceTriangles = MeshData.TrianglesOf(Mesh);
                    for (int i = 0; i < triangleCount; i++)
                    {
                        int3 triangle = sourceTriangles[i];
                        triangles[i] = mirrored ? triangle.xzy : triangle;
                    }

                    collider = MeshCollider.Create(vertices, triangles, Filter, Material);
                }

                *Result = collider.GetUnsafePtr();
            }
        }
    }
}
