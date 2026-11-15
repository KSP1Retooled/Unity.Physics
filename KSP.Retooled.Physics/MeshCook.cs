using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Validates a source mesh and copies it into a cooked mesh. The result is one relocatable block containing a <see
    /// cref="NativeStore.MeshData"/> header, vertices, triangles, and bounds. <see cref="ColliderBuilder"/> uses Unity
    /// Physics to create the geometry for each shape, since the scale, filter, and material stored in its blobs are not
    /// known during cooking.
    /// </summary>
    /// <remarks>
    /// Cooking runs on import workers and through Physics.BakeMesh in jobs, so this code must not access shared state.
    /// It must not allocate memory or call Unity Physics either. Unity Physics creation functions use Allocator.Temp,
    /// which is unavailable on threads outside the job system. As with query functions, the plugin stores Burst
    /// function pointers. Until asynchronous compilation finishes, these refer to stubs that run the same code in Mono.
    ///
    /// The block is also the stream that Unity stores in a pre-baked mesh asset (m_BakedConvexCollisionMesh and
    /// m_BakedTriangleCollisionMesh). Unity treats the bytes as opaque and returns them when loading, so we define the
    /// format. The output must be the same whether cooking runs in Burst or Mono. Thickening a flat mesh calculates new
    /// vertices, so use strict floating-point operations. Burst's default mode could fuse a multiply and an add,
    /// producing different bytes from Mono. Change <see cref="StreamVersion"/> when the layout changes.
    /// </remarks>
    [BurstCompile]
    internal static unsafe class MeshCook
    {
        internal const uint StreamMagic = 0x4853454d; // "MESH"
        internal const uint StreamVersion = 3;

        internal const int StatusOk = 0;
        internal const int StatusEmpty = 1;
        internal const int StatusBadVertices = 2;
        internal const int StatusBadIndices = 3;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void CookFunction(NativeStore.CookInput* input, NativeStore.CookOutput* output);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int ValidateFunction(byte* stream, long size, int convex);

        /// <summary>
        /// Writes the cooked mesh to output memory owned by the caller. If no output bytes are supplied, sets size to
        /// the required capacity. Otherwise, sets size to the number of bytes written, which may be smaller.
        /// </summary>
        [BurstCompile(DisableSafetyChecks = true, FloatMode = FloatMode.Strict)]
        [MonoPInvokeCallback(typeof(CookFunction))]
        internal static void Cook(NativeStore.CookInput* input, NativeStore.CookOutput* output)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.CookMesh.Auto();

            bool ranInMono = false;
            CookInMono(input, output, ref ranInMono);
            if (!ranInMono)
            {
                Run(input, output);
            }
        }

        /// <summary>
        /// Burst excludes this method, so it runs only in Mono. Catch exceptions here so that they do not leave the
        /// managed callback.
        /// </summary>
        [BurstDiscard]
        private static void CookInMono(NativeStore.CookInput* input, NativeStore.CookOutput* output, ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedCook();
            try
            {
                Run(input, output);
            }
            catch (Exception e)
            {
                output->status = StatusBadVertices;
                Debug.LogException(e);
            }
        }

        /// <summary>
        /// Checks whether the bytes describe a cooked mesh of the requested kind and can be read safely.
        /// </summary>
        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(ValidateFunction))]
        internal static int Validate(byte* stream, long size, int convex)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ValidateMesh.Auto();

            if (stream == null || size < sizeof(NativeStore.MeshData))
            {
                return 0;
            }

            NativeStore.MeshData header;
            UnsafeUtility.MemCpy(&header, stream, sizeof(NativeStore.MeshData));
            long expected = sizeof(NativeStore.MeshData) + header.vertexCount * (long)sizeof(float3) + header.triangleCount * (long)sizeof(int3);
            if (header.magic != StreamMagic
                || header.version != StreamVersion
                || (header.convex != 0) != (convex != 0)
                || size != expected)
            {
                return 0;
            }

            int* indices = (int*)(stream + sizeof(NativeStore.MeshData) + header.vertexCount * (long)sizeof(float3));
            long count = header.triangleCount * 3L;
            for (long i = 0; i < count; i++)
            {
                int index;
                UnsafeUtility.MemCpy(&index, indices + i, sizeof(int));
                if (index < 0 || index >= header.vertexCount)
                {
                    return 0;
                }
            }

            return 1;
        }

        private static void Run(NativeStore.CookInput* input, NativeStore.CookOutput* output)
        {
            // The input count is in indices, or in triangles when each element contains a whole triangle.
            bool convex = input->convex != 0;
            bool wholeTriangles = input->indexStride == 6 || input->indexStride == 12;
            long indexCount = input->indexCount * (wholeTriangles ? 3L : 1L);
            if (output->bytes == null)
            {
                // Allow space for twice as many vertices because thickening a flat convex mesh doubles the vertex
                // count.
                output->size = sizeof(NativeStore.MeshData)
                    + input->vertexCount * (convex ? 2L : 1L) * sizeof(float3)
                    + indexCount / 3 * sizeof(int3);
                return;
            }

            output->size = 0;
            NativeStore.MeshData* mesh = (NativeStore.MeshData*)output->bytes;
            float3* vertices = NativeStore.VerticesOf(mesh);
            output->status = ReadVertices(input, vertices);
            if (output->status != StatusOk)
            {
                return;
            }

            int vertexCount = (int)input->vertexCount;
            if (convex)
            {
                int spanning = FindSeeds(vertices, vertexCount, out int3 seeds);
                if (spanning == 3)
                {
                    vertexCount = Thicken(vertices, vertexCount, seeds);
                }
                else if (spanning < 3)
                {
                    output->status = StatusEmpty;
                    vertexCount = 0;
                }
            }

            int3* triangles = (int3*)(vertices + vertexCount);
            int triangleCount = 0;
            int triangleStatus = CookTriangles(input, vertices, indexCount, triangles, ref triangleCount);
            if (convex)
            {
                // The hull depends only on vertices. These triangles describe the source mesh, which may have no
                // triangles.
                if (triangleStatus == StatusBadIndices || vertexCount == 0)
                {
                    triangleCount = 0;
                }
            }
            else if (triangleStatus == StatusEmpty)
            {
                output->status = StatusEmpty;
                vertexCount = 0;
                triangleCount = 0;
            }
            else if (triangleStatus == StatusBadIndices)
            {
                output->status = StatusBadIndices;
                return;
            }

            WriteHeader(mesh, convex, vertices, vertexCount, triangles, triangleCount);
            output->size = sizeof(NativeStore.MeshData) + vertexCount * (long)sizeof(float3) + triangleCount * (long)sizeof(int3);
        }

        private static int ReadVertices(NativeStore.CookInput* input, float3* vertices)
        {
            uint stride = input->vertexStride != 0 ? input->vertexStride : (uint)sizeof(float3);
            if (input->vertices == null || input->vertexCount == 0 || stride < sizeof(float3))
            {
                return StatusBadVertices;
            }

            byte* source = (byte*)input->vertices;
            for (long i = 0; i < input->vertexCount; i++)
            {
                float3 vertex;
                UnsafeUtility.MemCpy(&vertex, source + i * stride, sizeof(float3));
                if (!math.all(math.isfinite(vertex)))
                {
                    return StatusBadVertices;
                }

                vertices[i] = vertex;
            }

            return StatusOk;
        }

        /// <summary>
        /// Copies the triangles and reverses their winding if flipped normals are requested. Retains degenerate
        /// triangles for Unity Physics to remove. The test for a usable triangle only checks that its three vertices
        /// have distinct positions; it does not check for collinearity. Reports the mesh as empty if no triangle passes
        /// that test, as Unity expects from cooking.
        /// </summary>
        private static int CookTriangles(NativeStore.CookInput* input, float3* vertices, long indexCount, int3* triangles, ref int triangleCount)
        {
            uint elementSize = input->indexStride;
            bool wholeTriangles = elementSize == 6 || elementSize == 12;
            bool wide = elementSize == 4 || elementSize == 12;
            if (input->indices == null || (!wholeTriangles && elementSize != 2 && elementSize != 4))
            {
                return StatusBadIndices;
            }

            ushort* narrow = (ushort*)input->indices;
            uint* wideIndices = (uint*)input->indices;
            bool usable = false;
            for (long i = 0; i + 2 < indexCount; i += 3)
            {
                int3 corners = default;
                for (int corner = 0; corner < 3; corner++)
                {
                    uint index = wide ? wideIndices[i + corner] : narrow[i + corner];
                    if (index >= input->vertexCount)
                    {
                        return StatusBadIndices;
                    }

                    corners[corner] = (int)index;
                }

                float3 a = vertices[corners.x];
                float3 b = vertices[corners.y];
                float3 c = vertices[corners.z];
                usable |= math.any(a != b) && math.any(b != c) && math.any(c != a);
                triangles[triangleCount++] = input->flippedNormals != 0 ? corners.xzy : corners;
            }

            return usable ? StatusOk : StatusEmpty;
        }

        /// <summary>
        /// Returns how many points are needed to span the input: 4 for a solid, 3 for a plane, or fewer for a line or a
        /// point, which cannot form a hull. For a plane, seeds contains the three points. Uses double precision to
        /// avoid additional errors when testing float input.
        /// </summary>
        private static int FindSeeds(float3* vertices, int count, out int3 seeds)
        {
            seeds = default;
            double3 lower = double.MaxValue;
            double3 upper = double.MinValue;
            double largest = 0.0;
            int first = 0;
            for (int i = 0; i < count; i++)
            {
                double3 p = vertices[i];
                first = p.x < vertices[first].x ? i : first;
                lower = math.min(lower, p);
                upper = math.max(upper, p);
                largest = math.max(largest, math.cmax(math.abs(p)));
            }

            // Allow for rounding in the float input so that a point on a plane is not classified as being off the
            // plane.
            double epsilon = 1e-6 * math.cmax(upper - lower) + 2e-7 * largest;

            double3 origin = vertices[first];
            int second = first;
            double best = epsilon * epsilon;
            for (int i = 0; i < count; i++)
            {
                double squared = math.lengthsq(vertices[i] - origin);
                if (squared > best)
                {
                    best = squared;
                    second = i;
                }
            }

            if (second == first)
            {
                return 1;
            }

            double3 axis = vertices[second] - origin;
            double axisSquared = math.lengthsq(axis);
            int third = first;
            best = epsilon * epsilon;
            for (int i = 0; i < count; i++)
            {
                double squared = math.lengthsq(math.cross(axis, vertices[i] - origin)) / axisSquared;
                if (squared > best)
                {
                    best = squared;
                    third = i;
                }
            }

            if (third == first)
            {
                return 2;
            }

            seeds = new int3(first, second, third);
            double3 normal = math.normalize(math.cross(axis, vertices[third] - origin));
            for (int i = 0; i < count; i++)
            {
                if (math.abs(math.dot(normal, vertices[i] - origin)) > epsilon)
                {
                    return 4;
                }
            }

            return 3;
        }

        /// <summary>
        /// Thickens a flat convex mesh into a slab, as PhysX does. The slab extends on each side of the plane by 5% of
        /// the smallest extent greater than 1e-6. Unity Physics would otherwise produce zero volume and therefore no
        /// mass. Returns the new vertex count, which is twice the original count.
        /// </summary>
        private static int Thicken(float3* vertices, int count, int3 seeds)
        {
            float3 lower = vertices[0];
            float3 upper = vertices[0];
            for (int i = 0; i < count; i++)
            {
                lower = math.min(lower, vertices[i]);
                upper = math.max(upper, vertices[i]);
            }

            double3 extent = upper - lower;
            double smallest = double.MaxValue;
            for (int axis = 0; axis < 3; axis++)
            {
                if (extent[axis] > 1e-6 && extent[axis] < smallest)
                {
                    smallest = extent[axis];
                }
            }

            double3 a = vertices[seeds.x];
            double3 normal = math.cross((double3)vertices[seeds.y] - a, (double3)vertices[seeds.z] - a);
            float3 offset = (float3)(normal * (0.05 * smallest / math.length(normal)));

            for (int i = 0; i < count; i++)
            {
                float3 vertex = vertices[i];
                vertices[i] = vertex + offset;
                vertices[count + i] = vertex - offset;
            }

            return count * 2;
        }

        /// <summary>
        /// A convex collider uses the hull of all vertices, while a triangle mesh uses only its triangles, as in PhysX.
        /// Exclude vertices that no triangle uses when calculating a triangle mesh's bounds.
        /// </summary>
        private static void WriteHeader(
            NativeStore.MeshData* mesh,
            bool convex,
            float3* vertices,
            int vertexCount,
            int3* triangles,
            int triangleCount)
        {
            float3 lower = float3.zero;
            float3 upper = float3.zero;
            if (vertexCount != 0)
            {
                lower = convex ? vertices[0] : vertices[triangles[0].x];
                upper = lower;
            }

            if (convex)
            {
                for (int i = 0; i < vertexCount; i++)
                {
                    lower = math.min(lower, vertices[i]);
                    upper = math.max(upper, vertices[i]);
                }
            }
            else
            {
                for (int i = 0; i < triangleCount; i++)
                {
                    int3 triangle = triangles[i];
                    lower = math.min(lower, math.min(vertices[triangle.x], math.min(vertices[triangle.y], vertices[triangle.z])));
                    upper = math.max(upper, math.max(vertices[triangle.x], math.max(vertices[triangle.y], vertices[triangle.z])));
                }
            }

            mesh->magic = StreamMagic;
            mesh->version = StreamVersion;
            mesh->convex = convex ? 1u : 0u;
            mesh->vertexCount = (uint)vertexCount;
            mesh->triangleCount = (uint)triangleCount;
            mesh->lower = lower;
            mesh->upper = upper;
        }
    }
}
