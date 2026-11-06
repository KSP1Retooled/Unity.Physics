using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Mesh cooking: checks a source mesh and copies it into a cooked mesh, one relocatable block holding a
    /// <see cref="NativeStore.MeshData"/> header, the vertices and the triangles, with bounds. The geometry itself is Unity
    /// Physics' to work out, per shape (<see cref="ColliderBuilder"/>), because its blobs carry a scale, a filter and a
    /// material that a cook doesn't know.
    /// </summary>
    /// <remarks>
    /// Cooks arrive from import workers and from Physics.BakeMesh in jobs, so nothing here may touch shared state. Nothing
    /// here allocates or calls into Unity Physics either: its creators use Allocator.Temp, and a thread the job system
    /// doesn't own has no temp allocator. As with the query functions, the plugin holds Burst function pointers, and until
    /// the async compile delivers they are stubs running the same code in Mono.
    ///
    /// The block is also the stream Unity stores in a pre-baked mesh asset (m_BakedConvexCollisionMesh and
    /// m_BakedTriangleCollisionMesh). Unity treats those bytes as opaque and hands them back at load, so the format is ours.
    /// A build must not depend on which form of this code ran, and thickening a flat mesh computes new vertices, so floats
    /// stay strict: Burst's default could fuse a multiply and an add, and give different bytes from Mono. Change
    /// <see cref="StreamVersion"/> with the layout.
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
        /// Writes the cooked mesh to the output, whose memory the caller owns. With no bytes given it only reports, in size, a
        /// capacity that is enough. Otherwise size becomes what was written, which may be less.
        /// </summary>
        [BurstCompile(DisableSafetyChecks = true, FloatMode = FloatMode.Strict)]
        [MonoPInvokeCallback(typeof(CookFunction))]
        internal static void Cook(NativeStore.CookInput* input, NativeStore.CookOutput* output)
        {
            bool ranInMono = false;
            CookInMono(input, output, ref ranInMono);
            if (!ranInMono)
            {
                Run(input, output);
            }
        }

        /// <summary>Burst drops this, so it only runs in the Mono form, which needs the exception ended here.</summary>
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

        /// <summary>Whether the bytes are a cooked mesh of this kind, safe to read.</summary>
        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(ValidateFunction))]
        internal static int Validate(byte* stream, long size, int convex)
        {
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
            // A stream is sized in indices, or in triangles when its element is a whole one.
            bool convex = input->convex != 0;
            bool wholeTriangles = input->indexStride == 6 || input->indexStride == 12;
            long indexCount = input->indexCount * (wholeTriangles ? 3L : 1L);
            if (output->bytes == null)
            {
                // A flat convex mesh doubles its vertices.
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
                // A hull comes from the vertices alone, so these triangles only describe the source mesh, which may have none.
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
        /// Copies the triangles, turned the right way out. A degenerate one stays for Unity Physics to drop, but a mesh with
        /// nothing else counts as empty, because Unity wants to hear that from the cook.
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
        /// How many points span the input: 4 for a solid, 3 for a plane, with those three in seeds, and fewer for a line or a
        /// point, which has no hull. Doubles keep the test honest for float input.
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

            // Float rounding of the input, so that a point on a plane isn't taken for one off it.
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
        /// PhysX gives a flat convex mesh a body: a slab reaching either side of the plane by 5% of the mesh's smallest extent.
        /// Unity Physics would leave it with no volume, so no mass. Returns the new count, which is double.
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
        /// A convex collider is the hull of every vertex and a triangle mesh only its triangles, as in PhysX, so a vertex that
        /// no triangle uses stays out of a triangle mesh's bounds.
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
