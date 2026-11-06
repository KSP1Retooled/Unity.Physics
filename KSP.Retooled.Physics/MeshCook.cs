using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Mesh cooking: turns a source mesh into a cooked mesh, one relocatable block holding a <see cref="NativeStore.MeshData"/>
    /// header, the vertices and the triangles. The welded vertices and clean triangles are kept, or the convex hull, with bounds
    /// and mass moments. The Unity Physics blobs come later, per shape (<see cref="ColliderBuilder"/>), because they carry
    /// a scale, a filter and a material that a cook doesn't know.
    /// </summary>
    /// <remarks>
    /// Cooks arrive from import workers and from Physics.BakeMesh in jobs, so nothing here may touch shared state, and only
    /// Allocator.Persistent is used because a thread the job system doesn't own has no temp allocator. As with the query
    /// functions, the plugin holds Burst function pointers, and until the async compile delivers they are stubs running the
    /// same code in Mono.
    ///
    /// The block is also the stream Unity stores in a pre-baked mesh asset (m_BakedConvexCollisionMesh and
    /// m_BakedTriangleCollisionMesh). Unity treats those bytes as opaque and hands them back at load, so the format is ours.
    /// A build must not depend on which form of this code ran, so floats stay strict: Burst's default could fuse a multiply
    /// and an add, and give different bytes from Mono. Change <see cref="StreamVersion"/> with the layout.
    /// </remarks>
    [BurstCompile]
    internal static unsafe class MeshCook
    {
        internal const uint StreamMagic = 0x4853454d; // "MESH"
        internal const uint StreamVersion = 2;

        internal const int StatusOk = 0;
        internal const int StatusEmpty = 1;
        internal const int StatusBadVertices = 2;
        internal const int StatusBadIndices = 3;

        private const uint EmptySlot = 0xffffffffu;
        private const int Unused = -1;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void CookFunction(NativeStore.CookInput* input, NativeStore.CookOutput* output);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int ValidateFunction(byte* stream, long size, int convex);

        /// <summary>
        /// Writes the cooked mesh to the output. With no bytes given it only reports, in size, a capacity that is enough,
        /// so the caller owns the memory and this code needs no allocator of the caller's.
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
            // Flat meshes double their vertices. A hull of n points has fewer than 2n triangles.
            int vertexCapacity = (int)input->vertexCount * 2;
            int triangleCapacity = math.max(vertexCapacity * 2, (int)input->indexCount);
            if (output->bytes == null)
            {
                output->size = sizeof(NativeStore.MeshData) + vertexCapacity * (long)sizeof(float3) + triangleCapacity * (long)sizeof(int3);
                return;
            }

            var vertices = new UnsafeList<float3>(math.max(vertexCapacity, 4), Allocator.Persistent);
            var remap = new UnsafeList<int>(math.max((int)input->vertexCount, 4), Allocator.Persistent);
            var triangles = new UnsafeList<int>(math.max(triangleCapacity * 3, 12), Allocator.Persistent);

            NativeStore.MeshMoments moments = default;
            int status = ReadVertices(input, ref vertices);
            if (status == StatusOk)
            {
                Weld(ref vertices, ref remap);
                status = input->convex != 0
                    ? CookConvex(ref vertices, ref triangles, ref moments)
                    : CookTriangles(input, ref vertices, ref remap, ref triangles, ref moments);
            }

            output->status = status;
            output->size = 0;
            if (status <= StatusEmpty)
            {
                output->size = Write(ref vertices, ref triangles, input->convex != 0, moments, output->bytes);
            }

            vertices.Dispose();
            remap.Dispose();
            triangles.Dispose();
        }

        private static int ReadVertices(NativeStore.CookInput* input, ref UnsafeList<float3> vertices)
        {
            uint stride = input->vertexStride != 0 ? input->vertexStride : (uint)sizeof(float3);
            if (input->vertices == null || input->vertexCount == 0 || stride < sizeof(float3))
            {
                return StatusBadVertices;
            }

            vertices.Resize((int)input->vertexCount);
            byte* source = (byte*)input->vertices;
            for (int i = 0; i < vertices.Length; i++)
            {
                float3 vertex;
                UnsafeUtility.MemCpy(&vertex, source + (long)i * stride, sizeof(float3));
                if (!math.all(math.isfinite(vertex)))
                {
                    return StatusBadVertices;
                }

                vertices[i] = vertex;
            }

            return StatusOk;
        }

        /// <summary>
        /// Merges exactly equal vertices, keeping first occurrences in order so the result depends only on the input. Afterwards
        /// remap takes an input index to its welded index.
        /// </summary>
        private static void Weld(ref UnsafeList<float3> vertices, ref UnsafeList<int> remap)
        {
            int capacity = 16;
            while (capacity < vertices.Length * 2)
            {
                capacity *= 2;
            }

            var table = new UnsafeList<uint>(capacity, Allocator.Persistent);
            table.Resize(capacity);
            for (int i = 0; i < capacity; i++)
            {
                table[i] = EmptySlot;
            }

            remap.Resize(vertices.Length);
            int unique = 0;
            for (int i = 0; i < vertices.Length; i++)
            {
                float3 vertex = vertices[i];

                // Adding zero turns -0 into +0, so the two weld.
                uint3 bits = math.asuint(vertex + 0.0f);
                int slot = (int)((bits.x * 73856093u ^ bits.y * 19349663u ^ bits.z * 83492791u) & (uint)(capacity - 1));
                while (true)
                {
                    uint found = table[slot];
                    if (found == EmptySlot)
                    {
                        table[slot] = (uint)unique;
                        remap[i] = unique;
                        vertices[unique++] = vertex;
                        break;
                    }

                    if (math.all(math.asuint(vertices[(int)found] + 0.0f) == bits))
                    {
                        remap[i] = (int)found;
                        break;
                    }

                    slot = (slot + 1) & (capacity - 1);
                }
            }

            vertices.Resize(unique);
            table.Dispose();
        }

        private static int CookConvex(
            ref UnsafeList<float3> vertices,
            ref UnsafeList<int> triangles,
            ref NativeStore.MeshMoments moments)
        {
            int3 seeds = default;
            int seedCount = BuildHull(ref vertices, ref triangles, ref seeds);
            if (seedCount == 3)
            {
                Thicken(ref vertices, seeds);
                seedCount = BuildHull(ref vertices, ref triangles, ref seeds);
            }

            if (seedCount != 4)
            {
                vertices.Clear();
                triangles.Clear();
                return StatusEmpty;
            }

            Compact(ref vertices, ref triangles);
            moments = ComputeMoments(ref vertices, ref triangles);
            return StatusOk;
        }

        private static int CookTriangles(
            NativeStore.CookInput* input,
            ref UnsafeList<float3> vertices,
            ref UnsafeList<int> remap,
            ref UnsafeList<int> triangles,
            ref NativeStore.MeshMoments moments)
        {
            // A stream is sized in indices, or in triangles when its element is a whole one.
            uint elementSize = input->indexStride;
            bool wholeTriangles = elementSize == 6 || elementSize == 12;
            bool wide = elementSize == 4 || elementSize == 12;
            if (input->indices == null || (!wholeTriangles && elementSize != 2 && elementSize != 4))
            {
                return StatusBadIndices;
            }

            long indexCount = input->indexCount * (wholeTriangles ? 3L : 1L);
            ushort* narrow = (ushort*)input->indices;
            uint* wideIndices = (uint*)input->indices;
            for (long i = 0; i + 2 < indexCount; i += 3)
            {
                int3 corners = default;
                for (int corner = 0; corner < 3; corner++)
                {
                    uint index = wide ? wideIndices[i + corner] : narrow[i + corner];
                    if (index >= remap.Length)
                    {
                        return StatusBadIndices;
                    }

                    corners[corner] = remap[(int)index];
                }

                if (corners.x == corners.y || corners.y == corners.z || corners.z == corners.x)
                {
                    continue;
                }

                triangles.Add(corners.x);
                triangles.Add(input->flippedNormals != 0 ? corners.z : corners.y);
                triangles.Add(input->flippedNormals != 0 ? corners.y : corners.z);
            }

            if (triangles.IsEmpty)
            {
                vertices.Clear();
                return StatusEmpty;
            }

            Compact(ref vertices, ref triangles);
            moments = ComputeMoments(ref vertices, ref triangles);
            return StatusOk;
        }

        private struct HullFace
        {
            public int A;
            public int B;
            public int C;
            public double3 Normal;
            public double Offset;

            public double Distance(double3 point)
            {
                return math.dot(Normal, point) - Offset;
            }
        }

        private static HullFace MakeFace(double3* points, int a, int b, int c)
        {
            double3 normal = math.cross(points[b] - points[a], points[c] - points[a]);
            double length = math.length(normal);
            if (length > 0.0)
            {
                normal /= length;
            }

            return new HullFace { A = a, B = b, C = c, Normal = normal, Offset = math.dot(normal, points[a]) };
        }

        private static ulong EdgeKey(int from, int to)
        {
            return ((ulong)(uint)from << 32) | (uint)to;
        }

        private static bool ContainsEdge(ulong* sorted, int count, ulong edge)
        {
            int low = 0;
            int high = count - 1;
            while (low <= high)
            {
                int middle = (low + high) / 2;
                if (sorted[middle] == edge)
                {
                    return true;
                }

                if (sorted[middle] < edge)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle - 1;
                }
            }

            return false;
        }

        /// <summary>
        /// Incremental convex hull, written to triangles. It is quadratic at worst, which collision meshes are too small to
        /// notice, and doubles keep the visibility tests honest for float input. Returns how many points span the input: 4 for
        /// a solid, whose hull was written; 3 for a plane, with those three in seeds; fewer for a line or a point.
        /// </summary>
        private static int BuildHull(ref UnsafeList<float3> vertices, ref UnsafeList<int> triangles, ref int3 seeds)
        {
            int count = vertices.Length;
            var pointStore = new UnsafeList<double3>(math.max(count, 1), Allocator.Persistent);
            pointStore.Resize(count);
            double3* points = pointStore.Ptr;

            double3 lower = double.MaxValue;
            double3 upper = double.MinValue;
            double largest = 0.0;
            int first = 0;
            for (int i = 0; i < count; i++)
            {
                double3 p = vertices[i];
                points[i] = p;
                first = p.x < points[first].x ? i : first;
                lower = math.min(lower, p);
                upper = math.max(upper, p);
                largest = math.max(largest, math.cmax(math.abs(p)));
            }

            // Float rounding of the input, so that a point on a face isn't taken for one outside it.
            double epsilon = 1e-6 * math.cmax(upper - lower) + 2e-7 * largest;

            int spanning = FindSeeds(points, count, first, epsilon, ref seeds, out int fourth);
            if (spanning == 4)
            {
                WriteHull(points, count, seeds, fourth, epsilon, ref triangles);
            }

            pointStore.Dispose();
            return spanning;
        }

        private static int FindSeeds(double3* points, int count, int first, double epsilon, ref int3 seeds, out int fourth)
        {
            fourth = first;
            int second = first;
            double best = epsilon * epsilon;
            for (int i = 0; i < count; i++)
            {
                double squared = math.lengthsq(points[i] - points[first]);
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

            double3 axis = points[second] - points[first];
            double axisSquared = math.lengthsq(axis);
            int third = first;
            best = epsilon * epsilon;
            for (int i = 0; i < count; i++)
            {
                double squared = math.lengthsq(math.cross(axis, points[i] - points[first])) / axisSquared;
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
            HullFace face = MakeFace(points, first, second, third);
            best = epsilon;
            for (int i = 0; i < count; i++)
            {
                double distance = math.abs(face.Distance(points[i]));
                if (distance > best)
                {
                    best = distance;
                    fourth = i;
                }
            }

            return fourth == first ? 3 : 4;
        }

        private static void WriteHull(double3* points, int count, int3 seeds, int fourth, double epsilon, ref UnsafeList<int> triangles)
        {
            // Wind the first tetrahedron outward: its fourth point has to lie behind the base.
            bool above = MakeFace(points, seeds.x, seeds.y, seeds.z).Distance(points[fourth]) > 0.0;
            int p0 = seeds.x;
            int p1 = above ? seeds.z : seeds.y;
            int p2 = above ? seeds.y : seeds.z;
            int p3 = fourth;

            var faces = new UnsafeList<HullFace>(64, Allocator.Persistent);
            var kept = new UnsafeList<HullFace>(64, Allocator.Persistent);
            var visibleEdges = new UnsafeList<ulong>(64, Allocator.Persistent);
            faces.Add(MakeFace(points, p0, p1, p2));
            faces.Add(MakeFace(points, p0, p3, p1));
            faces.Add(MakeFace(points, p1, p3, p2));
            faces.Add(MakeFace(points, p2, p3, p0));

            for (int point = 0; point < count; point++)
            {
                if (point == p0 || point == p1 || point == p2 || point == p3)
                {
                    continue;
                }

                visibleEdges.Clear();
                kept.Clear();
                for (int i = 0; i < faces.Length; i++)
                {
                    HullFace face = faces[i];
                    if (face.Distance(points[point]) > epsilon)
                    {
                        visibleEdges.Add(EdgeKey(face.A, face.B));
                        visibleEdges.Add(EdgeKey(face.B, face.C));
                        visibleEdges.Add(EdgeKey(face.C, face.A));
                    }
                    else
                    {
                        kept.Add(face);
                    }
                }

                if (visibleEdges.IsEmpty)
                {
                    continue;
                }

                // A directed edge whose reverse isn't visible lies on the horizon, and gets a new face to this point.
                NativeSortExtension.Sort(visibleEdges.Ptr, visibleEdges.Length);
                for (int i = 0; i < visibleEdges.Length; i++)
                {
                    ulong edge = visibleEdges[i];
                    int from = (int)(edge >> 32);
                    int to = (int)(uint)edge;
                    if (!ContainsEdge(visibleEdges.Ptr, visibleEdges.Length, EdgeKey(to, from)))
                    {
                        kept.Add(MakeFace(points, from, to, point));
                    }
                }

                UnsafeList<HullFace> swap = faces;
                faces = kept;
                kept = swap;
            }

            triangles.Clear();
            for (int i = 0; i < faces.Length; i++)
            {
                HullFace face = faces[i];
                triangles.Add(face.A);
                triangles.Add(face.B);
                triangles.Add(face.C);
            }

            faces.Dispose();
            kept.Dispose();
            visibleEdges.Dispose();
        }

        /// <summary>
        /// PhysX gives a flat convex mesh a body: a slab reaching either side of the plane by 5% of the mesh's smallest extent.
        /// </summary>
        private static void Thicken(ref UnsafeList<float3> vertices, int3 seeds)
        {
            float3 lower = vertices[0];
            float3 upper = vertices[0];
            for (int i = 0; i < vertices.Length; i++)
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

            int count = vertices.Length;
            vertices.Resize(count * 2);
            for (int i = 0; i < count; i++)
            {
                float3 vertex = vertices[i];
                vertices[i] = vertex + offset;
                vertices[count + i] = vertex - offset;
            }
        }

        /// <summary>Drops vertices no triangle uses, keeping the order of the rest.</summary>
        private static void Compact(ref UnsafeList<float3> vertices, ref UnsafeList<int> triangles)
        {
            var remap = new UnsafeList<int>(vertices.Length, Allocator.Persistent);
            remap.Resize(vertices.Length);
            for (int i = 0; i < remap.Length; i++)
            {
                remap[i] = Unused;
            }

            for (int i = 0; i < triangles.Length; i++)
            {
                remap[triangles[i]] = 0;
            }

            int next = 0;
            for (int i = 0; i < remap.Length; i++)
            {
                if (remap[i] != Unused)
                {
                    remap[i] = next;
                    vertices[next++] = vertices[i];
                }
            }

            vertices.Resize(next);
            for (int i = 0; i < triangles.Length; i++)
            {
                triangles[i] = remap[triangles[i]];
            }

            remap.Dispose();
        }

        private static double3x3 Outer(double3 a, double3 b)
        {
            return new double3x3(a * b.x, a * b.y, a * b.z);
        }

        /// <summary>
        /// Integrates over the solid a closed mesh bounds, as a sum of signed tetrahedra from the vertices' mean: each has a
        /// known covariance, and working near the solid keeps meshes far from their origin accurate.
        /// </summary>
        private static NativeStore.MeshMoments ComputeMoments(ref UnsafeList<float3> vertices, ref UnsafeList<int> triangles)
        {
            double3 mean = double3.zero;
            for (int i = 0; i < vertices.Length; i++)
            {
                mean += vertices[i];
            }

            mean /= math.max(vertices.Length, 1);

            double volume = 0.0;
            double3 weighted = double3.zero;
            double3x3 second = double3x3.zero;
            for (int i = 0; i + 2 < triangles.Length; i += 3)
            {
                double3 a = (double3)vertices[triangles[i]] - mean;
                double3 b = (double3)vertices[triangles[i + 1]] - mean;
                double3 c = (double3)vertices[triangles[i + 2]] - mean;
                double determinant = math.dot(a, math.cross(b, c));
                double3 sum = a + b + c;

                volume += determinant / 6.0;
                weighted += determinant / 24.0 * sum;

                // The integral of x x^T over a tetrahedron with one corner at the origin.
                second += determinant / 120.0 * (Outer(sum, sum) + Outer(a, a) + Outer(b, b) + Outer(c, c));
            }

            if (math.abs(volume) < 1e-18)
            {
                return default;
            }

            double3 centroid = weighted / volume;
            double3x3 covariance = second / volume - Outer(centroid, centroid);
            return new NativeStore.MeshMoments
            {
                volume = (float)math.abs(volume),
                centroid = (float3)(centroid + mean),
                covarianceDiagonal = new float3((float)covariance.c0.x, (float)covariance.c1.y, (float)covariance.c2.z),
                covarianceOffDiagonal = new float3((float)covariance.c1.x, (float)covariance.c2.x, (float)covariance.c2.y),
            };
        }

        private static long Write(
            ref UnsafeList<float3> vertices,
            ref UnsafeList<int> triangles,
            bool convex,
            NativeStore.MeshMoments moments,
            byte* output)
        {
            float3 lower = float3.zero;
            float3 upper = float3.zero;
            if (!vertices.IsEmpty)
            {
                lower = vertices[0];
                upper = vertices[0];
                for (int i = 0; i < vertices.Length; i++)
                {
                    lower = math.min(lower, vertices[i]);
                    upper = math.max(upper, vertices[i]);
                }
            }

            NativeStore.MeshData header = new NativeStore.MeshData
            {
                magic = StreamMagic,
                version = StreamVersion,
                convex = convex ? 1u : 0u,
                vertexCount = (uint)vertices.Length,
                triangleCount = (uint)(triangles.Length / 3),
                lower = lower,
                upper = upper,
                moments = moments,
            };

            long vertexBytes = vertices.Length * (long)sizeof(float3);
            long indexBytes = triangles.Length * (long)sizeof(int);
            UnsafeUtility.MemCpy(output, &header, sizeof(NativeStore.MeshData));
            if (vertexBytes != 0)
            {
                UnsafeUtility.MemCpy(output + sizeof(NativeStore.MeshData), vertices.Ptr, vertexBytes);
            }

            if (indexBytes != 0)
            {
                UnsafeUtility.MemCpy(output + sizeof(NativeStore.MeshData) + vertexBytes, triangles.Ptr, indexBytes);
            }

            return sizeof(NativeStore.MeshData) + vertexBytes + indexBytes;
        }
    }
}
