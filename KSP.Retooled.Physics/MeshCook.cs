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
        internal const int StatusNeedsCapacity = 4;

        private const uint EmptySlot = 0xffffffffu;
        private const int Unused = -1;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void CookFunction(NativeStore.CookInput* input, NativeStore.CookOutput* output);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int ValidateFunction(byte* stream, long size, int convex);

        /// <summary>
        /// Writes the cooked mesh to the output, whose memory the caller owns. With no bytes given it only reports, in size, a
        /// capacity that should be enough. Should a cook need more, nothing is written, the status says so, and size gives the
        /// capacity to call again with.
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
            // Flat meshes double their vertices. A hull of n points normally has fewer than 2n triangles.
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
                long size = sizeof(NativeStore.MeshData) + vertices.Length * (long)sizeof(float3) + triangles.Length * (long)sizeof(int);
                output->size = size;
                if (size > output->capacity)
                {
                    output->status = StatusNeedsCapacity;
                }
                else
                {
                    Write(ref vertices, ref triangles, input->convex != 0, moments, output->bytes);
                }
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

        /// <summary>A hull triangle, with the faces across AB, BC and CA. Stamp marks the last point that saw it.</summary>
        private struct HullFace
        {
            public int A;
            public int B;
            public int C;
            public int3 Neighbors;
            public double3 Normal;
            public double Offset;
            public int Stamp;
            public int Alive;

            public double Distance(double3 point)
            {
                return math.dot(Normal, point) - Offset;
            }
        }

        private static void SetGeometry(HullFace* face, double3* points, int a, int b, int c)
        {
            double3 normal = math.cross(points[b] - points[a], points[c] - points[a]);
            double length = math.length(normal);
            if (length > 0.0)
            {
                normal /= length;
            }

            face->A = a;
            face->B = b;
            face->C = c;
            face->Normal = normal;
            face->Offset = math.dot(normal, points[a]);
            face->Stamp = -1;
            face->Alive = 1;
        }

        private static double PlaneDistance(double3* points, int a, int b, int c, double3 point)
        {
            HullFace face = default;
            SetGeometry(&face, points, a, b, c);
            return face.Distance(point);
        }

        /// <summary>
        /// Incremental convex hull, written to triangles. It scans the faces for each point, which is quadratic at worst, and
        /// doubles keep the visibility tests honest for float input. Returns how many points span the input: 4 for a solid, whose
        /// hull was written; 3 for a plane, with those three in seeds; fewer for a line or a point.
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
            best = epsilon;
            for (int i = 0; i < count; i++)
            {
                double distance = math.abs(PlaneDistance(points, first, second, third, points[i]));
                if (distance > best)
                {
                    best = distance;
                    fourth = i;
                }
            }

            return fourth == first ? 3 : 4;
        }

        /// <summary>
        /// Grows the first tetrahedron one point at a time. What a point sees is flooded across neighbours from the face it
        /// sees best, because a plain test of every face can, near-coplanar, pick faces that aren't connected, and the surface
        /// then stops being closed and its face count can run away. A point whose horizon touches itself is within rounding of
        /// the hull, and it is left out.
        /// </summary>
        private static void WriteHull(double3* points, int count, int3 seeds, int fourth, double epsilon, ref UnsafeList<int> triangles)
        {
            // Wind the first tetrahedron outward: its fourth point has to lie behind the base.
            bool above = PlaneDistance(points, seeds.x, seeds.y, seeds.z, points[fourth]) > 0.0;
            int p0 = seeds.x;
            int p1 = above ? seeds.z : seeds.y;
            int p2 = above ? seeds.y : seeds.z;
            int p3 = fourth;

            var faces = new UnsafeList<HullFace>(64, Allocator.Persistent);
            faces.Resize(4);
            SetGeometry(faces.Ptr + 0, points, p0, p1, p2);
            SetGeometry(faces.Ptr + 1, points, p0, p3, p1);
            SetGeometry(faces.Ptr + 2, points, p1, p3, p2);
            SetGeometry(faces.Ptr + 3, points, p2, p3, p0);
            faces.Ptr[0].Neighbors = new int3(1, 2, 3);
            faces.Ptr[1].Neighbors = new int3(3, 2, 0);
            faces.Ptr[2].Neighbors = new int3(1, 3, 0);
            faces.Ptr[3].Neighbors = new int3(2, 1, 0);

            var free = new UnsafeList<int>(64, Allocator.Persistent);
            var visible = new UnsafeList<int>(64, Allocator.Persistent);
            var horizon = new UnsafeList<int3>(64, Allocator.Persistent); // from, to, the face beyond
            var created = new UnsafeList<int>(64, Allocator.Persistent);
            var startingAt = new UnsafeList<int>(count, Allocator.Persistent);
            var endingAt = new UnsafeList<int>(count, Allocator.Persistent);
            startingAt.Resize(count);
            endingAt.Resize(count);
            for (int i = 0; i < count; i++)
            {
                startingAt[i] = Unused;
                endingAt[i] = Unused;
            }

            for (int point = 0; point < count; point++)
            {
                if (point == p0 || point == p1 || point == p2 || point == p3)
                {
                    continue;
                }

                int nearest = FindVisibleFace(ref faces, points[point], epsilon);
                if (nearest == Unused)
                {
                    continue;
                }

                FloodVisible(ref faces, points[point], point, nearest, epsilon, ref visible, ref horizon);
                if (ClaimHorizon(ref horizon, ref startingAt, ref endingAt))
                {
                    AddFan(ref faces, points, point, ref visible, ref horizon, ref free, ref created, ref startingAt, ref endingAt);
                }

                for (int i = 0; i < horizon.Length; i++)
                {
                    startingAt[horizon[i].x] = Unused;
                    endingAt[horizon[i].y] = Unused;
                }
            }

            triangles.Clear();
            for (int i = 0; i < faces.Length; i++)
            {
                if (faces[i].Alive != 0)
                {
                    triangles.Add(faces[i].A);
                    triangles.Add(faces[i].B);
                    triangles.Add(faces[i].C);
                }
            }

            faces.Dispose();
            free.Dispose();
            visible.Dispose();
            horizon.Dispose();
            created.Dispose();
            startingAt.Dispose();
            endingAt.Dispose();
        }

        private static int FindVisibleFace(ref UnsafeList<HullFace> faces, double3 point, double epsilon)
        {
            int nearest = Unused;
            double best = epsilon;
            for (int i = 0; i < faces.Length; i++)
            {
                if (faces.Ptr[i].Alive != 0)
                {
                    double distance = faces.Ptr[i].Distance(point);
                    if (distance > best)
                    {
                        best = distance;
                        nearest = i;
                    }
                }
            }

            return nearest;
        }

        private static void FloodVisible(
            ref UnsafeList<HullFace> faces,
            double3 point,
            int stamp,
            int start,
            double epsilon,
            ref UnsafeList<int> visible,
            ref UnsafeList<int3> horizon)
        {
            visible.Clear();
            horizon.Clear();
            visible.Add(start);
            faces.Ptr[start].Stamp = stamp;
            for (int next = 0; next < visible.Length; next++)
            {
                HullFace* face = faces.Ptr + visible[next];
                for (int edge = 0; edge < 3; edge++)
                {
                    int neighbor = face->Neighbors[edge];
                    HullFace* other = faces.Ptr + neighbor;
                    if (other->Stamp != stamp && other->Distance(point) > epsilon)
                    {
                        other->Stamp = stamp;
                        visible.Add(neighbor);
                    }
                }
            }

            for (int i = 0; i < visible.Length; i++)
            {
                HullFace* face = faces.Ptr + visible[i];
                for (int edge = 0; edge < 3; edge++)
                {
                    int neighbor = face->Neighbors[edge];
                    if (faces.Ptr[neighbor].Stamp != stamp)
                    {
                        int from = edge == 0 ? face->A : edge == 1 ? face->B : face->C;
                        int to = edge == 0 ? face->B : edge == 1 ? face->C : face->A;
                        horizon.Add(new int3(from, to, neighbor));
                    }
                }
            }
        }

        /// <summary>Records where each horizon edge starts and ends, and returns false if the horizon isn't a simple loop.</summary>
        private static bool ClaimHorizon(ref UnsafeList<int3> horizon, ref UnsafeList<int> startingAt, ref UnsafeList<int> endingAt)
        {
            bool simple = true;
            for (int i = 0; i < horizon.Length; i++)
            {
                int3 edge = horizon[i];
                simple &= startingAt[edge.x] == Unused && endingAt[edge.y] == Unused;
                startingAt[edge.x] = i;
                endingAt[edge.y] = i;
            }

            return simple;
        }

        private static void AddFan(
            ref UnsafeList<HullFace> faces,
            double3* points,
            int point,
            ref UnsafeList<int> visible,
            ref UnsafeList<int3> horizon,
            ref UnsafeList<int> free,
            ref UnsafeList<int> created,
            ref UnsafeList<int> startingAt,
            ref UnsafeList<int> endingAt)
        {
            for (int i = 0; i < visible.Length; i++)
            {
                faces.Ptr[visible[i]].Alive = 0;
                free.Add(visible[i]);
            }

            created.Clear();
            for (int i = 0; i < horizon.Length; i++)
            {
                int slot;
                if (free.IsEmpty)
                {
                    slot = faces.Length;
                    faces.Resize(slot + 1);
                }
                else
                {
                    slot = free[free.Length - 1];
                    free.Resize(free.Length - 1);
                }

                created.Add(slot);
            }

            for (int i = 0; i < horizon.Length; i++)
            {
                int3 edge = horizon[i];
                HullFace* face = faces.Ptr + created[i];
                SetGeometry(face, points, edge.x, edge.y, point);
                face->Neighbors = new int3(edge.z, created[startingAt[edge.y]], created[endingAt[edge.x]]);

                // The face beyond the horizon pointed at a face that just went away.
                HullFace* beyond = faces.Ptr + edge.z;
                int across = beyond->A == edge.y ? 0 : beyond->B == edge.y ? 1 : 2;
                beyond->Neighbors[across] = created[i];
            }
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

        private static void Write(
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
        }
    }
}
