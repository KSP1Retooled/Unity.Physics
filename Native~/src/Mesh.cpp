// Mesh cooking. A cook normalises the source mesh into a Mesh record: welded vertices and clean triangles, or the convex
// hull, plus bounds and mass moments. The Unity Physics blobs are built from that record later, on the main thread, when
// a body needs them (ColliderBuilder.cs), because a blob carries the shape's scale, filter and material, which a cook
// doesn't know.
//
// Cooking stays in the plugin, with no managed call, because it arrives where the managed core can't be relied on: import
// workers during a content build, Physics.BakeMesh in jobs, and any thread while a domain reloads. Everything here works
// on local data, and the only shared state is a Mesh's reference count.
//
// The stream form is what Unity stores in a mesh asset when a mesh is pre-baked (m_BakedConvexCollisionMesh and
// m_BakedTriangleCollisionMesh). Unity treats it as opaque bytes and hands them back to CreateCollisionMeshFromStream on
// load, so the format is ours. It must stay a pure function of the source mesh, since content builds are deterministic.
// Change kStreamVersion with the layout: a stream of another version, or one PhysX wrote, is refused at load.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    static constexpr uint32_t kStreamMagic = 0x4853454d; // "MESH"
    static constexpr uint32_t kStreamVersion = 1;

    struct StreamHeader
    {
        uint32_t magic;
        uint32_t version;
        uint32_t convex;
        uint32_t vertexCount;
        uint32_t triangleCount;
        Float3 lower;
        Float3 upper;
        MeshMoments moments;
    };

    static_assert(sizeof(StreamHeader) == 84, "changing this layout changes every baked mesh: bump kStreamVersion");

    struct Double3
    {
        double x, y, z;
    };

    static Double3 ToDouble(const Float3& v)
    {
        return {v.x, v.y, v.z};
    }

    static Double3 Sub(const Double3& a, const Double3& b)
    {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    static Double3 Cross(const Double3& a, const Double3& b)
    {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    static double Dot(const Double3& a, const Double3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static uint32_t Bits(float value)
    {
        // Adding zero turns -0 into +0, so the two weld.
        const float normalised = value + 0.0f;
        uint32_t bits;
        std::memcpy(&bits, &normalised, sizeof(bits));
        return bits;
    }

    // Merges exactly equal vertices, keeping first occurrences in order so the result depends only on the input. remap
    // takes an input index to its welded index.
    static void Weld(std::vector<Float3>& vertices, std::vector<uint32_t>& remap)
    {
        constexpr uint32_t kEmpty = 0xffffffffu;

        size_t capacity = 16;
        while (capacity < vertices.size() * 2)
        {
            capacity *= 2;
        }

        std::vector<uint32_t> table(capacity, kEmpty);
        remap.resize(vertices.size());
        uint32_t unique = 0;
        for (size_t i = 0; i < vertices.size(); i++)
        {
            const Float3 vertex = vertices[i];
            const uint32_t x = Bits(vertex.x);
            const uint32_t y = Bits(vertex.y);
            const uint32_t z = Bits(vertex.z);
            size_t slot = (x * 73856093u ^ y * 19349663u ^ z * 83492791u) & (capacity - 1);
            while (true)
            {
                const uint32_t found = table[slot];
                if (found == kEmpty)
                {
                    table[slot] = unique;
                    remap[i] = unique;
                    vertices[unique++] = vertex;
                    break;
                }

                const Float3& other = vertices[found];
                if (Bits(other.x) == x && Bits(other.y) == y && Bits(other.z) == z)
                {
                    remap[i] = found;
                    break;
                }

                slot = (slot + 1) & (capacity - 1);
            }
        }

        vertices.resize(unique);
    }

    struct HullFace
    {
        uint32_t a, b, c;
        Double3 normal;
        double offset;
    };

    static HullFace MakeFace(const std::vector<Double3>& points, uint32_t a, uint32_t b, uint32_t c)
    {
        Double3 normal = Cross(Sub(points[b], points[a]), Sub(points[c], points[a]));
        const double length = std::sqrt(Dot(normal, normal));
        if (length > 0.0)
        {
            normal = {normal.x / length, normal.y / length, normal.z / length};
        }

        return {a, b, c, normal, Dot(normal, points[a])};
    }

    static double Distance(const HullFace& face, const Double3& point)
    {
        return Dot(face.normal, point) - face.offset;
    }

    static uint64_t EdgeKey(uint32_t from, uint32_t to)
    {
        return (static_cast<uint64_t>(from) << 32) | to;
    }

    // Incremental convex hull. It is quadratic at worst, which collision meshes are too small to notice, and double
    // precision keeps the visibility tests honest for float input. Returns false when the points are flat or collinear:
    // seeds then holds the points that span them, three for a plane and fewer otherwise.
    static bool BuildHull(const std::vector<Float3>& vertices, std::vector<uint32_t>& triangles, std::vector<uint32_t>& seeds)
    {
        std::vector<Double3> points(vertices.size());
        double largest = 0.0;
        Double3 lower = {1e300, 1e300, 1e300};
        Double3 upper = {-1e300, -1e300, -1e300};
        uint32_t first = 0;
        for (uint32_t i = 0; i < points.size(); i++)
        {
            const Double3 p = ToDouble(vertices[i]);
            points[i] = p;
            first = p.x < points[first].x ? i : first;
            lower = {std::min(lower.x, p.x), std::min(lower.y, p.y), std::min(lower.z, p.z)};
            upper = {std::max(upper.x, p.x), std::max(upper.y, p.y), std::max(upper.z, p.z)};
            largest = std::max({largest, std::abs(p.x), std::abs(p.y), std::abs(p.z)});
        }

        // Float rounding of the input, so that a point on a face isn't taken for one outside it.
        const Double3 extent = Sub(upper, lower);
        const double epsilon = 1e-6 * std::max({extent.x, extent.y, extent.z}) + 2e-7 * largest;

        seeds = {first};
        uint32_t second = first;
        double best = epsilon * epsilon;
        for (uint32_t i = 0; i < points.size(); i++)
        {
            const Double3 offset = Sub(points[i], points[first]);
            const double squared = Dot(offset, offset);
            if (squared > best)
            {
                best = squared;
                second = i;
            }
        }

        if (second == first)
        {
            return false;
        }

        seeds.push_back(second);
        const Double3 axis = Sub(points[second], points[first]);
        const double axisSquared = Dot(axis, axis);
        uint32_t third = first;
        best = epsilon * epsilon;
        for (uint32_t i = 0; i < points.size(); i++)
        {
            const Double3 arm = Cross(axis, Sub(points[i], points[first]));
            const double squared = Dot(arm, arm) / axisSquared;
            if (squared > best)
            {
                best = squared;
                third = i;
            }
        }

        if (third == first)
        {
            return false;
        }

        seeds.push_back(third);
        const HullFace base = MakeFace(points, first, second, third);
        uint32_t fourth = first;
        best = epsilon;
        for (uint32_t i = 0; i < points.size(); i++)
        {
            const double distance = std::abs(Distance(base, points[i]));
            if (distance > best)
            {
                best = distance;
                fourth = i;
            }
        }

        if (fourth == first)
        {
            return false;
        }

        // Wind the first tetrahedron outward: its fourth point has to lie behind the base.
        const bool above = Distance(base, points[fourth]) > 0.0;
        const uint32_t p0 = first;
        const uint32_t p1 = above ? third : second;
        const uint32_t p2 = above ? second : third;
        const uint32_t p3 = fourth;
        std::vector<HullFace> faces = {
            MakeFace(points, p0, p1, p2),
            MakeFace(points, p0, p3, p1),
            MakeFace(points, p1, p3, p2),
            MakeFace(points, p2, p3, p0),
        };

        std::vector<uint64_t> visibleEdges;
        std::vector<HullFace> kept;
        for (uint32_t point = 0; point < points.size(); point++)
        {
            if (point == p0 || point == p1 || point == p2 || point == p3)
            {
                continue;
            }

            visibleEdges.clear();
            kept.clear();
            for (const HullFace& face : faces)
            {
                if (Distance(face, points[point]) > epsilon)
                {
                    visibleEdges.push_back(EdgeKey(face.a, face.b));
                    visibleEdges.push_back(EdgeKey(face.b, face.c));
                    visibleEdges.push_back(EdgeKey(face.c, face.a));
                }
                else
                {
                    kept.push_back(face);
                }
            }

            if (visibleEdges.empty())
            {
                continue;
            }

            // A directed edge whose reverse isn't visible lies on the horizon, and gets a new face to this point.
            std::sort(visibleEdges.begin(), visibleEdges.end());
            for (const uint64_t edge : visibleEdges)
            {
                const uint32_t from = static_cast<uint32_t>(edge >> 32);
                const uint32_t to = static_cast<uint32_t>(edge);
                if (!std::binary_search(visibleEdges.begin(), visibleEdges.end(), EdgeKey(to, from)))
                {
                    kept.push_back(MakeFace(points, from, to, point));
                }
            }

            faces.swap(kept);
        }

        triangles.clear();
        triangles.reserve(faces.size() * 3);
        for (const HullFace& face : faces)
        {
            triangles.push_back(face.a);
            triangles.push_back(face.b);
            triangles.push_back(face.c);
        }

        return true;
    }

    // PhysX gives a flat convex mesh a body: a slab reaching either side of the plane by 5% of the mesh's smallest extent.
    // seeds are three points that span the plane.
    static void Thicken(std::vector<Float3>& vertices, const std::vector<uint32_t>& seeds)
    {
        Float3 lower = vertices[0];
        Float3 upper = vertices[0];
        for (const Float3& v : vertices)
        {
            lower = {std::min(lower.x, v.x), std::min(lower.y, v.y), std::min(lower.z, v.z)};
            upper = {std::max(upper.x, v.x), std::max(upper.y, v.y), std::max(upper.z, v.z)};
        }

        double smallest = 1e300;
        for (const double extent : {double(upper.x) - lower.x, double(upper.y) - lower.y, double(upper.z) - lower.z})
        {
            if (extent > 1e-6 && extent < smallest)
            {
                smallest = extent;
            }
        }

        const Double3 a = ToDouble(vertices[seeds[0]]);
        Double3 normal = Cross(Sub(ToDouble(vertices[seeds[1]]), a), Sub(ToDouble(vertices[seeds[2]]), a));
        const double reach = 0.05 * smallest / std::sqrt(Dot(normal, normal));
        const Float3 offset = {
            static_cast<float>(normal.x * reach),
            static_cast<float>(normal.y * reach),
            static_cast<float>(normal.z * reach),
        };

        const size_t count = vertices.size();
        vertices.resize(count * 2);
        for (size_t i = 0; i < count; i++)
        {
            const Float3 v = vertices[i];
            vertices[i] = {v.x + offset.x, v.y + offset.y, v.z + offset.z};
            vertices[count + i] = {v.x - offset.x, v.y - offset.y, v.z - offset.z};
        }
    }

    // Drops vertices no triangle uses, keeping the order of the rest.
    static void Compact(std::vector<Float3>& vertices, std::vector<uint32_t>& triangles)
    {
        constexpr uint32_t kUnused = 0xffffffffu;
        std::vector<uint32_t> remap(vertices.size(), kUnused);
        for (const uint32_t index : triangles)
        {
            remap[index] = 0;
        }

        uint32_t next = 0;
        for (size_t i = 0; i < vertices.size(); i++)
        {
            if (remap[i] != kUnused)
            {
                remap[i] = next;
                vertices[next++] = vertices[i];
            }
        }

        vertices.resize(next);
        for (uint32_t& index : triangles)
        {
            index = remap[index];
        }
    }

    // Integrates over the solid a closed mesh bounds, as a sum of signed tetrahedra from the vertices' mean: each has a
    // known covariance, and working near the solid keeps meshes far from their origin accurate.
    static MeshMoments ComputeMoments(const std::vector<Float3>& vertices, const std::vector<uint32_t>& triangles)
    {
        Double3 mean{};
        for (const Float3& vertex : vertices)
        {
            mean = {mean.x + vertex.x, mean.y + vertex.y, mean.z + vertex.z};
        }

        const double count = static_cast<double>(std::max<size_t>(vertices.size(), 1));
        mean = {mean.x / count, mean.y / count, mean.z / count};

        double volume = 0.0;
        Double3 weighted{};
        double second[3][3] = {};
        for (size_t i = 0; i + 2 < triangles.size(); i += 3)
        {
            const Double3 a = Sub(ToDouble(vertices[triangles[i]]), mean);
            const Double3 b = Sub(ToDouble(vertices[triangles[i + 1]]), mean);
            const Double3 c = Sub(ToDouble(vertices[triangles[i + 2]]), mean);
            const double determinant = Dot(a, Cross(b, c));
            const double* pa = &a.x;
            const double* pb = &b.x;
            const double* pc = &c.x;

            volume += determinant / 6.0;
            weighted = {
                weighted.x + determinant / 24.0 * (a.x + b.x + c.x),
                weighted.y + determinant / 24.0 * (a.y + b.y + c.y),
                weighted.z + determinant / 24.0 * (a.z + b.z + c.z),
            };

            // The integral of x_i x_j over a tetrahedron with one corner at the origin.
            for (int row = 0; row < 3; row++)
            {
                for (int column = 0; column < 3; column++)
                {
                    const double sums = (pa[row] + pb[row] + pc[row]) * (pa[column] + pb[column] + pc[column]);
                    const double squares = pa[row] * pa[column] + pb[row] * pb[column] + pc[row] * pc[column];
                    second[row][column] += determinant / 120.0 * (sums + squares);
                }
            }
        }

        MeshMoments moments{};
        if (std::abs(volume) < 1e-18)
        {
            return moments;
        }

        const Double3 centroid = {weighted.x / volume, weighted.y / volume, weighted.z / volume};
        const double* c = &centroid.x;
        const auto covariance = [&](int row, int column)
        {
            return static_cast<float>(second[row][column] / volume - c[row] * c[column]);
        };

        moments.volume = static_cast<float>(std::abs(volume));
        moments.centroid = {
            static_cast<float>(centroid.x + mean.x),
            static_cast<float>(centroid.y + mean.y),
            static_cast<float>(centroid.z + mean.z),
        };
        moments.covariance[0] = covariance(0, 0);
        moments.covariance[1] = covariance(1, 1);
        moments.covariance[2] = covariance(2, 2);
        moments.covariance[3] = covariance(0, 1);
        moments.covariance[4] = covariance(0, 2);
        moments.covariance[5] = covariance(1, 2);
        return moments;
    }

    static void Finish(Mesh& mesh, bool convex)
    {
        Float3 lower{};
        Float3 upper{};
        if (!mesh.vertices.empty())
        {
            lower = upper = mesh.vertices[0];
            for (const Float3& v : mesh.vertices)
            {
                lower = {std::min(lower.x, v.x), std::min(lower.y, v.y), std::min(lower.z, v.z)};
                upper = {std::max(upper.x, v.x), std::max(upper.y, v.y), std::max(upper.z, v.z)};
            }
        }

        mesh.data.vertices = mesh.vertices.data();
        mesh.data.triangles = mesh.triangles.data();
        mesh.data.vertexCount = static_cast<uint32_t>(mesh.vertices.size());
        mesh.data.triangleCount = static_cast<uint32_t>(mesh.triangles.size() / 3);
        mesh.data.convex = convex ? 1 : 0;
        mesh.data.lower = lower;
        mesh.data.upper = upper;
    }

    static bool ReadVertices(const Physics::CollisionMeshCookingDescriptor& descriptor, std::vector<Float3>& vertices)
    {
        const uint32_t stride = descriptor.vertexElementSize != 0 ? descriptor.vertexElementSize : sizeof(Float3);
        if (descriptor.vertexStream == nullptr || descriptor.vertexElementCount == 0 || stride < sizeof(Float3))
        {
            return false;
        }

        vertices.resize(descriptor.vertexElementCount);
        const char* source = static_cast<const char*>(descriptor.vertexStream);
        for (uint32_t i = 0; i < descriptor.vertexElementCount; i++)
        {
            std::memcpy(&vertices[i], source + static_cast<size_t>(i) * stride, sizeof(Float3));
            if (!std::isfinite(vertices[i].x) || !std::isfinite(vertices[i].y) || !std::isfinite(vertices[i].z))
            {
                return false;
            }
        }

        return true;
    }

    // Takes indices to welded indices, dropping triangles that collapsed and honouring flippedNormals.
    static bool ReadTriangles(
        const Physics::CollisionMeshCookingDescriptor& descriptor,
        const std::vector<uint32_t>& remap,
        std::vector<uint32_t>& triangles)
    {
        if (descriptor.indexStream == nullptr)
        {
            return false;
        }

        // A stream is sized in indices, or in triangles when its element is a whole one.
        const bool wholeTriangles = descriptor.indexElementSize == 6 || descriptor.indexElementSize == 12;
        const bool wide = descriptor.indexElementSize == 4 || descriptor.indexElementSize == 12;
        if (!wholeTriangles && descriptor.indexElementSize != 2 && descriptor.indexElementSize != 4)
        {
            return false;
        }

        const size_t indexCount = static_cast<size_t>(descriptor.indexElementCount) * (wholeTriangles ? 3 : 1);
        const uint16_t* narrowIndices = static_cast<const uint16_t*>(descriptor.indexStream);
        const uint32_t* wideIndices = static_cast<const uint32_t*>(descriptor.indexStream);
        triangles.reserve(indexCount);
        for (size_t i = 0; i + 2 < indexCount; i += 3)
        {
            uint32_t corners[3];
            for (size_t corner = 0; corner < 3; corner++)
            {
                const uint32_t index = wide ? wideIndices[i + corner] : narrowIndices[i + corner];
                if (index >= remap.size())
                {
                    return false;
                }

                corners[corner] = remap[index];
            }

            if (corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0])
            {
                continue;
            }

            triangles.push_back(corners[0]);
            triangles.push_back(corners[descriptor.flippedNormals ? 2 : 1]);
            triangles.push_back(corners[descriptor.flippedNormals ? 1 : 2]);
        }

        return true;
    }

    // Returns a mesh holding one reference, or null with an error. A mesh that reduces to nothing is still returned, as an
    // empty collider and with a warning: PhysX accepts such meshes, and a failure here would stop a content build.
    static Mesh* Cook(const Physics::CollisionMeshCookingDescriptor& descriptor, const char*& error, const char*& warning)
    {
        std::vector<Float3> vertices;
        if (!ReadVertices(descriptor, vertices))
        {
            error = "RetooledPhysics: the mesh has no vertices, or vertices that aren't finite";
            return nullptr;
        }

        std::vector<uint32_t> remap;
        Weld(vertices, remap);

        std::vector<uint32_t> triangles;
        std::vector<uint32_t> seeds;
        auto* mesh = new Mesh();
        if (descriptor.convex)
        {
            bool built = BuildHull(vertices, triangles, seeds);
            if (!built && seeds.size() == 3)
            {
                Thicken(vertices, seeds);
                built = BuildHull(vertices, triangles, seeds);
            }

            if (built)
            {
                Compact(vertices, triangles);
                mesh->moments = ComputeMoments(vertices, triangles);
            }
            else
            {
                vertices.clear();
                warning = "RetooledPhysics: the convex mesh is a point or a line, so it has no collider";
            }
        }
        else
        {
            if (!ReadTriangles(descriptor, remap, triangles))
            {
                delete mesh;
                error = "RetooledPhysics: the mesh's index stream can't be read";
                return nullptr;
            }

            if (triangles.empty())
            {
                vertices.clear();
                warning = "RetooledPhysics: the mesh has no usable triangles, so it has no collider";
            }
            else
            {
                Compact(vertices, triangles);
                mesh->moments = ComputeMoments(vertices, triangles);
            }
        }

        mesh->vertices = std::move(vertices);
        mesh->triangles = std::move(triangles);
        Finish(*mesh, descriptor.convex);
        return mesh;
    }

    static Physics::CookingError ResultOf(const char* warning)
    {
        return warning != nullptr ? Physics::CookingError::Warning : Physics::CookingError::Success;
    }

    void RetainMesh(Mesh* mesh)
    {
        if (mesh != nullptr)
        {
            mesh->references.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void ReleaseMesh(Mesh* mesh)
    {
        if (mesh != nullptr && mesh->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            delete mesh;
        }
    }

    static void CookCollisionMesh(void*, const Context&, ShapeCmd::CookCollisionMesh& command)
    {
        const char* error = nullptr;
        const char* warning = nullptr;
        Mesh* mesh = Cook(command.descriptor, error, warning);
        if (mesh == nullptr)
        {
            command.onMeshReadyCallback(Physics::CookingError::Failed, error, command.dataContext, nullptr);
            return;
        }

        command.onMeshReadyCallback(ResultOf(warning), warning, command.dataContext, mesh);
    }

    static void CookCollisionMeshStream(void*, const Context&, ShapeCmd::CookCollisionMeshStream& command)
    {
        const char* error = nullptr;
        const char* warning = nullptr;
        Mesh* mesh = Cook(command.descriptor, error, warning);
        if (mesh == nullptr)
        {
            command.onStreamReadyCallback(Physics::CookingError::Failed, error, command.dataContext, nullptr, 0);
            return;
        }

        StreamHeader header{};
        header.magic = kStreamMagic;
        header.version = kStreamVersion;
        header.convex = mesh->data.convex;
        header.vertexCount = mesh->data.vertexCount;
        header.triangleCount = mesh->data.triangleCount;
        header.lower = mesh->data.lower;
        header.upper = mesh->data.upper;
        header.moments = mesh->moments;

        const size_t vertexBytes = mesh->vertices.size() * sizeof(Float3);
        const size_t indexBytes = mesh->triangles.size() * sizeof(uint32_t);
        std::vector<char> stream(sizeof(header) + vertexBytes + indexBytes);
        char* end = stream.data();
        std::memcpy(end, &header, sizeof(header));
        end += sizeof(header);
        if (vertexBytes != 0)
        {
            std::memcpy(end, mesh->vertices.data(), vertexBytes);
            end += vertexBytes;
        }

        if (indexBytes != 0)
        {
            std::memcpy(end, mesh->triangles.data(), indexBytes);
        }

        ReleaseMesh(mesh);
        command.onStreamReadyCallback(ResultOf(warning), warning, command.dataContext, stream.data(), stream.size());
    }

    static void NoteRefusedStream(const uint8_t* stream, size_t size)
    {
        static std::atomic<bool> noted{false};
        if (noted.exchange(true))
        {
            return;
        }

        uint32_t magic = 0;
        if (stream != nullptr)
        {
            std::memcpy(&magic, stream, std::min(size, sizeof(magic)));
        }

        char text[192];
        std::snprintf(
            text,
            sizeof(text),
            "[RetooledPhysics] refused a baked collision mesh of %zu bytes starting %08x: it isn't in this backend's "
            "format, so it was probably built on another backend",
            size,
            magic);
        LogError(text);
    }

    // Unity's bytes may come from anywhere, including a build made on PhysX, so nothing is trusted until it is checked.
    static void CreateCollisionMeshFromStream(void*, const Context&, ShapeCmd::CreateCollisionMeshFromStream& command)
    {
        command.outCollisionMesh = nullptr;

        StreamHeader header{};
        const bool complete = command.stream != nullptr && command.streamSize >= sizeof(header);
        if (complete)
        {
            std::memcpy(&header, command.stream, sizeof(header));
        }

        const uint64_t vertexBytes = static_cast<uint64_t>(header.vertexCount) * sizeof(Float3);
        const uint64_t indexBytes = static_cast<uint64_t>(header.triangleCount) * 3 * sizeof(uint32_t);
        if (!complete
            || header.magic != kStreamMagic
            || header.version != kStreamVersion
            || (header.convex != 0) != command.convex
            || command.streamSize != sizeof(header) + vertexBytes + indexBytes)
        {
            NoteRefusedStream(command.stream, command.streamSize);
            return;
        }

        auto* mesh = new Mesh();
        mesh->vertices.resize(header.vertexCount);
        mesh->triangles.resize(static_cast<size_t>(header.triangleCount) * 3);
        const uint8_t* source = command.stream + sizeof(header);
        if (vertexBytes != 0)
        {
            std::memcpy(mesh->vertices.data(), source, vertexBytes);
        }

        if (indexBytes != 0)
        {
            std::memcpy(mesh->triangles.data(), source + vertexBytes, indexBytes);
        }

        for (const uint32_t index : mesh->triangles)
        {
            if (index >= header.vertexCount)
            {
                delete mesh;
                NoteRefusedStream(command.stream, command.streamSize);
                return;
            }
        }

        mesh->moments = header.moments;
        Finish(*mesh, header.convex != 0);
        command.outCollisionMesh = mesh;
    }

    static void DestroyCollisionMesh(void*, const Context&, ShapeCmd::DestroyCollisionMesh& command)
    {
        ReleaseMesh(static_cast<Mesh*>(command.collisionMesh));
    }

    static void ExtractCollisionMeshData(void* object, const Context&, ShapeCmd::ExtractCollisionMeshData& command)
    {
        const Shape* shape = static_cast<const Shape*>(object);
        if (shape == nullptr || !IsMeshGeometry(shape->data) || shape->data.mesh == nullptr)
        {
            return;
        }

        const MeshData& mesh = static_cast<const Mesh*>(shape->data.mesh)->data;
        Physics::SDKCollisionMeshData extracted{};
        extracted.vertices = &mesh.vertices->x;
        extracted.vertexCount = mesh.vertexCount;
        extracted.vertexStride = sizeof(Float3);
        extracted.indices = mesh.triangles;
        extracted.indexCount = static_cast<size_t>(mesh.triangleCount) * 3;
        extracted.indexStride = sizeof(uint32_t);
        command.callback(command.ctx, extracted);
    }

    // Ray hits don't name a triangle (see QueryKernel), so there is no cooked index to translate.
    static void RemapTriangleIndexToSourceIndex(void*, const Context&, ShapeCmd::RemapTriangleIndexToSourceIndex& command)
    {
        command.outSourceMeshTriangleIndex = command.physicsMeshTriangleIndex;
    }

    // Only a content build sends the stream commands through Unity, so these let a test take the same path directly.
    // Writes the stream into output when it fits and returns its size, or -1 if the cook failed.
    extern "C" UNITY_INTERFACE_EXPORT int32_t UNITY_INTERFACE_API RetooledPhysics_CookStream(
        const Float3* vertices,
        int32_t vertexCount,
        const int32_t* indices,
        int32_t indexCount,
        int32_t convex,
        uint8_t* output,
        int32_t capacity)
    {
        struct Result
        {
            uint8_t* output;
            int32_t capacity;
            int32_t size;
        };

        Result result = {output, capacity, -1};
        ShapeCmd::CookCollisionMeshStream command{};
        command.descriptor.vertexStream = const_cast<Float3*>(vertices);
        command.descriptor.vertexElementCount = static_cast<uint32_t>(vertexCount);
        command.descriptor.vertexElementSize = sizeof(Float3);
        command.descriptor.indexStream = const_cast<int32_t*>(indices);
        command.descriptor.indexElementCount = static_cast<uint32_t>(indexCount);
        command.descriptor.indexElementSize = sizeof(int32_t);
        command.descriptor.convex = convex != 0;
        command.dataContext = &result;
        command.onStreamReadyCallback = [](Physics::CookingError error, const char*, void* context, void* stream, size_t size)
        {
            auto* target = static_cast<Result*>(context);
            if ((error & Physics::CookingError::Failed) == Physics::CookingError::Failed)
            {
                return;
            }

            target->size = static_cast<int32_t>(size);
            if (target->size <= target->capacity)
            {
                std::memcpy(target->output, stream, size);
            }
        };

        CookCollisionMeshStream(nullptr, Context{}, command);
        return result.size;
    }

    // Returns what the stream loads as, with the reference Unity would hold, or null if it was refused. Release it with
    // RetooledPhysics_ReleaseMesh.
    extern "C" UNITY_INTERFACE_EXPORT const MeshData* UNITY_INTERFACE_API RetooledPhysics_LoadStream(
        const uint8_t* stream,
        int32_t size,
        int32_t convex)
    {
        ShapeCmd::CreateCollisionMeshFromStream command{};
        command.stream = const_cast<uint8_t*>(stream);
        command.streamSize = static_cast<size_t>(size);
        command.convex = convex != 0;
        CreateCollisionMeshFromStream(nullptr, Context{}, command);
        auto* mesh = static_cast<Mesh*>(command.outCollisionMesh);
        return mesh != nullptr ? &mesh->data : nullptr;
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_ReleaseMesh(const MeshData* mesh)
    {
        ReleaseMesh(reinterpret_cast<Mesh*>(const_cast<MeshData*>(mesh)));
    }

    template <typename T, Handler<T> Function>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Bound<T, Function>;
    }

    void BindMeshCommands(FunctionBinding* table)
    {
        Bind<ShapeCmd::CookCollisionMesh, CookCollisionMesh>(table);
        Bind<ShapeCmd::CookCollisionMeshStream, CookCollisionMeshStream>(table);
        Bind<ShapeCmd::CreateCollisionMeshFromStream, CreateCollisionMeshFromStream>(table);
        Bind<ShapeCmd::DestroyCollisionMesh, DestroyCollisionMesh>(table);
        Bind<ShapeCmd::ExtractCollisionMeshData, ExtractCollisionMeshData>(table);
        Bind<ShapeCmd::RemapTriangleIndexToSourceIndex, RemapTriangleIndexToSourceIndex>(table);
    }
}
