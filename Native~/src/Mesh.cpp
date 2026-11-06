// Mesh commands. MeshCook.cs does the cooking; this file unpacks the commands, owns the memory and counts the references.
//
// A cooked mesh is one relocatable block, and those bytes are also the stream Unity stores in a pre-baked mesh asset.
//
// While the managed core is away for a domain reload, a cook or a load can't run. Unity wants a handle at once, and a mesh
// handle is only read through Mesh::data, which may be null, so both are queued with a copy of their input and finish on
// attach. Until then the mesh has no collider, bounds or mass. Only a stream cook can't wait, because Unity stores its
// bytes straight away, so it fails. Destroying a mesh needs no managed code.

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    std::shared_mutex g_MeshLock;

    struct QueuedMesh
    {
        Mesh* mesh;
        CookInput input;
        std::vector<uint8_t> vertices;
        std::vector<uint8_t> indices;
        std::vector<uint8_t> stream;
        bool convex;
    };

    static std::mutex s_QueueLock;
    static std::vector<QueuedMesh> s_Queue;

    static CookInput Unpack(const Physics::CollisionMeshCookingDescriptor& descriptor)
    {
        CookInput input{};
        input.vertices = descriptor.vertexStream;
        input.indices = descriptor.indexStream;
        input.vertexCount = descriptor.vertexElementCount;
        input.indexCount = descriptor.indexElementCount;
        input.vertexStride = descriptor.vertexElementSize != 0 ? descriptor.vertexElementSize : sizeof(Float3);
        input.indexStride = descriptor.indexElementSize;
        input.convex = descriptor.convex ? 1 : 0;
        input.flippedNormals = descriptor.flippedNormals ? 1 : 0;
        return input;
    }

    static const char* CookMessage(int32_t status)
    {
        switch (status)
        {
            case kMeshCookOk:
                return nullptr;
            case kMeshCookEmpty:
                return "RetooledPhysics: the mesh reduces to a point, a line or nothing, so it has no collider";
            case kMeshCookBadVertices:
                return "RetooledPhysics: the mesh has no vertices, or vertices that aren't finite";
            default:
                return "RetooledPhysics: the mesh's index stream can't be read";
        }
    }

    // An empty mesh is a warning: PhysX accepts such meshes, and a failure here would stop a content build.
    static Physics::CookingError CookResult(int32_t status)
    {
        switch (status)
        {
            case kMeshCookOk:
                return Physics::CookingError::Success;
            case kMeshCookEmpty:
                return Physics::CookingError::Warning;
            default:
                return Physics::CookingError::Failed;
        }
    }

    // Returns the cooked block, or null on failure. Call with g_MeshLock held shared and the core attached.
    static MeshData* RunCook(const CookInput& input, int32_t& status, size_t& size)
    {
        CookOutput output{};
        g_Managed.cookMesh(&input, &output);

        output.bytes = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(output.size)));
        g_Managed.cookMesh(&input, &output);

        status = output.status;
        size = static_cast<size_t>(output.size);
        if (CookResult(status) == Physics::CookingError::Failed)
        {
            std::free(output.bytes);
            return nullptr;
        }

        return static_cast<MeshData*>(std::realloc(output.bytes, size));
    }

    static void Enqueue(QueuedMesh&& queued)
    {
        RetainMesh(queued.mesh); // the queue's own, so a mesh released meanwhile is still there to finish
        std::lock_guard lock(s_QueueLock);
        s_Queue.push_back(std::move(queued));
    }

    static void NoteRefusedStream(size_t size)
    {
        static std::atomic<bool> noted{false};
        if (!noted.exchange(true))
        {
            char text[160];
            std::snprintf(text, sizeof(text), "[RetooledPhysics] refused a baked collision mesh of %zu bytes: it was probably built on another backend", size);
            LogError(text);
        }
    }

    // Returns a copy of the stream if it is a valid mesh. Call with g_MeshLock held shared and the core attached.
    static MeshData* LoadStream(const uint8_t* stream, size_t size, bool convex)
    {
        if (g_Managed.validateMesh(stream, static_cast<int64_t>(size), convex ? 1 : 0) == 0)
        {
            NoteRefusedStream(size);
            return nullptr;
        }

        auto* data = static_cast<MeshData*>(std::malloc(size));
        std::memcpy(data, stream, size);
        return data;
    }

    void FinishQueuedMeshes()
    {
        std::vector<QueuedMesh> queue;
        {
            std::lock_guard lock(s_QueueLock);
            queue.swap(s_Queue);
        }

        std::shared_lock lock(g_MeshLock);
        for (QueuedMesh& queued : queue)
        {
            if (!queued.stream.empty())
            {
                queued.mesh->data = LoadStream(queued.stream.data(), queued.stream.size(), queued.convex);
            }
            else
            {
                int32_t status = 0;
                size_t size = 0;
                queued.mesh->data = RunCook(queued.input, status, size);
            }

            ReleaseMesh(queued.mesh);
        }
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
            std::free(mesh->data);
            delete mesh;
        }
    }

    static void CookCollisionMesh(void*, const Context&, ShapeCmd::CookCollisionMesh& command)
    {
        CookInput input = Unpack(command.descriptor);
        auto* mesh = new Mesh();
        int32_t status = kMeshCookOk;
        {
            std::shared_lock lock(g_MeshLock);
            if (g_Managed.cookMesh != nullptr)
            {
                size_t size = 0;
                mesh->data = RunCook(input, status, size);
            }
            else
            {
                QueuedMesh queued{mesh, input};
                const auto* vertices = static_cast<const uint8_t*>(input.vertices);
                const auto* indices = static_cast<const uint8_t*>(input.indices);
                queued.vertices.assign(vertices, vertices + static_cast<size_t>(input.vertexCount) * input.vertexStride);
                if (indices != nullptr)
                {
                    queued.indices.assign(indices, indices + static_cast<size_t>(input.indexCount) * input.indexStride);
                }

                queued.input.vertices = queued.vertices.data();
                queued.input.indices = queued.indices.data();
                Enqueue(std::move(queued));
            }
        }

        const Physics::CookingError result = CookResult(status);
        if (result == Physics::CookingError::Failed)
        {
            delete mesh;
            mesh = nullptr;
        }

        command.onMeshReadyCallback(result, CookMessage(status), command.dataContext, mesh);
    }

    static void CookCollisionMeshStream(void*, const Context&, ShapeCmd::CookCollisionMeshStream& command)
    {
        int32_t status = kMeshCookOk;
        size_t size = 0;
        MeshData* data = nullptr;
        {
            std::shared_lock lock(g_MeshLock);
            if (g_Managed.cookMesh == nullptr)
            {
                command.onStreamReadyCallback(
                    Physics::CookingError::Failed,
                    "RetooledPhysics: can't bake a mesh while the managed core is detached",
                    command.dataContext,
                    nullptr,
                    0);
                return;
            }

            data = RunCook(Unpack(command.descriptor), status, size);
        }

        command.onStreamReadyCallback(CookResult(status), CookMessage(status), command.dataContext, data, data != nullptr ? size : 0);
        std::free(data);
    }

    static void CreateCollisionMeshFromStream(void*, const Context&, ShapeCmd::CreateCollisionMeshFromStream& command)
    {
        std::shared_lock lock(g_MeshLock);
        if (g_Managed.validateMesh != nullptr)
        {
            MeshData* data = LoadStream(command.stream, command.streamSize, command.convex);
            command.outCollisionMesh = nullptr;
            if (data != nullptr)
            {
                auto* mesh = new Mesh();
                mesh->data = data;
                command.outCollisionMesh = mesh;
            }

            return;
        }

        auto* mesh = new Mesh();
        QueuedMesh queued{mesh};
        queued.stream.assign(command.stream, command.stream + command.streamSize);
        queued.convex = command.convex;
        Enqueue(std::move(queued));
        command.outCollisionMesh = mesh;
    }

    static void DestroyCollisionMesh(void*, const Context&, ShapeCmd::DestroyCollisionMesh& command)
    {
        ReleaseMesh(static_cast<Mesh*>(command.collisionMesh));
    }

    static void ExtractCollisionMeshData(void* object, const Context&, ShapeCmd::ExtractCollisionMeshData& command)
    {
        const Shape* shape = static_cast<const Shape*>(object);
        const MeshData* mesh = shape != nullptr ? MeshDataOf(shape->data) : nullptr;
        if (mesh == nullptr)
        {
            return;
        }

        const auto* vertices = reinterpret_cast<const Float3*>(mesh + 1);
        Physics::SDKCollisionMeshData extracted{};
        extracted.vertices = &vertices->x;
        extracted.vertexCount = mesh->vertexCount;
        extracted.vertexStride = sizeof(Float3);
        extracted.indices = vertices + mesh->vertexCount;
        extracted.indexCount = static_cast<size_t>(mesh->triangleCount) * 3;
        extracted.indexStride = sizeof(uint32_t);
        command.callback(command.ctx, extracted);
    }

    // Ray hits don't name a triangle (see QueryKernel), so there is no cooked index to translate.
    static void RemapTriangleIndexToSourceIndex(void*, const Context&, ShapeCmd::RemapTriangleIndexToSourceIndex& command)
    {
        command.outSourceMeshTriangleIndex = command.physicsMeshTriangleIndex;
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
