// Query commands. The geometry work happens in the managed core, in a Burst function native code calls directly, against
// a query world that holds nothing of the live store. Each handler here only unpacks a command into a QueryInput.
//
// Only the main thread rebuilds that world: it does so after a step, on a main-thread query, and when managed asks. A
// worker reads the world last published, because it can't build one, and waiting for the main thread could deadlock with
// the main thread completing the worker's job. So a worker sees the scene as of the last rebuild.

#include <cstdio>
#include <mutex>

#include "Commands.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

using namespace PhysicsCommands;

namespace Retooled
{
    static_assert(sizeof(RayCastHit) == sizeof(Physics::QueryHit));

    static World* WorldOf(void* object, const Context& ctx)
    {
        return static_cast<World*>(ctx.world != nullptr ? ctx.world : object);
    }

    static World* WorldOf(Shape* shape)
    {
        return shape != nullptr && shape->body != nullptr ? shape->body->world : nullptr;
    }

    static bool IsDirty(World* world)
    {
        return std::atomic_ref<uint32_t>(world->data.queryDirty).load(std::memory_order_acquire) != 0;
    }

    static void SyncQueryWorld(World* world)
    {
        std::unique_lock lock(world->queryLock);
        if (g_Managed.buildQueryWorld == nullptr || !IsDirty(world))
        {
            return;
        }

        // Any other answer leaves the world dirty, so the next query tries again.
        if (g_Managed.buildQueryWorld(&world->data) == kQueryWorldBuilt)
        {
            std::atomic_ref<uint32_t>(world->data.queryDirty).store(0, std::memory_order_release);
        }
    }

    void SyncAllQueryWorlds()
    {
        for (World* world : g_Worlds)
        {
            SyncQueryWorld(world);
        }
    }

    // Says once which thread first read a stale world. Nothing else can show that, because a native thread has no managed
    // frames to read it from.
    static void NoteStaleQuery()
    {
        static std::atomic<bool> noted{false};
        if (noted.exchange(true))
        {
            return;
        }

        char name[128] = "unnamed";
#ifdef _WIN32
        PWSTR description = nullptr;
        if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &description)) && description != nullptr)
        {
            WideCharToMultiByte(CP_UTF8, 0, description, -1, name, sizeof(name), nullptr, nullptr);
            LocalFree(description);
        }
#endif
        char text[192];
        std::snprintf(text, sizeof(text), "[RetooledPhysics] first query from a worker on a stale world, on thread '%s'", name);
        LogInfo(text);
    }

    // Runs one query under the shared lock. It reports nothing found if the core is detached or no world is published.
    static bool Query(World* world, QueryKind kind, const QueryInput& input, QueryOutput& output)
    {
        if (world == nullptr)
        {
            return false;
        }

        if (IsDirty(world))
        {
            if (std::this_thread::get_id() == g_MainThread)
            {
                SyncQueryWorld(world);
            }
            else
            {
                NoteStaleQuery();
            }
        }

        std::shared_lock lock(world->queryLock);
        return g_Managed.query != nullptr && world->data.queryWorld != nullptr
            && g_Managed.query(kind, world->data.queryWorld, &input, &output) != 0;
    }

    static void CopyHit(const QueryOutput& output, Physics::QueryHit& hit)
    {
        std::memcpy(static_cast<void*>(&hit), &output.hit, sizeof(hit));
    }

    // A hit on a shape the caller named comes from the live record, which the query function doesn't read ids from.
    static void CopyEngineObjectId(const Shape* shape, Physics::QueryHit& hit)
    {
        const char* owner = static_cast<const char*>(shape->data.userData);
        std::memcpy(&hit.userData, owner + g_Globals.engineObjectIdOffset, sizeof(hit.userData));
    }

    static void PackFilter(QueryInput& input, const Physics::QueryFilterData& filter)
    {
        input.layerMask = filter.mask;
        input.filterOptions = static_cast<uint32_t>(filter.options);
    }

    static void PackGeometry(QueryInput& input, const Physics::ShapeGeometry& geometry, const Physics::Pose& pose)
    {
        input.rotation = ToFloat4(pose.quat);
        input.position = ToFloat3(pose.pos);
        input.geometryType = static_cast<uint32_t>(geometry.type);
        switch (geometry.type)
        {
            case Physics::GeometryType::Box:
                input.extents = ToFloat3(geometry.boxGeometry.halfExtents);
                break;
            case Physics::GeometryType::Sphere:
                input.extents.x = geometry.sphereGeometry.radius;
                break;
            case Physics::GeometryType::Capsule:
                input.extents.x = geometry.capsuleGeometry.radius;
                input.extents.y = geometry.capsuleGeometry.height;
                break;
            default:
                break;
        }
    }

    static void RayCast(void* object, const Context& ctx, QueryData::RayCast& command)
    {
        command.hitCount = 0;

        QueryInput input{};
        input.position = ToFloat3(command.origin);
        input.direction = ToFloat3(command.direction);
        input.maxDistance = command.maxDistance;
        PackFilter(input, command.filter);

        QueryOutput output{};
        if (Query(WorldOf(object, ctx), kQueryRay, input, output))
        {
            CopyHit(output, command.hit);
            command.hitCount = 1;
        }
    }

    static void RaycastWithCollector(void* object, const Context& ctx, QueryData::RaycastWithCollector& command)
    {
        QueryInput input{};
        input.position = ToFloat3(command.origin);
        input.direction = ToFloat3(command.direction);
        input.maxDistance = command.maxDistance;
        input.collector = reinterpret_cast<void*>(command.collector);
        input.collectorContext = command.dataContext;
        PackFilter(input, command.filter);

        QueryOutput output{};
        Query(WorldOf(object, ctx), kQueryRayAll, input, output);
    }

    static void RayCastAgainstShape(void* object, const Context&, QueryData::RayCastAgainstShape& command)
    {
        command.hitCount = 0;

        Shape* shape = static_cast<Shape*>(object);
        QueryInput input{};
        input.position = ToFloat3(command.origin);
        input.direction = ToFloat3(command.direction);
        input.maxDistance = command.maxDistance;
        input.filterOptions = static_cast<uint32_t>(command.filterOptions);
        input.shape = shape;

        QueryOutput output{};
        if (Query(WorldOf(shape), kQueryRayAgainstShape, input, output))
        {
            CopyHit(output, command.hit);
            CopyEngineObjectId(shape, command.hit);
            command.hitCount = 1;
        }
    }

    static void Overlap(void* object, const Context& ctx, QueryData::Overlap& command)
    {
        command.shapeUserData = nullptr;

        QueryInput input{};
        PackGeometry(input, command.geometry, command.pose);
        PackFilter(input, command.filter);

        QueryOutput output{};
        if (Query(WorldOf(object, ctx), kQueryOverlapAny, input, output))
        {
            command.shapeUserData = output.shapeUserData;
        }
    }

    static void OverlapWithCollector(void* object, const Context& ctx, QueryData::OverlapWithCollector& command)
    {
        QueryInput input{};
        PackGeometry(input, command.geometry, command.pose);
        PackFilter(input, command.filter);
        input.collector = reinterpret_cast<void*>(command.collector);
        input.collectorContext = command.dataContext;

        QueryOutput output{};
        Query(WorldOf(object, ctx), kQueryOverlapAll, input, output);
    }

    static void OverlapWithBroadphaseCollector(void* object, const Context& ctx, QueryData::OverlapWithBroadphaseCollector& command)
    {
        QueryInput input{};
        PackGeometry(input, command.geometry, command.pose);
        PackFilter(input, command.filter);
        input.collector = reinterpret_cast<void*>(command.collector);
        input.collectorContext = command.dataContext;

        QueryOutput output{};
        Query(WorldOf(object, ctx), kQueryOverlapBroadphase, input, output);
    }

    static void ShapeCast(void* object, const Context& ctx, QueryData::ShapeCast& command)
    {
        command.hitCount = 0;

        QueryInput input{};
        PackGeometry(input, command.geometry, command.pose);
        PackFilter(input, command.filter);
        input.direction = ToFloat3(command.direction);
        input.maxDistance = command.maxDistance;

        QueryOutput output{};
        if (Query(WorldOf(object, ctx), kQueryCast, input, output))
        {
            CopyHit(output, command.hit);
            command.hitCount = 1;
        }
    }

    static void ShapeCastWithCollector(void* object, const Context& ctx, QueryData::ShapeCastWithCollector& command)
    {
        QueryInput input{};
        PackGeometry(input, command.geometry, command.pose);
        PackFilter(input, command.filter);
        input.direction = ToFloat3(command.direction);
        input.maxDistance = command.maxDistance;
        input.keepInitialOverlaps = command.keepInitialOverlaps ? 1 : 0;
        input.collector = reinterpret_cast<void*>(command.collector);
        input.collectorContext = command.dataContext;

        QueryOutput output{};
        Query(WorldOf(object, ctx), kQueryCastAll, input, output);
    }

    static void BodyCast(void* object, const Context& ctx, QueryData::BodyCast& command)
    {
        command.hitCount = 0;

        QueryInput input{};
        PackFilter(input, command.filter);
        input.direction = ToFloat3(command.direction);
        input.maxDistance = command.maxDistance;
        input.body = static_cast<Body*>(command.body);

        QueryOutput output{};
        if (input.body != nullptr && Query(input.body->world, kQueryBodyCast, input, output))
        {
            CopyHit(output, command.hit);
            command.hitCount = 1;
        }
    }

    static void BodyCastWithCollector(void* object, const Context& ctx, QueryData::BodyCastWithCollector& command)
    {
        QueryInput input{};
        PackFilter(input, command.filter);
        input.direction = ToFloat3(command.direction);
        input.maxDistance = command.maxDistance;
        input.body = static_cast<Body*>(command.body);
        input.collector = reinterpret_cast<void*>(command.collector);
        input.collectorContext = command.dataContext;

        QueryOutput output{};
        if (input.body != nullptr)
        {
            Query(input.body->world, kQueryBodyCastAll, input, output);
        }
    }

    static void ClosestPointOnShape(void* object, const Context&, QueryData::ClosestPointOnShape& command)
    {
        // A shape that isn't in the scene has no surface to offer, which reads like a point inside it.
        command.outPoint = command.point;
        command.outDistance = 0.0f;

        Shape* shape = static_cast<Shape*>(object);
        QueryInput input{};
        input.rotation = ToFloat4(command.pose.quat);
        input.position = ToFloat3(command.pose.pos);
        input.position2 = ToFloat3(command.point);
        input.shape = shape;

        QueryOutput output{};
        if (Query(WorldOf(shape), kQueryClosestPoint, input, output))
        {
            command.outPoint = ToVec3(output.hit.point);
            command.outDistance = output.hit.distance;
        }
    }

    static void ComputeShapePenetration(void*, const Context&, QueryData::ComputeShapePenetration& command)
    {
        command.areOverlapping = false;

        Shape* shapeA = static_cast<Shape*>(command.shapeA);
        Shape* shapeB = static_cast<Shape*>(command.shapeB);

        QueryInput input{};
        input.rotation = ToFloat4(command.poseA.quat);
        input.position = ToFloat3(command.poseA.pos);
        input.rotation2 = ToFloat4(command.poseB.quat);
        input.position2 = ToFloat3(command.poseB.pos);
        input.shape = shapeA;
        input.shape2 = shapeB;

        QueryOutput output{};
        World* world = WorldOf(shapeA);
        if (world != nullptr && WorldOf(shapeB) != nullptr && Query(world, kQueryPenetration, input, output))
        {
            command.outDirection = ToVec3(output.hit.normal);
            command.outDistance = output.hit.distance;
            command.areOverlapping = true;
        }
    }

    // Sweeps the geometry against the one shape the command addresses, which is how particles collide once the broadphase
    // has named the shapes near them.
    static void ShapeCastAgainstShape(void* object, const Context& ctx, QueryData::ShapeCastAgainstShape& command)
    {
        command.hitCount = 0;

        Shape* shape = static_cast<Shape*>(object);
        QueryInput input{};
        PackGeometry(input, command.targetGeometry, command.targetPose);
        PackFilter(input, command.filter);
        input.direction = ToFloat3(command.direction);
        input.maxDistance = command.maxDistance;
        input.keepInitialOverlaps = command.distanceAsDepth ? 1 : 0;
        input.shape = shape;

        QueryOutput output{};
        if (Query(ctx.world != nullptr ? static_cast<World*>(ctx.world) : WorldOf(shape), kQueryCastAgainstShape, input, output))
        {
            CopyHit(output, command.hit);
            CopyEngineObjectId(shape, command.hit);
            command.hitCount = 1;
        }
    }

    void BindQueryCommands(FunctionBinding* table)
    {
        table[static_cast<size_t>(QueryData::RayCast::command)] = &Bound<QueryData::RayCast, RayCast>;
        table[static_cast<size_t>(QueryData::RaycastWithCollector::command)] = &Bound<QueryData::RaycastWithCollector, RaycastWithCollector>;
        table[static_cast<size_t>(QueryData::RayCastAgainstShape::command)] = &Bound<QueryData::RayCastAgainstShape, RayCastAgainstShape>;
        table[static_cast<size_t>(QueryData::Overlap::command)] = &Bound<QueryData::Overlap, Overlap>;
        table[static_cast<size_t>(QueryData::OverlapWithCollector::command)] = &Bound<QueryData::OverlapWithCollector, OverlapWithCollector>;
        table[static_cast<size_t>(QueryData::OverlapWithBroadphaseCollector::command)] = &Bound<QueryData::OverlapWithBroadphaseCollector, OverlapWithBroadphaseCollector>;
        table[static_cast<size_t>(QueryData::ShapeCast::command)] = &Bound<QueryData::ShapeCast, ShapeCast>;
        table[static_cast<size_t>(QueryData::ShapeCastWithCollector::command)] = &Bound<QueryData::ShapeCastWithCollector, ShapeCastWithCollector>;
        table[static_cast<size_t>(QueryData::BodyCast::command)] = &Bound<QueryData::BodyCast, BodyCast>;
        table[static_cast<size_t>(QueryData::BodyCastWithCollector::command)] = &Bound<QueryData::BodyCastWithCollector, BodyCastWithCollector>;
        table[static_cast<size_t>(QueryData::ClosestPointOnShape::command)] = &Bound<QueryData::ClosestPointOnShape, ClosestPointOnShape>;
        table[static_cast<size_t>(QueryData::ComputeShapePenetration::command)] = &Bound<QueryData::ComputeShapePenetration, ComputeShapePenetration>;

        table[static_cast<size_t>(QueryData::ShapeCastAgainstShape::command)] = &Bound<QueryData::ShapeCastAgainstShape, ShapeCastAgainstShape>;
    }
}
