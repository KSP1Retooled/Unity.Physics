// Query commands. The geometry work happens in the managed core. Queries can arrive from worker threads, so a query
// first makes sure the world's query structures are current, under the exclusive lock and only when something changed,
// and then runs under the shared lock.

#include <mutex>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    static World* WorldOf(void* object, const Context& ctx)
    {
        return static_cast<World*>(ctx.world != nullptr ? ctx.world : object);
    }

    static void SyncQueryWorld(World* world)
    {
        std::atomic_ref<uint32_t> dirty(world->data.queryDirty);
        if (dirty.load(std::memory_order_acquire) == 0)
        {
            return;
        }

        std::unique_lock lock(world->queryLock);
        if (dirty.load(std::memory_order_acquire) != 0)
        {
            g_Managed.buildQueryWorld(&world->data);
            dirty.store(0, std::memory_order_release);
        }
    }

    static void RayCast(void* object, const Context& ctx, QueryData::RayCast& command)
    {
        command.hitCount = 0;
        World* world = WorldOf(object, ctx);
        if (g_Managed.rayCast == nullptr || world == nullptr)
        {
            return;
        }

        SyncQueryWorld(world);

        const RayCastInput input = {
            {command.origin.x, command.origin.y, command.origin.z},
            {command.direction.x, command.direction.y, command.direction.z},
            command.maxDistance,
            command.filter.mask,
            static_cast<uint32_t>(command.filter.options),
        };

        RayCastHit hit{};
        std::shared_lock lock(world->queryLock);
        if (g_Managed.rayCast(&world->data, &input, &hit) != 0)
        {
            static_assert(sizeof(RayCastHit) == sizeof(Physics::QueryHit));
            std::memcpy(static_cast<void*>(&command.hit), &hit, sizeof(hit));
            command.hitCount = 1;
        }
    }

    template <typename T>
    static void BindStub(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &ZeroedStub<T, kQueryCategory>;
    }

    void BindQueryCommands(FunctionBinding* table)
    {
        table[static_cast<size_t>(QueryData::RayCast::command)] = &Bound<QueryData::RayCast, RayCast>;

        BindStub<QueryData::ComputeShapePenetration>(table);
        BindStub<QueryData::ClosestPointOnShape>(table);
        BindStub<QueryData::RayCastAgainstShape>(table);
        BindStub<QueryData::RaycastWithCollector>(table);
        BindStub<QueryData::Overlap>(table);
        BindStub<QueryData::OverlapWithCollector>(table);
        BindStub<QueryData::OverlapWithBroadphaseCollector>(table);
        BindStub<QueryData::ShapeCast>(table);
        BindStub<QueryData::ShapeCastAgainstShape>(table);
        BindStub<QueryData::ShapeCastWithCollector>(table);
        BindStub<QueryData::BodyCast>(table);
        BindStub<QueryData::BodyCastWithCollector>(table);
    }
}
