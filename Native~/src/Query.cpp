// Query commands. The geometry work happens in the managed core, in a Burst function native code calls directly, against
// a query world that holds nothing of the live store.
//
// Only the main thread rebuilds that world: it does so after a step, on a main-thread query, and when managed asks. A
// worker reads the world last published, because it can't build one, and waiting for the main thread could deadlock with
// the main thread completing the worker's job. So a worker sees the scene as of the last rebuild. New query commands can
// use Query below to get the same rules.

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
    static World* WorldOf(void* object, const Context& ctx)
    {
        return static_cast<World*>(ctx.world != nullptr ? ctx.world : object);
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

    // Runs query(queryWorld) under the shared lock. It does nothing if the core is detached or no world is published,
    // which reads as no hit.
    template <typename Function>
    static void Query(World* world, Function query)
    {
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
        if (g_Managed.rayCast != nullptr && world->data.queryWorld != nullptr)
        {
            query(world->data.queryWorld);
        }
    }

    static void RayCast(void* object, const Context& ctx, QueryData::RayCast& command)
    {
        command.hitCount = 0;
        World* world = WorldOf(object, ctx);
        if (world == nullptr)
        {
            return;
        }

        const RayCastInput input = {
            {command.origin.x, command.origin.y, command.origin.z},
            {command.direction.x, command.direction.y, command.direction.z},
            command.maxDistance,
            command.filter.mask,
            static_cast<uint32_t>(command.filter.options),
        };

        Query(world, [&](void* queryWorld)
        {
            RayCastHit hit{};
            if (g_Managed.rayCast(queryWorld, &input, &hit) != 0)
            {
                static_assert(sizeof(RayCastHit) == sizeof(Physics::QueryHit));
                std::memcpy(static_cast<void*>(&command.hit), &hit, sizeof(hit));
                command.hitCount = 1;
            }
        });
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
