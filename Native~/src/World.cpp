// World commands. They address the world through the object handle. A world and its settings stay here because Unity
// creates and configures the scene before any managed code exists; a world's bodies and counts run in WorldKernel.cs.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <mutex>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    // WorldKernel.BodyCounts writes these through a pointer.
    static_assert(offsetof(Physics::SimulationStepStats, dynamicBodies) == 0);
    static_assert(offsetof(Physics::SimulationStepStats, activeDynamicBodies) == 4);
    static_assert(offsetof(Physics::SimulationStepStats, kinematicBodies) == 8);
    static_assert(offsetof(Physics::SimulationStepStats, activeKinematicBodies) == 12);
    static_assert(offsetof(Physics::SimulationStepStats, staticBodies) == 16);

    static World* AsWorld(void* object)
    {
        return static_cast<World*>(object);
    }

    static void CreateWorld(void*, const Context&, WorldCmd::CreateWorld& command)
    {
        World* world = new World();
        world->userData = command.userData;
        world->data.owner = world;
        world->data.gravity = ToFloat3(command.gravity);
        world->data.bounceThreshold = command.bounceThreshold;
        world->data.queryDirty = 1;
        g_Worlds.push_back(world);
        command.outWorld = world;
    }

    // Managed frees the bodies. Detached, which only happens after quitting here, they go with the process.
    static void DestroyWorld(void* object, const Context&, WorldCmd::DestroyWorld&)
    {
        World* world = AsWorld(object);
        if (g_Managed.destroyWorld != nullptr)
        {
            // Waits out a query still reading what managed is about to free.
            std::unique_lock lock(world->queryLock);
            g_Managed.destroyWorld(&world->data);
        }

        g_Worlds.erase(std::find(g_Worlds.begin(), g_Worlds.end(), world));
        std::free(world->data.bodies);
        std::free(world->data.deadColliders);
        delete world;
    }

    static void SetGravity(void* object, const Context&, WorldCmd::SetGravity& command)
    {
        AsWorld(object)->data.gravity = ToFloat3(command.value);
    }

    static void GetGravity(void* object, const Context&, WorldCmd::GetGravity& command)
    {
        command.value = ToVec3(AsWorld(object)->data.gravity);
    }

    static void SetBounceThreshold(void* object, const Context&, WorldCmd::SetBounceThreshold& command)
    {
        AsWorld(object)->data.bounceThreshold = command.value;
    }

    static void GetBounceThreshold(void* object, const Context&, WorldCmd::GetBounceThreshold& command)
    {
        command.value = AsWorld(object)->data.bounceThreshold;
    }

    static void SetScratchBufferChunkCount(void*, const Context&, WorldCmd::SetScratchBufferChunkCount&)
    {
    }

    static void Simulate(void* object, const Context&, WorldCmd::Simulate& command)
    {
        World* world = AsWorld(object);
        if (g_Managed.simulate == nullptr)
        {
            command.ranSimulation = false;
            return;
        }

        std::unique_lock lock(world->queryLock);
        g_Managed.simulate(&world->data, command.deltaTime);
        command.ranSimulation = true;
    }

    static void GetLastSimulationStepResults(void*, const Context&, WorldCmd::GetLastSimulationStepResults& command)
    {
        command.bodyContactPairs = nullptr;
        command.bodyContactPairsCount = 0;
        command.triggerEvents = nullptr;
        command.triggerEventsCount = 0;
        command.jointBreakEvents = nullptr;
        command.jointBreakEventsCount = 0;
    }

    static void ReleaseLastSimulationStepData(void*, const Context&, WorldCmd::ReleaseLastSimulationStepData&)
    {
    }

    static void GetUserData(void* object, const Context&, WorldCmd::GetUserData& command)
    {
        command.userData = AsWorld(object)->userData;
    }

    static void AddActorsToScene(void*, const Context&, WorldCmd::AddActorsToScene&)
    {
        // A body is part of the scene from CreateBody on, whether or not Unity batches it through here.
    }

    static void CreateArticulation(void*, const Context&, WorldCmd::CreateArticulation& command)
    {
        LogUnimplementedOnce(kWorldCategory, static_cast<uint32_t>(PhysicsCommands::World::CreateArticulation));
        command.outArticulation = nullptr;
    }

    static void DestroyArticulation(void*, const Context&, WorldCmd::DestoryArticulation&)
    {
    }

    static void CreateJoint(void*, const Context&, WorldCmd::CreateJoint& command)
    {
        command.outJoint = CreateJointRecord(command);
    }

    static void DestroyJoint(void*, const Context&, WorldCmd::DestroyJoint& command)
    {
        DestroyJointRecord(static_cast<Joint*>(command.value));
    }

    static void GetArticulationCount(void*, const Context&, WorldCmd::GetArticulationCount& command)
    {
        command.value = 0;
    }

    static void GetArticulations(void*, const Context&, WorldCmd::GetArticulations& command)
    {
        command.written = 0;
    }

    template <typename T, Handler<T> Function>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Bound<T, Function>;
    }

    // A command that runs in the kernel, addressed by its world's WorldData, which the handle also points at.
    template <typename T, auto Pack, auto Unpack = None>
    static void Forward(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Forwarded<T, &ManagedCallbacks::worldCommand, Pack, Unpack, true>;
    }

    void BindWorldCommands(FunctionBinding* table)
    {
        Bind<WorldCmd::CreateWorld, CreateWorld>(table);
        Bind<WorldCmd::DestroyWorld, DestroyWorld>(table);
        Bind<WorldCmd::SetGravity, SetGravity>(table);
        Bind<WorldCmd::GetGravity, GetGravity>(table);
        Bind<WorldCmd::SetBounceThreshold, SetBounceThreshold>(table);
        Bind<WorldCmd::GetBounceThreshold, GetBounceThreshold>(table);
        Bind<WorldCmd::SetScratchBufferChunkCount, SetScratchBufferChunkCount>(table);
        Bind<WorldCmd::Simulate, Simulate>(table);
        Bind<WorldCmd::GetLastSimulationStepResults, GetLastSimulationStepResults>(table);
        Bind<WorldCmd::ReleaseLastSimulationStepData, ReleaseLastSimulationStepData>(table);
        Bind<WorldCmd::GetUserData, GetUserData>(table);
        Bind<WorldCmd::AddActorsToScene, AddActorsToScene>(table);
        Bind<WorldCmd::CreateArticulation, CreateArticulation>(table);
        Bind<WorldCmd::DestoryArticulation, DestroyArticulation>(table);
        Bind<WorldCmd::CreateJoint, CreateJoint>(table);
        Bind<WorldCmd::DestroyJoint, DestroyJoint>(table);
        Bind<WorldCmd::GetArticulationCount, GetArticulationCount>(table);
        Bind<WorldCmd::GetArticulations, GetArticulations>(table);

        Forward<WorldCmd::GetBasicStats, None, [](const CommandArgs& a, auto& c) { c.value = Physics::WorldStats(0, a.value, 0); }>(table);
        Forward<WorldCmd::GetLastSimulationStepStats, [](CommandArgs& a, auto& c)
        {
            c.value = Physics::SimulationStepStats();
            a.pointer = &c.value;
        }>(table);
        Forward<WorldCmd::CreateBody, [](CommandArgs& a, const auto& c)
        {
            PackPose(a, c.pose);
            a.pointer = c.userData;
            a.scalar = c.mass;
            a.scalar2 = c.linearDamping;
            a.scalar3 = c.angularDamping;
            a.flags = static_cast<uint32_t>(c.flags);
            a.enabled = c.isStatic;
        }, [](const CommandArgs& a, auto& c) { c.outBody = a.pointer; }>(table);
        Forward<WorldCmd::DestroyBody, [](CommandArgs& a, const auto& c) { a.pointer = c.value; }>(table);
        Forward<WorldCmd::GetAllShapesByLayerMask, [](CommandArgs& a, const auto& c) { a.value = c.layerMask; }, [](const CommandArgs& a, auto& c)
        {
            c.callback(static_cast<void**>(a.pointer), static_cast<size_t>(a.size));
            std::free(a.pointer);
        }>(table);
    }
}
