// World commands. They address the world through the object handle, and bodies are created here because the dense
// array belongs to the world.

#include <algorithm>
#include <cstdlib>
#include <mutex>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    // Rigidbody.maxLinearVelocity's default; Unity sends the rest of a body's limits after creating it.
    static constexpr float kDefaultMaxLinearVelocity = 1e16f;
    static constexpr float kDefaultMaxAngularVelocity = 7.0f;
    static constexpr float kDefaultMaxDepenetrationVelocity = 1e32f;
    static constexpr float kDefaultSleepThreshold = 0.005f;
    static constexpr float kDefaultWakeCounter = 0.4f;

    static World* AsWorld(void* object)
    {
        return static_cast<World*>(object);
    }

    static void GrowBodies(WorldData& data)
    {
        const uint32_t capacity = std::max<uint32_t>(32, data.bodyCapacity * 2);
        data.bodies = static_cast<BodyData*>(std::realloc(data.bodies, capacity * sizeof(BodyData)));
        data.bodyCapacity = capacity;
    }

    static void CreateWorld(void*, const Context&, WorldCmd::CreateWorld& command)
    {
        World* world = new World();
        world->userData = command.userData;
        world->data.owner = world;
        world->data.gravity = {command.gravity.x, command.gravity.y, command.gravity.z};
        world->data.bounceThreshold = command.bounceThreshold;
        world->data.queryDirty = 1;
        g_Worlds.push_back(world);
        command.outWorld = world;
    }

    static void DestroyWorld(void* object, const Context&, WorldCmd::DestroyWorld&)
    {
        World* world = AsWorld(object);
        if (g_Managed.destroyWorld != nullptr)
        {
            // Waits out a query still reading what managed is about to free.
            std::unique_lock lock(world->queryLock);
            g_Managed.destroyWorld(&world->data);
        }

        // Unity destroys bodies before their world, so this only frees storage. A shape can outlive its body.
        for (uint32_t i = 0; i < world->data.bodyCount; i++)
        {
            Body* body = world->data.bodies[i].owner;
            DetachAllShapes(body);
            delete body;
        }

        g_Worlds.erase(std::find(g_Worlds.begin(), g_Worlds.end(), world));
        std::free(world->data.bodies);
        std::free(world->data.deadColliders);
        delete world;
    }

    static void SetGravity(void* object, const Context&, WorldCmd::SetGravity& command)
    {
        AsWorld(object)->data.gravity = {command.value.x, command.value.y, command.value.z};
    }

    static void GetGravity(void* object, const Context&, WorldCmd::GetGravity& command)
    {
        const Float3& gravity = AsWorld(object)->data.gravity;
        command.value = Physics::Vec3(gravity.x, gravity.y, gravity.z);
    }

    static void SetBounceThreshold(void* object, const Context&, WorldCmd::SetBounceThreshold& command)
    {
        AsWorld(object)->data.bounceThreshold = command.value;
    }

    static void GetBounceThreshold(void* object, const Context&, WorldCmd::GetBounceThreshold& command)
    {
        command.value = AsWorld(object)->data.bounceThreshold;
    }

    static void GetBasicStats(void* object, const Context&, WorldCmd::GetBasicStats& command)
    {
        command.value = Physics::WorldStats(0, AsWorld(object)->data.bodyCount, 0);
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

    static void GetLastSimulationStats(void* object, const Context&, WorldCmd::GetLastSimulationStepStats& command)
    {
        command.value = Physics::SimulationStepStats();
        command.value.dynamicBodies = AsWorld(object)->data.bodyCount;
    }

    static void GetUserData(void* object, const Context&, WorldCmd::GetUserData& command)
    {
        command.userData = AsWorld(object)->userData;
    }

    static void CreateBody(void* object, const Context&, WorldCmd::CreateBody& command)
    {
        World* world = AsWorld(object);
        std::unique_lock lock(world->queryLock);

        WorldData& data = world->data;
        if (data.bodyCount == data.bodyCapacity)
        {
            GrowBodies(data);
        }

        Body* body = new Body();
        body->world = world;
        body->index = data.bodyCount++;
        body->userData = command.userData;

        BodyData hot{};
        hot.rotation = {command.pose.quat.x, command.pose.quat.y, command.pose.quat.z, command.pose.quat.w};
        hot.position = {command.pose.pos.x, command.pose.pos.y, command.pose.pos.z};
        hot.flags = static_cast<uint32_t>(command.flags) | (command.isStatic ? kBodyStatic : 0u) | kBodyColliderDirty;
        hot.mass = command.mass;
        hot.inverseMass = command.mass > 0.0f ? 1.0f / command.mass : 0.0f;
        hot.inertia = {1.0f, 1.0f, 1.0f};
        hot.inverseInertia = {1.0f, 1.0f, 1.0f};
        hot.inertiaRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        hot.linearDamping = command.linearDamping;
        hot.angularDamping = command.angularDamping;
        hot.maxLinearVelocity = kDefaultMaxLinearVelocity;
        hot.maxAngularVelocity = kDefaultMaxAngularVelocity;
        hot.maxDepenetrationVelocity = kDefaultMaxDepenetrationVelocity;
        hot.wakeCounter = kDefaultWakeCounter;
        hot.sleepThreshold = kDefaultSleepThreshold;
        hot.targetRotation = hot.rotation;
        hot.targetPosition = hot.position;
        hot.owner = body;
        data.bodies[body->index] = hot;

        data.staticsVersion++;
        MarkQueryDirty(world);
        command.outBody = body;
    }

    static void AddActorsToScene(void*, const Context&, WorldCmd::AddActorsToScene&)
    {
        // A body is part of the scene from CreateBody on, whether or not Unity batches it through here.
    }

    static void DestroyBody(void* object, const Context&, WorldCmd::DestroyBody& command)
    {
        World* world = AsWorld(object);
        Body* body = static_cast<Body*>(command.value);
        DetachAllShapes(body);

        std::unique_lock lock(world->queryLock);
        WorldData& data = world->data;
        BodyData& hot = data.bodies[body->index];
        if (hot.collider != nullptr)
        {
            if (data.deadColliderCount == data.deadColliderCapacity)
            {
                data.deadColliderCapacity = std::max<uint32_t>(32, data.deadColliderCapacity * 2);
                data.deadColliders = static_cast<void**>(std::realloc(data.deadColliders, data.deadColliderCapacity * sizeof(void*)));
            }

            data.deadColliders[data.deadColliderCount++] = hot.collider;
        }

        const uint32_t last = --data.bodyCount;
        if (body->index != last)
        {
            hot = data.bodies[last];
            hot.owner->index = body->index;
        }

        delete body;
        data.staticsVersion++;
        MarkQueryDirty(world);
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

    static void GetAllShapesByLayerMask(void* object, const Context&, WorldCmd::GetAllShapesByLayerMask& command)
    {
        const WorldData& data = AsWorld(object)->data;
        std::vector<void*> found;
        for (uint32_t i = 0; i < data.bodyCount; i++)
        {
            const BodyData& hot = data.bodies[i];
            for (uint32_t s = 0; s < hot.shapeCount; s++)
            {
                const ShapeData& shape = hot.shapes[s]->data;
                if ((shape.belongsTo & command.layerMask) != 0)
                {
                    found.push_back(shape.userData);
                }
            }
        }

        command.callback(found.data(), found.size());
    }

    template <typename T, Handler<T> Function>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Bound<T, Function>;
    }

    void BindWorldCommands(FunctionBinding* table)
    {
        Bind<WorldCmd::CreateWorld, CreateWorld>(table);
        Bind<WorldCmd::DestroyWorld, DestroyWorld>(table);
        Bind<WorldCmd::SetGravity, SetGravity>(table);
        Bind<WorldCmd::GetGravity, GetGravity>(table);
        Bind<WorldCmd::SetBounceThreshold, SetBounceThreshold>(table);
        Bind<WorldCmd::GetBounceThreshold, GetBounceThreshold>(table);
        Bind<WorldCmd::GetBasicStats, GetBasicStats>(table);
        Bind<WorldCmd::SetScratchBufferChunkCount, SetScratchBufferChunkCount>(table);
        Bind<WorldCmd::Simulate, Simulate>(table);
        Bind<WorldCmd::GetLastSimulationStepResults, GetLastSimulationStepResults>(table);
        Bind<WorldCmd::ReleaseLastSimulationStepData, ReleaseLastSimulationStepData>(table);
        Bind<WorldCmd::GetLastSimulationStepStats, GetLastSimulationStats>(table);
        Bind<WorldCmd::GetUserData, GetUserData>(table);
        Bind<WorldCmd::CreateBody, CreateBody>(table);
        Bind<WorldCmd::AddActorsToScene, AddActorsToScene>(table);
        Bind<WorldCmd::DestroyBody, DestroyBody>(table);
        Bind<WorldCmd::CreateArticulation, CreateArticulation>(table);
        Bind<WorldCmd::DestoryArticulation, DestroyArticulation>(table);
        Bind<WorldCmd::CreateJoint, CreateJoint>(table);
        Bind<WorldCmd::DestroyJoint, DestroyJoint>(table);
        Bind<WorldCmd::GetArticulationCount, GetArticulationCount>(table);
        Bind<WorldCmd::GetArticulations, GetArticulations>(table);
        Bind<WorldCmd::GetAllShapesByLayerMask, GetAllShapesByLayerMask>(table);
    }
}
