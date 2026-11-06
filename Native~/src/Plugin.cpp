// Plugin entry: registers the backend with Unity and routes commands to the per-category tables. Every slot of every
// table holds a handler before Register, as the plugin API requires.

#include <array>
#include <atomic>
#include <cstdio>
#include <utility>

#include "IUnityLog.h"

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    StoreGlobals g_Globals{};
    ManagedCallbacks g_Managed{};
    std::vector<World*> g_Worlds;
    std::unordered_set<uint32_t> g_IgnoredPairs;

    static IUnityLog* s_Log;
    static bool s_Registered;

    static std::array<FunctionBinding, static_cast<size_t>(PhysicsCommands::SDK::Count)> s_Sdk;
    static std::array<FunctionBinding, static_cast<size_t>(PhysicsCommands::World::Count)> s_World;
    static std::array<FunctionBinding, static_cast<size_t>(PhysicsCommands::Body::Count)> s_Body;
    static std::array<FunctionBinding, static_cast<size_t>(PhysicsCommands::Shape::Count)> s_Shape;
    static std::array<FunctionBinding, static_cast<size_t>(PhysicsCommands::Query::Count)> s_Query;
    static std::array<FunctionBinding, static_cast<size_t>(PhysicsCommands::Joint::Count)> s_Joint;
    static std::array<FunctionBinding, static_cast<size_t>(PhysicsCommands::Articulation::Count)> s_Articulation;
    static std::array<FunctionBinding, static_cast<size_t>(PhysXExt::PhysXExtension::Count)> s_Extension;

    void LogInfo(const char* message)
    {
        if (s_Log != nullptr)
        {
            s_Log->Log(kUnityLogTypeLog, message, __FILE__, __LINE__);
        }
    }

    void LogError(const char* message)
    {
        if (s_Log != nullptr)
        {
            s_Log->Log(kUnityLogTypeError, message, __FILE__, __LINE__);
        }
    }

    void LogUnimplementedOnce(uint32_t category, uint32_t id)
    {
        static std::atomic<bool> seen[16][256];
        if (category < 16 && id < 256 && !seen[category][id].exchange(true))
        {
            char text[96];
            std::snprintf(text, sizeof(text), "[RetooledPhysics] unimplemented command %u.%u", category, id);
            LogInfo(text);
        }
    }

    static void BindTables()
    {
        FillWithStubs<kSdkCategory, static_cast<uint32_t>(PhysicsCommands::SDK::Count)>(s_Sdk);
        FillWithStubs<kWorldCategory, static_cast<uint32_t>(PhysicsCommands::World::Count)>(s_World);
        FillWithStubs<kBodyCategory, static_cast<uint32_t>(PhysicsCommands::Body::Count)>(s_Body);
        FillWithStubs<kShapeCategory, static_cast<uint32_t>(PhysicsCommands::Shape::Count)>(s_Shape);
        FillWithStubs<kQueryCategory, static_cast<uint32_t>(PhysicsCommands::Query::Count)>(s_Query);
        FillWithStubs<kJointCategory, static_cast<uint32_t>(PhysicsCommands::Joint::Count)>(s_Joint);
        FillWithStubs<kArticulationCategory, static_cast<uint32_t>(PhysicsCommands::Articulation::Count)>(s_Articulation);
        FillWithStubs<kExtensionCategory, static_cast<uint32_t>(PhysXExt::PhysXExtension::Count)>(s_Extension);

        BindSdkCommands(s_Sdk.data());
        BindWorldCommands(s_World.data());
        BindBodyCommands(s_Body.data());
        BindShapeCommands(s_Shape.data());
        BindQueryCommands(s_Query.data());
        BindJointCommands(s_Joint.data());
        BindExtensionCommands(s_Extension.data());
    }

    static void ProcessCommand(void* object, const Context& ctx, const uint32_t function, Command& command)
    {
        const uint32_t id = function & kReverseFullMask;
#ifdef RETOOLED_VEHICLE_TRACE
        if ((function & kFullMask) != kPhysXExtensionMask)
        {
            TraceCommand(object, ctx, function);
        }
#endif
        switch (function & kFullMask)
        {
            case kSDKFunctionMask:
                s_Sdk[id](object, ctx, command);
                break;
            case kWorldFunctionMask:
                s_World[id](object, ctx, command);
                break;
            case kBodyFunctionMask:
                s_Body[id](object, ctx, command);
                break;
            case kShapeFunctionMask:
                s_Shape[id](object, ctx, command);
                break;
            case kQueryFunctionMask:
                s_Query[id](object, ctx, command);
                break;
            case kJointFunctionMask:
                s_Joint[id](object, ctx, command);
                break;
            case kArticulationFunctionMask:
                s_Articulation[id](object, ctx, command);
                break;
            case kPhysXExtensionMask:
                s_Extension[id](object, ctx, command);
                break;
            default:
                break;
        }
    }

    extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* interfaces)
    {
        if (s_Registered)
        {
            return;
        }

        s_Log = interfaces->Get<IUnityLog>();
        IUnityPhysicsBackendSystem* system = interfaces->Get<IUnityPhysicsBackendSystem>();
        if (system == nullptr)
        {
            LogError("[RetooledPhysics] this editor has no physics backend interface");
            return;
        }

        BindTables();
        s_Registered = system->Register(reinterpret_cast<BackendEntryPoint>(ProcessCommand));
        if (!s_Registered)
        {
            LogError("[RetooledPhysics] backend registration failed");
        }
    }

    extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
    {
        g_Managed = {};
        s_Log = nullptr;
    }

    // The managed core attaches through these. Sizes are indexed by NativeStore.StructId.
    extern "C" UNITY_INTERFACE_EXPORT int32_t UNITY_INTERFACE_API RetooledPhysics_StructSize(int32_t id)
    {
        switch (id)
        {
            case 0:
                return sizeof(BodyData);
            case 1:
                return sizeof(ShapeData);
            case 2:
                return sizeof(MaterialData);
            case 3:
                return sizeof(WorldData);
            case 4:
                return sizeof(StoreGlobals);
            case 5:
                return sizeof(RayCastInput);
            case 6:
                return sizeof(RayCastHit);
            case 7:
                return sizeof(ManagedCallbacks);
            default:
                return -1;
        }
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_Attach(const ManagedCallbacks* callbacks)
    {
        g_Managed = *callbacks;
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_Detach()
    {
        g_Managed = {};
    }

    extern "C" UNITY_INTERFACE_EXPORT const StoreGlobals* UNITY_INTERFACE_API RetooledPhysics_Globals()
    {
        return &g_Globals;
    }

    extern "C" UNITY_INTERFACE_EXPORT int32_t UNITY_INTERFACE_API RetooledPhysics_WorldCount()
    {
        return static_cast<int32_t>(g_Worlds.size());
    }

    extern "C" UNITY_INTERFACE_EXPORT WorldData* UNITY_INTERFACE_API RetooledPhysics_GetWorld(int32_t index)
    {
        return &g_Worlds[index]->data;
    }
}
