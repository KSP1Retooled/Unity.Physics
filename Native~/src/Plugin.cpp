// Plugin entry: registers the backend with Unity and routes commands to the per-category tables. Every slot of every
// table holds a handler before Register, as the plugin API requires.

#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "IUnityLog.h"

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    StoreGlobals g_Globals{};
    ManagedCallbacks g_Managed{};
    std::thread::id g_MainThread;
    std::vector<World*> g_Worlds;
    std::atomic<World*> g_ReadBackWorld{nullptr};

    static IUnityLog* s_Log;
    static bool s_Registered;
    static std::atomic<bool> s_Attached;

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

    static void LogWarning(const char* message)
    {
        if (s_Log != nullptr)
        {
            s_Log->Log(kUnityLogTypeWarning, message, __FILE__, __LINE__);
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

    // Commands received while managed code was detached, counted until the next attach.
    static std::atomic<uint32_t> s_Detached[16][256];

    static void NoteDetachedCommand(uint32_t category, uint32_t id)
    {
        if (category < 16 && id < 256)
        {
            s_Detached[category][id].fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Commands dropped because they required the managed core. Report their names at the next attach.
    static std::atomic<uint32_t> s_Dropped[16][256];

    // Set by NoteDroppedCommand while ProcessCommand runs a command on this thread.
    static thread_local bool t_Dropped;

    void NoteDroppedCommand()
    {
        t_Dropped = true;
    }

    // Appends "category.id x count" for every non-zero count, resets the counts, and returns whether there were any.
    static bool TakeCounts(std::atomic<uint32_t> (&counts)[16][256], std::string& text)
    {
        bool any = false;
        for (uint32_t category = 0; category < 16; category++)
        {
            for (uint32_t id = 0; id < 256; id++)
            {
                const uint32_t count = counts[category][id].exchange(0, std::memory_order_relaxed);
                if (count != 0)
                {
                    text += " " + std::to_string(category) + "." + std::to_string(id) + "x" + std::to_string(count);
                    any = true;
                }
            }
        }

        return any;
    }

    static void LogDetachedCommands()
    {
        std::string text = "[RetooledPhysics] commands while detached (category.id x count):";
        if (TakeCounts(s_Detached, text))
        {
            LogInfo(text.c_str());
        }

        std::string dropped = "[RetooledPhysics] commands that were lost because the managed core was detached (category.id x count):";
        if (TakeCounts(s_Dropped, dropped))
        {
            LogWarning(dropped.c_str());
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
        BindArticulationCommands(s_Articulation.data());
        BindExtensionCommands(s_Extension.data());
    }

#ifdef RETOOLED_COMMAND_TRACE
    // How many commands of each category and id arrived, on the main thread ([0]) and on other threads ([1]), since the
    // last read. Tests read and reset the counts between the stages of a step to see which commands each stage sends.
    static std::atomic<uint32_t> s_Trace[2][16][256];

    void NoteForwardedPose()
    {
        const uint32_t thread = std::this_thread::get_id() == g_MainThread ? 0 : 1;
        s_Trace[thread][15][static_cast<uint32_t>(BodyCmd::GetPose::command)].fetch_add(1, std::memory_order_relaxed);
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_ReadCommandTrace(uint32_t* counts)
    {
        for (uint32_t thread = 0; thread < 2; thread++)
        {
            for (uint32_t category = 0; category < 16; category++)
            {
                for (uint32_t id = 0; id < 256; id++)
                {
                    counts[(thread * 16 + category) * 256 + id] = s_Trace[thread][category][id].exchange(0, std::memory_order_relaxed);
                }
            }
        }
    }
#endif

    static void ProcessCommand(void* object, const Context& ctx, const uint32_t function, Command& command)
    {
        const uint32_t id = function & kReverseFullMask;
        const uint32_t category = (function & kFullMask) >> kMaskOffsetSize;
        if (!s_Attached.load(std::memory_order_relaxed))
        {
            NoteDetachedCommand(category, id);
        }

        // A command can come back in here from a callback that it makes, so keep the flag of the outer command.
        const bool outerDropped = t_Dropped;
        t_Dropped = false;

#ifdef RETOOLED_COMMAND_TRACE
        if (id < 256)
        {
            const uint32_t thread = std::this_thread::get_id() == g_MainThread ? 0 : 1;
            s_Trace[thread][(function & kFullMask) >> kMaskOffsetSize][id].fetch_add(1, std::memory_order_relaxed);
        }
#endif

        // Keep the read-back open while Unity reads body and link transforms and flags.
        const uint32_t mask = function & kFullMask;
        const bool bodyRead = mask == kBodyFunctionMask
            && (id == static_cast<uint32_t>(BodyCmd::GetPose::command) || id == static_cast<uint32_t>(BodyCmd::GetFlags::command)
                || id == static_cast<uint32_t>(BodyCmd::GetIsSleeping::command));
        const bool linkRead = mask == kArticulationFunctionMask
            && (id == static_cast<uint32_t>(Articulation::GetLinkPose) || id == static_cast<uint32_t>(Articulation::GetLinkFlags)
                || id == static_cast<uint32_t>(Articulation::GetIsSleeping));
        if (!bodyRead && !linkRead)
        {
            CloseReadBack();
        }

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

        if (t_Dropped && category < 16 && id < 256)
        {
            s_Dropped[category][id].fetch_add(1, std::memory_order_relaxed);
        }

        t_Dropped = outerDropped;
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

        InitializeGlobals();
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

    // The managed core attaches through these. FirstSizeMismatch reads the sizes in this order and the command counts after
    // them, which catches a plugin API that numbers its commands differently from the kernels' enums.
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
                return sizeof(QueryInput);
            case 6:
                return sizeof(RayCastHit);
            case 7:
                return sizeof(ManagedCallbacks);
            case 8:
                return sizeof(MeshData);
            case 9:
                return sizeof(CookInput);
            case 10:
                return sizeof(CookOutput);
            case 11:
                return sizeof(Body);
            case 12:
                return sizeof(Shape);
            case 13:
                return sizeof(CommandArgs);
            case 14:
                return static_cast<int32_t>(PhysicsCommands::Body::Count);
            case 15:
                return static_cast<int32_t>(PhysicsCommands::Shape::Count);
            case 16:
                return static_cast<int32_t>(PhysicsCommands::World::Count);
            case 17:
                return sizeof(JointBreak);
            case 18:
                return static_cast<int32_t>(PhysicsCommands::Joint::Count);
            case 19:
                return sizeof(HeightField);
            case 20:
                return VehicleCommandCount();
            case 21:
                return static_cast<int32_t>(PhysicsCommands::Articulation::Count);
            default:
                return -1;
        }
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_Attach(const ManagedCallbacks* callbacks)
    {
        {
            std::unique_lock lock(g_MeshLock);
            g_Managed = *callbacks;
        }

        g_MainThread = std::this_thread::get_id();
        s_Attached.store(true, std::memory_order_relaxed);
        LogDetachedCommands();
        FinishQueuedMeshes();
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_SyncQueryWorlds()
    {
        SyncAllQueryWorlds();
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_Detach()
    {
        s_Attached.store(false, std::memory_order_relaxed);
        {
            std::unique_lock lock(g_MeshLock);
            g_Managed = {};
        }

        // A query reads the callbacks under the world's lock, so passing through each lock waits out any query already
        // running, and managed can free what those queries read.
        for (World* world : g_Worlds)
        {
            std::unique_lock lock(world->queryLock);
        }
    }

    extern "C" UNITY_INTERFACE_EXPORT const StoreGlobals* UNITY_INTERFACE_API RetooledPhysics_Globals()
    {
        return &g_Globals;
    }

    extern "C" UNITY_INTERFACE_EXPORT int32_t UNITY_INTERFACE_API RetooledPhysics_WorldCount()
    {
        return static_cast<int32_t>(g_Worlds.size());
    }

    // Whether the caller is the thread that attached. Unity sends some commands from job workers, and a Burst kernel has
    // no other way to tell which thread it is on.
    extern "C" UNITY_INTERFACE_EXPORT int32_t UNITY_INTERFACE_API RetooledPhysics_OnMainThread()
    {
        return std::this_thread::get_id() == g_MainThread ? 1 : 0;
    }

    extern "C" UNITY_INTERFACE_EXPORT WorldData* UNITY_INTERFACE_API RetooledPhysics_GetWorld(int32_t index)
    {
        return &g_Worlds[index]->data;
    }
}
