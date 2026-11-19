// SDK lifecycle and materials. Material commands address the material through the object handle.

#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
#ifdef RETOOLED_STORE_QUARANTINE
    struct RetiredStoreBlock
    {
        void* address;
        uint64_t releaseStep;
    };

    // Construct only when checking is enabled. Managed jobs may allocate at the same time.
    struct StoreQuarantine
    {
        std::mutex mutex;
        std::unordered_map<void*, size_t> live;
        std::deque<RetiredStoreBlock> retired;
        uint64_t step = 0;

        ~StoreQuarantine()
        {
            for (const RetiredStoreBlock& block : retired)
            {
                std::free(block.address);
            }
        }
    };

    static StoreQuarantine s_Quarantine;

    // Call with the quarantine locked. The pointer remains allocated but will read as NaN through a float view.
    static void RetireStoreBlock(StoreQuarantine& quarantine, void* memory)
    {
        if (memory == nullptr)
        {
            return;
        }

        auto entry = quarantine.live.find(memory);
        if (entry == quarantine.live.end())
        {
            LogError("[RetooledPhysics] store allocator received a release for a non-live allocation");
            std::abort();
        }

        std::memset(memory, 0xff, entry->second);
        quarantine.retired.push_back({ memory, quarantine.step + 4 });
        quarantine.live.erase(entry);
    }

    static void ReleaseCheckedStoreBlock(void* memory)
    {
        StoreQuarantine& quarantine = s_Quarantine;
        std::lock_guard lock(quarantine.mutex);
        RetireStoreBlock(quarantine, memory);
    }

    static void* ReallocateCheckedStoreBlock(void* memory, uint64_t size)
    {
        StoreQuarantine& quarantine = s_Quarantine;
        std::lock_guard lock(quarantine.mutex);
        if (memory != nullptr && size == 0)
        {
            RetireStoreBlock(quarantine, memory);
            return nullptr;
        }

        void* replacement = std::malloc(static_cast<size_t>(size));
        if (replacement == nullptr)
        {
            return nullptr;
        }

        if (memory != nullptr)
        {
            auto entry = quarantine.live.find(memory);
            if (entry == quarantine.live.end())
            {
                LogError("[RetooledPhysics] store allocator received a resize for a non-live allocation");
                std::abort();
            }
            size_t oldSize = entry->second;
            size_t copySize = oldSize < size ? oldSize : static_cast<size_t>(size);
            std::memcpy(replacement, memory, copySize);
            RetireStoreBlock(quarantine, memory);
        }

        quarantine.live.emplace(replacement, static_cast<size_t>(size));
        return replacement;
    }

    void AdvanceStoreQuarantine()
    {
        StoreQuarantine& quarantine = s_Quarantine;
        std::lock_guard lock(quarantine.mutex);
        ++quarantine.step;
        while (!quarantine.retired.empty() && quarantine.retired.front().releaseStep <= quarantine.step)
        {
            std::free(quarantine.retired.front().address);
            quarantine.retired.pop_front();
        }
    }
#endif

    // PhysicMaterial's defaults; Unity overwrites them for every material it creates.
    static constexpr float kDefaultFriction = 0.6f;

    static MaterialData s_DefaultMaterial = {
        kDefaultFriction,
        kDefaultFriction,
        0.0f,
        static_cast<uint32_t>(Physics::MaterialCombineMode::Average),
        static_cast<uint32_t>(Physics::MaterialCombineMode::Average),
        0,
        nullptr,
    };

    MaterialData* DefaultMaterial()
    {
        return &s_DefaultMaterial;
    }

    void RetainMaterial(MaterialData* material)
    {
        if (material != nullptr && material != &s_DefaultMaterial)
        {
            std::atomic_ref<uint32_t>(material->references).fetch_add(1, std::memory_order_relaxed);
        }
    }

    void ReleaseMaterial(MaterialData* material)
    {
        if (material != nullptr
            && material != &s_DefaultMaterial
            && std::atomic_ref<uint32_t>(material->references).fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            delete material;
        }
    }

    void InitializeGlobals()
    {
        g_Globals.defaultMaterial = &s_DefaultMaterial;
#ifdef RETOOLED_STORE_QUARANTINE
        g_Globals.reallocate = ReallocateCheckedStoreBlock;
        g_Globals.release = ReleaseCheckedStoreBlock;
        LogInfo("[RetooledPhysics] store memory quarantine enabled");
#else
        g_Globals.reallocate = [](void* memory, uint64_t size) { return std::realloc(memory, static_cast<size_t>(size)); };
        g_Globals.release = std::free;
#endif
        g_Globals.retainMesh = RetainMesh;
        g_Globals.releaseMesh = ReleaseMesh;
        g_Globals.retainMaterial = RetainMaterial;
        g_Globals.releaseMaterial = ReleaseMaterial;
    }

    static void InitializeSdk(void*, const Context&, SdkData::InitializeSdk& command)
    {
        g_Globals.engineObjectIdOffset = command.engineObjectIdOffset;
        command.result = true;
    }

    static void ShutdownSdk(void*, const Context&, SdkData::ShutdownSdk&)
    {
    }

    static void GetIntegrationInfo(void*, const Context&, SdkData::GetSdkIntegrationInfo& command)
    {
        Physics::IntegrationInfo& info = command.value;
        info.id = kBackendId;
        info.version[0] = 1;
        info.features = Physics::FeatureSupport::DynamicsSupport | Physics::FeatureSupport::ArticulationSupport
            | Physics::FeatureSupport::PhysXVehicleSupport;
        std::strcpy(info.name, "RetooledPhysics");
        std::strcpy(info.desc, "Unity Physics backend for KSP Retooled");
    }

    static void SetErrorVerbosityLevel(void*, const Context&, SdkData::SetErrorVerbosityLevel&)
    {
    }

    static void ConnectVisualDebugger(void*, const Context&, SdkData::ConnectVisualDebugger& command)
    {
        command.result = false;
    }

    static void DisconnectVisualDebugger(void*, const Context&, SdkData::DisconnectVisualDebugger&)
    {
    }

    static void GetIsVisualDebuggerConnected(void*, const Context&, SdkData::GetIsVisualDebuggerConnected& command)
    {
        command.value = false;
    }

    static void SetVisualDebuggerViewport(void*, const Context&, SdkData::SetVisualDebuggerViewport&)
    {
    }

    static void CreateMaterial(void*, const Context&, MaterialCmd::CreateMaterial& command)
    {
        auto* material = new Retooled::MaterialData(s_DefaultMaterial);
        material->userData = command.userData;
        material->references = 1;
        command.outMaterial = material;
    }

    static void ReleaseMaterialCommand(void* object, const Context&, MaterialCmd::ReleaseMaterial&)
    {
        ReleaseMaterial(static_cast<Retooled::MaterialData*>(object));
    }

    static void GetDefaultMaterial(void*, const Context&, MaterialCmd::GetDefaultMaterial& command)
    {
        command.value = &s_DefaultMaterial;
    }

    static void SetImprovedPatchFriction(void*, const Context&, MaterialCmd::SetImprovedPatchFrictionEnabledGlobal&)
    {
    }

    static void SetMaterialConfiguration(void* object, const Context&, MaterialCmd::SetMaterialConfiguration& command)
    {
        auto* material = static_cast<Retooled::MaterialData*>(object);
        material->dynamicFriction = command.dynamicFriction;
        material->staticFriction = command.staticFriction;
        material->bounciness = command.bounciness;
        g_Globals.materialVersion++;
    }

    static void GetMaterialConfiguration(void* object, const Context&, MaterialCmd::GetMaterialConfiguration& command)
    {
        auto* material = static_cast<Retooled::MaterialData*>(object);
        command.dynamicFriction = material->dynamicFriction;
        command.staticFriction = material->staticFriction;
        command.bounciness = material->bounciness;
    }

    static void SetMaterialCombineModes(void* object, const Context&, MaterialCmd::SetMaterialCombineModes& command)
    {
        auto* material = static_cast<Retooled::MaterialData*>(object);
        material->frictionCombine = static_cast<uint32_t>(command.friction);
        material->bounceCombine = static_cast<uint32_t>(command.bounciness);
        g_Globals.materialVersion++;
    }

    static void GetMaterialCombineModes(void* object, const Context&, MaterialCmd::GetMaterialCombineModes& command)
    {
        auto* material = static_cast<Retooled::MaterialData*>(object);
        command.friction = static_cast<Physics::MaterialCombineMode>(material->frictionCombine);
        command.bounciness = static_cast<Physics::MaterialCombineMode>(material->bounceCombine);
    }

    template <typename T, Handler<T> Function>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Bound<T, Function>;
    }

    void BindSdkCommands(FunctionBinding* table)
    {
        Bind<SdkData::InitializeSdk, InitializeSdk>(table);
        Bind<SdkData::ShutdownSdk, ShutdownSdk>(table);
        Bind<SdkData::GetSdkIntegrationInfo, GetIntegrationInfo>(table);
        Bind<SdkData::SetErrorVerbosityLevel, SetErrorVerbosityLevel>(table);
        Bind<SdkData::ConnectVisualDebugger, ConnectVisualDebugger>(table);
        Bind<SdkData::DisconnectVisualDebugger, DisconnectVisualDebugger>(table);
        Bind<SdkData::GetIsVisualDebuggerConnected, GetIsVisualDebuggerConnected>(table);
        Bind<SdkData::SetVisualDebuggerViewport, SetVisualDebuggerViewport>(table);
        Bind<MaterialCmd::CreateMaterial, CreateMaterial>(table);
        Bind<MaterialCmd::ReleaseMaterial, ReleaseMaterialCommand>(table);
        Bind<MaterialCmd::GetDefaultMaterial, GetDefaultMaterial>(table);
        Bind<MaterialCmd::SetImprovedPatchFrictionEnabledGlobal, SetImprovedPatchFriction>(table);
        Bind<MaterialCmd::SetMaterialConfiguration, SetMaterialConfiguration>(table);
        Bind<MaterialCmd::GetMaterialConfiguration, GetMaterialConfiguration>(table);
        Bind<MaterialCmd::SetMaterialCombineModes, SetMaterialCombineModes>(table);
        Bind<MaterialCmd::GetMaterialCombineModes, GetMaterialCombineModes>(table);
    }
}
