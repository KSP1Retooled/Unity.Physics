// SDK lifecycle and materials. Material commands address the material through the object handle.

#include <cstring>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
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
        info.features = Physics::FeatureSupport::DynamicsSupport;
#ifdef RETOOLED_VEHICLE_TRACE
        info.features |= Physics::FeatureSupport::PhysXVehicleSupport;
#endif
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
        command.outMaterial = material;
    }

    static void ReleaseMaterial(void* object, const Context&, MaterialCmd::ReleaseMaterial&)
    {
        auto* material = static_cast<Retooled::MaterialData*>(object);
        if (material != &s_DefaultMaterial)
        {
            delete material;
        }
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
        Bind<MaterialCmd::ReleaseMaterial, ReleaseMaterial>(table);
        Bind<MaterialCmd::GetDefaultMaterial, GetDefaultMaterial>(table);
        Bind<MaterialCmd::SetImprovedPatchFrictionEnabledGlobal, SetImprovedPatchFriction>(table);
        Bind<MaterialCmd::SetMaterialConfiguration, SetMaterialConfiguration>(table);
        Bind<MaterialCmd::GetMaterialConfiguration, GetMaterialConfiguration>(table);
        Bind<MaterialCmd::SetMaterialCombineModes, SetMaterialCombineModes>(table);
        Bind<MaterialCmd::GetMaterialCombineModes, GetMaterialCombineModes>(table);
    }
}
