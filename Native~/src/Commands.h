// What the per-category files share: the binding tables, the logging path and the handler helpers.

#pragma once

#include <cstring>
#include <utility>

#include "PhysicsCommands.h"

// This header uses uint32_t and PhysicsCommands without including them itself.
#include "IUnityPhysicsBackendSystem.h"

#include "PhysicsBodyCommandData.h"
#include "PhysicsGeneralCommandData.h"
#include "PhysicsJointCommandData.h"
#include "PhysicsMaterialCommandData.h"
#include "PhysicsQueryCommandData.h"
#include "PhysicsShapeCommandData.h"
#include "PhysicsWorldCommandData.h"
#include "PhysXPhysicsExtensionCommands.h"

#include "Store.h"

namespace Retooled
{
    using PhysicsCommands::Command;
    using PhysicsCommands::Context;
    using PhysicsCommands::FunctionBinding;

    // The plugin API's command structs. Their namespaces share names with the store's records.
    namespace BodyCmd = PhysicsCommands::BodyData;
    namespace JointCmd = PhysicsCommands::JointData;
    namespace MaterialCmd = PhysicsCommands::MaterialData;
    namespace ShapeCmd = PhysicsCommands::ShapeData;
    namespace WorldCmd = PhysicsCommands::WorldData;

    constexpr uint32_t kBackendId = 0x52545048; // "RTPH"

    // Everything goes to the Unity log, or nowhere before the plugin has loaded.
    void LogInfo(const char* message);
    void LogError(const char* message);

    // Logs the first time an unimplemented command arrives, so that a gap shows up once instead of on every call.
    void LogUnimplementedOnce(uint32_t category, uint32_t id);

    // Binds a handler to the command of the same name: the handler's parameter names the command struct, and that
    // struct's `command` constant picks the table slot.
    template <typename T>
    using Handler = void (*)(void* object, const Context& ctx, T& command);

    template <typename T, Handler<T> Function>
    void Bound(void* object, const Context& ctx, Command& command)
    {
        Function(object, ctx, static_cast<T&>(command));
    }

    template <uint32_t Id, uint32_t Category>
    void SilentStub(void*, const Context&, Command&)
    {
        LogUnimplementedOnce(Category, Id);
    }

    template <uint32_t Category, uint32_t Count, typename Table>
    void FillWithStubs(Table& table)
    {
        [&]<uint32_t... Ids>(std::integer_sequence<uint32_t, Ids...>)
        {
            ((table[Ids] = &SilentStub<Ids, Category>), ...);
        }(std::make_integer_sequence<uint32_t, Count>{});
    }

    constexpr uint32_t kSdkCategory = PhysicsCommands::kSDKFunctionMask >> PhysicsCommands::kMaskOffsetSize;
    constexpr uint32_t kWorldCategory = PhysicsCommands::kWorldFunctionMask >> PhysicsCommands::kMaskOffsetSize;
    constexpr uint32_t kBodyCategory = PhysicsCommands::kBodyFunctionMask >> PhysicsCommands::kMaskOffsetSize;
    constexpr uint32_t kShapeCategory = PhysicsCommands::kShapeFunctionMask >> PhysicsCommands::kMaskOffsetSize;
    constexpr uint32_t kQueryCategory = PhysicsCommands::kQueryFunctionMask >> PhysicsCommands::kMaskOffsetSize;
    constexpr uint32_t kJointCategory = PhysicsCommands::kJointFunctionMask >> PhysicsCommands::kMaskOffsetSize;
    constexpr uint32_t kArticulationCategory = PhysicsCommands::kArticulationFunctionMask >> PhysicsCommands::kMaskOffsetSize;
    constexpr uint32_t kExtensionCategory = PhysicsCommands::kPhysXExtensionMask >> PhysicsCommands::kMaskOffsetSize;

    // Each file fills its own table. Plugin.cpp owns the tables and has already put a stub in every slot.
    void BindSdkCommands(FunctionBinding* table);
    void BindWorldCommands(FunctionBinding* table);
    void BindBodyCommands(FunctionBinding* table);
    void BindShapeCommands(FunctionBinding* table);
    void BindMeshCommands(FunctionBinding* table);
    void BindQueryCommands(FunctionBinding* table);
    void BindJointCommands(FunctionBinding* table);
    void BindExtensionCommands(FunctionBinding* table);

    // How many vehicle commands the plugin API has, which VehicleKernel.cs numbers the same way.
    int32_t VehicleCommandCount();

#ifdef RETOOLED_COMMAND_TRACE
    // Counts a GetPose that the plugin forwards to the kernel, as category 15 in the trace.
    void NoteForwardedPose();
#endif

#ifdef RETOOLED_VEHICLE_TRACE
    void BindVehicleTrace(FunctionBinding* table);
    void TraceCommand(void* object, const Context& ctx, uint32_t function);
#endif

    inline bool IsMeshGeometry(const ShapeData& shape)
    {
        const auto type = static_cast<Physics::GeometryType>(shape.geometryType);
        return type == Physics::GeometryType::ConvexMesh || type == Physics::GeometryType::TriangleMesh;
    }

    // The cooked mesh of a shape, or null if it has none yet.
    inline const MeshData* MeshDataOf(const ShapeData& shape)
    {
        return IsMeshGeometry(shape) && shape.mesh != nullptr ? static_cast<const Mesh*>(shape.mesh)->data : nullptr;
    }

    // Runs the cooks and loads that arrived while the core was detached. Main thread, after attaching.
    void FinishQueuedMeshes();

    // Meshes and materials are reference counted (see Store.h), and these accept null where a reference may be missing.
    void RetainMesh(Mesh* mesh);
    void ReleaseMesh(Mesh* mesh);
    void RetainMaterial(MaterialData* material);
    void ReleaseMaterial(MaterialData* material);

    // What a shape uses when Unity gives it no material.
    MaterialData* DefaultMaterial();

    // Fills the plugin's half of StoreGlobals. Before any command, so a kernel always finds it.
    void InitializeGlobals();

    inline Float3 ToFloat3(const Physics::Vec3& v)
    {
        return {v.x, v.y, v.z};
    }

    inline Float4 ToFloat4(const Physics::Vec4& v)
    {
        return {v.x, v.y, v.z, v.w};
    }

    inline Physics::Vec3 ToVec3(const Float3& v)
    {
        return Physics::Vec3(v.x, v.y, v.z);
    }

    inline Physics::Vec4 ToVec4(const Float4& v)
    {
        return Physics::Vec4(v.x, v.y, v.z, v.w);
    }

    inline void PackPose(CommandArgs& args, const Physics::Pose& pose)
    {
        args.position = ToFloat3(pose.pos);
        args.rotation = ToFloat4(pose.quat);
    }

    inline Physics::Pose UnpackPose(const CommandArgs& args)
    {
        return Physics::Pose(args.position, args.rotation);
    }

    // The unpacking the command files repeat. Pack fills CommandArgs from the command, and Unpack the command from it.
    inline constexpr auto None = [](auto&, auto&) {};
    inline constexpr auto ScalarIn = [](CommandArgs& args, const auto& command) { args.scalar = command.value; };
    inline constexpr auto ScalarOut = [](const CommandArgs& args, auto& command) { command.value = args.scalar; };
    inline constexpr auto VectorIn = [](CommandArgs& args, const auto& command) { args.vector = ToFloat3(command.value); };
    inline constexpr auto VectorOut = [](const CommandArgs& args, auto& command) { command.value = ToVec3(args.vector); };
    inline constexpr auto PoseIn = [](CommandArgs& args, const auto& command) { PackPose(args, command.value); };
    inline constexpr auto PoseOut = [](const CommandArgs& args, auto& command) { command.value = UnpackPose(args); };
    inline constexpr auto RotationIn = [](CommandArgs& args, const auto& command) { args.rotation = ToFloat4(command.value); };
    inline constexpr auto RotationOut = [](const CommandArgs& args, auto& command) { command.value = ToVec4(args.rotation); };

    // Runs a command in its kernel. Without a kernel, or for a handle Unity never got because it was created without one,
    // it is a no-op that answers zero and a rotation that is at least valid. That only happens while the core is away for
    // a domain reload, when no command is expected.
    template <typename T, CommandKernel ManagedCallbacks::*Kernel, auto Pack, auto Unpack, bool NeedsObject>
    void Forwarded(void* object, const Context&, Command& base)
    {
        T& command = static_cast<T&>(base);
        CommandArgs args{};
        args.rotation.w = 1.0f;

        const CommandKernel kernel = g_Managed.*Kernel;
        if (kernel != nullptr && (object != nullptr || !NeedsObject))
        {
            Pack(args, command);
            kernel(static_cast<uint32_t>(T::command), object, &args, &g_Globals);
        }

        Unpack(args, command);
    }
}
