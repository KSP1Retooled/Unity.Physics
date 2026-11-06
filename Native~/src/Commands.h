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

    // A command that is not implemented yet and has outputs: Unity reads the fields back, so they are zeroed, and zero
    // means "nothing" throughout the command set (no hit, no count, a null handle, false).
    template <typename T, uint32_t Category>
    void ZeroedStub(void*, const Context&, Command& command)
    {
        LogUnimplementedOnce(Category, static_cast<uint32_t>(T::command));
        std::memset(static_cast<void*>(&static_cast<T&>(command)), 0, sizeof(T));
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

#ifdef RETOOLED_VEHICLE_TRACE
    void BindVehicleTrace(FunctionBinding* table);
    void TraceCommand(void* object, const Context& ctx, uint32_t function);
#endif

    // The handlers World.cpp forwards to, so that each kind of record is created and destroyed in its own file.
    Joint* CreateJointRecord(const WorldCmd::CreateJoint& command);
    void DestroyJointRecord(Joint* joint);

    // Detaches a body's shapes when the body goes away; the shapes outlive it until Unity destroys them.
    void DetachAllShapes(Body* body);

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

    // The straight part of a capsule. Despite its name in the plugin API, CapsuleGeometry::height holds half of this,
    // like PxCapsuleGeometry::halfHeight.
    inline float CapsuleCylinderLength(const ShapeData& shape)
    {
        return 2.0f * shape.extents.y;
    }

    // What the mass calculation needs of one shape, in the shape's own frame: its volume, its centre of mass, and its
    // inertia per unit mass about that centre. False for shapes the calculation skips.
    bool ShapeMassProperties(const ShapeData& shape, float& volume, Float3& center, float (&inertiaPerUnitMass)[3][3]);

    // The shape's world-space axis-aligned bounds under a body pose.
    void ShapeWorldBounds(const ShapeData& shape, const Float3& bodyPosition, const Float4& bodyRotation, Float3& min, Float3& max);

    // Small rigid-transform helpers.
    Float3 Rotate(const Float4& q, const Float3& v);
    Float4 Multiply(const Float4& a, const Float4& b);

    // Unity gives a static body's shape a world-space pose in AttachShape and SetPose, unlike a dynamic body's, whatever
    // the plugin API's comments say. The store keeps every shape relative to its body.
    void StoreStaticShapePose(ShapeData& shape, const BodyData& body, const Physics::Pose& worldPose);
}
