// Shape commands. Each one is unpacked here and runs in ShapeKernel.cs. Mesh.cpp holds the mesh commands. Height field
// commands are stubs, so a TerrainCollider has no collision.

#include <new>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    // The union is the plugin API's, and a kernel sees one shape of data for every geometry: a type, three numbers and a handle.
    static void PackGeometry(CommandArgs& args, const Physics::ShapeGeometry& geometry)
    {
        args.mode = static_cast<uint32_t>(geometry.type);
        switch (geometry.type)
        {
            case Physics::GeometryType::Box:
                args.vector = ToFloat3(geometry.boxGeometry.halfExtents);
                break;
            case Physics::GeometryType::Sphere:
                args.vector.x = geometry.sphereGeometry.radius;
                break;
            case Physics::GeometryType::Capsule:
                args.vector.x = geometry.capsuleGeometry.radius;
                args.vector.y = geometry.capsuleGeometry.height;
                break;
            case Physics::GeometryType::ConvexMesh:
            case Physics::GeometryType::TriangleMesh:
                args.vector = ToFloat3(geometry.meshGeometry.scale);
                args.pointer = geometry.meshGeometry.mesh;
                break;
            case Physics::GeometryType::Terrain:
                args.vector = ToFloat3(geometry.heightFieldGeometry.scale);
                args.pointer = geometry.heightFieldGeometry.heightField;
                break;
            default:
                break;
        }
    }

    static Physics::ShapeGeometry UnpackGeometry(const CommandArgs& args)
    {
        Physics::ShapeGeometry geometry{};
        geometry.type = static_cast<Physics::GeometryType>(args.mode);
        switch (geometry.type)
        {
            case Physics::GeometryType::Box:
                geometry.boxGeometry.halfExtents = ToVec3(args.vector);
                break;
            case Physics::GeometryType::Sphere:
                geometry.sphereGeometry.radius = args.vector.x;
                break;
            case Physics::GeometryType::Capsule:
                geometry.capsuleGeometry.radius = args.vector.x;
                geometry.capsuleGeometry.height = args.vector.y;
                break;
            case Physics::GeometryType::ConvexMesh:
            case Physics::GeometryType::TriangleMesh:
                geometry.meshGeometry.mesh = args.pointer;
                geometry.meshGeometry.scale = ToVec3(args.vector);
                break;
            case Physics::GeometryType::Terrain:
                geometry.heightFieldGeometry.heightField = args.pointer;
                geometry.heightFieldGeometry.scale = ToVec3(args.vector);
                break;
            default:
                break;
        }

        return geometry;
    }

    template <typename T, auto Pack, auto Unpack = None, bool NeedsObject = true>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Forwarded<T, &ManagedCallbacks::shapeCommand, Pack, Unpack, NeedsObject>;
    }

    static constexpr auto BoundsOut = [](const CommandArgs& a, auto& c)
    {
        c.center = ToVec3(a.vector);
        c.extents = ToVec3(a.vector2);
    };

    // A kernel can't construct the plugin API's type, whose defaults aren't all zero.
    static constexpr auto CreatedOut = [](const CommandArgs& a, auto& c)
    {
        c.value = a.pointer;
        if (c.value != nullptr)
        {
            new (&static_cast<Shape*>(c.value)->filter) Physics::FilterData();
        }
    };

    static void SetFilterData(void* object, const Context& ctx, Command& base)
    {
        auto& command = static_cast<ShapeCmd::SetFilterData&>(base);
        if (object != nullptr)
        {
            static_cast<Shape*>(object)->filter = command.data;
        }

        Forwarded<ShapeCmd::SetFilterData, &ManagedCallbacks::shapeCommand, [](CommandArgs& a, const auto& c)
        {
            a.value = c.data.belongsTo;
            a.value2 = c.data.collidesWith;
            a.mode = c.data.ignorePairID;
            a.enabled = c.data.hasCollisionDisabled;
        }, None, true>(object, ctx, base);
    }

    static void GetFilterData(void* object, const Context&, Command& base)
    {
        if (object != nullptr)
        {
            static_cast<ShapeCmd::GetFilterData&>(base).data = static_cast<Shape*>(object)->filter;
        }
    }

    static void SetSupportedMessages(void* object, const Context&, Command& base)
    {
        if (object != nullptr)
        {
            static_cast<Shape*>(object)->filter.pairFlags = static_cast<ShapeCmd::SetSupportedMessages&>(base).flags;
        }
    }

    static void SetModifiableContacts(void*, const Context&, Command&)
    {
    }

    template <typename T>
    static void BindStub(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &ZeroedStub<T, kShapeCategory>;
    }

    void BindShapeCommands(FunctionBinding* table)
    {
        table[static_cast<size_t>(ShapeCmd::SetFilterData::command)] = &SetFilterData;
        table[static_cast<size_t>(ShapeCmd::GetFilterData::command)] = &GetFilterData;
        table[static_cast<size_t>(ShapeCmd::SetSupportedMessages::command)] = &SetSupportedMessages;
        table[static_cast<size_t>(ShapeCmd::SetModifiableContacts::command)] = &SetModifiableContacts;

        Bind<ShapeCmd::CreateShape, [](CommandArgs& a, const auto& c)
        {
            PackGeometry(a, c.geom);
            a.pointer2 = c.materialPtr;
            a.pointer3 = c.userData;
            a.flags = static_cast<uint32_t>(c.flags);
        }, CreatedOut, false>(table);
        Bind<ShapeCmd::DestroyShape, None>(table);
        Bind<ShapeCmd::GetActorDescriptor, None, [](const CommandArgs& a, auto& c)
        {
            c.value = Physics::ActorDescriptor{};
            c.value.actor = a.pointer;
            c.value.userData = a.pointer2;
            c.value.type = static_cast<Physics::ActorType>(a.mode);
        }>(table);
        Bind<ShapeCmd::GetUserData, None, [](const CommandArgs& a, auto& c) { c.value = a.pointer; }>(table);
        Bind<ShapeCmd::GetFlags, None, [](const CommandArgs& a, auto& c) { c.flags = static_cast<Physics::ShapeFlags>(a.flags); }>(table);
        Bind<ShapeCmd::SetFlags, [](CommandArgs& a, const auto& c) { a.flags = static_cast<uint32_t>(c.flags); }>(table);
        Bind<ShapeCmd::SetMaterial, [](CommandArgs& a, const auto& c) { a.pointer = c.value; }>(table);
        Bind<ShapeCmd::GetLocalPose, None, PoseOut>(table);
        Bind<ShapeCmd::GetPose, None, PoseOut>(table);
        Bind<ShapeCmd::SetPose, PoseIn>(table);
        Bind<ShapeCmd::SetContactOffset, ScalarIn>(table);
        Bind<ShapeCmd::GetWorldBounds, None, BoundsOut>(table);
        Bind<ShapeCmd::GetBoundsAtPose, [](CommandArgs& a, const auto& c) { PackPose(a, c.pose); }, BoundsOut>(table);
        Bind<ShapeCmd::ShouldIgnoreCollision, [](CommandArgs& a, const auto& c)
        {
            a.value = c.ignoreId0;
            a.value2 = c.ignoreId1;
        }, [](const CommandArgs& a, auto& c) { c.value = a.enabled != 0; }, false>(table);
        Bind<ShapeCmd::IgnoreCollision, [](CommandArgs& a, const auto& c)
        {
            a.value = c.ignoreId0;
            a.value2 = c.ignoreId1;
            a.enabled = c.value;
        }, None, false>(table);
        Bind<ShapeCmd::CleanupIgnoredColliders, [](CommandArgs& a, const auto& c) { a.value = c.ignoreID; }, None, false>(table);
        Bind<ShapeCmd::GetGeometryType, None, [](const CommandArgs& a, auto& c) { c.value = static_cast<Physics::GeometryType>(a.mode); }>(table);
        Bind<ShapeCmd::SetGeometry, [](CommandArgs& a, const auto& c) { PackGeometry(a, c.value); }>(table);
        Bind<ShapeCmd::GetGeometry, None, [](const CommandArgs& a, auto& c) { c.value = UnpackGeometry(a); }>(table);

        BindMeshCommands(table);

        BindStub<ShapeCmd::CreateHeightField>(table);
    }
}
