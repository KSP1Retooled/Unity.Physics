// Shape commands. Each one is unpacked here and runs in ShapeKernel.cs, except for the few that only read what the plugin
// holds itself. Mesh.cpp holds the mesh commands.

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
            a.size = static_cast<uint64_t>(c.data.pairFlags);
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
            Shape* shape = static_cast<Shape*>(object);
            shape->filter.pairFlags = static_cast<ShapeCmd::SetSupportedMessages&>(base).flags;
            shape->messages = static_cast<uint32_t>(shape->filter.pairFlags);
        }
    }

    static void SetModifiableContacts(void*, const Context&, Command&)
    {
    }

    // Answer frequent actor descriptor reads natively using only cold records and row-owned flags, which workers
    // may read.
    static void GetActorDescriptor(void* object, const Context&, Command& base)
    {
        Physics::ActorDescriptor& descriptor = static_cast<ShapeCmd::GetActorDescriptor&>(base).value;
        descriptor = Physics::ActorDescriptor{};
        const Shape* shape = static_cast<const Shape*>(object);
        if (shape == nullptr || shape->body == nullptr)
        {
            return;
        }

        const Body* body = shape->body;
        const bool isStatic = (BodyRow(body->world->data, body->index)->flags & kBodyStatic) != 0;
        descriptor.actor = shape->body;
        descriptor.userData = body->userData;
        descriptor.type = isStatic ? Physics::ActorType::Static : Physics::ActorType::Dynamic;
    }

    static void PackHeightField(CommandArgs& args, const Physics::HeightFieldDescriptor& descriptor)
    {
        args.pointer = const_cast<int16_t*>(descriptor.heights);
        args.pointer2 = const_cast<uint8_t*>(descriptor.holes);
        args.value = static_cast<uint32_t>(descriptor.resolution);
        args.mode = static_cast<uint32_t>(descriptor.holesCount);
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
        table[static_cast<size_t>(ShapeCmd::GetActorDescriptor::command)] = &GetActorDescriptor;
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
        }, [](const CommandArgs& a, auto& c)
        {
            // The ids are in and out: 0 asks for a new one, which Unity keeps on the collider and puts in its filter data.
            c.ignoreId0 = static_cast<uint16_t>(a.value);
            c.ignoreId1 = static_cast<uint16_t>(a.value2);
        }, false>(table);
        Bind<ShapeCmd::CleanupIgnoredColliders, [](CommandArgs& a, const auto& c) { a.value = c.ignoreID; }, None, false>(table);
        Bind<ShapeCmd::GetGeometryType, None, [](const CommandArgs& a, auto& c) { c.value = static_cast<Physics::GeometryType>(a.mode); }>(table);
        Bind<ShapeCmd::SetGeometry, [](CommandArgs& a, const auto& c) { PackGeometry(a, c.value); }>(table);
        Bind<ShapeCmd::GetGeometry, None, [](const CommandArgs& a, auto& c) { c.value = UnpackGeometry(a); }>(table);

        Bind<ShapeCmd::CreateHeightField, [](CommandArgs& a, const auto& c) { PackHeightField(a, c.descriptor); },
            [](const CommandArgs& a, auto& c) { c.heightField = a.pointer; }, false>(table);
        Bind<ShapeCmd::UpdateHeightFieldRegion, [](CommandArgs& a, const auto& c)
        {
            PackHeightField(a, c.descriptor);
            a.pointer3 = c.heightField;
        }, None, false>(table);
        Bind<ShapeCmd::DestroyHeightField, [](CommandArgs& a, const auto& c) { a.pointer = c.heightField; }, None, false>(table);

        BindMeshCommands(table);
    }
}
