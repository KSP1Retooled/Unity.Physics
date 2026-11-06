// Shape commands. They address the shape through the object handle, and the owning body and world through the context.
// Mesh.cpp holds the mesh commands. Height field commands are stubs, so a TerrainCollider has no collision.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    static Shape* AsShape(void* object)
    {
        return static_cast<Shape*>(object);
    }

    static void TouchBody(Shape* shape)
    {
        if (shape->body != nullptr)
        {
            MarkColliderDirty(shape->body);
        }
    }

    static void SetShapeMaterial(Shape* shape, MaterialData* material)
    {
        MaterialData* next = material != nullptr ? material : DefaultMaterial();
        RetainMaterial(next);
        ReleaseMaterial(shape->data.material);
        shape->data.material = next;
    }

    static void StoreGeometry(Shape* shape, const Physics::ShapeGeometry& geometry)
    {
        ShapeData& data = shape->data;
        Mesh* previousMesh = IsMeshGeometry(data) ? static_cast<Mesh*>(data.mesh) : nullptr;
        data.geometryType = static_cast<uint32_t>(geometry.type);
        data.extents = {};
        data.mesh = nullptr;
        switch (geometry.type)
        {
            case Physics::GeometryType::Box:
            {
                const Physics::Vec3& half = geometry.boxGeometry.halfExtents;
                data.extents = {half.x, half.y, half.z};
                break;
            }
            case Physics::GeometryType::Sphere:
                data.extents.x = geometry.sphereGeometry.radius;
                break;
            case Physics::GeometryType::Capsule:
                data.extents.x = geometry.capsuleGeometry.radius;
                data.extents.y = geometry.capsuleGeometry.height;
                break;
            case Physics::GeometryType::ConvexMesh:
            case Physics::GeometryType::TriangleMesh:
            {
                const Physics::Vec3& scale = geometry.meshGeometry.scale;
                data.extents = {scale.x, scale.y, scale.z};
                data.mesh = geometry.meshGeometry.mesh;
                RetainMesh(static_cast<Mesh*>(data.mesh));
                break;
            }
            case Physics::GeometryType::Terrain:
            {
                const Physics::Vec3& scale = geometry.heightFieldGeometry.scale;
                data.extents = {scale.x, scale.y, scale.z};
                data.mesh = geometry.heightFieldGeometry.heightField;
                break;
            }
            default:
                break;
        }

        ReleaseMesh(previousMesh);
    }

    static Physics::ShapeGeometry LoadGeometry(const ShapeData& data)
    {
        Physics::ShapeGeometry geometry{};
        geometry.type = static_cast<Physics::GeometryType>(data.geometryType);
        switch (geometry.type)
        {
            case Physics::GeometryType::Box:
                geometry.boxGeometry.halfExtents = Physics::Vec3(data.extents.x, data.extents.y, data.extents.z);
                break;
            case Physics::GeometryType::Sphere:
                geometry.sphereGeometry.radius = data.extents.x;
                break;
            case Physics::GeometryType::Capsule:
                geometry.capsuleGeometry.radius = data.extents.x;
                geometry.capsuleGeometry.height = data.extents.y;
                break;
            case Physics::GeometryType::ConvexMesh:
            case Physics::GeometryType::TriangleMesh:
                geometry.meshGeometry.mesh = data.mesh;
                geometry.meshGeometry.scale = Physics::Vec3(data.extents.x, data.extents.y, data.extents.z);
                break;
            case Physics::GeometryType::Terrain:
                geometry.heightFieldGeometry.heightField = data.mesh;
                geometry.heightFieldGeometry.scale = Physics::Vec3(data.extents.x, data.extents.y, data.extents.z);
                break;
            default:
                break;
        }

        return geometry;
    }

    static void SetDiagonal(float (&tensor)[3][3], float x, float y, float z)
    {
        std::memset(tensor, 0, sizeof(tensor));
        tensor[0][0] = x;
        tensor[1][1] = y;
        tensor[2][2] = z;
    }

    // A scale s moves x to s x, so the covariance's (i, j) entry gains s_i s_j. The inertia of a unit mass about its
    // centre is trace(C) I - C.
    static bool MeshMassProperties(const ShapeData& shape, float& volume, Float3& center, float (&inertiaPerUnitMass)[3][3])
    {
        const MeshData* mesh = MeshDataOf(shape);
        if (mesh == nullptr || mesh->moments.volume <= 0.0f)
        {
            return false;
        }

        const MeshMoments& moments = mesh->moments;
        const Float3& s = shape.extents;
        volume = moments.volume * std::abs(s.x * s.y * s.z);
        center = {moments.centroid.x * s.x, moments.centroid.y * s.y, moments.centroid.z * s.z};

        const float xx = moments.covarianceDiagonal.x * s.x * s.x;
        const float yy = moments.covarianceDiagonal.y * s.y * s.y;
        const float zz = moments.covarianceDiagonal.z * s.z * s.z;
        const float xy = moments.covarianceOffDiagonal.x * s.x * s.y;
        const float xz = moments.covarianceOffDiagonal.y * s.x * s.z;
        const float yz = moments.covarianceOffDiagonal.z * s.y * s.z;
        inertiaPerUnitMass[0][0] = yy + zz;
        inertiaPerUnitMass[1][1] = xx + zz;
        inertiaPerUnitMass[2][2] = xx + yy;
        inertiaPerUnitMass[0][1] = inertiaPerUnitMass[1][0] = -xy;
        inertiaPerUnitMass[0][2] = inertiaPerUnitMass[2][0] = -xz;
        inertiaPerUnitMass[1][2] = inertiaPerUnitMass[2][1] = -yz;
        return volume > 0.0f;
    }

    bool ShapeMassProperties(const ShapeData& shape, float& volume, Float3& center, float (&inertiaPerUnitMass)[3][3])
    {
        const float pi = std::numbers::pi_v<float>;
        center = {};
        switch (static_cast<Physics::GeometryType>(shape.geometryType))
        {
            case Physics::GeometryType::Box:
            {
                const Float3& h = shape.extents;
                volume = 8.0f * h.x * h.y * h.z;
                SetDiagonal(
                    inertiaPerUnitMass,
                    (h.y * h.y + h.z * h.z) / 3.0f,
                    (h.x * h.x + h.z * h.z) / 3.0f,
                    (h.x * h.x + h.y * h.y) / 3.0f);
                return true;
            }
            case Physics::GeometryType::Sphere:
            {
                const float r = shape.extents.x;
                volume = 4.0f / 3.0f * pi * r * r * r;
                const float i = 0.4f * r * r;
                SetDiagonal(inertiaPerUnitMass, i, i, i);
                return true;
            }
            case Physics::GeometryType::Capsule:
            {
                // The axis is local X, as in PhysX. A cylinder of length h between two hemispheres.
                const float r = shape.extents.x;
                const float h = CapsuleCylinderLength(shape);
                const float cylinder = pi * r * r * h;
                const float spheres = 4.0f / 3.0f * pi * r * r * r;
                volume = cylinder + spheres;
                const float axial = cylinder * r * r / 2.0f + spheres * 0.4f * r * r;
                const float transverse = cylinder * (h * h / 12.0f + r * r / 4.0f)
                    + spheres * (0.4f * r * r + h * h / 4.0f + 3.0f * h * r / 8.0f);
                SetDiagonal(inertiaPerUnitMass, axial / volume, transverse / volume, transverse / volume);
                return true;
            }
            case Physics::GeometryType::ConvexMesh:
            case Physics::GeometryType::TriangleMesh:
                return MeshMassProperties(shape, volume, center, inertiaPerUnitMass);
            default:
                return false;
        }
    }

    // The half extents of the world-aligned box around a box of the given half extents.
    static Float3 RotatedHalfExtents(const Float4& rotation, const Float3& half)
    {
        const Float3 x = Rotate(rotation, {half.x, 0, 0});
        const Float3 y = Rotate(rotation, {0, half.y, 0});
        const Float3 z = Rotate(rotation, {0, 0, half.z});
        return {
            std::abs(x.x) + std::abs(y.x) + std::abs(z.x),
            std::abs(x.y) + std::abs(y.y) + std::abs(z.y),
            std::abs(x.z) + std::abs(y.z) + std::abs(z.z),
        };
    }

    void ShapeWorldBounds(const ShapeData& shape, const Float3& bodyPosition, const Float4& bodyRotation, Float3& lower, Float3& upper)
    {
        const Float3 offset = Rotate(bodyRotation, shape.localPosition);
        Float3 center = {bodyPosition.x + offset.x, bodyPosition.y + offset.y, bodyPosition.z + offset.z};
        const Float4 rotation = Multiply(bodyRotation, shape.localRotation);

        Float3 half{};
        switch (static_cast<Physics::GeometryType>(shape.geometryType))
        {
            case Physics::GeometryType::Box:
                half = RotatedHalfExtents(rotation, shape.extents);
                break;
            case Physics::GeometryType::Sphere:
                half = {shape.extents.x, shape.extents.x, shape.extents.x};
                break;
            case Physics::GeometryType::Capsule:
            {
                const float r = shape.extents.x;
                const Float3 axis = Rotate(rotation, {CapsuleCylinderLength(shape) / 2.0f, 0, 0});
                half = {std::abs(axis.x) + r, std::abs(axis.y) + r, std::abs(axis.z) + r};
                break;
            }
            case Physics::GeometryType::ConvexMesh:
            case Physics::GeometryType::TriangleMesh:
            {
                const MeshData* cooked = MeshDataOf(shape);
                if (cooked == nullptr)
                {
                    break;
                }

                const MeshData& mesh = *cooked;
                const Float3& s = shape.extents;
                const Float3 middle = {
                    (mesh.lower.x + mesh.upper.x) * 0.5f * s.x,
                    (mesh.lower.y + mesh.upper.y) * 0.5f * s.y,
                    (mesh.lower.z + mesh.upper.z) * 0.5f * s.z,
                };
                const Float3 shifted = Rotate(rotation, middle);
                center = {center.x + shifted.x, center.y + shifted.y, center.z + shifted.z};
                half = RotatedHalfExtents(rotation, {
                    (mesh.upper.x - mesh.lower.x) * 0.5f * std::abs(s.x),
                    (mesh.upper.y - mesh.lower.y) * 0.5f * std::abs(s.y),
                    (mesh.upper.z - mesh.lower.z) * 0.5f * std::abs(s.z),
                });
                break;
            }
            default:
                break;
        }

        lower = {center.x - half.x, center.y - half.y, center.z - half.z};
        upper = {center.x + half.x, center.y + half.y, center.z + half.z};
    }

    static void CreateShape(void*, const Context&, ShapeCmd::CreateShape& command)
    {
        Shape* shape = new Shape();
        shape->data.owner = shape;
        shape->data.localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        shape->data.flags = static_cast<uint32_t>(command.flags);
        SetShapeMaterial(shape, static_cast<MaterialData*>(command.materialPtr));
        shape->data.userData = command.userData;
        StoreGeometry(shape, command.geom);
        command.value = shape;
    }

    static void DestroyShape(void* object, const Context&, ShapeCmd::DestroyShape&)
    {
        Shape* shape = AsShape(object);
        if (shape->body != nullptr)
        {
            Body* body = shape->body;
            body->shapes.erase(std::remove(body->shapes.begin(), body->shapes.end(), shape), body->shapes.end());
            BodyData& hot = Hot(body);
            hot.shapes = body->shapes.data();
            hot.shapeCount = static_cast<uint32_t>(body->shapes.size());
            MarkColliderDirty(body);
        }

        if (IsMeshGeometry(shape->data))
        {
            ReleaseMesh(static_cast<Mesh*>(shape->data.mesh));
        }

        ReleaseMaterial(shape->data.material);
        delete shape;
    }

    static void GetActorDescriptor(void* object, const Context&, ShapeCmd::GetActorDescriptor& command)
    {
        Body* body = AsShape(object)->body;
        command.value = Physics::ActorDescriptor{};
        command.value.actor = body;
        if (body != nullptr)
        {
            command.value.userData = body->userData;
            command.value.type = (Hot(body).flags & kBodyStatic) != 0 ? Physics::ActorType::Static : Physics::ActorType::Dynamic;
        }
    }

    static void GetUserData(void* object, const Context&, ShapeCmd::GetUserData& command)
    {
        command.value = AsShape(object)->data.userData;
    }

    static void GetFlags(void* object, const Context&, ShapeCmd::GetFlags& command)
    {
        command.flags = static_cast<Physics::ShapeFlags>(AsShape(object)->data.flags);
    }

    static void SetFlags(void* object, const Context&, ShapeCmd::SetFlags& command)
    {
        Shape* shape = AsShape(object);
        shape->data.flags = static_cast<uint32_t>(command.flags);
        TouchBody(shape);
    }

    static void GetFilterData(void* object, const Context&, ShapeCmd::GetFilterData& command)
    {
        command.data = AsShape(object)->filter;
    }

    static void SetFilterData(void* object, const Context&, ShapeCmd::SetFilterData& command)
    {
        Shape* shape = AsShape(object);
        shape->filter = command.data;
        shape->data.belongsTo = command.data.belongsTo;
        shape->data.collidesWith = command.data.collidesWith;
        shape->data.ignorePairId = command.data.ignorePairID;
        shape->data.collisionDisabled = command.data.hasCollisionDisabled;
        TouchBody(shape);
    }

    static void SetModifiableContacts(void*, const Context&, ShapeCmd::SetModifiableContacts&)
    {
    }

    static void SetSupportedMessages(void* object, const Context&, ShapeCmd::SetSupportedMessages& command)
    {
        AsShape(object)->filter.pairFlags = command.flags;
    }

    static void SetMaterial(void* object, const Context&, ShapeCmd::SetMaterial& command)
    {
        Shape* shape = AsShape(object);
        SetShapeMaterial(shape, static_cast<MaterialData*>(command.value));
        TouchBody(shape);
    }

    static void GetLocalPose(void* object, const Context&, ShapeCmd::GetLocalPose& command)
    {
        const ShapeData& data = AsShape(object)->data;
        command.value = Physics::Pose(data.localPosition, data.localRotation);
    }

    static void GetPose(void* object, const Context&, ShapeCmd::GetPose& command)
    {
        const Shape* shape = AsShape(object);
        const ShapeData& data = shape->data;
        if (shape->body == nullptr)
        {
            command.value = Physics::Pose(data.localPosition, data.localRotation);
            return;
        }

        const BodyData& hot = Hot(shape->body);
        const Float3 offset = Rotate(hot.rotation, data.localPosition);
        const Float3 position = {hot.position.x + offset.x, hot.position.y + offset.y, hot.position.z + offset.z};
        command.value = Physics::Pose(position, Multiply(hot.rotation, data.localRotation));
    }

    static void SetPose(void* object, const Context&, ShapeCmd::SetPose& command)
    {
        Shape* shape = AsShape(object);
        if (shape->body != nullptr && (Hot(shape->body).flags & kBodyStatic) != 0)
        {
            StoreStaticShapePose(shape->data, Hot(shape->body), command.value);
        }
        else
        {
            shape->data.localPosition = {command.value.pos.x, command.value.pos.y, command.value.pos.z};
            shape->data.localRotation = {command.value.quat.x, command.value.quat.y, command.value.quat.z, command.value.quat.w};
        }

        TouchBody(shape);
    }

    static void SetContactOffset(void* object, const Context&, ShapeCmd::SetContactOffset& command)
    {
        AsShape(object)->contactOffset = command.value;
    }

    static void GetWorldBounds(void* object, const Context&, ShapeCmd::GetWorldBounds& command)
    {
        const Shape* shape = AsShape(object);
        Float3 position{};
        Float4 rotation{0.0f, 0.0f, 0.0f, 1.0f};
        if (shape->body != nullptr)
        {
            const BodyData& hot = Hot(shape->body);
            position = hot.position;
            rotation = hot.rotation;
        }

        Float3 lower, upper;
        ShapeWorldBounds(shape->data, position, rotation, lower, upper);
        command.center = Physics::Vec3((lower.x + upper.x) * 0.5f, (lower.y + upper.y) * 0.5f, (lower.z + upper.z) * 0.5f);
        command.extents = Physics::Vec3((upper.x - lower.x) * 0.5f, (upper.y - lower.y) * 0.5f, (upper.z - lower.z) * 0.5f);
    }

    static void GetBoundsAtPose(void* object, const Context&, ShapeCmd::GetBoundsAtPose& command)
    {
        // The pose replaces the shape's whole world pose, so the shape's own local pose takes no part.
        ShapeData data = AsShape(object)->data;
        data.localPosition = {};
        data.localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        const Float3 position = {command.pose.pos.x, command.pose.pos.y, command.pose.pos.z};
        const Float4 rotation = {command.pose.quat.x, command.pose.quat.y, command.pose.quat.z, command.pose.quat.w};

        Float3 lower, upper;
        ShapeWorldBounds(data, position, rotation, lower, upper);
        command.center = Physics::Vec3((lower.x + upper.x) * 0.5f, (lower.y + upper.y) * 0.5f, (lower.z + upper.z) * 0.5f);
        command.extents = Physics::Vec3((upper.x - lower.x) * 0.5f, (upper.y - lower.y) * 0.5f, (upper.z - lower.z) * 0.5f);
    }

    static uint32_t IgnorePairKey(uint16_t a, uint16_t b)
    {
        return (static_cast<uint32_t>(std::min(a, b)) << 16) | std::max(a, b);
    }

    static void ShouldIgnoreCollision(void*, const Context&, ShapeCmd::ShouldIgnoreCollision& command)
    {
        command.value = g_IgnoredPairs.count(IgnorePairKey(command.ignoreId0, command.ignoreId1)) != 0;
    }

    static void IgnoreCollision(void*, const Context&, ShapeCmd::IgnoreCollision& command)
    {
        const uint32_t key = IgnorePairKey(command.ignoreId0, command.ignoreId1);
        if (command.value)
        {
            g_IgnoredPairs.insert(key);
        }
        else
        {
            g_IgnoredPairs.erase(key);
        }
    }

    static void CleanupIgnoredColliders(void*, const Context&, ShapeCmd::CleanupIgnoredColliders& command)
    {
        for (auto it = g_IgnoredPairs.begin(); it != g_IgnoredPairs.end();)
        {
            const bool matches = (*it >> 16) == command.ignoreID || (*it & 0xffff) == command.ignoreID;
            it = matches ? g_IgnoredPairs.erase(it) : std::next(it);
        }
    }

    static void GetGeometryType(void* object, const Context&, ShapeCmd::GetGeometryType& command)
    {
        command.value = static_cast<Physics::GeometryType>(AsShape(object)->data.geometryType);
    }

    static void SetGeometry(void* object, const Context&, ShapeCmd::SetGeometry& command)
    {
        Shape* shape = AsShape(object);
        StoreGeometry(shape, command.value);
        TouchBody(shape);
    }

    static void GetGeometry(void* object, const Context&, ShapeCmd::GetGeometry& command)
    {
        command.value = LoadGeometry(AsShape(object)->data);
    }

    template <typename T, Handler<T> Function>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Bound<T, Function>;
    }

    template <typename T>
    static void BindStub(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &ZeroedStub<T, kShapeCategory>;
    }

    void BindShapeCommands(FunctionBinding* table)
    {
        Bind<ShapeCmd::CreateShape, CreateShape>(table);
        Bind<ShapeCmd::DestroyShape, DestroyShape>(table);
        Bind<ShapeCmd::GetActorDescriptor, GetActorDescriptor>(table);
        Bind<ShapeCmd::GetUserData, GetUserData>(table);
        Bind<ShapeCmd::GetFlags, GetFlags>(table);
        Bind<ShapeCmd::SetFlags, SetFlags>(table);
        Bind<ShapeCmd::GetFilterData, GetFilterData>(table);
        Bind<ShapeCmd::SetFilterData, SetFilterData>(table);
        Bind<ShapeCmd::SetModifiableContacts, SetModifiableContacts>(table);
        Bind<ShapeCmd::SetSupportedMessages, SetSupportedMessages>(table);
        Bind<ShapeCmd::SetMaterial, SetMaterial>(table);
        Bind<ShapeCmd::GetLocalPose, GetLocalPose>(table);
        Bind<ShapeCmd::GetPose, GetPose>(table);
        Bind<ShapeCmd::SetPose, SetPose>(table);
        Bind<ShapeCmd::SetContactOffset, SetContactOffset>(table);
        Bind<ShapeCmd::GetWorldBounds, GetWorldBounds>(table);
        Bind<ShapeCmd::GetBoundsAtPose, GetBoundsAtPose>(table);
        Bind<ShapeCmd::ShouldIgnoreCollision, ShouldIgnoreCollision>(table);
        Bind<ShapeCmd::IgnoreCollision, IgnoreCollision>(table);
        Bind<ShapeCmd::CleanupIgnoredColliders, CleanupIgnoredColliders>(table);
        Bind<ShapeCmd::GetGeometryType, GetGeometryType>(table);
        Bind<ShapeCmd::SetGeometry, SetGeometry>(table);
        Bind<ShapeCmd::GetGeometry, GetGeometry>(table);

        BindMeshCommands(table);

        BindStub<ShapeCmd::CreateHeightField>(table);
    }
}
