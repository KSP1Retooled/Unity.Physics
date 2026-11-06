// Body commands. They address the body through the object handle. Everything here reads or writes the store; the
// forces collected by AddForce and AddTorque are consumed by the managed step.

#include <algorithm>
#include <cmath>
#include <limits>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    static constexpr float kWakeCounterReset = 0.4f;

    static Body* AsBody(void* object)
    {
        return static_cast<Body*>(object);
    }

    static Float3 ToFloat3(const Physics::Vec3& v)
    {
        return {v.x, v.y, v.z};
    }

    static Float4 ToFloat4(const Physics::Vec4& v)
    {
        return {v.x, v.y, v.z, v.w};
    }

    static Physics::Vec3 ToVec3(const Float3& v)
    {
        return Physics::Vec3(v.x, v.y, v.z);
    }

    static Physics::Vec4 ToVec4(const Float4& v)
    {
        return Physics::Vec4(v.x, v.y, v.z, v.w);
    }

    static Float3 Add(const Float3& a, const Float3& b)
    {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    static Float3 Subtract(const Float3& a, const Float3& b)
    {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    static Float3 Scale(const Float3& a, float s)
    {
        return {a.x * s, a.y * s, a.z * s};
    }

    static Float3 Cross(const Float3& a, const Float3& b)
    {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    static float Dot(const Float3& a, const Float3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static float InverseOrZero(float value)
    {
        return value > 0.0f ? 1.0f / value : 0.0f;
    }

    Float3 Rotate(const Float4& q, const Float3& v)
    {
        const Float3 axis = {q.x, q.y, q.z};
        const Float3 t = Scale(Cross(axis, v), 2.0f);
        return Add(Add(v, Scale(t, q.w)), Cross(axis, t));
    }

    Float4 Multiply(const Float4& a, const Float4& b)
    {
        return {
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        };
    }

    void StoreStaticShapePose(ShapeData& shape, const BodyData& body, const Physics::Pose& worldPose)
    {
        const Float4 inverse = {-body.rotation.x, -body.rotation.y, -body.rotation.z, body.rotation.w};
        shape.localPosition = Rotate(inverse, Subtract(ToFloat3(worldPose.pos), body.position));
        shape.localRotation = Multiply(inverse, ToFloat4(worldPose.quat));
    }

    static void Wake(BodyData& hot)
    {
        hot.flags &= ~kBodySleeping;
        hot.wakeCounter = std::max(hot.wakeCounter, kWakeCounterReset);
    }

    static void SetInertia(BodyData& hot, const Float3& inertia)
    {
        hot.inertia = inertia;
        hot.inverseInertia = {InverseOrZero(inertia.x), InverseOrZero(inertia.y), InverseOrZero(inertia.z)};
    }

    static void SetFlags(void* object, const Context&, BodyCmd::SetFlags& command)
    {
        BodyData& hot = Hot(AsBody(object));
        const uint32_t bits = static_cast<uint32_t>(command.value) & kBodyPublicFlagsMask;
        if (command.enabled)
        {
            hot.flags |= bits;
        }
        else
        {
            hot.flags &= ~bits;
        }

        if ((hot.flags & kBodyKinematic) == 0)
        {
            hot.flags &= ~kBodyHasKinematicTarget;
        }

        MarkPoseChanged(AsBody(object), hot.flags);
    }

    static void GetFlags(void* object, const Context&, BodyCmd::GetFlags& command)
    {
        command.value = static_cast<Physics::BodyFlags>(Hot(AsBody(object)).flags & kBodyPublicFlagsMask);
    }

    static void SetMass(void* object, const Context&, BodyCmd::SetMass& command)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.mass = command.value;
        hot.inverseMass = InverseOrZero(command.value);
    }

    static void GetMass(void* object, const Context&, BodyCmd::GetMass& command)
    {
        command.value = Hot(AsBody(object)).mass;
    }

    static void SetDamping(void* object, const Context&, BodyCmd::SetDamping& command)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.linearDamping = command.linear;
        hot.angularDamping = command.angular;
    }

    static void GetDamping(void* object, const Context&, BodyCmd::GetDamping& command)
    {
        const BodyData& hot = Hot(AsBody(object));
        command.linear = hot.linearDamping;
        command.angular = hot.angularDamping;
    }

    static void SetPose(void* object, const Context&, BodyCmd::SetPose& command)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.position = ToFloat3(command.value.pos);
        hot.rotation = ToFloat4(command.value.quat);
        if (command.awake)
        {
            Wake(hot);
        }

        MarkPoseChanged(AsBody(object), hot.flags);
    }

    static void GetPose(void* object, const Context&, BodyCmd::GetPose& command)
    {
        const BodyData& hot = Hot(AsBody(object));
        command.value = Physics::Pose(hot.position, hot.rotation);
    }

    static void PublishShapes(Body* body)
    {
        BodyData& hot = Hot(body);
        hot.shapes = body->shapes.data();
        hot.shapeCount = static_cast<uint32_t>(body->shapes.size());
        MarkColliderDirty(body);
    }

    static void AttachShape(void* object, const Context&, BodyCmd::AttachShape& command)
    {
        Body* body = AsBody(object);
        Shape* shape = static_cast<Shape*>(command.shape);
        shape->body = body;
        const BodyData& hot = Hot(body);
        if ((hot.flags & kBodyStatic) != 0 && !command.forceLocalPose)
        {
            StoreStaticShapePose(shape->data, hot, command.pose);
        }
        else
        {
            shape->data.localPosition = ToFloat3(command.pose.pos);
            shape->data.localRotation = ToFloat4(command.pose.quat);
        }

        body->shapes.push_back(shape);
        PublishShapes(body);
    }

    static void DetachShape(void* object, const Context&, BodyCmd::DetachShape& command)
    {
        Body* body = AsBody(object);
        Shape* shape = static_cast<Shape*>(command.shape);
        body->shapes.erase(std::remove(body->shapes.begin(), body->shapes.end(), shape), body->shapes.end());
        shape->body = nullptr;
        PublishShapes(body);
    }

    void DetachAllShapes(Body* body)
    {
        for (Shape* shape : body->shapes)
        {
            shape->body = nullptr;
        }

        body->shapes.clear();
        PublishShapes(body);
    }

    static void GetShapeCount(void* object, const Context&, BodyCmd::GetShapeCount& command)
    {
        command.value = AsBody(object)->shapes.size();
    }

    static void GetShapes(void* object, const Context&, BodyCmd::GetShapes& command)
    {
        const Body* body = AsBody(object);
        command.written = std::min(command.bufferSize, body->shapes.size());
        for (size_t i = 0; i < command.written; i++)
        {
            command.buffer[i] = body->shapes[i];
        }
    }

    static void GetShapesUserData(void* object, const Context&, BodyCmd::GetShapesUserData& command)
    {
        const Body* body = AsBody(object);
        command.written = std::min(command.bufferSize, body->shapes.size());
        for (size_t i = 0; i < command.written; i++)
        {
            command.buffer[i] = body->shapes[i]->data.userData;
        }
    }

    static void GetUserData(void* object, const Context&, BodyCmd::GetUserData& command)
    {
        command.value = AsBody(object)->userData;
    }

    static void SetCollisionDetectionMode(void* object, const Context&, BodyCmd::SetCollisionDetectionMode& command)
    {
        AsBody(object)->collisionDetectionMode = static_cast<int32_t>(command.value);
    }

    static void AddForce(void* object, const Context&, BodyCmd::AddForce& command)
    {
        BodyData& hot = Hot(AsBody(object));
        if ((hot.flags & (kBodyKinematic | kBodyStatic)) != 0)
        {
            return;
        }

        const Float3 value = ToFloat3(command.value);
        switch (command.mode)
        {
            case Physics::ForceMode::Force:
                hot.force = Add(hot.force, value);
                break;
            case Physics::ForceMode::Impulse:
                hot.impulse = Add(hot.impulse, value);
                break;
            case Physics::ForceMode::VelocityChange:
                hot.velocityChange = Add(hot.velocityChange, value);
                break;
            case Physics::ForceMode::Acceleration:
                hot.acceleration = Add(hot.acceleration, value);
                break;
        }

        if (command.awake)
        {
            Wake(hot);
        }
    }

    static void AddTorque(void* object, const Context&, BodyCmd::AddTorque& command)
    {
        BodyData& hot = Hot(AsBody(object));
        if ((hot.flags & (kBodyKinematic | kBodyStatic)) != 0)
        {
            return;
        }

        const Float3 value = ToFloat3(command.value);
        switch (command.mode)
        {
            case Physics::ForceMode::Force:
                hot.torque = Add(hot.torque, value);
                break;
            case Physics::ForceMode::Impulse:
                hot.angularImpulse = Add(hot.angularImpulse, value);
                break;
            case Physics::ForceMode::VelocityChange:
                hot.angularVelocityChange = Add(hot.angularVelocityChange, value);
                break;
            case Physics::ForceMode::Acceleration:
                hot.angularAcceleration = Add(hot.angularAcceleration, value);
                break;
        }

        if (command.awake)
        {
            Wake(hot);
        }
    }

    static void ClearForce(void* object, const Context&, BodyCmd::ClearForce&)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.force = {};
        hot.acceleration = {};
        hot.impulse = {};
        hot.velocityChange = {};
    }

    static void ClearTorque(void* object, const Context&, BodyCmd::ClearTorque&)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.torque = {};
        hot.angularAcceleration = {};
        hot.angularImpulse = {};
        hot.angularVelocityChange = {};
    }

    static void SetLinearVelocity(void* object, const Context&, BodyCmd::SetLinearVelocity& command)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.linearVelocity = ToFloat3(command.value);
        if (command.awake)
        {
            Wake(hot);
        }
    }

    static void GetLinearVelocity(void* object, const Context&, BodyCmd::GetLinearVelocity& command)
    {
        command.value = ToVec3(Hot(AsBody(object)).linearVelocity);
    }

    static void SetAngularVelocity(void* object, const Context&, BodyCmd::SetAngularVelocity& command)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.angularVelocity = ToFloat3(command.value);
        if (command.awake)
        {
            Wake(hot);
        }
    }

    static void GetAngularVelocity(void* object, const Context&, BodyCmd::GetAngularVelocity& command)
    {
        command.value = ToVec3(Hot(AsBody(object)).angularVelocity);
    }

    static void SetMaxLinearVelocity(void* object, const Context&, BodyCmd::SetMaxLinearVelocity& command)
    {
        Hot(AsBody(object)).maxLinearVelocity = command.value;
    }

    static void GetMaxLinearVelocity(void* object, const Context&, BodyCmd::GetMaxLinearVelocity& command)
    {
        command.value = Hot(AsBody(object)).maxLinearVelocity;
    }

    static void SetMaxAngularVelocity(void* object, const Context&, BodyCmd::SetMaxAngularVelocity& command)
    {
        Hot(AsBody(object)).maxAngularVelocity = command.value;
    }

    static void GetMaxAngularVelocity(void* object, const Context&, BodyCmd::GetMaxAngularVelocity& command)
    {
        command.value = Hot(AsBody(object)).maxAngularVelocity;
    }

    static void SetMaxDepenetrationVelocity(void* object, const Context&, BodyCmd::SetMaxDepenetrationVelocity& command)
    {
        Hot(AsBody(object)).maxDepenetrationVelocity = command.value;
    }

    static void GetMaxDepenetrationVelocity(void* object, const Context&, BodyCmd::GetMaxDepenetrationVelocity& command)
    {
        command.value = Hot(AsBody(object)).maxDepenetrationVelocity;
    }

    static void SetInertiaTensor(void* object, const Context&, BodyCmd::SetInertiaTensor& command)
    {
        SetInertia(Hot(AsBody(object)), ToFloat3(command.value));
    }

    static void GetInertiaTensor(void* object, const Context&, BodyCmd::GetInertiaTensor& command)
    {
        command.value = ToVec3(Hot(AsBody(object)).inertia);
    }

    static void SetInertiaTensorRotation(void* object, const Context&, BodyCmd::SetInertiaTensorRotation& command)
    {
        Hot(AsBody(object)).inertiaRotation = ToFloat4(command.value);
    }

    static void GetInertiaTensorRotation(void* object, const Context&, BodyCmd::GetInertiaTensorRotation& command)
    {
        command.value = ToVec4(Hot(AsBody(object)).inertiaRotation);
    }

    static void SetLocalCenterOfMass(void* object, const Context&, BodyCmd::SetLocalCenterOfMass& command)
    {
        Hot(AsBody(object)).localCenterOfMass = ToFloat3(command.value);
    }

    static void GetLocalCenterOfMass(void* object, const Context&, BodyCmd::GetLocalCenterOfMass& command)
    {
        command.value = ToVec3(Hot(AsBody(object)).localCenterOfMass);
    }

    static void GetWorldInertiaTensorMatrix(void* object, const Context&, BodyCmd::GetWorldInertiaTensorMatrix& command)
    {
        const BodyData& hot = Hot(AsBody(object));
        const Float4 rotation = Multiply(hot.rotation, hot.inertiaRotation);
        const Float3 axes[3] = {Rotate(rotation, {1, 0, 0}), Rotate(rotation, {0, 1, 0}), Rotate(rotation, {0, 0, 1})};
        const float moments[3] = {hot.inertia.x, hot.inertia.y, hot.inertia.z};

        for (int row = 0; row < 3; row++)
        {
            for (int column = 0; column < 3; column++)
            {
                float sum = 0.0f;
                for (int axis = 0; axis < 3; axis++)
                {
                    const float* a = &axes[axis].x;
                    sum += a[row] * moments[axis] * a[column];
                }

                command.value[row * 3 + column] = sum;
            }
        }
    }

    struct Matrix3
    {
        double m[3][3] = {};
    };

    // Cyclic Jacobi: rotates a symmetric matrix until it is diagonal, collecting the rotation in `axes`, whose columns
    // end up as the eigenvectors.
    static void Diagonalize(Matrix3& a, Matrix3& axes)
    {
        axes = {};
        axes.m[0][0] = axes.m[1][1] = axes.m[2][2] = 1.0;
        for (int sweep = 0; sweep < 32; sweep++)
        {
            const double offDiagonal = std::abs(a.m[0][1]) + std::abs(a.m[0][2]) + std::abs(a.m[1][2]);
            if (offDiagonal < 1e-14 * (std::abs(a.m[0][0]) + std::abs(a.m[1][1]) + std::abs(a.m[2][2])))
            {
                return;
            }

            for (int p = 0; p < 2; p++)
            {
                for (int q = p + 1; q < 3; q++)
                {
                    if (a.m[p][q] == 0.0)
                    {
                        continue;
                    }

                    const double theta = (a.m[q][q] - a.m[p][p]) / (2.0 * a.m[p][q]);
                    const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                    const double c = 1.0 / std::sqrt(t * t + 1.0);
                    const double s = t * c;

                    for (int k = 0; k < 3; k++)
                    {
                        const double akp = a.m[k][p];
                        const double akq = a.m[k][q];
                        a.m[k][p] = c * akp - s * akq;
                        a.m[k][q] = s * akp + c * akq;
                    }

                    for (int k = 0; k < 3; k++)
                    {
                        const double apk = a.m[p][k];
                        const double aqk = a.m[q][k];
                        a.m[p][k] = c * apk - s * aqk;
                        a.m[q][k] = s * apk + c * aqk;
                    }

                    for (int k = 0; k < 3; k++)
                    {
                        const double vkp = axes.m[k][p];
                        const double vkq = axes.m[k][q];
                        axes.m[k][p] = c * vkp - s * vkq;
                        axes.m[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
    }

    static Float4 QuaternionFromAxes(const Matrix3& axes)
    {
        const auto& m = axes.m;
        const double trace = m[0][0] + m[1][1] + m[2][2];
        double x, y, z, w;
        if (trace > 0.0)
        {
            const double s = std::sqrt(trace + 1.0) * 2.0;
            w = 0.25 * s;
            x = (m[2][1] - m[1][2]) / s;
            y = (m[0][2] - m[2][0]) / s;
            z = (m[1][0] - m[0][1]) / s;
        }
        else if (m[0][0] > m[1][1] && m[0][0] > m[2][2])
        {
            const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
            w = (m[2][1] - m[1][2]) / s;
            x = 0.25 * s;
            y = (m[0][1] + m[1][0]) / s;
            z = (m[0][2] + m[2][0]) / s;
        }
        else if (m[1][1] > m[2][2])
        {
            const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
            w = (m[0][2] - m[2][0]) / s;
            x = (m[0][1] + m[1][0]) / s;
            y = 0.25 * s;
            z = (m[1][2] + m[2][1]) / s;
        }
        else
        {
            const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
            w = (m[1][0] - m[0][1]) / s;
            x = (m[0][2] + m[2][0]) / s;
            y = (m[1][2] + m[2][1]) / s;
            z = 0.25 * s;
        }

        return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), static_cast<float>(w)};
    }

    // Distributes the body's mass over its non-trigger shapes by volume, as PhysX's updateMassAndInertia does.
    static void RecomputeMassProperties(void* object, const Context&, BodyCmd::RecomputeMassProperties& command)
    {
        Body* body = AsBody(object);
        BodyData& hot = Hot(body);
        const Physics::MassPropertiesOverride& overrides = command.overrides;

        struct Piece
        {
            const ShapeData* shape;
            float volume;
            Float3 center; // in the body's frame
            float inertiaPerUnitMass[3][3]; // in the shape's frame
        };

        std::vector<Piece> pieces;
        float totalVolume = 0.0f;
        for (const Shape* shape : body->shapes)
        {
            Piece piece{};
            Float3 localCenter{};
            if ((shape->data.flags & static_cast<uint32_t>(Physics::ShapeFlags::Trigger)) == 0
                && ShapeMassProperties(shape->data, piece.volume, localCenter, piece.inertiaPerUnitMass))
            {
                piece.shape = &shape->data;
                piece.center = Add(shape->data.localPosition, Rotate(shape->data.localRotation, localCenter));
                pieces.push_back(piece);
                totalVolume += piece.volume;
            }
        }

        if (totalVolume <= 0.0f)
        {
            hot.localCenterOfMass = overrides.overrideCenterOfMass ? ToFloat3(overrides.centerOfMass) : Float3{};
            SetInertia(hot, overrides.overrideInertiaTensor ? ToFloat3(overrides.inertiaTensor) : Float3{1.0f, 1.0f, 1.0f});
            hot.inertiaRotation = overrides.overrideInertiaTensor ? ToFloat4(overrides.inertiaTensorRotation) : Float4{0, 0, 0, 1};
            return;
        }

        Float3 center{};
        for (const Piece& piece : pieces)
        {
            center = Add(center, Scale(piece.center, piece.volume / totalVolume));
        }

        if (overrides.overrideCenterOfMass)
        {
            center = ToFloat3(overrides.centerOfMass);
        }

        hot.localCenterOfMass = center;
        if (overrides.overrideInertiaTensor)
        {
            SetInertia(hot, ToFloat3(overrides.inertiaTensor));
            hot.inertiaRotation = ToFloat4(overrides.inertiaTensorRotation);
            return;
        }

        Matrix3 tensor;
        for (const Piece& piece : pieces)
        {
            const double pieceMass = hot.mass * piece.volume / totalVolume;
            const Float4& q = piece.shape->localRotation;
            const Float3 axes[3] = {Rotate(q, {1, 0, 0}), Rotate(q, {0, 1, 0}), Rotate(q, {0, 0, 1})};
            const Float3 offset = Subtract(piece.center, center);
            const float* d = &offset.x;
            const double offsetSquared = Dot(offset, offset);

            for (int row = 0; row < 3; row++)
            {
                for (int column = 0; column < 3; column++)
                {
                    double value = (row == column ? offsetSquared : 0.0) - static_cast<double>(d[row]) * d[column];
                    for (int first = 0; first < 3; first++)
                    {
                        for (int second = 0; second < 3; second++)
                        {
                            value += static_cast<double>((&axes[first].x)[row]) * piece.inertiaPerUnitMass[first][second]
                                * (&axes[second].x)[column];
                        }
                    }

                    tensor.m[row][column] += pieceMass * value;
                }
            }
        }

        Matrix3 principalAxes;
        Diagonalize(tensor, principalAxes);

        // Eigenvectors may come out left-handed, and a quaternion can only express a proper rotation.
        const double determinant = principalAxes.m[0][0] * (principalAxes.m[1][1] * principalAxes.m[2][2] - principalAxes.m[1][2] * principalAxes.m[2][1])
            - principalAxes.m[0][1] * (principalAxes.m[1][0] * principalAxes.m[2][2] - principalAxes.m[1][2] * principalAxes.m[2][0])
            + principalAxes.m[0][2] * (principalAxes.m[1][0] * principalAxes.m[2][1] - principalAxes.m[1][1] * principalAxes.m[2][0]);
        if (determinant < 0.0)
        {
            for (int row = 0; row < 3; row++)
            {
                principalAxes.m[row][2] = -principalAxes.m[row][2];
            }
        }

        SetInertia(hot, {static_cast<float>(tensor.m[0][0]), static_cast<float>(tensor.m[1][1]), static_cast<float>(tensor.m[2][2])});
        hot.inertiaRotation = QuaternionFromAxes(principalAxes);
    }

    static void GetAccumulatedForce(void* object, const Context&, BodyCmd::GetAccumulatedForce& command)
    {
        const BodyData& hot = Hot(AsBody(object));
        const float perStep = command.deltaTime > 0.0f ? 1.0f / command.deltaTime : 0.0f;
        Float3 total = Add(hot.force, Scale(hot.acceleration, hot.mass));
        total = Add(total, Scale(hot.impulse, perStep));
        total = Add(total, Scale(hot.velocityChange, hot.mass * perStep));
        command.value = ToVec3(total);
    }

    // Acceleration and velocity-change torques go through the principal axes, because they carry no inertia of their own.
    static Float3 ApplyInertia(const BodyData& hot, const Float3& angular)
    {
        const Float4 rotation = Multiply(hot.rotation, hot.inertiaRotation);
        const Float4 inverse = {-rotation.x, -rotation.y, -rotation.z, rotation.w};
        const Float3 local = Rotate(inverse, angular);
        return Rotate(rotation, {local.x * hot.inertia.x, local.y * hot.inertia.y, local.z * hot.inertia.z});
    }

    static void GetAccumulatedTorque(void* object, const Context&, BodyCmd::GetAccumulatedTorque& command)
    {
        const BodyData& hot = Hot(AsBody(object));
        const float perStep = command.deltaTime > 0.0f ? 1.0f / command.deltaTime : 0.0f;
        Float3 total = Add(hot.torque, ApplyInertia(hot, hot.angularAcceleration));
        total = Add(total, Scale(hot.angularImpulse, perStep));
        total = Add(total, Scale(ApplyInertia(hot, hot.angularVelocityChange), perStep));
        command.value = ToVec3(total);
    }

    static Float3 WorldCenterOfMass(const BodyData& hot)
    {
        return Add(hot.position, Rotate(hot.rotation, hot.localCenterOfMass));
    }

    static Float3 VelocityAt(const BodyData& hot, const Float3& worldPoint)
    {
        return Add(hot.linearVelocity, Cross(hot.angularVelocity, Subtract(worldPoint, WorldCenterOfMass(hot))));
    }

    static void GetLocalPointVelocity(void* object, const Context&, BodyCmd::GetLocalPointVelocity& command)
    {
        const BodyData& hot = Hot(AsBody(object));
        const Float3 worldPoint = Add(hot.position, Rotate(hot.rotation, ToFloat3(command.localPoint)));
        command.value = ToVec3(VelocityAt(hot, worldPoint));
    }

    static void GetWorldPointVelocity(void* object, const Context&, BodyCmd::GetWorldPointVelocity& command)
    {
        command.value = ToVec3(VelocityAt(Hot(AsBody(object)), ToFloat3(command.worldPoint)));
    }

    static void GetClosestWorldPointOnBounds(void* object, const Context&, BodyCmd::GetClosestWorldPointOnBounds& command)
    {
        const Body* body = AsBody(object);
        const BodyData& hot = Hot(AsBody(object));
        const float infinity = std::numeric_limits<float>::infinity();
        Float3 lower = {infinity, infinity, infinity};
        Float3 upper = {-infinity, -infinity, -infinity};
        for (const Shape* shape : body->shapes)
        {
            Float3 shapeLower, shapeUpper;
            ShapeWorldBounds(shape->data, hot.position, hot.rotation, shapeLower, shapeUpper);
            lower = {std::min(lower.x, shapeLower.x), std::min(lower.y, shapeLower.y), std::min(lower.z, shapeLower.z)};
            upper = {std::max(upper.x, shapeUpper.x), std::max(upper.y, shapeUpper.y), std::max(upper.z, shapeUpper.z)};
        }

        const Float3 position = ToFloat3(command.worldPosition);
        Float3 closest = position;
        if (!body->shapes.empty())
        {
            closest = {
                std::clamp(position.x, lower.x, upper.x),
                std::clamp(position.y, lower.y, upper.y),
                std::clamp(position.z, lower.z, upper.z),
            };
        }

        const Float3 offset = Subtract(closest, position);
        command.outClosestWorldPositionOnBounds = ToVec3(closest);
        command.outSqrDistance = Dot(offset, offset);
    }

    static void SetKinematicTarget(void* object, const Context&, BodyCmd::SetKinematicTarget& command)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.targetPosition = ToFloat3(command.value.pos);
        hot.targetRotation = ToFloat4(command.value.quat);
        hot.flags |= kBodyHasKinematicTarget;
        Wake(hot);
    }

    static void GetKinematicTarget(void* object, const Context&, BodyCmd::GetKinematicTarget& command)
    {
        const BodyData& hot = Hot(AsBody(object));
        command.value = (hot.flags & kBodyHasKinematicTarget) != 0
            ? Physics::Pose(hot.targetPosition, hot.targetRotation)
            : Physics::Pose(hot.position, hot.rotation);
    }

    static void GetIsSleeping(void* object, const Context&, BodyCmd::GetIsSleeping& command)
    {
        command.value = (Hot(AsBody(object)).flags & kBodySleeping) != 0;
    }

    static void Sleep(void* object, const Context&, BodyCmd::Sleep&)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.flags |= kBodySleeping;
        hot.wakeCounter = 0.0f;
        hot.linearVelocity = {};
        hot.angularVelocity = {};
    }

    static void WakeUp(void* object, const Context&, BodyCmd::WakeUp&)
    {
        Wake(Hot(AsBody(object)));
    }

    static void SetWakeCounter(void* object, const Context&, BodyCmd::SetWakeCounter& command)
    {
        BodyData& hot = Hot(AsBody(object));
        hot.wakeCounter = command.value;
        if (command.value > 0.0f)
        {
            hot.flags &= ~kBodySleeping;
        }
    }

    static void GetWakeCounter(void* object, const Context&, BodyCmd::GetWakeCounter& command)
    {
        command.value = Hot(AsBody(object)).wakeCounter;
    }

    static void SetSleepThresholdOverride(void* object, const Context&, BodyCmd::SetSleepThresholdOverride& command)
    {
        Hot(AsBody(object)).sleepThreshold = command.value;
    }

    static void GetSleepThresholdOverride(void* object, const Context&, BodyCmd::GetSleepThresholdOverride& command)
    {
        command.value = Hot(AsBody(object)).sleepThreshold;
    }

    static void SetSolverIterationsOverride(void* object, const Context&, BodyCmd::SetSolverIterationsOverride& command)
    {
        Body* body = AsBody(object);
        body->positionIterations = command.positionIterations;
        body->velocityIterations = command.velocityIterations;
    }

    static void GetSolverIterationsOverride(void* object, const Context&, BodyCmd::GetSolverIterationsOverride& command)
    {
        const Body* body = AsBody(object);
        command.positionIterations = body->positionIterations;
        command.velocityIterations = body->velocityIterations;
    }

    template <typename T, Handler<T> Function>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Bound<T, Function>;
    }

    void BindBodyCommands(FunctionBinding* table)
    {
        Bind<BodyCmd::SetFlags, Retooled::SetFlags>(table);
        Bind<BodyCmd::GetFlags, Retooled::GetFlags>(table);
        Bind<BodyCmd::SetMass, Retooled::SetMass>(table);
        Bind<BodyCmd::GetMass, Retooled::GetMass>(table);
        Bind<BodyCmd::SetDamping, Retooled::SetDamping>(table);
        Bind<BodyCmd::GetDamping, Retooled::GetDamping>(table);
        Bind<BodyCmd::SetPose, Retooled::SetPose>(table);
        Bind<BodyCmd::GetPose, Retooled::GetPose>(table);
        Bind<BodyCmd::AttachShape, Retooled::AttachShape>(table);
        Bind<BodyCmd::DetachShape, Retooled::DetachShape>(table);
        Bind<BodyCmd::GetShapeCount, Retooled::GetShapeCount>(table);
        Bind<BodyCmd::GetShapes, Retooled::GetShapes>(table);
        Bind<BodyCmd::GetShapesUserData, Retooled::GetShapesUserData>(table);
        Bind<BodyCmd::GetUserData, Retooled::GetUserData>(table);
        Bind<BodyCmd::SetCollisionDetectionMode, Retooled::SetCollisionDetectionMode>(table);
        Bind<BodyCmd::AddForce, Retooled::AddForce>(table);
        Bind<BodyCmd::AddTorque, Retooled::AddTorque>(table);
        Bind<BodyCmd::ClearForce, Retooled::ClearForce>(table);
        Bind<BodyCmd::ClearTorque, Retooled::ClearTorque>(table);
        Bind<BodyCmd::SetLinearVelocity, Retooled::SetLinearVelocity>(table);
        Bind<BodyCmd::GetLinearVelocity, Retooled::GetLinearVelocity>(table);
        Bind<BodyCmd::SetAngularVelocity, Retooled::SetAngularVelocity>(table);
        Bind<BodyCmd::GetAngularVelocity, Retooled::GetAngularVelocity>(table);
        Bind<BodyCmd::SetMaxLinearVelocity, Retooled::SetMaxLinearVelocity>(table);
        Bind<BodyCmd::GetMaxLinearVelocity, Retooled::GetMaxLinearVelocity>(table);
        Bind<BodyCmd::SetMaxAngularVelocity, Retooled::SetMaxAngularVelocity>(table);
        Bind<BodyCmd::GetMaxAngularVelocity, Retooled::GetMaxAngularVelocity>(table);
        Bind<BodyCmd::SetMaxDepenetrationVelocity, Retooled::SetMaxDepenetrationVelocity>(table);
        Bind<BodyCmd::GetMaxDepenetrationVelocity, Retooled::GetMaxDepenetrationVelocity>(table);
        Bind<BodyCmd::SetInertiaTensor, Retooled::SetInertiaTensor>(table);
        Bind<BodyCmd::GetInertiaTensor, Retooled::GetInertiaTensor>(table);
        Bind<BodyCmd::SetInertiaTensorRotation, Retooled::SetInertiaTensorRotation>(table);
        Bind<BodyCmd::GetInertiaTensorRotation, Retooled::GetInertiaTensorRotation>(table);
        Bind<BodyCmd::SetLocalCenterOfMass, Retooled::SetLocalCenterOfMass>(table);
        Bind<BodyCmd::GetLocalCenterOfMass, Retooled::GetLocalCenterOfMass>(table);
        Bind<BodyCmd::GetWorldInertiaTensorMatrix, Retooled::GetWorldInertiaTensorMatrix>(table);
        Bind<BodyCmd::RecomputeMassProperties, Retooled::RecomputeMassProperties>(table);
        Bind<BodyCmd::GetAccumulatedForce, Retooled::GetAccumulatedForce>(table);
        Bind<BodyCmd::GetAccumulatedTorque, Retooled::GetAccumulatedTorque>(table);
        Bind<BodyCmd::GetLocalPointVelocity, Retooled::GetLocalPointVelocity>(table);
        Bind<BodyCmd::GetWorldPointVelocity, Retooled::GetWorldPointVelocity>(table);
        Bind<BodyCmd::GetClosestWorldPointOnBounds, Retooled::GetClosestWorldPointOnBounds>(table);
        Bind<BodyCmd::SetKinematicTarget, Retooled::SetKinematicTarget>(table);
        Bind<BodyCmd::GetKinematicTarget, Retooled::GetKinematicTarget>(table);
        Bind<BodyCmd::GetIsSleeping, Retooled::GetIsSleeping>(table);
        Bind<BodyCmd::Sleep, Retooled::Sleep>(table);
        Bind<BodyCmd::WakeUp, Retooled::WakeUp>(table);
        Bind<BodyCmd::SetWakeCounter, Retooled::SetWakeCounter>(table);
        Bind<BodyCmd::GetWakeCounter, Retooled::GetWakeCounter>(table);
        Bind<BodyCmd::SetSleepThresholdOverride, Retooled::SetSleepThresholdOverride>(table);
        Bind<BodyCmd::GetSleepThresholdOverride, Retooled::GetSleepThresholdOverride>(table);
        Bind<BodyCmd::SetSolverIterationsOverride, Retooled::SetSolverIterationsOverride>(table);
        Bind<BodyCmd::GetSolverIterationsOverride, Retooled::GetSolverIterationsOverride>(table);
    }
}
