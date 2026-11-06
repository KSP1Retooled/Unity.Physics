// Body commands. Each one is unpacked here and runs in BodyKernel.cs, which addresses the body through the object handle.

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    template <typename T, auto Pack, auto Unpack = None>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Forwarded<T, &ManagedCallbacks::bodyCommand, Pack, Unpack, true>;
    }

    static constexpr auto VectorAwakeIn = [](CommandArgs& a, const auto& c)
    {
        a.vector = ToFloat3(c.value);
        a.enabled = c.awake;
    };

    static constexpr auto ForceIn = [](CommandArgs& a, const auto& c)
    {
        a.vector = ToFloat3(c.value);
        a.mode = static_cast<uint32_t>(c.mode);
        a.enabled = c.awake;
    };

    static constexpr auto BufferIn = [](CommandArgs& a, const auto& c)
    {
        a.pointer = c.buffer;
        a.size = c.bufferSize;
    };

    static constexpr auto WrittenOut = [](const CommandArgs& a, auto& c) { c.written = static_cast<size_t>(a.size); };
    static constexpr auto DeltaTimeIn = [](CommandArgs& a, const auto& c) { a.scalar = c.deltaTime; };
    static constexpr auto ShapeIn = [](CommandArgs& a, const auto& c) { a.pointer = c.shape; };

    void BindBodyCommands(FunctionBinding* table)
    {
        Bind<BodyCmd::SetFlags, [](CommandArgs& a, const auto& c)
        {
            a.flags = static_cast<uint32_t>(c.value);
            a.enabled = c.enabled;
        }>(table);
        Bind<BodyCmd::GetFlags, None, [](const CommandArgs& a, auto& c) { c.value = static_cast<Physics::BodyFlags>(a.flags); }>(table);
        Bind<BodyCmd::SetMass, ScalarIn>(table);
        Bind<BodyCmd::GetMass, None, ScalarOut>(table);
        Bind<BodyCmd::SetDamping, [](CommandArgs& a, const auto& c)
        {
            a.scalar = c.linear;
            a.scalar2 = c.angular;
        }>(table);
        Bind<BodyCmd::GetDamping, None, [](const CommandArgs& a, auto& c)
        {
            c.linear = a.scalar;
            c.angular = a.scalar2;
        }>(table);
        Bind<BodyCmd::SetPose, [](CommandArgs& a, const auto& c)
        {
            PackPose(a, c.value);
            a.enabled = c.awake;
        }>(table);
        Bind<BodyCmd::GetPose, None, PoseOut>(table);
        Bind<BodyCmd::AttachShape, [](CommandArgs& a, const auto& c)
        {
            a.pointer = c.shape;
            PackPose(a, c.pose);
            a.enabled = c.forceLocalPose;
        }>(table);
        Bind<BodyCmd::DetachShape, ShapeIn>(table);
        Bind<BodyCmd::GetShapeCount, None, [](const CommandArgs& a, auto& c) { c.value = static_cast<size_t>(a.size); }>(table);
        Bind<BodyCmd::GetShapes, BufferIn, WrittenOut>(table);
        Bind<BodyCmd::GetShapesUserData, BufferIn, WrittenOut>(table);
        Bind<BodyCmd::GetUserData, None, [](const CommandArgs& a, auto& c) { c.value = a.pointer; }>(table);
        Bind<BodyCmd::SetCollisionDetectionMode, [](CommandArgs& a, const auto& c) { a.mode = static_cast<uint32_t>(c.value); }>(table);
        Bind<BodyCmd::AddForce, ForceIn>(table);
        Bind<BodyCmd::AddTorque, ForceIn>(table);
        Bind<BodyCmd::ClearForce, None>(table);
        Bind<BodyCmd::ClearTorque, None>(table);
        Bind<BodyCmd::SetLinearVelocity, VectorAwakeIn>(table);
        Bind<BodyCmd::GetLinearVelocity, None, VectorOut>(table);
        Bind<BodyCmd::SetAngularVelocity, VectorAwakeIn>(table);
        Bind<BodyCmd::GetAngularVelocity, None, VectorOut>(table);
        Bind<BodyCmd::SetMaxLinearVelocity, ScalarIn>(table);
        Bind<BodyCmd::GetMaxLinearVelocity, None, ScalarOut>(table);
        Bind<BodyCmd::SetMaxAngularVelocity, ScalarIn>(table);
        Bind<BodyCmd::GetMaxAngularVelocity, None, ScalarOut>(table);
        Bind<BodyCmd::SetMaxDepenetrationVelocity, ScalarIn>(table);
        Bind<BodyCmd::GetMaxDepenetrationVelocity, None, ScalarOut>(table);
        Bind<BodyCmd::SetInertiaTensor, VectorIn>(table);
        Bind<BodyCmd::GetInertiaTensor, None, VectorOut>(table);
        Bind<BodyCmd::SetInertiaTensorRotation, RotationIn>(table);
        Bind<BodyCmd::GetInertiaTensorRotation, None, RotationOut>(table);
        Bind<BodyCmd::SetLocalCenterOfMass, VectorIn>(table);
        Bind<BodyCmd::GetLocalCenterOfMass, None, VectorOut>(table);
        Bind<BodyCmd::GetWorldInertiaTensorMatrix, [](CommandArgs& a, auto& c) { a.pointer = c.value; }>(table);
        Bind<BodyCmd::RecomputeMassProperties, [](CommandArgs& a, const auto& c)
        {
            const Physics::MassPropertiesOverride& overrides = c.overrides;
            a.vector = ToFloat3(overrides.centerOfMass);
            a.vector2 = ToFloat3(overrides.inertiaTensor);
            a.rotation = ToFloat4(overrides.inertiaTensorRotation);
            a.flags = (overrides.overrideCenterOfMass ? 1u : 0u) | (overrides.overrideInertiaTensor ? 2u : 0u);
        }>(table);
        Bind<BodyCmd::GetAccumulatedForce, DeltaTimeIn, VectorOut>(table);
        Bind<BodyCmd::GetAccumulatedTorque, DeltaTimeIn, VectorOut>(table);
        Bind<BodyCmd::GetLocalPointVelocity, [](CommandArgs& a, const auto& c) { a.vector = ToFloat3(c.localPoint); }, VectorOut>(table);
        Bind<BodyCmd::GetWorldPointVelocity, [](CommandArgs& a, const auto& c) { a.vector = ToFloat3(c.worldPoint); }, VectorOut>(table);
        Bind<BodyCmd::GetClosestWorldPointOnBounds, [](CommandArgs& a, const auto& c) { a.vector = ToFloat3(c.worldPosition); },
            [](const CommandArgs& a, auto& c)
            {
                c.outClosestWorldPositionOnBounds = ToVec3(a.vector);
                c.outSqrDistance = a.scalar;
            }>(table);
        Bind<BodyCmd::SetKinematicTarget, PoseIn>(table);
        Bind<BodyCmd::GetKinematicTarget, None, PoseOut>(table);
        Bind<BodyCmd::GetIsSleeping, None, [](const CommandArgs& a, auto& c) { c.value = a.enabled != 0; }>(table);
        Bind<BodyCmd::Sleep, None>(table);
        Bind<BodyCmd::WakeUp, None>(table);
        Bind<BodyCmd::SetWakeCounter, ScalarIn>(table);
        Bind<BodyCmd::GetWakeCounter, None, ScalarOut>(table);
        Bind<BodyCmd::SetSleepThresholdOverride, ScalarIn>(table);
        Bind<BodyCmd::GetSleepThresholdOverride, None, ScalarOut>(table);
        Bind<BodyCmd::SetSolverIterationsOverride, [](CommandArgs& a, const auto& c)
        {
            a.value = c.positionIterations;
            a.value2 = c.velocityIterations;
        }>(table);
        Bind<BodyCmd::GetSolverIterationsOverride, None, [](const CommandArgs& a, auto& c)
        {
            c.positionIterations = a.value;
            c.velocityIterations = a.value2;
        }>(table);
    }
}
