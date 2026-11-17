// Body commands. Each one is unpacked here and runs in BodyKernel.cs, which addresses the body through the object handle,
// except for the few that only read the row's own fields.

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

    // The flags live in the row itself, so these need nothing of managed. Unity asks both of every body in each step's
    // transform read.
    static uint32_t RowFlags(void* object)
    {
        const Body* body = static_cast<const Body*>(object);
        return body != nullptr ? BodyRow(body->world->data, body->index)->flags : 0;
    }

    static void GetFlags(void* object, const Context&, Command& base)
    {
        constexpr uint32_t publicFlags = kBodyGravity | kBodyKinematic | kBodySimulated;
        static_cast<BodyCmd::GetFlags&>(base).value = static_cast<Physics::BodyFlags>(RowFlags(object) & publicFlags);
    }

    static void GetIsSleeping(void* object, const Context&, Command& base)
    {
        static_cast<BodyCmd::GetIsSleeping&>(base).value = (RowFlags(object) & kBodySleeping) != 0;
    }

    // Read pose pointers directly on workers and during main-thread read-back, when managed would not refresh
    // them.
    // Other main-thread reads go through the kernel to refresh pointers after structural changes.
    static void GetPose(void* object, const Context& ctx, Command& base)
    {
        const Body* body = static_cast<const Body*>(object);
        if (body != nullptr
            && (std::this_thread::get_id() != g_MainThread || g_ReadBackWorld.load(std::memory_order_relaxed) == body->world))
        {
            const BodyPose* pose = BodyRow(body->world->data, body->index)->pose;
            static_cast<BodyCmd::GetPose&>(base).value = Physics::Pose(pose->position.x, pose->position.y, pose->position.z,
                pose->rotation.x, pose->rotation.y, pose->rotation.z, pose->rotation.w);
            return;
        }

#ifdef RETOOLED_COMMAND_TRACE
        NoteForwardedPose();
#endif
        Forwarded<BodyCmd::GetPose, &ManagedCallbacks::bodyCommand, None, PoseOut, true>(object, ctx, base);
    }

    void BindBodyCommands(FunctionBinding* table)
    {
        table[static_cast<size_t>(BodyCmd::GetFlags::command)] = &GetFlags;
        table[static_cast<size_t>(BodyCmd::GetIsSleeping::command)] = &GetIsSleeping;
        table[static_cast<size_t>(BodyCmd::GetPose::command)] = &GetPose;

        Bind<BodyCmd::SetFlags, [](CommandArgs& a, const auto& c)
        {
            a.flags = static_cast<uint32_t>(c.value);
            a.enabled = c.enabled;
        }>(table);
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
