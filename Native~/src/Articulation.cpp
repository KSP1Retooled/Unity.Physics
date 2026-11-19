#include "Commands.h"
#include "PhysicsArticulationCommandData.h"

using namespace PhysicsCommands;

namespace Retooled
{
    namespace Tree = ArticulationData;
    namespace Link = ArticulationLinkData;
    namespace Inbound = ArticulationJointData;

    // The world kernel shares the store's ownership and allocation functions with the articulation commands.
    constexpr uint32_t ArticulationCategory = 0x10000u;

    template <typename T, auto Pack, auto Unpack>
    static void Forward(void* object, const Context& ctx, Command& base)
    {
        T& command = static_cast<T&>(base);
        CommandArgs args{};
        args.rotation.w = 1.0f;
        if (g_Managed.worldCommand == nullptr)
        {
            NoteDroppedCommand();
        }
        else
        {
            constexpr bool isLinkCommand = T::command < Articulation::CreateLink;
            void* handle = isLinkCommand && ctx.body != nullptr ? ctx.body : object;
            if (handle != nullptr)
            {
                Pack(args, command);
                g_Managed.worldCommand(ArticulationCategory | static_cast<uint32_t>(T::command), handle, &args, &g_Globals);
            }
        }

        Unpack(args, command);
    }

    template <typename T, auto Pack = None, auto Unpack = None>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Forward<T, Pack, Unpack>;
    }

    static constexpr auto FlagsIn = [](CommandArgs& a, const auto& c)
    {
        a.flags = static_cast<uint32_t>(c.value);
        a.enabled = c.enabled;
    };
    static constexpr auto FlagsOut = [](const CommandArgs& a, auto& c) { c.value = static_cast<decltype(c.value)>(a.flags); };
    static constexpr auto CountOut = [](const CommandArgs& a, auto& c) { c.value = static_cast<decltype(c.value)>(a.size); };
    static constexpr auto HandleOut = [](const CommandArgs& a, auto& c) { c.value = a.pointer; };
    static constexpr auto HandleIn = [](CommandArgs& a, const auto& c) { a.pointer = c.value; };
    static constexpr auto BufferIn = [](CommandArgs& a, const auto& c)
    {
        a.pointer = c.buffer;
        a.size = c.bufferSize;
    };
    static constexpr auto WrittenOut = [](const CommandArgs& a, auto& c) { c.written = static_cast<size_t>(a.size); };
    static constexpr auto ForceIn = [](CommandArgs& a, const auto& c)
    {
        a.vector = ToFloat3(c.value);
        a.mode = static_cast<uint32_t>(c.mode);
        a.enabled = c.awake;
    };
    static constexpr auto VectorAwakeIn = [](CommandArgs& a, const auto& c)
    {
        a.vector = ToFloat3(c.value);
        a.enabled = c.awake;
    };
    static constexpr auto ReducedIn = [](CommandArgs& a, const auto& c)
    {
        a.vector = ToFloat3(c.value.coordinate);
        a.value = c.value.dofCount;
    };
    static constexpr auto ReducedOut = [](const CommandArgs& a, auto& c)
    {
        c.value.coordinate = ToVec3(a.vector);
        c.value.dofCount = static_cast<int>(a.value);
    };
    static constexpr auto FloatsIn = [](CommandArgs& a, const auto& c)
    {
        a.pointer = const_cast<float*>(c.value.data);
        a.size = c.value.size;
    };
    static constexpr auto FloatsOut = [](const CommandArgs& a, auto& c)
    {
        c.value = { static_cast<const float*>(a.pointer), static_cast<size_t>(a.size), a.pointer != nullptr };
    };
    static constexpr auto LinkIn = [](CommandArgs& a, const auto& c) { a.pointer = c.link; };
    static constexpr auto ElapsedIn = [](CommandArgs& a, const auto& c) { a.scalar = c.deltaTime; };

    static const Body* LinkBody(void* object, const Context& ctx)
    {
        return static_cast<const Body*>(ctx.body != nullptr ? ctx.body : object);
    }

    static void GetLinkFlags(void* object, const Context& ctx, Command& base)
    {
        const Body* body = LinkBody(object, ctx);
        const uint32_t flags = body != nullptr ? BodyRow(body->world->data, body->index)->flags : 0;
        static_cast<Link::GetFlags&>(base).value = static_cast<Physics::BodyFlags>(flags & kBodyGravity);
    }

    static void GetLinkPose(void* object, const Context& ctx, Command& base)
    {
        const Body* body = LinkBody(object, ctx);
        if (body != nullptr && (std::this_thread::get_id() != g_MainThread ||
            g_ReadBackWorld.load(std::memory_order_relaxed) == body->world))
        {
            const BodyPose* pose = BodyRow(body->world->data, body->index)->pose;
            static_cast<Link::GetPose&>(base).value = Physics::Pose(pose->position.x, pose->position.y, pose->position.z,
                pose->rotation.x, pose->rotation.y, pose->rotation.z, pose->rotation.w);
            return;
        }

        Forward<Link::GetPose, None, PoseOut>(object, ctx, base);
    }

    void BindArticulationCommands(FunctionBinding* table)
    {
        Bind<Link::GetLinkIndex, None, CountOut>(table);
        Bind<Link::GetLinkJoint, None, HandleOut>(table);
        Bind<Link::SetFlags, FlagsIn>(table);
        table[static_cast<size_t>(Link::GetFlags::command)] = &GetLinkFlags;
        Bind<Link::SetMass, ScalarIn>(table);
        Bind<Link::GetMass, None, ScalarOut>(table);
        Bind<Link::SetDamping, [](CommandArgs& a, const auto& c)
        {
            a.scalar = c.linear;
            a.scalar2 = c.angular;
        }>(table);
        Bind<Link::GetDamping, None, [](const CommandArgs& a, auto& c)
        {
            c.linear = a.scalar;
            c.angular = a.scalar2;
        }>(table);
        Bind<Link::SetPose, [](CommandArgs& a, const auto& c)
        {
            PackPose(a, c.value);
            a.enabled = c.awake;
        }>(table);
        table[static_cast<size_t>(Link::GetPose::command)] = &GetLinkPose;
        Bind<Link::AttachShape, [](CommandArgs& a, const auto& c)
        {
            a.pointer = c.shape;
            PackPose(a, c.pose);
            a.enabled = 1;
        }>(table);
        Bind<Link::DetachShape, [](CommandArgs& a, const auto& c) { a.pointer = c.shape; }>(table);
        Bind<Link::GetShapeCount, None, CountOut>(table);
        Bind<Link::GetShapesUserData, BufferIn, WrittenOut>(table);
        Bind<Link::GetUserData, None, HandleOut>(table);
        Bind<Link::SetCollisionDetectionMode, [](CommandArgs& a, const auto& c) { a.mode = static_cast<uint32_t>(c.value); }>(table);
        Bind<Link::AddForce, ForceIn>(table);
        Bind<Link::AddTorque, ForceIn>(table);
        Bind<Link::ClearForce>(table);
        Bind<Link::ClearTorque>(table);
        Bind<Link::SetLinearVelocity, VectorAwakeIn>(table);
        Bind<Link::GetLinearVelocity, None, VectorOut>(table);
        Bind<Link::SetAngularVelocity, VectorAwakeIn>(table);
        Bind<Link::GetAngularVelocity, None, VectorOut>(table);
        Bind<Link::SetMaxLinearVelocity, ScalarIn>(table);
        Bind<Link::GetMaxLinearVelocity, None, ScalarOut>(table);
        Bind<Link::SetMaxAngularVelocity, ScalarIn>(table);
        Bind<Link::GetMaxAngularVelocity, None, ScalarOut>(table);
        Bind<Link::SetMaxDepenetrationVelocity, ScalarIn>(table);
        Bind<Link::GetMaxDepenetrationVelocity, None, ScalarOut>(table);
        Bind<Link::SetInertiaTensor, VectorIn>(table);
        Bind<Link::GetInertiaTensor, None, VectorOut>(table);
        Bind<Link::SetInertiaTensorRotation, RotationIn>(table);
        Bind<Link::GetInertiaTensorRotation, None, RotationOut>(table);
        Bind<Link::SetLocalCenterOfMass, VectorIn>(table);
        Bind<Link::GetLocalCenterOfMass, None, VectorOut>(table);
        Bind<Link::GetWorldInertiaTensorMatrix, [](CommandArgs& a, const auto& c) { a.pointer = const_cast<float*>(c.value); }>(table);
        Bind<Link::RecomputeMassProperties, [](CommandArgs& a, const auto& c)
        {
            const auto& overrides = c.overrides;
            a.vector = ToFloat3(overrides.centerOfMass);
            a.vector2 = ToFloat3(overrides.inertiaTensor);
            a.rotation = ToFloat4(overrides.inertiaTensorRotation);
            a.flags = (overrides.overrideCenterOfMass ? 1u : 0u) | (overrides.overrideInertiaTensor ? 2u : 0u);
        }>(table);
        Bind<Link::GetAccumulatedForce, ElapsedIn, VectorOut>(table);
        Bind<Link::GetAccumulatedTorque, ElapsedIn, VectorOut>(table);
        Bind<Link::GetLocalPointVelocity, [](CommandArgs& a, const auto& c) { a.vector = ToFloat3(c.localPoint); }, VectorOut>(table);
        Bind<Link::GetWorldPointVelocity, [](CommandArgs& a, const auto& c) { a.vector = ToFloat3(c.worldPoint); }, VectorOut>(table);
        Bind<Link::GetClosestWorldPoint, [](CommandArgs& a, const auto& c)
        {
            PackPose(a, c.enginePose);
            a.vector = ToFloat3(c.worldPosition);
        }, [](const CommandArgs& a, auto& c)
        {
            c.outClosestWorldPosition = ToVec3(a.vector);
            c.outSqrDistance = a.scalar;
        }>(table);

        Bind<Inbound::SetType, [](CommandArgs& a, const auto& c) { a.value = static_cast<uint32_t>(c.value); }>(table);
        Bind<Inbound::GetType, None, [](const CommandArgs& a, auto& c) { c.value = static_cast<Physics::JointType>(a.value); }>(table);
        Bind<Inbound::SetAxisLock, [](CommandArgs& a, const auto& c)
        {
            a.mode = static_cast<uint32_t>(c.axis);
            a.value = static_cast<uint32_t>(c.value);
        }>(table);
        Bind<Inbound::SetLimit, [](CommandArgs& a, const auto& c)
        {
            a.mode = static_cast<uint32_t>(c.axis);
            a.scalar = c.min;
            a.scalar2 = c.max;
        }>(table);
        Bind<Inbound::SetActorLocalPose, [](CommandArgs& a, const auto& c)
        {
            a.mode = static_cast<uint32_t>(c.target);
            PackPose(a, c.value);
        }>(table);
        Bind<Inbound::SetAxisMotorConfiguration, [](CommandArgs& a, const auto& c)
        {
            a.mode = static_cast<uint32_t>(c.axis);
            a.vector = { c.value.lowerLimit, c.value.upperLimit, c.value.stiffness };
            a.vector2 = { c.value.damping, c.value.forceLimit, c.value.target };
            a.scalar = c.value.targetVelocity;
            a.value = static_cast<uint32_t>(c.value.type);
        }>(table);
        Bind<Inbound::SetFriction, ScalarIn>(table);
        Bind<Inbound::SetPosition, ReducedIn>(table);
        Bind<Inbound::GetPosition, None, ReducedOut>(table);
        Bind<Inbound::SetVelocity, ReducedIn>(table);
        Bind<Inbound::GetVelocity, None, ReducedOut>(table);
        Bind<Inbound::SetAcceleration, ReducedIn>(table);
        Bind<Inbound::GetAcceleration, None, ReducedOut>(table);
        Bind<Inbound::SetForce, ReducedIn>(table);
        Bind<Inbound::GetForce, None, ReducedOut>(table);
        Bind<Inbound::GetMotorForce, None, ReducedOut>(table);
        Bind<Inbound::GetForcesForAcceleration, [](CommandArgs& a, const auto& c)
        {
            a.vector = ToFloat3(c.acceleration.coordinate);
            a.value = c.acceleration.dofCount;
        }, ReducedOut>(table);
        Bind<Inbound::SetMaxVelocity, ScalarIn>(table);
        Bind<Inbound::GetMaxVelocity, None, ScalarOut>(table);
        Bind<Inbound::GetDofCount, None, CountOut>(table);

        Bind<Tree::CreateLink, [](CommandArgs& a, const auto& c)
        {
            a.pointer = c.parentLink;
            a.pointer2 = c.userData;
            PackPose(a, c.pose);
        }, [](const CommandArgs& a, auto& c) { c.outNewLink = a.pointer; }>(table);
        Bind<Tree::DestroyLink, HandleIn>(table);
        Bind<Tree::GetLinkCount, None, CountOut>(table);
        // The editor publishes the ID but has no command data struct for GetLinks.
        table[static_cast<size_t>(Articulation::GetLinks)] = [](void*, const Context&, Command&)
        {
            LogUnimplementedOnce(kArticulationCategory, static_cast<uint32_t>(Articulation::GetLinks));
        };
        Bind<Tree::GetLinksUserData, BufferIn, WrittenOut>(table);
        Bind<Tree::GetLinkParent, LinkIn, HandleOut>(table);
        Bind<Tree::GetLinkParentUserData, LinkIn, HandleOut>(table);
        Bind<Tree::GetLinkChildCount, LinkIn, CountOut>(table);
        Bind<Tree::GetLinkChildrenUserData, [](CommandArgs& a, const auto& c)
        {
            a.pointer = c.link;
            a.pointer2 = c.buffer;
            a.size = c.bufferSize;
        }, WrittenOut>(table);
        Bind<Tree::SetFlags, FlagsIn>(table);
        Bind<Tree::GetFlags, None, FlagsOut>(table);
        Bind<Tree::GetUserData, None, HandleOut>(table);
        Bind<Tree::GetIsSleeping, None, [](const CommandArgs& a, auto& c) { c.value = a.enabled != 0; }>(table);
        Bind<Tree::Sleep>(table);
        Bind<Tree::WakeUp>(table);
        Bind<Tree::SetSleepThresholdOverride, ScalarIn>(table);
        Bind<Tree::GetSleepThresholdOverride, None, ScalarOut>(table);
        Bind<Tree::SetSolverIterationsOverride, [](CommandArgs& a, const auto& c)
        {
            a.value = c.positionIterations;
            a.value2 = c.velocityIterations;
        }>(table);
        Bind<Tree::GetSolverIterationsOverride, None, [](const CommandArgs& a, auto& c)
        {
            c.positionIterations = a.value;
            c.velocityIterations = a.value2;
        }>(table);
        Bind<Tree::GetDenseJacobianAsBuffer, None, [](const CommandArgs& a, auto& c)
        {
            c.rowCount = a.value;
            c.colCount = a.value2;
            FloatsOut(a, c);
        }>(table);
        Bind<Tree::SetJointsPositions, FloatsIn>(table);
        Bind<Tree::GetJointsPositions, None, FloatsOut>(table);
        Bind<Tree::SetJointsVelocities, FloatsIn>(table);
        Bind<Tree::GetJointsVelocities, None, FloatsOut>(table);
        Bind<Tree::SetJointsAccelerations, FloatsIn>(table);
        Bind<Tree::GetJointsAccelerations, None, FloatsOut>(table);
        Bind<Tree::SetJointsForces, FloatsIn>(table);
        Bind<Tree::GetJointsForces, None, FloatsOut>(table);
        Bind<Tree::GetJointsDriveForces, None, FloatsOut>(table);
        Bind<Tree::GetJointsGravityForces, None, FloatsOut>(table);
        Bind<Tree::GetJointsCoriolisCentrifugalForces, None, FloatsOut>(table);
        Bind<Tree::GetJointsExternalForces, ElapsedIn, FloatsOut>(table);
        table[static_cast<size_t>(Tree::ReleaseDataBuffer::command)] = [](void*, const Context&, Command& base)
        {
            const auto& buffer = static_cast<Tree::ReleaseDataBuffer&>(base).value;
            if (buffer.sdkOwnedMemory)
            {
                g_Globals.release(const_cast<float*>(buffer.data));
            }
        };
    }
}
