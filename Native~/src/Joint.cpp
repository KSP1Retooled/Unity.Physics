// Joint commands. Each one is unpacked here and runs in JointKernel.cs, which addresses the joint through the object handle.
// CreateJoint and DestroyJoint are world commands, in World.cpp.

#include <cstddef>

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    static_assert(sizeof(Physics::ActorDescriptor) == 24, "NativeStore.ActorDescriptor mirrors this layout");
    static_assert(offsetof(Physics::ActorDescriptor, type) == 16, "NativeStore.ActorDescriptor mirrors this layout");

    template <typename T, auto Pack, auto Unpack = None>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Forwarded<T, &ManagedCallbacks::jointCommand, Pack, Unpack, true>;
    }

    static constexpr auto AxisIn = [](CommandArgs& a, const auto& c) { a.value = static_cast<uint32_t>(c.axis); };

    static constexpr auto LimitIn = [](CommandArgs& a, const auto& c)
    {
        a.scalar = c.min;
        a.scalar2 = c.max;
        a.vector = {c.bounciness, c.bounceMinVelocity, c.contactDistance};
    };

    void BindJointCommands(FunctionBinding* table)
    {
        Bind<JointCmd::SetFlags, [](CommandArgs& a, const auto& c)
        {
            a.flags = static_cast<uint32_t>(c.value);
            a.enabled = c.enabled;
        }>(table);
        Bind<JointCmd::GetFlags, None, [](const CommandArgs& a, auto& c) { c.value = static_cast<Physics::JointFlags>(a.flags); }>(table);
        Bind<JointCmd::GetUserData, None, [](const CommandArgs& a, auto& c) { c.value = a.pointer; }>(table);
        Bind<JointCmd::SetActorLocalPose, [](CommandArgs& a, const auto& c)
        {
            PackPose(a, c.value);
            a.value = static_cast<uint32_t>(c.target);
        }>(table);
        Bind<JointCmd::GetActorLocalPose, [](CommandArgs& a, const auto& c) { a.value = static_cast<uint32_t>(c.target); }, PoseOut>(table);
        Bind<JointCmd::SetBreakForce, [](CommandArgs& a, const auto& c)
        {
            a.scalar = c.force;
            a.scalar2 = c.torque;
        }>(table);
        Bind<JointCmd::GetIsConstraintValid, None, [](const CommandArgs& a, auto& c) { c.value = a.enabled != 0; }>(table);
        Bind<JointCmd::GetForces, None, [](const CommandArgs& a, auto& c)
        {
            c.linear = ToVec3(a.vector);
            c.angular = ToVec3(a.vector2);
        }>(table);
        Bind<JointCmd::SetInvMassAndInertiaScale, [](CommandArgs& a, const auto& c)
        {
            a.value = static_cast<uint32_t>(c.target);
            a.scalar = c.value;
        }>(table);
        Bind<JointCmd::SetActors, [](CommandArgs& a, const auto& c) { a.pointer = const_cast<Physics::ActorDescriptor*>(c.actors); }>(table);
        Bind<JointCmd::GetActors, [](CommandArgs& a, auto& c) { a.pointer = c.actors; }>(table);
        Bind<JointCmd::WakeUpActors, None>(table);
        Bind<JointCmd::Set6DofLimit, [](CommandArgs& a, const auto& c)
        {
            AxisIn(a, c);
            LimitIn(a, c);
        }>(table);
        Bind<JointCmd::Get6DofLimit, AxisIn, [](const CommandArgs& a, auto& c)
        {
            c.min = a.scalar;
            c.max = a.scalar2;
            c.bounciness = a.vector.x;
            c.bounceMinVelocity = a.vector.y;
            c.contactDistance = a.vector.z;
        }>(table);
        Bind<JointCmd::Set6DofSpringLimit, [](CommandArgs& a, const auto& c)
        {
            AxisIn(a, c);
            a.scalar = c.spring;
            a.scalar2 = c.damping;
        }>(table);
        Bind<JointCmd::Set6DofAxisLock, [](CommandArgs& a, const auto& c)
        {
            AxisIn(a, c);
            a.mode = static_cast<uint32_t>(c.value);
        }>(table);
        Bind<JointCmd::Set6DofAxisMotorConfiguration, [](CommandArgs& a, const auto& c)
        {
            AxisIn(a, c);
            a.scalar = c.positionSpring;
            a.scalar2 = c.positionDamper;
            a.scalar3 = c.maximumForce;
            a.enabled = c.usesAcceleration;
        }>(table);
        Bind<JointCmd::Set6DofMotorTargetPosition, VectorIn>(table);
        Bind<JointCmd::Set6DofMotorTargetRotation, RotationIn>(table);
        Bind<JointCmd::Set6DofMotorTargetLinearVelocity, VectorIn>(table);
        Bind<JointCmd::Set6DofMotorTargetAngularVelocity, VectorIn>(table);
        Bind<JointCmd::Set6DofProjectionTolerance, [](CommandArgs& a, const auto& c)
        {
            a.scalar = c.linear;
            a.scalar2 = c.angular;
        }>(table);
        Bind<JointCmd::SetDistanceLimit, [](CommandArgs& a, const auto& c)
        {
            a.scalar = c.min;
            a.scalar2 = c.max;
        }>(table);
        Bind<JointCmd::SetDistanceSpringLimit, [](CommandArgs& a, const auto& c)
        {
            a.scalar = c.spring;
            a.scalar2 = c.damping;
        }>(table);
        Bind<JointCmd::SetDistanceErrorTolerance, ScalarIn>(table);
        Bind<JointCmd::SetHingeLimitEnabled, [](CommandArgs& a, const auto& c) { a.enabled = c.value; }>(table);
        Bind<JointCmd::SetHingeMotorConfiguration, [](CommandArgs& a, const auto& c)
        {
            a.scalar = c.maxForce;
            a.enabled = c.freeSpin;
        }>(table);
        Bind<JointCmd::SetHingeMotorTargetVelocity, ScalarIn>(table);
        Bind<JointCmd::SetHingeLimit, LimitIn>(table);
    }
}
