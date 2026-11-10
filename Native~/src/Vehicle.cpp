// Vehicle commands, which back WheelCollider. Each one is unpacked here and runs in VehicleKernel.cs. Per-wheel commands
// carry no handle: Unity passes the vehicle id in its place, so the world comes from the context. Character controllers,
// immediate mode and contact modification stay unadvertised, so their slots keep the logging stubs Plugin.cpp installed.
// The development trace build records what Unity sends instead.

#include <cstddef>
#include <cstdint>

#include "Commands.h"
#include "PhysXVehicleCommandData.h"

using namespace PhysicsCommands;
using namespace PhysicsCommands::PhysXExt;

namespace Retooled
{
    // VehicleKernel writes these through pointers, so NativeStore.cs mirrors their layout.
    static_assert(sizeof(Physics::PhysXExt::WheelQueryResult) == 80, "NativeStore.WheelQueryResult mirrors this layout");
    static_assert(offsetof(Physics::PhysXExt::WheelQueryResult, userData) == 72, "NativeStore.WheelQueryResult mirrors this layout");
    static_assert(sizeof(Physics::PhysXExt::WheelSuspensionData) == 20, "NativeStore.WheelSuspension mirrors this layout");
    static_assert(sizeof(Physics::PhysXExt::WheelData) == 16, "NativeStore.WheelBody mirrors this layout");
    static_assert(sizeof(Physics::PhysXExt::WheelFrictionCurve) == 20, "NativeStore.WheelFrictionCurve mirrors this layout");
    static_assert(sizeof(Physics::FilterData) == 16, "VehicleKernel copies the filter as four words");

    constexpr uint32_t kFirstVehicleCommand = static_cast<uint32_t>(PhysXExtension::CreateVehicle);

    // The id within the vehicle commands, which VehicleCommand in VehicleKernel.cs numbers from zero.
    template <typename T>
    constexpr uint32_t VehicleId()
    {
        return static_cast<uint32_t>(T::command) - kFirstVehicleCommand;
    }

#ifndef RETOOLED_VEHICLE_TRACE
    // The kernel gets the world's data as the handle, the addressed vehicle's id in value2 and the body in pointer3. Pack can
    // overwrite value2 for the commands that name a vehicle themselves.
    template <typename T, auto Pack, auto Unpack>
    static void Forwarded(void* object, const Context& ctx, Command& base)
    {
        T& command = static_cast<T&>(base);
        CommandArgs args{};
        args.rotation.w = 1.0f;

        World* world = static_cast<World*>(ctx.world != nullptr ? ctx.world : object);
        const CommandKernel kernel = g_Managed.vehicleCommand;
        if (kernel != nullptr && world != nullptr)
        {
            args.value2 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(object));
            args.pointer3 = ctx.body;
            Pack(args, command);
            kernel(VehicleId<T>(), &world->data, &args, &g_Globals);
        }

        Unpack(args, command);
    }

    template <typename T, auto Pack, auto Unpack = None>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Forwarded<T, Pack, Unpack>;
    }

    static constexpr auto WheelIn = [](CommandArgs& a, const auto& c) { a.value = c.wheelId; };

    // The kernel writes these results through the pointer, into the command itself.
    template <typename F>
    static void* Out(const F& field)
    {
        return const_cast<F*>(&field);
    }

    void BindExtensionCommands(FunctionBinding* table)
    {
        using namespace VehicleData;

        Bind<CreateVehicle, [](CommandArgs& a, const auto& c)
        {
            a.value = c.sourceVehicleId;
            a.mode = c.wheelCount;
            a.pointer = c.wheelUserDatasBuffer;
        }, [](const CommandArgs& a, auto& c)
        {
            c.outVehicleId = a.value;
            c.outWrittenUserDatas = static_cast<uint32_t>(a.size);
        }>(table);
        Bind<DestroyVehicle, [](CommandArgs& a, const auto& c) { a.value2 = c.vehicleId; }>(table);
        Bind<SetVehicleWheelEnabled, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.enabled = c.enabled;
            a.pointer = c.wheelFrictionData;
            a.pointer2 = c.wheelUserData;
        }, [](const CommandArgs& a, auto& c) { c.outAllWheelsDisabled = a.flags != 0; }>(table);
        Bind<GetFirstDisabledWheelFromVehicle, None, [](const CommandArgs& a, auto& c)
        {
            c.outWheelIndex = a.value;
            c.outWheelCount = a.mode;
        }>(table);
        Bind<SetVehicleHasImplicitSprungMasses, [](CommandArgs& a, const auto& c) { a.enabled = c.hasSprungMasses; }>(table);
        Bind<GetVehicleHasImplicitSprungMasses, None, [](const CommandArgs& a, auto& c) { c.outHasSprungMasses = a.enabled != 0; }>(table);
        Bind<GetVehicleWheelQueryResult, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.pointer = Out(c.outHit);
        }>(table);
        Bind<SetVehicleWheelFilterData, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.pointer = Out(c.filterData);
        }>(table);
        Bind<GetVehicleWheelFilterData, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.pointer = Out(c.outFilterData);
        }>(table);
        Bind<GetVehicleWheelCount, None, [](const CommandArgs& a, auto& c) { c.outCount = a.value; }>(table);
        Bind<GetVehicleWheelsUserData, [](CommandArgs& a, const auto& c)
        {
            a.pointer = c.buffer;
            a.mode = c.bufferSize;
        }, [](const CommandArgs& a, auto& c) { c.outWritten = a.value; }>(table);
        Bind<ResetVehicleQueryResultsForShape, [](CommandArgs& a, const auto& c) { a.pointer = c.shape; }>(table);
        Bind<SetVehicleLongitudinalSpeedSubSteps, [](CommandArgs& a, const auto& c)
        {
            a.scalar = c.longitudinalSpeed;
            a.value = c.belowSpeedSubsteps;
            a.mode = c.aboveSpeedSubsteps;
        }>(table);
        Bind<SetVehicleWheelComOffset, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.vector = ToFloat3(c.wheelCentreOffset);
            a.vector2 = ToFloat3(c.forceAppPointOffset);
        }>(table);
        Bind<GetVehicleWheelComOffset, WheelIn, [](const CommandArgs& a, auto& c)
        {
            c.outWheelCentreOffset = ToVec3(a.vector);
            c.outForceAppPointOffset = ToVec3(a.vector2);
        }>(table);
        Bind<SetVehicleSuspensionExpansionLimitEnabled, [](CommandArgs& a, const auto& c) { a.enabled = c.enabled; }>(table);
        Bind<GetVehicleSuspensionExpansionLimitEnabled, None, [](const CommandArgs& a, auto& c) { c.outEnabled = a.enabled != 0; }>(table);
        Bind<SetVehicleWheelSuspensionData, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.pointer = Out(c.suspension);
        }>(table);
        Bind<GetVehicleWheelSuspensionData, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.pointer = Out(c.outSuspension);
        }>(table);
        Bind<SetVehicleWheelData, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.pointer = Out(c.wheelData);
        }>(table);
        Bind<GetVehicleWheelData, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.pointer = Out(c.outWheelData);
        }>(table);
        Bind<GetVehicleWheelLocalPose, WheelIn, [](const CommandArgs& a, auto& c) { c.pose = UnpackPose(a); }>(table);
        Bind<SetVehicleWheelRotationSpeed, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.scalar = c.speed;
        }>(table);
        Bind<GetVehicleWheelRotationSpeed, WheelIn, [](const CommandArgs& a, auto& c) { c.outSpeed = a.scalar; }>(table);
        Bind<SetVehicleWheelMotorTorque, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.scalar = c.torque;
        }>(table);
        Bind<GetVehicleWheelMotorTorque, WheelIn, [](const CommandArgs& a, auto& c) { c.outTorque = a.scalar; }>(table);
        Bind<SetVehicleWheelBrakeTorque, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.scalar = c.brkTorque;
        }>(table);
        Bind<GetVehicleWheelBrakeTorque, WheelIn, [](const CommandArgs& a, auto& c) { c.outBrkTorque = a.scalar; }>(table);
        Bind<SetVehicleWheelSteerAngle, [](CommandArgs& a, const auto& c)
        {
            WheelIn(a, c);
            a.scalar = c.angle;
        }>(table);
        Bind<GetVehicleWheelSteerAngle, WheelIn, [](const CommandArgs& a, auto& c) { c.outAngle = a.scalar; }>(table);
        Bind<CalculateVehicleSprungMasses, [](CommandArgs& a, const auto& c)
        {
            a.pointer = c.wheelIds;
            a.pointer2 = c.wheelOffsets;
            a.mode = c.wheelCount;
        }>(table);
    }
#else
    void BindExtensionCommands(FunctionBinding* table)
    {
        BindVehicleTrace(table);
    }
#endif

    int32_t VehicleCommandCount()
    {
        return static_cast<int32_t>(PhysXExtension::CalculateVehicleSprungMasses) - static_cast<int32_t>(kFirstVehicleCommand) + 1;
    }
}
