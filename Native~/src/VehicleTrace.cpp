// Development recording of what Unity sends a backend that advertises vehicles: the PhysX extension commands with their
// addressing and fields, and, once armed by a mark, every other command between them with its addressing. It keeps just
// enough wheel state to answer the getters, so that Unity carries on as it would with PhysX. The file is written to
// Library/RetooledPhysics/vehicle-trace.txt, relative to the project folder a batch editor runs in.

#ifdef RETOOLED_VEHICLE_TRACE

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "Commands.h"
#include "PhysXVehicleCommandData.h"

using namespace PhysicsCommands;
using namespace PhysicsCommands::PhysXExt;
using Physics::PhysXExt::kInvalidWheelId;
using Physics::PhysXExt::kMaxVehicleWheelsCount;

namespace Retooled
{
    struct TraceWheel
    {
        bool enabled = false;
        void* userData = nullptr;
        Physics::PhysXExt::WheelSuspensionData suspension{};
        Physics::PhysXExt::WheelData wheel{};
        Physics::Vec3 centreOffset{};
        Physics::Vec3 forceOffset{};
        Physics::FilterData filter{};
        float speed = 0.0f;
        float motor = 0.0f;
        float brake = 0.0f;
        float steer = 0.0f;
    };

    struct TraceVehicle
    {
        uint32_t wheelCount = 0;
        // Unity never turns this on for a new vehicle, and it reads it back before distributing the sprung mass.
        bool implicitSprungMasses = true;
        bool expansionLimit = false;
        TraceWheel wheels[kMaxVehicleWheelsCount];
    };

    static std::recursive_mutex s_Lock;
    static std::FILE* s_File;
    static bool s_Armed;
    static uint32_t s_NextVehicle;
    static uint32_t s_Line;
    static std::map<uint32_t, TraceVehicle> s_Vehicles;
    static std::map<void*, std::string> s_Names;

    static std::FILE* File()
    {
        if (s_File == nullptr)
        {
            s_File = std::fopen("Library/RetooledPhysics/vehicle-trace.txt", "w");
        }

        return s_File;
    }

    static std::string Numbered(const char* kind, size_t number)
    {
        return std::string(kind) + "#" + std::to_string(number);
    }

    // A stable name for a handle: pointers differ per run, and the dense index of a body moves as bodies go.
    static const std::string& Name(void* handle)
    {
        static const std::string none = "null";
        if (handle == nullptr)
        {
            return none;
        }

        // Too small for a pointer: this is a vehicle id, which Unity passes in place of the object handle.
        if (reinterpret_cast<uintptr_t>(handle) < 0x10000)
        {
            auto id = s_Names.emplace(handle, Numbered("vehicle-id", reinterpret_cast<uintptr_t>(handle)));
            return id.first->second;
        }

        auto known = s_Names.find(handle);
        if (known != s_Names.end())
        {
            return known->second;
        }

        std::string name;
        for (size_t w = 0; w < g_Worlds.size() && name.empty(); w++)
        {
            if (g_Worlds[w] == handle)
            {
                name = Numbered("world", s_Names.size());
            }

            const WorldData& data = g_Worlds[w]->data;
            for (uint32_t b = 0; b < data.bodyCount && name.empty(); b++)
            {
                const BodyData& hot = data.bodies[b];
                if (hot.owner == handle)
                {
                    name = Numbered((hot.flags & kBodyStatic) != 0 ? "static" : "body", s_Names.size());
                }

                for (uint32_t s = 0; s < hot.shapeCount && name.empty(); s++)
                {
                    if (hot.shapes[s] == handle)
                    {
                        name = Numbered("shape", s_Names.size());
                    }
                }
            }
        }

        if (name.empty())
        {
            name = Numbered("unknown", s_Names.size());
        }

        return s_Names[handle] = name;
    }

    static void Write(const char* kind, void* object, const Context& ctx, const char* format, ...)
    {
        std::lock_guard lock(s_Lock);
        std::FILE* file = File();
        if (file == nullptr)
        {
            return;
        }

        std::fprintf(file, "%05u %-40s object=%s ctx.world=%s ctx.body=%s ", ++s_Line, kind, Name(object).c_str(),
            Name(ctx.world).c_str(), Name(ctx.body).c_str());

        std::va_list arguments;
        va_start(arguments, format);
        std::vfprintf(file, format, arguments);
        va_end(arguments);

        std::fputc('\n', file);
        std::fflush(file);
    }

    void TraceCommand(void* object, const Context& ctx, uint32_t function)
    {
        if (s_Armed)
        {
            char kind[32];
            std::snprintf(kind, sizeof(kind), "cmd %u.%u", function >> kMaskOffsetSize, function & kReverseFullMask);
            Write(kind, object, ctx, "");
        }
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_TraceMark(const char* text)
    {
        std::lock_guard lock(s_Lock);
        s_Armed = true;
        if (File() != nullptr)
        {
            std::fprintf(File(), "\n=== %s\n", text);
            std::fflush(File());
        }
    }

    // Per-wheel commands carry their vehicle's id in the object handle, and vehicle 0 therefore looks like a null object.
    static TraceVehicle* VehicleOf(void* object, const Context&)
    {
        auto vehicle = s_Vehicles.find(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(object)));
        return vehicle != s_Vehicles.end() ? &vehicle->second : nullptr;
    }

    static TraceWheel* WheelOf(void* object, const Context& ctx, uint32_t wheelId)
    {
        static TraceWheel nothing;
        TraceVehicle* vehicle = VehicleOf(object, ctx);
        return vehicle != nullptr && wheelId < kMaxVehicleWheelsCount ? &vehicle->wheels[wheelId] : &nothing;
    }

    static void CreateVehicle(void* object, const Context& ctx, VehicleData::CreateVehicle& command)
    {
        std::lock_guard lock(s_Lock);
        TraceVehicle created;
        created.wheelCount = command.wheelCount;
        command.outWrittenUserDatas = 0;

        auto source = s_Vehicles.find(command.sourceVehicleId);
        if (source != s_Vehicles.end())
        {
            created = source->second;
            created.wheelCount = command.wheelCount;
            uint32_t carried = std::min(source->second.wheelCount, command.wheelCount);
            for (uint32_t i = 0; i < carried && command.wheelUserDatasBuffer != nullptr; i++)
            {
                command.wheelUserDatasBuffer[i] = source->second.wheels[i].userData;
            }

            command.outWrittenUserDatas = carried;
        }

        command.outVehicleId = s_NextVehicle++;
        s_Vehicles[command.outVehicleId] = created;
        Write("CreateVehicle", object, ctx, "source=%d wheelCount=%u buffer=%s -> id=%u written=%u",
            static_cast<int32_t>(command.sourceVehicleId), command.wheelCount, command.wheelUserDatasBuffer != nullptr ? "yes" : "no",
            command.outVehicleId, command.outWrittenUserDatas);
    }

    static void DestroyVehicle(void* object, const Context& ctx, VehicleData::DestroyVehicle& command)
    {
        std::lock_guard lock(s_Lock);
        s_Vehicles.erase(command.vehicleId);
        Write("DestroyVehicle", object, ctx, "id=%u", command.vehicleId);
    }

    static void SetWheelEnabled(void* object, const Context& ctx, VehicleData::SetVehicleWheelEnabled& command)
    {
        std::lock_guard lock(s_Lock);
        TraceWheel* wheel = WheelOf(object, ctx, command.wheelId);
        wheel->enabled = command.enabled;
        wheel->userData = command.wheelUserData;

        TraceVehicle* vehicle = VehicleOf(object, ctx);
        command.outAllWheelsDisabled = true;
        for (uint32_t i = 0; vehicle != nullptr && i < vehicle->wheelCount; i++)
        {
            command.outAllWheelsDisabled = command.outAllWheelsDisabled && !vehicle->wheels[i].enabled;
        }

        Write("SetVehicleWheelEnabled", object, ctx, "wheel=%u enabled=%d friction=%s userData=%s -> allDisabled=%d", command.wheelId,
            command.enabled, command.wheelFrictionData != nullptr ? "yes" : "no", command.wheelUserData != nullptr ? "yes" : "no",
            command.outAllWheelsDisabled);
    }

    static void GetFirstDisabled(void* object, const Context& ctx, VehicleData::GetFirstDisabledWheelFromVehicle& command)
    {
        std::lock_guard lock(s_Lock);
        TraceVehicle* vehicle = VehicleOf(object, ctx);
        command.outWheelIndex = kInvalidWheelId;
        command.outWheelCount = vehicle != nullptr ? vehicle->wheelCount : 0;
        for (uint32_t i = 0; vehicle != nullptr && i < vehicle->wheelCount; i++)
        {
            if (!vehicle->wheels[i].enabled)
            {
                command.outWheelIndex = i;
                break;
            }
        }

        Write("GetFirstDisabledWheelFromVehicle", object, ctx, "-> index=%d count=%u", static_cast<int32_t>(command.outWheelIndex),
            command.outWheelCount);
    }

    static void SetImplicitSprungMasses(void* object, const Context& ctx, VehicleData::SetVehicleHasImplicitSprungMasses& command)
    {
        std::lock_guard lock(s_Lock);
        if (TraceVehicle* vehicle = VehicleOf(object, ctx))
        {
            vehicle->implicitSprungMasses = command.hasSprungMasses;
        }

        Write("SetVehicleHasImplicitSprungMasses", object, ctx, "value=%d", command.hasSprungMasses);
    }

    static void GetImplicitSprungMasses(void* object, const Context& ctx, VehicleData::GetVehicleHasImplicitSprungMasses& command)
    {
        std::lock_guard lock(s_Lock);
        TraceVehicle* vehicle = VehicleOf(object, ctx);
        command.outHasSprungMasses = vehicle != nullptr && vehicle->implicitSprungMasses;
        Write("GetVehicleHasImplicitSprungMasses", object, ctx, "-> %d", command.outHasSprungMasses);
    }

    static void GetQueryResult(void* object, const Context& ctx, VehicleData::GetVehicleWheelQueryResult& command)
    {
        std::lock_guard lock(s_Lock);
        std::memset(static_cast<void*>(&command.outHit), 0, sizeof(command.outHit));
        Write("GetVehicleWheelQueryResult", object, ctx, "wheel=%u -> no hit", command.wheelId);
    }

    static void SetFilter(void* object, const Context& ctx, VehicleData::SetVehicleWheelFilterData& command)
    {
        std::lock_guard lock(s_Lock);
        WheelOf(object, ctx, command.wheelId)->filter = command.filterData;

        uint32_t words[4];
        std::memcpy(words, &command.filterData, sizeof(words));
        Write("SetVehicleWheelFilterData", object, ctx, "wheel=%u words=%08x %08x %08x %08x", command.wheelId, words[0], words[1],
            words[2], words[3]);
    }

    static void GetFilter(void* object, const Context& ctx, VehicleData::GetVehicleWheelFilterData& command)
    {
        std::lock_guard lock(s_Lock);
        command.outFilterData = WheelOf(object, ctx, command.wheelId)->filter;
        Write("GetVehicleWheelFilterData", object, ctx, "wheel=%u", command.wheelId);
    }

    static void GetWheelCount(void* object, const Context& ctx, VehicleData::GetVehicleWheelCount& command)
    {
        std::lock_guard lock(s_Lock);
        TraceVehicle* vehicle = VehicleOf(object, ctx);
        command.outCount = vehicle != nullptr ? vehicle->wheelCount : 0;
        Write("GetVehicleWheelCount", object, ctx, "-> %u", command.outCount);
    }

    static void GetWheelsUserData(void* object, const Context& ctx, VehicleData::GetVehicleWheelsUserData& command)
    {
        std::lock_guard lock(s_Lock);
        TraceVehicle* vehicle = VehicleOf(object, ctx);
        command.outWritten = 0;
        // Unity dereferences every pointer it gets back, so disabled slots stay out.
        for (uint32_t i = 0; vehicle != nullptr && i < vehicle->wheelCount && command.outWritten < command.bufferSize; i++)
        {
            if (vehicle->wheels[i].enabled)
            {
                command.buffer[command.outWritten++] = vehicle->wheels[i].userData;
            }
        }

        Write("GetVehicleWheelsUserData", object, ctx, "buffer=%u -> %u", command.bufferSize, command.outWritten);
    }

    static void ResetQueryResults(void* object, const Context& ctx, VehicleData::ResetVehicleQueryResultsForShape& command)
    {
        std::lock_guard lock(s_Lock);
        Write("ResetVehicleQueryResultsForShape", object, ctx, "shape=%s", Name(command.shape).c_str());
    }

    static void SetSubSteps(void* object, const Context& ctx, VehicleData::SetVehicleLongitudinalSpeedSubSteps& command)
    {
        std::lock_guard lock(s_Lock);
        Write("SetVehicleLongitudinalSpeedSubSteps", object, ctx, "speed=%g below=%u above=%u", command.longitudinalSpeed,
            command.belowSpeedSubsteps, command.aboveSpeedSubsteps);
    }

    static void SetComOffset(void* object, const Context& ctx, VehicleData::SetVehicleWheelComOffset& command)
    {
        std::lock_guard lock(s_Lock);
        TraceWheel* wheel = WheelOf(object, ctx, command.wheelId);
        wheel->centreOffset = command.wheelCentreOffset;
        wheel->forceOffset = command.forceAppPointOffset;
        Write("SetVehicleWheelComOffset", object, ctx, "wheel=%u centre=(%g %g %g) force=(%g %g %g)", command.wheelId,
            command.wheelCentreOffset.x, command.wheelCentreOffset.y, command.wheelCentreOffset.z, command.forceAppPointOffset.x,
            command.forceAppPointOffset.y, command.forceAppPointOffset.z);
    }

    static void GetComOffset(void* object, const Context& ctx, VehicleData::GetVehicleWheelComOffset& command)
    {
        std::lock_guard lock(s_Lock);
        TraceWheel* wheel = WheelOf(object, ctx, command.wheelId);
        command.outWheelCentreOffset = wheel->centreOffset;
        command.outForceAppPointOffset = wheel->forceOffset;
        Write("GetVehicleWheelComOffset", object, ctx, "wheel=%u", command.wheelId);
    }

    static void SetExpansionLimit(void* object, const Context& ctx, VehicleData::SetVehicleSuspensionExpansionLimitEnabled& command)
    {
        std::lock_guard lock(s_Lock);
        if (TraceVehicle* vehicle = VehicleOf(object, ctx))
        {
            vehicle->expansionLimit = command.enabled;
        }

        Write("SetVehicleSuspensionExpansionLimitEnabled", object, ctx, "value=%d", command.enabled);
    }

    static void GetExpansionLimit(void* object, const Context& ctx, VehicleData::GetVehicleSuspensionExpansionLimitEnabled& command)
    {
        std::lock_guard lock(s_Lock);
        TraceVehicle* vehicle = VehicleOf(object, ctx);
        command.outEnabled = vehicle != nullptr && vehicle->expansionLimit;
        Write("GetVehicleSuspensionExpansionLimitEnabled", object, ctx, "-> %d", command.outEnabled);
    }

    static void SetSuspension(void* object, const Context& ctx, VehicleData::SetVehicleWheelSuspensionData& command)
    {
        std::lock_guard lock(s_Lock);
        WheelOf(object, ctx, command.wheelId)->suspension = command.suspension;
        Write("SetVehicleWheelSuspensionData", object, ctx, "wheel=%u spring=%g damper=%g compression=%g droop=%g sprung=%g", command.wheelId,
            command.suspension.springStrength, command.suspension.springDamperRate, command.suspension.maxCompression,
            command.suspension.maxDroop, command.suspension.sprungMass);
    }

    static void GetSuspension(void* object, const Context& ctx, VehicleData::GetVehicleWheelSuspensionData& command)
    {
        std::lock_guard lock(s_Lock);
        command.outSuspension = WheelOf(object, ctx, command.wheelId)->suspension;
        Write("GetVehicleWheelSuspensionData", object, ctx, "wheel=%u", command.wheelId);
    }

    static void SetWheelData(void* object, const Context& ctx, VehicleData::SetVehicleWheelData& command)
    {
        std::lock_guard lock(s_Lock);
        WheelOf(object, ctx, command.wheelId)->wheel = command.wheelData;
        Write("SetVehicleWheelData", object, ctx, "wheel=%u radius=%g mass=%g moi=%g damping=%g", command.wheelId, command.wheelData.radius,
            command.wheelData.mass, command.wheelData.momentOfInertia, command.wheelData.dampingRate);
    }

    static void GetWheelData(void* object, const Context& ctx, VehicleData::GetVehicleWheelData& command)
    {
        std::lock_guard lock(s_Lock);
        command.outWheelData = WheelOf(object, ctx, command.wheelId)->wheel;
        Write("GetVehicleWheelData", object, ctx, "wheel=%u", command.wheelId);
    }

    static void GetLocalPose(void* object, const Context& ctx, VehicleData::GetVehicleWheelLocalPose& command)
    {
        std::lock_guard lock(s_Lock);
        Write("GetVehicleWheelLocalPose", object, ctx, "wheel=%u", command.wheelId);
    }

    static void SetRotationSpeed(void* object, const Context& ctx, VehicleData::SetVehicleWheelRotationSpeed& command)
    {
        std::lock_guard lock(s_Lock);
        WheelOf(object, ctx, command.wheelId)->speed = command.speed;
        Write("SetVehicleWheelRotationSpeed", object, ctx, "wheel=%u value=%g", command.wheelId, command.speed);
    }

    static void GetRotationSpeed(void* object, const Context& ctx, VehicleData::GetVehicleWheelRotationSpeed& command)
    {
        std::lock_guard lock(s_Lock);
        command.outSpeed = WheelOf(object, ctx, command.wheelId)->speed;
        Write("GetVehicleWheelRotationSpeed", object, ctx, "wheel=%u", command.wheelId);
    }

    static void SetMotorTorque(void* object, const Context& ctx, VehicleData::SetVehicleWheelMotorTorque& command)
    {
        std::lock_guard lock(s_Lock);
        WheelOf(object, ctx, command.wheelId)->motor = command.torque;
        Write("SetVehicleWheelMotorTorque", object, ctx, "wheel=%u value=%g", command.wheelId, command.torque);
    }

    static void GetMotorTorque(void* object, const Context& ctx, VehicleData::GetVehicleWheelMotorTorque& command)
    {
        std::lock_guard lock(s_Lock);
        command.outTorque = WheelOf(object, ctx, command.wheelId)->motor;
        Write("GetVehicleWheelMotorTorque", object, ctx, "wheel=%u", command.wheelId);
    }

    static void SetBrakeTorque(void* object, const Context& ctx, VehicleData::SetVehicleWheelBrakeTorque& command)
    {
        std::lock_guard lock(s_Lock);
        WheelOf(object, ctx, command.wheelId)->brake = command.brkTorque;
        Write("SetVehicleWheelBrakeTorque", object, ctx, "wheel=%u value=%g", command.wheelId, command.brkTorque);
    }

    static void GetBrakeTorque(void* object, const Context& ctx, VehicleData::GetVehicleWheelBrakeTorque& command)
    {
        std::lock_guard lock(s_Lock);
        command.outBrkTorque = WheelOf(object, ctx, command.wheelId)->brake;
        Write("GetVehicleWheelBrakeTorque", object, ctx, "wheel=%u", command.wheelId);
    }

    static void SetSteerAngle(void* object, const Context& ctx, VehicleData::SetVehicleWheelSteerAngle& command)
    {
        std::lock_guard lock(s_Lock);
        WheelOf(object, ctx, command.wheelId)->steer = command.angle;
        Write("SetVehicleWheelSteerAngle", object, ctx, "wheel=%u value=%g", command.wheelId, command.angle);
    }

    static void GetSteerAngle(void* object, const Context& ctx, VehicleData::GetVehicleWheelSteerAngle& command)
    {
        std::lock_guard lock(s_Lock);
        command.outAngle = WheelOf(object, ctx, command.wheelId)->steer;
        Write("GetVehicleWheelSteerAngle", object, ctx, "wheel=%u", command.wheelId);
    }

    static void CalculateSprungMasses(void* object, const Context& ctx, VehicleData::CalculateVehicleSprungMasses& command)
    {
        std::lock_guard lock(s_Lock);
        Write("CalculateVehicleSprungMasses", object, ctx, "wheelCount=%u firstWheel=%u", command.wheelCount,
            command.wheelCount > 0 ? command.wheelIds[0] : 0);
    }

    template <typename T, Handler<T> Function>
    static void Bind(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &Bound<T, Function>;
    }

    void BindVehicleTrace(FunctionBinding* table)
    {
        Bind<VehicleData::CreateVehicle, CreateVehicle>(table);
        Bind<VehicleData::DestroyVehicle, DestroyVehicle>(table);
        Bind<VehicleData::SetVehicleWheelEnabled, SetWheelEnabled>(table);
        Bind<VehicleData::GetFirstDisabledWheelFromVehicle, GetFirstDisabled>(table);
        Bind<VehicleData::SetVehicleHasImplicitSprungMasses, SetImplicitSprungMasses>(table);
        Bind<VehicleData::GetVehicleHasImplicitSprungMasses, GetImplicitSprungMasses>(table);
        Bind<VehicleData::GetVehicleWheelQueryResult, GetQueryResult>(table);
        Bind<VehicleData::SetVehicleWheelFilterData, SetFilter>(table);
        Bind<VehicleData::GetVehicleWheelFilterData, GetFilter>(table);
        Bind<VehicleData::GetVehicleWheelCount, GetWheelCount>(table);
        Bind<VehicleData::GetVehicleWheelsUserData, GetWheelsUserData>(table);
        Bind<VehicleData::ResetVehicleQueryResultsForShape, ResetQueryResults>(table);
        Bind<VehicleData::SetVehicleLongitudinalSpeedSubSteps, SetSubSteps>(table);
        Bind<VehicleData::SetVehicleWheelComOffset, SetComOffset>(table);
        Bind<VehicleData::GetVehicleWheelComOffset, GetComOffset>(table);
        Bind<VehicleData::SetVehicleSuspensionExpansionLimitEnabled, SetExpansionLimit>(table);
        Bind<VehicleData::GetVehicleSuspensionExpansionLimitEnabled, GetExpansionLimit>(table);
        Bind<VehicleData::SetVehicleWheelSuspensionData, SetSuspension>(table);
        Bind<VehicleData::GetVehicleWheelSuspensionData, GetSuspension>(table);
        Bind<VehicleData::SetVehicleWheelData, SetWheelData>(table);
        Bind<VehicleData::GetVehicleWheelData, GetWheelData>(table);
        Bind<VehicleData::GetVehicleWheelLocalPose, GetLocalPose>(table);
        Bind<VehicleData::SetVehicleWheelRotationSpeed, SetRotationSpeed>(table);
        Bind<VehicleData::GetVehicleWheelRotationSpeed, GetRotationSpeed>(table);
        Bind<VehicleData::SetVehicleWheelMotorTorque, SetMotorTorque>(table);
        Bind<VehicleData::GetVehicleWheelMotorTorque, GetMotorTorque>(table);
        Bind<VehicleData::SetVehicleWheelBrakeTorque, SetBrakeTorque>(table);
        Bind<VehicleData::GetVehicleWheelBrakeTorque, GetBrakeTorque>(table);
        Bind<VehicleData::SetVehicleWheelSteerAngle, SetSteerAngle>(table);
        Bind<VehicleData::GetVehicleWheelSteerAngle, GetSteerAngle>(table);
        Bind<VehicleData::CalculateVehicleSprungMasses, CalculateSprungMasses>(table);
    }
}

#endif
