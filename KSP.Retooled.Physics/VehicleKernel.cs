using KSP.Retooled.Physics.Interop;
using System;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Vehicles;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Vehicles;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Vehicle command IDs shared with Vehicle.cpp. Attach verifies the count.
    /// </summary>
    internal enum VehicleCommand : uint
    {
        CreateVehicle,
        DestroyVehicle,
        SetVehicleWheelEnabled,
        GetFirstDisabledWheelFromVehicle,
        SetVehicleHasImplicitSprungMasses,
        GetVehicleHasImplicitSprungMasses,
        GetVehicleWheelQueryResult,
        SetVehicleWheelFilterData,
        GetVehicleWheelFilterData,
        GetVehicleWheelCount,
        GetVehicleWheelsUserData,
        ResetVehicleQueryResultsForShape,
        SetVehicleLongitudinalSpeedSubSteps,
        SetVehicleWheelComOffset,
        GetVehicleWheelComOffset,
        SetVehicleSuspensionExpansionLimitEnabled,
        GetVehicleSuspensionExpansionLimitEnabled,
        SetVehicleWheelSuspensionData,
        GetVehicleWheelSuspensionData,
        SetVehicleWheelData,
        GetVehicleWheelData,
        GetVehicleWheelLocalPose,
        SetVehicleWheelRotationSpeed,
        GetVehicleWheelRotationSpeed,
        SetVehicleWheelMotorTorque,
        GetVehicleWheelMotorTorque,
        SetVehicleWheelBrakeTorque,
        GetVehicleWheelBrakeTorque,
        SetVehicleWheelSteerAngle,
        GetVehicleWheelSteerAngle,
        CalculateVehicleSprungMasses,
        Count,
    }

    /// <summary>
    /// Handle WheelCollider commands on the main thread outside simulation.
    /// Store vehicle settings, wheel slots and last-step hits; WheelSolver.WheelJob updates them during the step.
    /// </summary>
    [BurstCompile]
    internal static unsafe class VehicleKernel
    {
        // The value returned by GetFirstDisabledWheelFromVehicle when every slot is in use. Unity interprets it as -1.
        private const uint NoIndex = uint.MaxValue;

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(ManagedCallbacks.CommandFunction))]
        internal static void Run(uint command, void* handle, CommandArgs* args, StoreGlobals* globals)
        {
            // A WheelCollider's state is read every frame, and a span costs about as much as such a read, so reads run
            // without the category's span, as in BodyKernel.
            if (IsRead((VehicleCommand)command))
            {
                Dispatch(command, handle, args, globals);
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.VehicleCommand.Auto();
            Dispatch(command, handle, args, globals);
        }

        /// <summary>Whether the command only reads a vehicle. Reads run without the category's span.</summary>
        internal static bool IsRead(VehicleCommand command)
        {
            switch (command)
            {
                case VehicleCommand.GetFirstDisabledWheelFromVehicle:
                case VehicleCommand.GetVehicleHasImplicitSprungMasses:
                case VehicleCommand.GetVehicleWheelQueryResult:
                case VehicleCommand.GetVehicleWheelFilterData:
                case VehicleCommand.GetVehicleWheelCount:
                case VehicleCommand.GetVehicleWheelsUserData:
                case VehicleCommand.GetVehicleWheelComOffset:
                case VehicleCommand.GetVehicleSuspensionExpansionLimitEnabled:
                case VehicleCommand.GetVehicleWheelSuspensionData:
                case VehicleCommand.GetVehicleWheelData:
                case VehicleCommand.GetVehicleWheelLocalPose:
                case VehicleCommand.GetVehicleWheelRotationSpeed:
                case VehicleCommand.GetVehicleWheelMotorTorque:
                case VehicleCommand.GetVehicleWheelBrakeTorque:
                case VehicleCommand.GetVehicleWheelSteerAngle:
                    return true;
                default:
                    return false;
            }
        }

        private static void Dispatch(uint command, void* handle, CommandArgs* args, StoreGlobals* globals)
        {
            bool ranInMono = false;
            RunInMono(command, handle, args, globals, ref ranInMono);
            if (!ranInMono)
            {
                Execute((VehicleCommand)command, (WorldData*)handle, args, globals);
            }
        }

        /// <summary>
        /// Burst omits this method, so it runs only in Mono. The Mono version must catch the exception here.
        /// </summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, CommandArgs* args, StoreGlobals* globals,
            ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedCommand();
            try
            {
                Execute((VehicleCommand)command, (WorldData*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(VehicleCommand command, WorldData* world, CommandArgs* a, StoreGlobals* globals)
        {
            switch (command)
            {
                case VehicleCommand.CreateVehicle:
                    Create(world, a, globals);
                    return;
                case VehicleCommand.DestroyVehicle:
                    Destroy(world, a->value2, globals);
                    return;
                case VehicleCommand.ResetVehicleQueryResultsForShape:
                    ForgetShape(world, (Shape*)a->pointer);
                    return;
            }

            VehicleData* vehicle = Find(world, a->value2);
            if (vehicle == null)
            {
                return;
            }

            WheelSlot* wheel = a->value < vehicle->wheelCount ? VehicleData.WheelsOf(vehicle) + a->value : null;
            switch (command)
            {
                case VehicleCommand.SetVehicleWheelEnabled:
                    if (wheel != null)
                    {
                        Enable(vehicle, wheel, a);
                    }

                    a->flags = AllDisabled(vehicle) ? 1u : 0u;
                    return;
                case VehicleCommand.GetFirstDisabledWheelFromVehicle:
                    a->value = FirstDisabled(vehicle);
                    a->mode = vehicle->wheelCount;
                    return;
                case VehicleCommand.SetVehicleHasImplicitSprungMasses:
                    vehicle->implicitSprungMasses = a->enabled;
                    return;
                case VehicleCommand.GetVehicleHasImplicitSprungMasses:
                    a->enabled = vehicle->implicitSprungMasses;
                    return;
                case VehicleCommand.GetVehicleWheelCount:
                    a->value = vehicle->wheelCount;
                    return;
                case VehicleCommand.GetVehicleWheelsUserData:
                    a->value = WriteUserData(vehicle, (void**)a->pointer, a->mode);
                    return;
                case VehicleCommand.SetVehicleLongitudinalSpeedSubSteps:
                    vehicle->substepSpeed = a->scalar;
                    vehicle->substepsBelow = a->value;
                    vehicle->substepsAbove = a->mode;
                    return;
                case VehicleCommand.SetVehicleSuspensionExpansionLimitEnabled:
                    vehicle->expansionLimit = a->enabled;
                    return;
                case VehicleCommand.GetVehicleSuspensionExpansionLimitEnabled:
                    a->enabled = vehicle->expansionLimit;
                    return;
                case VehicleCommand.CalculateVehicleSprungMasses:
                    SprungMasses(vehicle, (uint*)a->pointer, (float3*)a->pointer2, (int)a->mode);
                    return;
            }

            if (wheel == null)
            {
                return;
            }

            switch (command)
            {
                case VehicleCommand.GetVehicleWheelQueryResult:
                    *(WheelQueryResult*)a->pointer = wheel->grounded != 0 ? wheel->hit : default;
                    break;
                case VehicleCommand.SetVehicleWheelFilterData:
                    SetFilter(wheel, (uint*)a->pointer);
                    break;
                case VehicleCommand.GetVehicleWheelFilterData:
                    GetFilter(wheel, (uint*)a->pointer);
                    break;
                case VehicleCommand.SetVehicleWheelComOffset:
                    wheel->centreOffset = a->vector;
                    wheel->forceOffset = a->vector2;
                    break;
                case VehicleCommand.GetVehicleWheelComOffset:
                    a->vector = wheel->centreOffset;
                    a->vector2 = wheel->forceOffset;
                    break;
                case VehicleCommand.SetVehicleWheelSuspensionData:
                    wheel->suspension = *(WheelSuspension*)a->pointer;
                    wheel->jounce = math.clamp(wheel->jounce, -wheel->suspension.maxDroop, wheel->suspension.maxCompression);
                    break;
                case VehicleCommand.GetVehicleWheelSuspensionData:
                    *(WheelSuspension*)a->pointer = wheel->suspension;
                    break;
                case VehicleCommand.SetVehicleWheelData:
                    wheel->wheel = *(WheelBody*)a->pointer;
                    break;
                case VehicleCommand.GetVehicleWheelData:
                    *(WheelBody*)a->pointer = wheel->wheel;
                    break;
                case VehicleCommand.GetVehicleWheelLocalPose:
                    LocalPose(vehicle, wheel, a);
                    break;
                case VehicleCommand.SetVehicleWheelRotationSpeed:
                    wheel->rotationSpeed = a->scalar;
                    break;
                case VehicleCommand.GetVehicleWheelRotationSpeed:
                    a->scalar = wheel->rotationSpeed;
                    break;
                case VehicleCommand.SetVehicleWheelMotorTorque:
                    wheel->motorTorque = a->scalar;
                    WakeFor(vehicle, a->scalar);
                    break;
                case VehicleCommand.GetVehicleWheelMotorTorque:
                    a->scalar = wheel->motorTorque;
                    break;
                case VehicleCommand.SetVehicleWheelBrakeTorque:
                    wheel->brakeTorque = a->scalar;
                    break;
                case VehicleCommand.GetVehicleWheelBrakeTorque:
                    a->scalar = wheel->brakeTorque;
                    break;
                case VehicleCommand.SetVehicleWheelSteerAngle:
                    wheel->steerAngle = a->scalar;
                    break;
                case VehicleCommand.GetVehicleWheelSteerAngle:
                    a->scalar = wheel->steerAngle;
                    break;
            }
        }

        private static VehicleData* Find(WorldData* world, uint id)
        {
            return id < world->vehicleCount ? world->vehicles[id] : null;
        }

        /// <summary>
        /// Create vehicle slots, preserving indices when copying from a smaller source vehicle.
        /// Unity rebinds wheels in slot order before destroying the source.
        /// </summary>
        private static void Create(WorldData* world, CommandArgs* a, StoreGlobals* globals)
        {
            uint id = 0;
            while (id < world->vehicleCount && world->vehicles[id] != null)
            {
                id++;
            }

            if (id == world->vehicleCount)
            {
                if (world->vehicleCount == world->vehicleCapacity)
                {
                    world->vehicleCapacity = StoreMemory.Grown(world->vehicleCapacity);
                    world->vehicles = (VehicleData**)StoreMemory.Reallocate(globals, world->vehicles, world->vehicleCapacity * (ulong)sizeof(void*));
                }

                world->vehicleCount++;
            }

            uint wheelCount = a->mode;
            ulong size = (ulong)sizeof(VehicleData) + wheelCount * (ulong)sizeof(WheelSlot);
            var vehicle = (VehicleData*)StoreMemory.Reallocate(globals, null, size);
            UnsafeUtility.MemClear(vehicle, (long)size);
            vehicle->body = (Body*)a->pointer3;
            vehicle->wheelCount = wheelCount;
            vehicle->implicitSprungMasses = 1;
            world->vehicles[id] = vehicle;

            uint written = 0;
            VehicleData* source = Find(world, a->value);
            if (source != null)
            {
                vehicle->implicitSprungMasses = source->implicitSprungMasses;
                vehicle->expansionLimit = source->expansionLimit;
                vehicle->substepSpeed = source->substepSpeed;
                vehicle->substepsBelow = source->substepsBelow;
                vehicle->substepsAbove = source->substepsAbove;

                uint kept = math.min(source->wheelCount, wheelCount);
                UnsafeUtility.MemCpy(VehicleData.WheelsOf(vehicle), VehicleData.WheelsOf(source), kept * (long)sizeof(WheelSlot));
                written = WriteUserData(vehicle, (void**)a->pointer, kept);
            }

            a->value = id;
            a->size = written;
        }

        private static void Destroy(WorldData* world, uint id, StoreGlobals* globals)
        {
            VehicleData* vehicle = Find(world, id);
            if (vehicle == null)
            {
                return;
            }

            StoreMemory.Release(globals, vehicle);
            world->vehicles[id] = null;
            while (world->vehicleCount > 0 && world->vehicles[world->vehicleCount - 1] == null)
            {
                world->vehicleCount--;
            }
        }

        /// <summary>
        /// Enables or disables a wheel and clears its previous hit. The wheel starts at full droop with no rotation
        /// speed.
        /// </summary>
        private static void Enable(VehicleData* vehicle, WheelSlot* wheel, CommandArgs* a)
        {
            wheel->enabled = a->enabled;
            wheel->userData = a->enabled != 0 ? a->pointer2 : null;
            wheel->grounded = 0;
            wheel->hit = default;
            wheel->hitShape = null;
            wheel->rotationSpeed = 0.0f;
            wheel->jounce = -wheel->suspension.maxDroop;
        }

        private static bool AllDisabled(VehicleData* vehicle)
        {
            return FirstEnabled(vehicle) == NoIndex;
        }

        private static uint FirstEnabled(VehicleData* vehicle)
        {
            WheelSlot* wheels = VehicleData.WheelsOf(vehicle);
            for (uint i = 0; i < vehicle->wheelCount; i++)
            {
                if (wheels[i].enabled != 0)
                {
                    return i;
                }
            }

            return NoIndex;
        }

        private static uint FirstDisabled(VehicleData* vehicle)
        {
            WheelSlot* wheels = VehicleData.WheelsOf(vehicle);
            for (uint i = 0; i < vehicle->wheelCount; i++)
            {
                if (wheels[i].enabled == 0)
                {
                    return i;
                }
            }

            return NoIndex;
        }

        /// <summary>
        /// Return enabled wheels' user data in slot order among the first limit slots. Omit disabled slots rather
        /// than returning null pointers.
        /// </summary>
        private static uint WriteUserData(VehicleData* vehicle, void** buffer, uint limit)
        {
            if (buffer == null)
            {
                return 0;
            }

            WheelSlot* wheels = VehicleData.WheelsOf(vehicle);
            uint written = 0;
            uint slots = math.min(limit, vehicle->wheelCount);
            for (uint i = 0; i < slots; i++)
            {
                if (wheels[i].enabled != 0 && wheels[i].userData != null)
                {
                    buffer[written++] = wheels[i].userData;
                }
            }

            return written;
        }

        private static void SetFilter(WheelSlot* wheel, uint* words)
        {
            wheel->collidesWith = words[0];
            wheel->belongsTo = words[1];
            wheel->filterFlags = words[2];
            wheel->filterIds = words[3];
        }

        private static void GetFilter(WheelSlot* wheel, uint* words)
        {
            words[0] = wheel->collidesWith;
            words[1] = wheel->belongsTo;
            words[2] = wheel->filterFlags;
            words[3] = wheel->filterIds;
        }

        /// <summary>
        /// Detach a destroyed body from its vehicle to prevent stale pointer access.
        /// Keep the vehicle allocation until Unity destroys it.
        /// </summary>
        internal static void ForgetBody(WorldData* world, Body* body)
        {
            for (uint v = 0; v < world->vehicleCount; v++)
            {
                VehicleData* vehicle = world->vehicles[v];
                if (vehicle != null && vehicle->body == body)
                {
                    vehicle->body = null;
                }
            }
        }

        /// <summary>
        /// Clears every hit on a shape that is changing or being destroyed, so that the shape is not read through an
        /// old hit afterward.
        /// </summary>
        private static void ForgetShape(WorldData* world, Shape* shape)
        {
            for (uint v = 0; v < world->vehicleCount; v++)
            {
                VehicleData* vehicle = world->vehicles[v];
                if (vehicle == null)
                {
                    continue;
                }

                WheelSlot* wheels = VehicleData.WheelsOf(vehicle);
                for (uint i = 0; i < vehicle->wheelCount; i++)
                {
                    if (wheels[i].hitShape == shape)
                    {
                        wheels[i].hitShape = null;
                        wheels[i].hit = default;
                        wheels[i].grounded = 0;
                    }
                }
            }
        }

        /// <summary>
        /// Returns the wheel's pose relative to its body. The centre is raised by the jounce. The rotation applies
        /// steering first, then spin about the axle.
        /// </summary>
        private static void LocalPose(VehicleData* vehicle, WheelSlot* wheel, CommandArgs* a)
        {
            float3 centreOfMass = float3.zero;
            if (vehicle->body != null)
            {
                centreOfMass = (*StoreOps.Hot(vehicle->body)).Mass()->Transform.pos;
            }

            a->position = centreOfMass + wheel->centreOffset + new float3(0.0f, wheel->jounce, 0.0f);
            a->rotation = math.mul(quaternion.RotateY(wheel->steerAngle), quaternion.RotateX(wheel->rotationAngle)).value;
        }

        /// <summary>
        /// Wakes the body when motor torque is non-zero. The game uses this to prevent a vehicle from falling asleep.
        /// </summary>
        private static void WakeFor(VehicleData* vehicle, float torque)
        {
            if (torque != 0.0f && vehicle->body != null)
            {
                BodyData* hot = StoreOps.Hot(vehicle->body);
                if ((hot->flags & BodyFlags.Sleeping) != 0)
                {
                    BodyKinematics.Wake(hot);
                }
            }
        }

        private static void SprungMasses(VehicleData* vehicle, uint* ids, float3* offsets, int count)
        {
            if (count <= 0 || vehicle->body == null)
            {
                return;
            }

            SprungMass.SprungMasses(vehicle, StoreOps.Hot(vehicle->body), ids, offsets, count);
        }
    }
}
