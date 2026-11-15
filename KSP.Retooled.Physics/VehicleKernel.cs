using System;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Defines the plugin API's vehicle commands, starting with CreateVehicle and using the ids passed by Vehicle.cpp.
    /// Attach checks that the number of commands matches the plugin's count.
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
    /// Implements WheelCollider commands and stores the data that Unity supplies for each vehicle, its wheel slots, and
    /// the hits from the last step. Unity creates one vehicle per body. To add slots, Unity creates a larger vehicle
    /// using the old one as the source. <see cref="WheelJob"/> updates the wheels during each step. Commands arrive on
    /// the main thread outside a step.
    /// </summary>
    [BurstCompile]
    internal static unsafe class VehicleKernel
    {
        // The value returned by GetFirstDisabledWheelFromVehicle when every slot is in use. Unity interprets it as -1.
        private const uint NoIndex = uint.MaxValue;

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(NativeStore.CommandFunction))]
        internal static void Run(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.VehicleCommand.Auto();

            bool ranInMono = false;
            RunInMono(command, handle, args, globals, ref ranInMono);
            if (!ranInMono)
            {
                Execute((VehicleCommand)command, (NativeStore.WorldData*)handle, args, globals);
            }
        }

        /// <summary>
        /// Burst omits this method, so it runs only in Mono. The Mono version must catch the exception here.
        /// </summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals,
            ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedCommand();
            try
            {
                Execute((VehicleCommand)command, (NativeStore.WorldData*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(VehicleCommand command, NativeStore.WorldData* world, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
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
                    ForgetShape(world, (NativeStore.Shape*)a->pointer);
                    return;
            }

            NativeStore.VehicleData* vehicle = Find(world, a->value2);
            if (vehicle == null)
            {
                return;
            }

            NativeStore.WheelSlot* wheel = a->value < vehicle->wheelCount ? NativeStore.WheelsOf(vehicle) + a->value : null;
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
                    *(NativeStore.WheelQueryResult*)a->pointer = wheel->grounded != 0 ? wheel->hit : default;
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
                    wheel->suspension = *(NativeStore.WheelSuspension*)a->pointer;
                    wheel->jounce = math.clamp(wheel->jounce, -wheel->suspension.maxDroop, wheel->suspension.maxCompression);
                    break;
                case VehicleCommand.GetVehicleWheelSuspensionData:
                    *(NativeStore.WheelSuspension*)a->pointer = wheel->suspension;
                    break;
                case VehicleCommand.SetVehicleWheelData:
                    wheel->wheel = *(NativeStore.WheelBody*)a->pointer;
                    break;
                case VehicleCommand.GetVehicleWheelData:
                    *(NativeStore.WheelBody*)a->pointer = wheel->wheel;
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

        private static NativeStore.VehicleData* Find(NativeStore.WorldData* world, uint id)
        {
            return id < world->vehicleCount ? world->vehicles[id] : null;
        }

        /// <summary>
        /// Creates a vehicle with the requested number of slots. When Unity grows a vehicle, the existing slots are
        /// copied from the source without changing their indices. Unity then rebinds the wheels using the returned user
        /// data in slot order and destroys the source vehicle.
        /// </summary>
        private static void Create(NativeStore.WorldData* world, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
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
                    world->vehicleCapacity = StoreOps.Grown(world->vehicleCapacity);
                    world->vehicles = (NativeStore.VehicleData**)StoreOps.Reallocate(globals, world->vehicles, world->vehicleCapacity * (ulong)sizeof(void*));
                }

                world->vehicleCount++;
            }

            uint wheelCount = a->mode;
            ulong size = (ulong)sizeof(NativeStore.VehicleData) + wheelCount * (ulong)sizeof(NativeStore.WheelSlot);
            var vehicle = (NativeStore.VehicleData*)StoreOps.Reallocate(globals, null, size);
            UnsafeUtility.MemClear(vehicle, (long)size);
            vehicle->body = (NativeStore.Body*)a->pointer3;
            vehicle->wheelCount = wheelCount;
            vehicle->implicitSprungMasses = 1;
            world->vehicles[id] = vehicle;

            uint written = 0;
            NativeStore.VehicleData* source = Find(world, a->value);
            if (source != null)
            {
                vehicle->implicitSprungMasses = source->implicitSprungMasses;
                vehicle->expansionLimit = source->expansionLimit;
                vehicle->substepSpeed = source->substepSpeed;
                vehicle->substepsBelow = source->substepsBelow;
                vehicle->substepsAbove = source->substepsAbove;

                uint kept = math.min(source->wheelCount, wheelCount);
                UnsafeUtility.MemCpy(NativeStore.WheelsOf(vehicle), NativeStore.WheelsOf(source), kept * (long)sizeof(NativeStore.WheelSlot));
                written = WriteUserData(vehicle, (void**)a->pointer, kept);
            }

            a->value = id;
            a->size = written;
        }

        private static void Destroy(NativeStore.WorldData* world, uint id, NativeStore.StoreGlobals* globals)
        {
            NativeStore.VehicleData* vehicle = Find(world, id);
            if (vehicle == null)
            {
                return;
            }

            StoreOps.Release(globals, vehicle);
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
        private static void Enable(NativeStore.VehicleData* vehicle, NativeStore.WheelSlot* wheel, NativeStore.CommandArgs* a)
        {
            wheel->enabled = a->enabled;
            wheel->userData = a->enabled != 0 ? a->pointer2 : null;
            wheel->grounded = 0;
            wheel->hit = default;
            wheel->hitShape = null;
            wheel->rotationSpeed = 0.0f;
            wheel->jounce = -wheel->suspension.maxDroop;
        }

        private static bool AllDisabled(NativeStore.VehicleData* vehicle)
        {
            return FirstEnabled(vehicle) == NoIndex;
        }

        private static uint FirstEnabled(NativeStore.VehicleData* vehicle)
        {
            NativeStore.WheelSlot* wheels = NativeStore.WheelsOf(vehicle);
            for (uint i = 0; i < vehicle->wheelCount; i++)
            {
                if (wheels[i].enabled != 0)
                {
                    return i;
                }
            }

            return NoIndex;
        }

        private static uint FirstDisabled(NativeStore.VehicleData* vehicle)
        {
            NativeStore.WheelSlot* wheels = NativeStore.WheelsOf(vehicle);
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
        /// Returns the user data for enabled wheels among the first <paramref name="limit"/> slots, in slot order.
        /// Disabled slots are omitted instead of returning null pointers, because Unity follows every returned pointer.
        /// </summary>
        private static uint WriteUserData(NativeStore.VehicleData* vehicle, void** buffer, uint limit)
        {
            if (buffer == null)
            {
                return 0;
            }

            NativeStore.WheelSlot* wheels = NativeStore.WheelsOf(vehicle);
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

        private static void SetFilter(NativeStore.WheelSlot* wheel, uint* words)
        {
            wheel->collidesWith = words[0];
            wheel->belongsTo = words[1];
            wheel->filterFlags = words[2];
            wheel->filterIds = words[3];
        }

        private static void GetFilter(NativeStore.WheelSlot* wheel, uint* words)
        {
            words[0] = wheel->collidesWith;
            words[1] = wheel->belongsTo;
            words[2] = wheel->filterFlags;
            words[3] = wheel->filterIds;
        }

        /// <summary>
        /// Saves each vehicle body's velocity before the forces for this step are applied. Suspension damping depends
        /// on the speed at which the body approaches the ground. If that calculation included gravity from the current
        /// step, it would indicate constant compression and hold the body above its rest position.
        /// </summary>
        internal static void RecordStartVelocities(NativeStore.WorldData* world)
        {
            for (uint v = 0; v < world->vehicleCount; v++)
            {
                NativeStore.VehicleData* vehicle = world->vehicles[v];
                if (vehicle != null && vehicle->body != null)
                {
                    NativeStore.BodyData* hot = world->bodies + vehicle->body->index;
                    vehicle->startLinear = hot->velocity->Linear;
                    vehicle->startAngular = StoreOps.WorldAngularVelocity(*hot);
                }
            }
        }

        /// <summary>
        /// Unity normally destroys a body's vehicle before destroying the body. This method prevents the step from
        /// following a vehicle's pointer to a body that has already been destroyed. The vehicle remains allocated until
        /// Unity destroys it, but no wheels are updated for it.
        /// </summary>
        internal static void ForgetBody(NativeStore.WorldData* world, NativeStore.Body* body)
        {
            for (uint v = 0; v < world->vehicleCount; v++)
            {
                NativeStore.VehicleData* vehicle = world->vehicles[v];
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
        private static void ForgetShape(NativeStore.WorldData* world, NativeStore.Shape* shape)
        {
            for (uint v = 0; v < world->vehicleCount; v++)
            {
                NativeStore.VehicleData* vehicle = world->vehicles[v];
                if (vehicle == null)
                {
                    continue;
                }

                NativeStore.WheelSlot* wheels = NativeStore.WheelsOf(vehicle);
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
        private static void LocalPose(NativeStore.VehicleData* vehicle, NativeStore.WheelSlot* wheel, NativeStore.CommandArgs* a)
        {
            float3 centreOfMass = float3.zero;
            if (vehicle->body != null)
            {
                centreOfMass = StoreOps.Hot(vehicle->body)->mass->Transform.pos;
            }

            a->position = centreOfMass + wheel->centreOffset + new float3(0.0f, wheel->jounce, 0.0f);
            a->rotation = math.mul(quaternion.RotateY(wheel->steerAngle), quaternion.RotateX(wheel->rotationAngle)).value;
        }

        /// <summary>
        /// Wakes the body when motor torque is non-zero. The game uses this to prevent a vehicle from falling asleep.
        /// </summary>
        private static void WakeFor(NativeStore.VehicleData* vehicle, float torque)
        {
            if (torque != 0.0f && vehicle->body != null)
            {
                NativeStore.BodyData* hot = StoreOps.Hot(vehicle->body);
                if ((hot->flags & NativeStore.BodyFlags.Sleeping) != 0)
                {
                    JointKernel.WakeRow(hot);
                }
            }
        }

        /// <summary>
        /// Distributes the body's mass among the wheels so that they support it level. The assigned masses sum to the
        /// body's mass and balance about its centre of mass in the ground plane. If these conditions allow multiple
        /// solutions, the solution has the smallest sum of squared masses, as in PhysX's PxVehicleComputeSprungMasses.
        /// The offsets are measured from the centre of mass along the body's axes.
        /// </summary>
        private static void SprungMasses(NativeStore.VehicleData* vehicle, uint* ids, float3* offsets, int count)
        {
            if (count <= 0 || vehicle->body == null)
            {
                return;
            }

            NativeStore.BodyData* hot = StoreOps.Hot(vehicle->body);
            float inverseMass = hot->mass->InverseMass;
            float mass = inverseMass > 0.0f ? 1.0f / inverseMass : 0.0f;
            NativeStore.WheelSlot* wheels = NativeStore.WheelsOf(vehicle);
            for (int i = 0; i < count; i++)
            {
                if (ids[i] < vehicle->wheelCount)
                {
                    wheels[ids[i]].suspension.sprungMass = SprungMass(offsets, count, i, mass);
                }
            }
        }

        internal static float SprungMass(float3* offsets, int count, int index, float mass)
        {
            if (count == 1)
            {
                return mass;
            }

            // The matrix for the balance equations has rows for 1, x and z, with one column per wheel. The solution of
            // rows * m = (mass, 0, 0) with the smallest sum of squared masses is rows^T * (rows * rows^T)^-1 * (mass,
            // 0, 0). Only the first column of that inverse is needed.
            double3x3 gram = default;
            for (int i = 0; i < count; i++)
            {
                double3 row = new double3(1.0, offsets[i].x, offsets[i].z);
                gram += new double3x3(row * row.x, row * row.y, row * row.z);
            }

            double determinant = math.determinant(gram);
            double3 own = new double3(1.0, offsets[index].x, offsets[index].z);
            if (math.abs(determinant) > 1e-9 * math.abs(gram.c0.x * gram.c1.y * gram.c2.z))
            {
                double3 column = math.inverse(gram).c0;
                return (float)(mass * math.dot(own, column));
            }

            // The wheels lie on a line, so only the moment along that line can be balanced.
            double2 axis = LineOf(offsets, count);
            if (math.lengthsq(axis) == 0.0)
            {
                return mass / count;
            }

            double2x2 line = default;
            for (int i = 0; i < count; i++)
            {
                double2 row = new double2(1.0, math.dot(axis, new double2(offsets[i].x, offsets[i].z)));
                line += new double2x2(row * row.x, row * row.y);
            }

            double2 ownOnLine = new double2(1.0, math.dot(axis, new double2(own.y, own.z)));
            return math.abs(math.determinant(line)) > 1e-12 ? (float)(mass * math.dot(ownOnLine, math.inverse(line).c0)) : mass / count;
        }

        private static double2 LineOf(float3* offsets, int count)
        {
            for (int i = 1; i < count; i++)
            {
                double2 along = new double2(offsets[i].x - offsets[0].x, offsets[i].z - offsets[0].z);
                if (math.lengthsq(along) > 1e-12)
                {
                    return math.normalize(along);
                }
            }

            return double2.zero;
        }
    }
}
