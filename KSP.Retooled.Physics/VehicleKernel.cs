using System;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The plugin API's vehicle commands, from CreateVehicle, numbered as Vehicle.cpp passes them. Attach checks the count
    /// against the plugin's.
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
    /// What a WheelCollider's commands do: keeping what Unity sends about each body's vehicle, its wheel slots and the last
    /// step's hits. Unity makes one vehicle per body and grows it by making a bigger one from the old. The step drives the
    /// wheels in <see cref="WheelJob"/>. Commands arrive on the main thread outside a step.
    /// </summary>
    [BurstCompile]
    internal static unsafe class VehicleKernel
    {
        // What GetFirstDisabledWheelFromVehicle answers when every slot is taken, which Unity reads as -1.
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

        /// <summary>Burst drops this, so it only runs in the Mono form, which needs the exception ended here.</summary>
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
        /// A vehicle with the slots asked for, which takes over the source's slots in the same places when Unity grows one.
        /// Unity then rebinds the wheels whose user data comes back, in slot order, and destroys the source itself.
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

        /// <summary>A wheel switched on starts out hanging, with no hit and no jounce from its last life.</summary>
        private static void Enable(NativeStore.VehicleData* vehicle, NativeStore.WheelSlot* wheel, NativeStore.CommandArgs* a)
        {
            wheel->enabled = a->enabled;
            wheel->userData = a->enabled != 0 ? a->pointer2 : null;
            wheel->grounded = 0;
            wheel->hasJounce = 0;
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
        /// The user data of the enabled wheels among the first <paramref name="limit"/> slots, in slot order. A disabled slot
        /// is left out rather than written as null, because Unity follows every pointer it gets back.
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
        /// Unity destroys a body's vehicle before the body, so this only guards the step from following a vehicle to a body
        /// that is gone. The vehicle stays until Unity destroys it, and drives nothing.
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

        /// <summary>Drops every hit on a shape that is changing or going away, so nothing reads it after.</summary>
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
        /// The wheel's pose in the body's space: its centre risen by the jounce, turned by the steer and then by its spin about
        /// its axle.
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

        /// <summary>A motor torque keeps its body awake, which is how the game keeps a vehicle from going to sleep.</summary>
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
        /// Shares the body's mass among the wheels so that they hold it level: the masses sum to the body's, and balance about
        /// its centre of mass across the ground plane. With more wheels than that fixes, it is the smallest such set, which is
        /// what PhysX's PxVehicleComputeSprungMasses gives. The offsets are from the centre of mass, along the body's axes.
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

            // The rows of the balance are 1, x and z per wheel. The smallest solution of rows * m = (mass, 0, 0) is
            // rows^T * (rows * rows^T)^-1 * (mass, 0, 0), which needs only the first column of that inverse.
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

            // The wheels stand in a line, so only the moment along it can balance.
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
