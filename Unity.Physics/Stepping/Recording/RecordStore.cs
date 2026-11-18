using System;
using System.Collections.Generic;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Vehicles;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.Recording
{
    internal static unsafe class RecordStore
    {
        internal static void Write(StoreStepper* stepper, RecordedStep step, RecordedGeometry geometry, RecordingWriter writer)
        {
            WorldData* world = stepper->Store;
            StoreGlobals* globals = stepper->Globals;
            writer.Clear();
            writer.WriteValue(RecordedPhysicsSettings.Read());
            writer.WriteValue(world->gravity);
            writer.Write(world->bounceThreshold);
            writer.Write(world->staticsVersion);
            writer.Write(globals->materialVersion);
            writer.Write(stepper->RecorderBuiltMaterialVersion);
            writer.Write(globals->ignoreVersion);
            WriteMaterial(writer, globals->defaultMaterial);
            IgnoreTable.WriteRecording(writer, globals);

            var addresses = new List<ulong>(checked((int)world->bodyCount));
            for (uint i = 0; i < world->bodyCount; i++)
            {
                BodyData* body = world->bodies + i;
                for (uint s = 0; s < body->shapeCount; s++)
                {
                    addresses.Add((ulong)body->shapes[s]);
                }
            }

            for (uint i = 0; i < world->removedShapeCount; i++)
            {
                addresses.Add((ulong)world->removedShapes[i]);
            }

            addresses.Sort();
            writer.Write(addresses.Count);
            foreach (ulong address in addresses)
            {
                writer.Write(address);
            }

            writer.Write(world->removedShapeCount);
            for (uint i = 0; i < world->removedShapeCount; i++)
            {
                writer.Write((ulong)world->removedShapes[i]);
            }

            writer.Write(world->bodyCount);
            for (uint i = 0; i < world->bodyCount; i++)
            {
                WriteBody(writer, world->bodies + i, globals, step, geometry);
            }

            writer.Write(world->jointCount);
            writer.Write(geometry.CaptureJoints(world, globals, step));
            writer.WriteList(*JointLoadStates.Of(world));

            writer.Write(world->vehicleCount);
            for (uint i = 0; i < world->vehicleCount; i++)
            {
                VehicleData* source = world->vehicles[i];
                if (source == null)
                {
                    writer.Write(RecordingBinary.EmptyVehicle);
                    continue;
                }

                VehicleData vehicle = *source;
                writer.Write(vehicle.body == null ? -1 : checked((int)vehicle.body->index));
                vehicle.body = null;
                writer.WriteValue(vehicle);
                WheelSlot* wheels = VehicleData.WheelsOf(source);
                for (uint w = 0; w < vehicle.wheelCount; w++)
                {
                    WheelSlot wheel = wheels[w];
                    writer.Write(FlightRecorder.EngineId(wheel.userData, globals->engineObjectIdOffset));
                    writer.Write((ulong)wheel.hitShape);
                    wheel.userData = null;
                    wheel.hitShape = null;
                    writer.WriteValue(wheel);
                }
            }

            stepper->WriteContactCheckpoint(writer);
            writer.Write(geometry.CaptureBuild(stepper, step));
            step.Input = writer.ToArray();
        }

        private static void WriteBody(RecordingWriter writer, BodyData* row, StoreGlobals* globals, RecordedStep step,
            RecordedGeometry geometry)
        {
            writer.WriteValue(row->flags);
            writer.Write(geometry.CaptureMovingCompound(row, step));
            writer.Write(row->serial);
            writer.Write(row->jointEnds);
            Body* body = row->owner;
            writer.Write(body != null);
            if (body != null)
            {
                writer.Write((ulong)body);
                writer.Write(FlightRecorder.EngineId(body->userData, globals->engineObjectIdOffset));
                writer.Write(body->serial);
                writer.Write(body->collisionDetectionMode);
                writer.Write(body->positionIterations);
                writer.Write(body->velocityIterations);
            }

            writer.WriteValue(new BodyHome
            {
                pose = *row->pose,
                velocity = *row->velocity,
                mass = *row->mass,
                bodyMass = *row->bodyMass,
                damping = *row->damping,
                limits = *row->limits,
                sleep = *row->sleep,
                forces = *row->forces,
                target = *row->target,
            });

            writer.Write(row->shapeCount);
            for (uint i = 0; i < row->shapeCount; i++)
            {
                Shape* source = (Shape*)row->shapes[i];
                Shape shape = *source;
                writer.Write((ulong)source);
                writer.Write(FlightRecorder.EngineId(shape.data.userData, globals->engineObjectIdOffset));
                writer.Write(geometry.Capture(&shape.data, step));
                WriteMaterial(writer, shape.data.material);
                shape.body = null;
                shape.data.owner = null;
                shape.data.mesh = null;
                shape.data.material = null;
                shape.data.userData = null;
                shape.nextOnField = null;
                writer.WriteValue(shape);
            }
        }

        private static void WriteMaterial(RecordingWriter writer, MaterialData* material)
        {
            writer.Write(material != null);
            if (material != null)
            {
                MaterialData value = *material;
                value.userData = null;
                writer.WriteValue(value);
            }
        }
    }
}
