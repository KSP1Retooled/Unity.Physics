using System;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Vehicles;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.Recording
{
    internal sealed unsafe class RecordStore
    {
        private RecordedStoreRow[] rows = Array.Empty<RecordedStoreRow>();
        private RecordedStoreShape[] shapes = Array.Empty<RecordedStoreShape>();

        internal void Write(StoreStepper* stepper, RecordedStep step, RecordedGeometry geometry, RecordingWriter writer)
        {
            WorldData* world = stepper->Store;
            StoreGlobals* globals = stepper->Globals;
            writer.WriteValue(RecordedPhysicsSettings.Read());
            writer.WriteValue(world->gravity);
            writer.Write(world->bounceThreshold);
            writer.Write(world->staticsVersion);
            writer.Write(globals->materialVersion);
            writer.Write(stepper->RecorderBuiltMaterialVersion);
            writer.Write(globals->ignoreVersion);
            WriteMaterial(writer, globals->defaultMaterial);
            IgnoreTable.WriteRecording(writer, globals);

            writer.Write(world->removedShapeCount);
            writer.WriteBlock(world->removedShapes, checked((int)world->removedShapeCount * sizeof(ulong)));
            WriteBodies(world, globals, writer, step, geometry);

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
            step.Input = writer.Buffer;
            step.InputLength = writer.Length;
        }

        private void WriteBodies(WorldData* world, StoreGlobals* globals, RecordingWriter writer, RecordedStep step,
            RecordedGeometry geometry)
        {
            int bodyCount = checked((int)world->bodyCount);
            int shapeCount = 0;
            for (int i = 0; i < bodyCount; i++)
            {
                shapeCount = checked(shapeCount + (int)world->bodies[i].shapeCount);
            }

            if (rows.Length < bodyCount)
            {
                rows = new RecordedStoreRow[bodyCount];
            }
            if (shapes.Length < shapeCount)
            {
                shapes = new RecordedStoreShape[shapeCount];
            }

            fixed (RecordedStoreRow* bodyRows = rows)
            fixed (RecordedStoreShape* shapeRows = shapes)
            {
                RecordedStoreRow.Capture(world, globals->engineObjectIdOffset, bodyRows);
                RecordedStoreShape.Capture(world, globals->engineObjectIdOffset, shapeRows);
                for (int i = 0; i < bodyCount; i++)
                {
                    if ((bodyRows[i].Flags & BodyFlags.CollidersMoved) != 0)
                    {
                        bodyRows[i].MovingCompound = geometry.CaptureMovingCompound(world->bodies + i, step);
                    }
                }
                for (int i = 0; i < shapeCount; i++)
                {
                    if (shapeRows[i].Shape.data.geometryType >= StoreConstants.GeometryConvexMesh)
                    {
                        shapeRows[i].GeometryId = geometry.Capture((ShapeData*)shapeRows[i].Address, step);
                    }
                }

                writer.Write(bodyCount);
                writer.WriteBlock(bodyRows, checked(bodyCount * sizeof(RecordedStoreRow)));
                writer.Write(geometry.ShareShapes(shapeRows, shapeCount, step));
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
