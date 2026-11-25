using System;
using System.Collections.Generic;
using System.IO;
using KSP.Retooled.Physics.Interop;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Physics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Stepping;
using Unity.Physics.Stepping.Recording;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Vehicles;
using Unity.Physics.Store.World;
using Mesh = Unity.Physics.Store.Shapes.Mesh;

namespace KSP.Retooled.Physics.Tests.Tools
{
    /// <summary>Owns an offline store for one captured step.</summary>
    internal sealed unsafe class ReplayStore : IDisposable
    {
        private const int MaxRecords = 1 << 24;
        private readonly List<nint> allocations = new List<nint>();
        private readonly Dictionary<ulong, nint> shapes = new Dictionary<ulong, nint>();
        private readonly Dictionary<ulong, nint> bodiesByAddress = new Dictionary<ulong, nint>();
        private readonly Dictionary<ulong, nint> jointsByAddress = new Dictionary<ulong, nint>();
        private readonly Dictionary<int, nint> geometry = new Dictionary<int, nint>();
        private readonly StoreGlobals* globals;
        private readonly WorldData* world;
        internal StoreStepper* Stepper { get; private set; }
        internal WorldData* World => world;

        internal ReplayStore()
        {
            globals = StoreMemory.Allocate<StoreGlobals>(NativeLibrary.RetooledPhysics_Globals());
            var native = NativeLibrary.RetooledPhysics_Globals();
            globals->reallocate = native->reallocate;
            globals->release = native->release;
            world = New<WorldData>();
        }

        internal void Read(BinaryReader reader, Dictionary<int, byte[]> payloads, int version)
        {
            var settings = reader.ReadValue<RecordedPhysicsSettings>();
            if (version == RecordingBinary.VersionBeforeServoEdges)
            {
                // The servo switch occupies a byte that version 11 left as padding.
                settings.ServoEdges = false;
            }
            settings.Apply();
            world->gravity = reader.ReadValue<Unity.Mathematics.float3>();
            world->bounceThreshold = reader.ReadSingle();
            world->staticsVersion = reader.ReadUInt32();
            globals->materialVersion = reader.ReadUInt32();
            uint builtMaterial = reader.ReadUInt32();
            globals->ignoreVersion = reader.ReadUInt32();
            globals->defaultMaterial = ReadMaterial(reader);
            byte[] filters = payloads[reader.ReadInt32()];
            using (var filterReader = new BinaryReader(new MemoryStream(filters, sizeof(int), filters.Length - sizeof(int), writable: false)))
            {
                IgnoreTable.ReadRecording(filterReader, globals);
            }

            ReadBodies(reader, payloads);

            int joints = reader.ReadCount(MaxRecords);
            world->joints = (JointData**)New<nint>(joints);
            world->jointCount = world->jointCapacity = (uint)joints;
            var jointRecords = New<JointData>(joints);
            var jointLoads = New<JointLoadState>(joints);
            int jointsId = reader.ReadInt32();
            if (joints != 0)
            {
                byte[] bytes = payloads[jointsId];
                if (bytes.Length != checked(sizeof(int) + joints * sizeof(RecordedJoint)))
                {
                    throw new InvalidDataException("Invalid joint records.");
                }

                fixed (byte* payload = bytes)
                {
                    var recordsOfJoints = (RecordedJoint*)(payload + sizeof(int));
                    for (int i = 0; i < joints; i++)
                    {
                        ref RecordedJoint source = ref recordsOfJoints[i];
                        JointData* joint = jointRecords + i;
                        *joint = source.Data;
                        joint->body0 = Owner(source.Body0);
                        joint->body1 = Owner(source.Body1);
                        joint->userData = UserData(source.EngineId);
                        world->joints[i] = joint;
                        jointsByAddress.Add(source.Address, (nint)joint);
                    }
                }
            }

            if (reader.ReadCount(MaxRecords) != joints)
            {
                throw new InvalidDataException("Invalid joint loads.");
            }

            reader.ReadBlock(jointLoads, checked(joints * sizeof(JointLoadState)));
            int vehicles = reader.ReadCount(MaxRecords);
            world->vehicles = (VehicleData**)New<nint>(vehicles);
            world->vehicleCount = world->vehicleCapacity = (uint)vehicles;
            for (int i = 0; i < vehicles; i++)
            {
                int row = reader.ReadInt32();
                if (row == RecordingBinary.EmptyVehicle)
                {
                    continue;
                }

                Body* owner = Owner(row);
                VehicleData value = reader.ReadValue<VehicleData>();
                if (value.wheelCount > MaxRecords)
                {
                    throw new InvalidDataException("Invalid wheel count.");
                }

                var vehicle = (VehicleData*)New<byte>(checked(sizeof(VehicleData) + (int)value.wheelCount * sizeof(WheelSlot)));
                world->vehicles[i] = vehicle;
                *vehicle = value;
                vehicle->body = owner;
                WheelSlot* wheels = VehicleData.WheelsOf(vehicle);
                for (uint w = 0; w < vehicle->wheelCount; w++)
                {
                    void* userData = ReadUserData(reader);
                    Shape* hit = (Shape*)ResolveShape(reader.ReadUInt64());
                    wheels[w] = reader.ReadValue<WheelSlot>();
                    wheels[w].userData = userData;
                    wheels[w].hitShape = hit;
                }
            }

            var state = New<StoreStepper>();
            *state = new StoreStepper(world, globals, New<QueryWorld>());
            Stepper = state;
            state->RecorderBuiltMaterialVersion = builtMaterial;
            UnsafeUtility.MemCpy(JointLoadStates.Of(world)->Ptr, jointLoads, (long)joints * sizeof(JointLoadState));
            state->ReadContactCheckpoint(reader, ResolveShape);
            byte[] build = payloads[reader.ReadInt32()];
            using var buildReader = new BinaryReader(new MemoryStream(build, sizeof(int), build.Length - sizeof(int), writable: false));
            state->ReadBuildCheckpoint(buildReader);
            byte[] vessels = payloads[reader.ReadInt32()];
            using var vesselReader = new BinaryReader(new MemoryStream(vessels, sizeof(int), vessels.Length - sizeof(int), writable: false));
            state->ReadVesselCheckpoint(vesselReader, ResolveBody, ResolveJoint);
        }

        private nint ResolveBody(ulong address)
        {
            if (!bodiesByAddress.TryGetValue(address, out nint result))
            {
                result = (nint)New<Body>();
                bodiesByAddress.Add(address, result);
            }

            return result;
        }

        private nint ResolveJoint(ulong address)
        {
            if (!jointsByAddress.TryGetValue(address, out nint result))
            {
                result = (nint)New<JointData>();
                jointsByAddress.Add(address, result);
            }

            return result;
        }

        private void ReadBodies(BinaryReader reader, Dictionary<int, byte[]> payloads)
        {
            int removed = reader.ReadCount(MaxRecords);
            var removedAddresses = new ulong[removed];
            for (int i = 0; i < removed; i++)
            {
                removedAddresses[i] = (ulong)reader.ReadValue<nuint>();
            }

            int count = reader.ReadCount(MaxRecords);
            var rows = new RecordedStoreRow[count];
            fixed (RecordedStoreRow* records = rows)
            {
                reader.ReadBlock(records, checked(count * sizeof(RecordedStoreRow)));
                int shapeId = reader.ReadInt32();
                byte[] chunk = shapeId == 0 ? Array.Empty<byte>() : payloads[shapeId];
                int shapeBytes = shapeId == 0 ? 0 : chunk.Length - sizeof(int);
                if (shapeBytes % sizeof(RecordedStoreShape) != 0)
                {
                    throw new InvalidDataException("Invalid shape records.");
                }

                int shapeCount = shapeBytes / sizeof(RecordedStoreShape);
                fixed (byte* bytes = chunk)
                {
                    var shapeRows = (RecordedStoreShape*)(bytes + sizeof(int));
                    DefineShapeAddresses(shapeRows, shapeCount, removedAddresses);

                    world->removedShapes = (Shape**)New<nint>(removed);
                    world->removedShapeCount = world->removedShapeCapacity = (uint)removed;
                    for (int i = 0; i < removed; i++)
                    {
                        world->removedShapes[i] = (Shape*)ResolveShape(removedAddresses[i]);
                    }

                    world->bodies = New<BodyData>(count);
                    world->bodyCount = world->bodyCapacity = (uint)count;
                    Body* owners = New<Body>(count);
                    int firstShape = 0;
                    for (int i = 0; i < count; i++)
                    {
                        ref RecordedStoreRow source = ref records[i];
                        if (source.ShapeCount > shapeCount - firstShape)
                        {
                            throw new InvalidDataException("Invalid body shape count.");
                        }

                        ReadBody(payloads, i, owners + i, source, shapeRows + firstShape);
                        firstShape += (int)source.ShapeCount;
                    }

                    if (firstShape != shapeCount)
                    {
                        throw new InvalidDataException("Extra shape records.");
                    }
                }
            }
        }

        // Contact keys order shape pointers. Allocate in the same relative order as the recorded addresses.
        private void DefineShapeAddresses(RecordedStoreShape* shapes, int count, ulong[] removed)
        {
            var addresses = new ulong[checked(count + removed.Length)];
            for (int i = 0; i < count; i++)
            {
                addresses[i] = shapes[i].Address;
            }

            Array.Copy(removed, 0, addresses, count, removed.Length);
            Array.Sort(addresses);
            Shape* records = New<Shape>(addresses.Length);
            for (int i = 0; i < addresses.Length; i++)
            {
                this.shapes.TryAdd(addresses[i], (nint)(records + i));
            }
        }

        private void ReadBody(Dictionary<int, byte[]> payloads, int index, Body* storage, in RecordedStoreRow source,
            RecordedStoreShape* shapes)
        {
            BodyData* row = world->bodies + index;
            row->flags = source.Flags;
            int oldCompound = source.MovingCompound;
            if (oldCompound == 0)
            {
                row->flags |= BodyFlags.ColliderDirty;
            }

            row->serial = source.Serial;
            row->jointEnds = source.JointEnds;
            if (source.Address != 0)
            {
                row->owner = storage;
                storage->world = world;
                storage->index = (uint)index;
                bodiesByAddress.Add(source.Address, (nint)storage);
                storage->userData = UserData(source.EngineId);
                storage->serial = source.OwnerSerial;
                storage->collisionDetectionMode = source.CollisionDetectionMode;
                storage->positionIterations = source.PositionIterations;
                storage->velocityIterations = source.VelocityIterations;
            }

            storage->home = source.Home;
            row->pose = &storage->home.pose;
            row->velocity = &storage->home.velocity;
            row->mass = &storage->home.mass;
            row->bodyMass = &storage->home.bodyMass;
            row->damping = &storage->home.damping;
            row->limits = &storage->home.limits;
            row->sleep = &storage->home.sleep;
            row->forces = &storage->home.forces;
            row->target = &storage->home.target;
            row->collider = &storage->home.collider;
            if (oldCompound != 0)
            {
                byte[] data = payloads[oldCompound];
                fixed (byte* blob = data)
                {
                    ColliderBuilder.SetCollider(row, BlobAssetReference<Collider>.Create(blob + sizeof(int), data.Length - sizeof(int)));
                }
            }

            int count = (int)source.ShapeCount;
            row->shapes = (ShapeData**)New<nint>(count);
            row->shapeCount = (uint)count;
            storage->shapes = row->shapes;
            storage->shapeCount = storage->shapeCapacity = (uint)count;
            for (int i = 0; i < count; i++)
            {
                ref RecordedStoreShape record = ref shapes[i];
                var shape = (Shape*)ResolveShape(record.Address);
                if (shape == null)
                {
                    throw new InvalidDataException("Missing shape.");
                }

                *shape = record.Shape;
                shape->body = row->owner;
                shape->data.owner = shape;
                shape->data.mesh = ReadGeometry(record.GeometryId, payloads);
                if (record.HasMaterial)
                {
                    shape->data.material = New<MaterialData>();
                    *shape->data.material = record.Material;
                }

                shape->data.userData = UserData(record.EngineId);
                row->shapes[i] = &shape->data;
            }
        }

        private nint ResolveShape(ulong address)
        {
            if (address == 0)
            {
                return 0;
            }

            if (!shapes.TryGetValue(address, out nint pointer))
            {
                pointer = (nint)New<Shape>();
                shapes.Add(address, pointer);
            }

            return pointer;
        }

        private void* ReadGeometry(int id, Dictionary<int, byte[]> payloads)
        {
            if (id == 0)
            {
                return null;
            }

            if (geometry.TryGetValue(id, out nint existing))
            {
                return (void*)existing;
            }

            using var reader = new BinaryReader(new MemoryStream(payloads[id], writable: false));
            if (reader.ReadInt32() != id)
            {
                throw new InvalidDataException("Invalid geometry ID.");
            }

            uint type = reader.ReadUInt32();
            bool heightField = type == StoreConstants.GeometryHeightField;
            int header = reader.ReadInt32();
            int first = reader.ReadCount(payloads[id].Length);
            int second = reader.ReadCount(payloads[id].Length);
            if (header != (heightField ? sizeof(HeightField) : sizeof(Mesh)))
            {
                throw new InvalidDataException("Geometry layout does not match this build.");
            }

            byte* handle = New<byte>(header);
            byte* data = New<byte>(first);
            byte* cells = New<byte>(second);
            reader.ReadBlock(handle, header);
            reader.ReadBlock(data, first);
            reader.ReadBlock(cells, second);
            if (heightField)
            {
                ((HeightField*)handle)->heights = (short*)data;
                ((HeightField*)handle)->cells = cells;
            }
            else
            {
                ((Mesh*)handle)->data = (MeshData*)data;
            }

            geometry.Add(id, (nint)handle);
            return handle;
        }

        private Body* Owner(int row)
        {
            if (row == -1)
            {
                return null;
            }

            if (row < 0 || row >= world->bodyCount)
            {
                throw new InvalidDataException($"Invalid body row {row}.");
            }

            return world->bodies[row].owner;
        }

        private void* ReadUserData(BinaryReader reader) => UserData(reader.ReadUInt64());

        private void* UserData(ulong id)
        {
            if (id == 0)
            {
                return null;
            }

            ulong* result = New<ulong>();
            *result = id;
            return result;
        }

        private MaterialData* ReadMaterial(BinaryReader reader)
        {
            if (!reader.ReadBoolean())
            {
                return null;
            }

            MaterialData* result = New<MaterialData>();
            *result = reader.ReadValue<MaterialData>();
            return result;
        }

        private T* New<T>(int count = 1) where T : unmanaged
        {
            if (count == 0)
            {
                return null;
            }

            long bytes = (long)count * sizeof(T);
            void* result = StoreMemory.Reallocate(globals, null, (ulong)bytes);
            if (result == null)
            {
                throw new OutOfMemoryException();
            }

            allocations.Add((nint)result);
            UnsafeUtility.MemClear(result, bytes);
            return (T*)result;
        }

        public void Dispose()
        {
            if (Stepper != null)
            {
                Stepper->Dispose();
            }

            IgnoreTable.FreeRecording(globals);
            StoreMemory.Release(globals, world->jointBreaks);
            StoreMemory.Release(globals, world->contactBodies);
            StoreMemory.Release(globals, world->contactShapes);
            StoreMemory.Release(globals, world->contactPoints);
            StoreMemory.Release(globals, world->triggers);
            for (int i = allocations.Count - 1; i >= 0; i--)
            {
                StoreMemory.Release(globals, (void*)allocations[i]);
            }

            StoreMemory.Release(globals, globals);
        }
    }
}
