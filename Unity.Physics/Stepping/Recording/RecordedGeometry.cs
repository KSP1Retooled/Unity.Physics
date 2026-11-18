using System;
using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using StoreMesh = Unity.Physics.Store.Shapes.Mesh;
using Unity.Physics.Store.World;
using Unity.Physics.Store.Versions;

namespace Unity.Physics.Stepping.Recording
{
    /// <summary>Shares unchanged cooked geometry between frames. Entries for unseen handles expire after each step.</summary>
    internal sealed unsafe class RecordedGeometry
    {
        private readonly Dictionary<nuint, Entry> entries = new Dictionary<nuint, Entry>();
        private readonly List<nuint> expired = new List<nuint>();
        private int nextId;
        private byte[] jointBuffer = Array.Empty<byte>();
        private byte[] joints;
        private uint jointTopology;
        private uint jointSettings;
        private byte[] build;
        private byte[] vessels;
        private uint vesselVersion;
        private byte[] shapes;
        private byte[] filters;
        private uint filterVersion;

        internal int CaptureFilters(StoreGlobals* globals, RecordedStep step)
        {
            if (filters != null && filterVersion == globals->ignoreVersion)
            {
                step.Geometry.Add(filters);
                fixed (byte* payload = filters)
                {
                    return *(int*)payload;
                }
            }

            buildBuffer.Clear();
            IgnoreTable.WriteRecording(buildBuffer, globals);
            fixed (byte* source = buildBuffer.Buffer)
            {
                int id = Share(ref filters, source, buildBuffer.Length, step);
                filterVersion = globals->ignoreVersion;
                return id;
            }
        }

        internal int ShareShapes(RecordedStoreShape* rows, int count, RecordedStep step)
        {
            return count == 0 ? 0 : Share(ref shapes, rows, checked(count * sizeof(RecordedStoreShape)), step);
        }
        private readonly RecordingWriter buildBuffer = new RecordingWriter();

        internal int CaptureVessels(StoreStepper* stepper, RecordedStep step)
        {
            uint version = stepper->RecorderVesselCompilationVersion;
            if (vessels != null && vesselVersion == version)
            {
                step.Geometry.Add(vessels);
                fixed (byte* payload = vessels)
                {
                    return *(int*)payload;
                }
            }

            buildBuffer.Clear();
            stepper->WriteVesselCheckpoint(buildBuffer);
            fixed (byte* source = buildBuffer.Buffer)
            {
                int id = Share(ref vessels, source, buildBuffer.Length, step);
                vesselVersion = version;
                return id;
            }
        }

        internal int CaptureBuild(StoreStepper* stepper, RecordedStep step)
        {
            buildBuffer.Clear();
            stepper->WriteBuildCheckpoint(buildBuffer);
            fixed (byte* source = buildBuffer.Buffer)
            {
                return Share(ref build, source, buildBuffer.Length, step);
            }
        }

        internal int CaptureJoints(WorldData* world, StoreGlobals* globals, RecordedStep step)
        {
            int bytes = checked((int)world->jointCount * sizeof(RecordedJoint));
            if (bytes == 0)
            {
                return 0;
            }

            uint topology = JointTopology.Version;
            uint settings = JointSettings.Version;
            const int header = sizeof(int);
            if (joints != null && joints.Length == bytes + header && jointTopology == topology && jointSettings == settings)
            {
                fixed (byte* previous = joints)
                {
                    if (RecordedJoint.MatchesTargetsAndEnds(world, globals->engineObjectIdOffset, (RecordedJoint*)(previous + header)) != 0)
                    {
                        step.Geometry.Add(joints);
                        return *(int*)previous;
                    }
                }
            }

            jointTopology = topology;
            jointSettings = settings;
            if (jointBuffer.Length < bytes)
            {
                jointBuffer = new byte[bytes];
            }

            fixed (byte* source = jointBuffer)
            {
                RecordedJoint.Capture(world, globals->engineObjectIdOffset, (RecordedJoint*)source);
                return Share(ref joints, source, bytes, step);
            }
        }

        private int Share(ref byte[] previous, void* source, int bytes, RecordedStep step)
        {
            const int header = sizeof(int);
            fixed (byte* old = previous)
            {
                if (old != null && previous.Length == bytes + header
                    && UnsafeUtility.MemCmp(old + header, source, bytes) == 0)
                {
                    step.Geometry.Add(previous);
                    return *(int*)old;
                }
            }

            int id = checked(++nextId);
            previous = new byte[checked(bytes + header)];
            fixed (byte* destination = previous)
            {
                *(int*)destination = id;
                UnsafeUtility.MemCpy(destination + header, source, bytes);
            }

            step.Geometry.Add(previous);
            return id;
        }

        private struct Entry
        {
            internal int Id;
            internal long Step;
            internal byte[] Bytes;
            internal uint Version;
            internal uint Type;
            internal nuint Data;
        }

        internal int Capture(ShapeData* shape, RecordedStep step)
        {
            if (shape->mesh == null || shape->geometryType < StoreConstants.GeometryConvexMesh)
            {
                return 0;
            }

            nuint address = (nuint)shape->mesh;
            entries.TryGetValue(address, out Entry entry);
            if (entry.Step == step.Number && entry.Id != 0)
            {
                return entry.Id;
            }

            uint type = shape->geometryType;
            uint version = QuerySnapshotVersion.Version;
            bool sameVersion = entry.Id != 0 && entry.Version == version && entry.Type == type;
            if (type == StoreConstants.GeometryHeightField)
            {
                HeightField field = *(HeightField*)shape->mesh;
                short* heights = field.heights;
                byte* cells = field.cells;
                if (sameVersion && entry.Data == (nuint)heights)
                {
                    return UseEntry(address, ref entry, step);
                }

                entry.Data = (nuint)heights;
                field.heights = null;
                field.cells = null;
                field.firstShape = null;
                field.references = 0;
                int samples = checked((int)field.resolution * (int)field.resolution);
                int cellCount = checked((int)(field.resolution - 1) * (int)(field.resolution - 1));
                Update(ref entry, type, &field, sizeof(HeightField), heights, samples * sizeof(short), cells,
                    cells == null ? 0 : cellCount);
            }
            else
            {
                StoreMesh mesh = *(StoreMesh*)shape->mesh;
                MeshData* data = mesh.data;
                mesh.data = null;
                // A cook publishes an immutable stream. The data pointer can change when a queued cook finishes.
                if (sameVersion && entry.Data == (nuint)data)
                {
                    const int headerBytes = sizeof(int) * 5;
                    fixed (byte* previous = entry.Bytes)
                    {
                        if (UnsafeUtility.MemCmp(previous + headerBytes, &mesh, sizeof(StoreMesh)) == 0)
                        {
                            return UseEntry(address, ref entry, step);
                        }
                    }
                }

                entry.Data = (nuint)data;
                int length = data == null ? 0 : checked(sizeof(MeshData)
                    + (int)data->vertexCount * sizeof(float3) + (int)data->triangleCount * sizeof(int3));
                Update(ref entry, type, &mesh, sizeof(StoreMesh), data, length, null, 0);
            }

            entry.Version = version;
            entry.Type = type;
            return UseEntry(address, ref entry, step);
        }

        private int UseEntry(nuint address, ref Entry entry, RecordedStep step)
        {
            entry.Step = step.Number;
            entries[address] = entry;
            step.Geometry.Add(entry.Bytes);
            return entry.Id;
        }

        internal int CaptureMovingCompound(BodyData* body, RecordedStep step)
        {
            if ((body->flags & (BodyFlags.ColliderDirty | BodyFlags.CollidersMoved)) != BodyFlags.CollidersMoved
                || !body->collider->Value.IsCreated || body->collider->Value.Value.Type != ColliderType.Compound)
            {
                return 0;
            }

            var collider = (Collider*)body->collider->Value.GetUnsafePtr();
            // Collider blobs and cooked geometry have separate keys. Their addresses are aligned.
            nuint address = (nuint)collider | 1;
            entries.TryGetValue(address, out Entry entry);
            entry.Id = Share(ref entry.Bytes, collider, collider->MemorySize, step);
            entry.Step = step.Number;
            entries[address] = entry;
            return entry.Id;
        }

        private void Update(ref Entry entry, uint type, void* header, int headerSize, void* first, int firstSize,
            void* second, int secondSize)
        {
            int start = sizeof(int) * 5;
            int length = checked(start + headerSize + firstSize + secondSize);
            fixed (byte* old = entry.Bytes)
            {
                if (old != null && entry.Bytes.Length == length && *(uint*)(old + sizeof(int)) == type
                    && *(int*)(old + 2 * sizeof(int)) == headerSize
                    && *(int*)(old + 3 * sizeof(int)) == firstSize
                    && *(int*)(old + 4 * sizeof(int)) == secondSize
                    && UnsafeUtility.MemCmp(old + start, header, headerSize) == 0
                    && (firstSize == 0 || UnsafeUtility.MemCmp(old + start + headerSize, first, firstSize) == 0)
                    && (secondSize == 0 || UnsafeUtility.MemCmp(old + start + headerSize + firstSize, second, secondSize) == 0))
                {
                    return;
                }
            }

            entry.Id = checked(++nextId);
            entry.Bytes = new byte[length];
            fixed (byte* bytes = entry.Bytes)
            {
                ((int*)bytes)[0] = entry.Id;
                ((uint*)bytes)[1] = type;
                ((int*)bytes)[2] = headerSize;
                ((int*)bytes)[3] = firstSize;
                ((int*)bytes)[4] = secondSize;
                UnsafeUtility.MemCpy(bytes + start, header, headerSize);
                if (firstSize != 0)
                {
                    UnsafeUtility.MemCpy(bytes + start + headerSize, first, firstSize);
                }

                if (secondSize != 0)
                {
                    UnsafeUtility.MemCpy(bytes + start + headerSize + firstSize, second, secondSize);
                }
            }
        }

        internal void Finish(long step)
        {
            foreach (var pair in entries)
            {
                if (pair.Value.Step != step)
                {
                    expired.Add(pair.Key);
                }
            }

            foreach (nuint key in expired)
            {
                entries.Remove(key);
            }

            expired.Clear();
        }
    }
}
