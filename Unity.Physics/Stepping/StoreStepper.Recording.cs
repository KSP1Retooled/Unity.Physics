using System;
using System.IO;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Dynamics.Vessels.Detection;
using Unity.Physics.Stepping.Recording;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Versions;

namespace Unity.Physics.Stepping
{
    internal unsafe partial struct StoreStepper
    {
        private NativeArray<RecordedIsland> recordedIslands;
        private NativeArray<RecordedSolverPath> recordedPaths;
        private NativeArray<byte> restoredMotion;
        internal byte* RecorderRestoredMotion { get; private set; }

        internal void BeginRecorderStep()
        {
            forcesApplied.Complete();
            VesselData.vesselsFilled.Complete();
            if (!recordedIslands.IsCreated || recordedIslands.Length < store->bodyCount)
            {
                FreeRecorderIslands();
                recordedIslands = new NativeArray<RecordedIsland>((int)store->bodyCount, Allocator.Persistent);
                recordedPaths = new NativeArray<RecordedSolverPath>((int)store->bodyCount, Allocator.Persistent);
                restoredMotion = new NativeArray<byte>((int)store->bodyCount, Allocator.Persistent);
            }

            var output = (RecordedIsland*)recordedIslands.GetUnsafePtr();
            UnsafeUtility.MemClear(output, (long)store->bodyCount * sizeof(RecordedIsland));
            SolverCaches.directSolverCache.RecorderIslands = output;
            var paths = (RecordedSolverPath*)recordedPaths.GetUnsafePtr();
            UnsafeUtility.MemClear(paths, (long)store->bodyCount * sizeof(RecordedSolverPath));
            SolverCaches.directSolverCache.RecorderPaths = paths;
            RecorderRestoredMotion = (byte*)restoredMotion.GetUnsafePtr();
            UnsafeUtility.MemClear(RecorderRestoredMotion, store->bodyCount);
        }

        internal void StopRecorderStep()
        {
            SolverCaches.directSolverCache.RecorderIslands = null;
            SolverCaches.directSolverCache.RecorderPaths = null;
            RecorderRestoredMotion = null;
        }

        internal RecordedIsland* RecorderIslands => SolverCaches.directSolverCache.RecorderIslands;

        internal readonly uint RecorderContactStep => Contacts.stepCount;

        internal uint RecorderBuiltMaterialVersion
        {
            readonly get => Colliders.builtMaterialVersion;
            set => Colliders.builtMaterialVersion = value;
        }

        private void FreeRecorderIslands()
        {
            if (recordedIslands.IsCreated)
            {
                recordedIslands.Dispose();
                recordedIslands = default;
                recordedPaths.Dispose();
                recordedPaths = default;
                restoredMotion.Dispose();
                restoredMotion = default;
            }
        }

        internal readonly void WriteContactCheckpoint(RecordingWriter writer)
        {
            writer.Write(Contacts.stepCount);
            writer.Write(Contacts.HasPendingRefilter);
            writer.Write(Contacts.lastImpulses.Count);
            foreach (var pair in Contacts.lastImpulses)
            {
                writer.Write((ulong)pair.Key.ShapeA);
                writer.Write((ulong)pair.Key.ShapeB);
                writer.Write(pair.Key.KeyA);
                writer.Write(pair.Key.KeyB);
                writer.Write(pair.Value);
            }

            writer.Write(Contacts.touching.Count);
            foreach (var pair in Contacts.touching)
            {
                writer.Write((ulong)pair.Key.First);
                writer.Write((ulong)pair.Key.Second);
                writer.Write(pair.Value.LastSeen);
                writer.Write(pair.Value.Trigger);
                writer.Write(pair.Value.FirstIsTrigger);
                writer.Write(pair.Value.Kinematic);
            }

            SolverCaches.directSolverCache.WriteRecording(writer);
        }

        internal readonly void WriteBuildCheckpoint(RecordingWriter writer)
        {
            bool canReuse = Joints.grouped && Joints.groupedTopology == JointTopology.Version;
            writer.Write(canReuse);
            if (canReuse)
            {
                writer.WriteList(Joints.jointGroups);
                writer.WriteList(Joints.jointGroupMembers);
                writer.WriteList(Joints.jointUnits);
                writer.WriteList(Joints.jointRoles);
                writer.WriteList(Joints.jointsWritten);
            }

            writer.Write(VesselCompiler.RecordingId);
        }

        internal void ReadBuildCheckpoint(BinaryReader reader)
        {
            Joints.grouped = reader.ReadBoolean();
            if (Joints.grouped)
            {
                Joints.groupedTopology = JointTopology.Version;
                reader.ReadList(Joints.jointGroups);
                reader.ReadList(Joints.jointGroupMembers);
                reader.ReadList(Joints.jointUnits);
                reader.ReadList(Joints.jointRoles);
                reader.ReadList(Joints.jointsWritten);
            }

            VesselCompiler.RecordingId = reader.ReadInt64();
        }

        internal readonly uint RecorderVesselCompilationVersion => VesselData.CompilationVersion;

        internal readonly void WriteVesselCheckpoint(RecordingWriter writer) => VesselData.WriteRecording(writer);

        internal void ReadVesselCheckpoint(BinaryReader reader, Func<ulong, nint> resolveBody, Func<ulong, nint> resolveJoint)
            => VesselData.ReadRecording(reader, resolveBody, resolveJoint);

        internal void ReadContactCheckpoint(BinaryReader reader, Func<ulong, nint> resolveShape)
        {
            Contacts.stepCount = reader.ReadUInt32();
            Contacts.RestorePendingRefilter(reader.ReadBoolean());
            int impulses = reader.ReadCount(1 << 24);
            for (int i = 0; i < impulses; i++)
            {
                var key = new PersistentManifoldKey
                {
                    ShapeA = (ShapeData*)resolveShape(reader.ReadUInt64()),
                    ShapeB = (ShapeData*)resolveShape(reader.ReadUInt64()),
                    KeyA = reader.ReadUInt32(),
                    KeyB = reader.ReadUInt32(),
                };
                Contacts.lastImpulses.Add(key, reader.ReadSingle());
            }

            int touching = reader.ReadCount(1 << 24);
            for (int i = 0; i < touching; i++)
            {
                var key = new ShapePairKey
                {
                    First = (IntPtr)resolveShape(reader.ReadUInt64()),
                    Second = (IntPtr)resolveShape(reader.ReadUInt64()),
                };
                Contacts.touching.Add(key, new ShapePairState
                {
                    LastSeen = reader.ReadUInt32(),
                    Trigger = reader.ReadBoolean(),
                    FirstIsTrigger = reader.ReadBoolean(),
                    Kinematic = reader.ReadByte(),
                });
            }

            SolverCaches.directSolverCache.ReadRecording(reader);
        }
    }
}
