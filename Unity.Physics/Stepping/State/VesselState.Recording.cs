using System;
using System.IO;
using Unity.Collections;
using Unity.Physics.Dynamics.Vessels.Detection;
using Unity.Physics.Stepping.Recording;
using Unity.Physics.Store.Versions;

namespace Unity.Physics.Stepping.State
{
    internal unsafe partial struct VesselState
    {
        internal readonly void WriteRecording(RecordingWriter writer)
        {
            writer.Write(VesselCompiler.RecordingId);
            writer.Write(compiled);
            writer.Write(cacheFilled);
            writer.Write(cachedSettings != JointSettings.Version);
            writer.Write(tombstoneExpiry);
            writer.WriteList(vesselBuild.Vessels);
            writer.WriteList(vesselBuild.PartStoreBodies);
            writer.WriteList(vesselBuild.PartUnits);
            writer.WriteList(vesselBuild.PartParents);
            writer.WriteList(vesselBuild.PartIsBody0);
            writer.WriteList(vesselBuild.PartVesselFirst);
            writer.WriteList(vesselBuild.ClosureUnits);
            writer.WriteList(vesselBuild.ClosurePartA);
            writer.WriteList(vesselBuild.ClosurePartB);
            writer.WriteList(vesselBuild.PartTombstones);
            writer.WriteList(vesselBuild.ClosureTombstones);
            writer.WriteList(vesselBuild.UnitVessels);
            writer.WriteList(vesselBuild.BodyVessels);
            writer.WriteList(vesselBuild.PartBodyRecords);
            writer.WriteList(vesselBuild.PartJointRecords);
            writer.WriteList(vesselBuild.ClosureJointRecords);
            writer.WriteList(vesselBuild.PartBodySerials);
            writer.WriteList(vesselBuild.PartJointSerials);
            writer.WriteList(vesselBuild.ClosureJointSerials);
            writer.WriteList(vesselBuild.EdgeCache);
            writer.WriteList(vesselBuild.ClosureCache);
            writer.WriteList(vesselBuild.EdgeCacheFrames);
            writer.WriteList(vesselBuild.ClosureCacheFrames);
            writer.WriteList(vesselBuild.EdgeCacheStamps);
            writer.WriteList(vesselBuild.ClosureCacheStamps);
            writer.Write(vesselBuild.PairUnits.Count());
            foreach (var pair in vesselBuild.PairUnits)
            {
                writer.Write(pair.Key);
                writer.Write(pair.Value);
            }
        }

        internal void ReadRecording(BinaryReader reader, Func<ulong, nint> resolveBody, Func<ulong, nint> resolveJoint)
        {
            VesselCompiler.RecordingId = reader.ReadInt64();
            compiled = reader.ReadBoolean();
            cacheFilled = reader.ReadBoolean();
            uint settings = JointSettings.Version;
            cachedSettings = reader.ReadBoolean() ? unchecked(settings - 1) : settings;
            tombstoneExpiry = reader.ReadInt32();
            reader.ReadList(vesselBuild.Vessels);
            reader.ReadList(vesselBuild.PartStoreBodies);
            reader.ReadList(vesselBuild.PartUnits);
            reader.ReadList(vesselBuild.PartParents);
            reader.ReadList(vesselBuild.PartIsBody0);
            reader.ReadList(vesselBuild.PartVesselFirst);
            reader.ReadList(vesselBuild.ClosureUnits);
            reader.ReadList(vesselBuild.ClosurePartA);
            reader.ReadList(vesselBuild.ClosurePartB);
            reader.ReadList(vesselBuild.PartTombstones);
            reader.ReadList(vesselBuild.ClosureTombstones);
            reader.ReadList(vesselBuild.UnitVessels);
            reader.ReadList(vesselBuild.BodyVessels);
            reader.ReadList(vesselBuild.PartBodyRecords);
            reader.ReadList(vesselBuild.PartJointRecords);
            reader.ReadList(vesselBuild.ClosureJointRecords);
            reader.ReadList(vesselBuild.PartBodySerials);
            reader.ReadList(vesselBuild.PartJointSerials);
            reader.ReadList(vesselBuild.ClosureJointSerials);
            reader.ReadList(vesselBuild.EdgeCache);
            reader.ReadList(vesselBuild.ClosureCache);
            reader.ReadList(vesselBuild.EdgeCacheFrames);
            reader.ReadList(vesselBuild.ClosureCacheFrames);
            reader.ReadList(vesselBuild.EdgeCacheStamps);
            reader.ReadList(vesselBuild.ClosureCacheStamps);
            int pairs = reader.ReadCount(1 << 24);
            for (int i = 0; i < pairs; i++)
            {
                vesselBuild.PairUnits.Add(reader.ReadUInt64(), reader.ReadInt32());
            }

            Resolve(vesselBuild.PartBodyRecords, resolveBody);
            Resolve(vesselBuild.PartJointRecords, resolveJoint);
            Resolve(vesselBuild.ClosureJointRecords, resolveJoint);
        }

        private static void Resolve(NativeList<IntPtr> list, Func<ulong, nint> resolve)
        {
            for (int i = 0; i < list.Length; i++)
            {
                ulong address = (ulong)list[i];
                list[i] = address == 0 ? IntPtr.Zero : (IntPtr)resolve(address);
            }
        }
    }
}
