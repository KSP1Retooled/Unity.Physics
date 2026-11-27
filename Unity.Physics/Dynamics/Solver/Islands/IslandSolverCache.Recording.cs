using System.Collections.Generic;
using System.IO;
using Unity.Collections;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Stepping.Recording;

namespace Unity.Physics.Dynamics.Solver.Islands
{
    internal unsafe partial struct IslandSolverCache
    {
        internal readonly void WriteRecording(RecordingWriter writer)
        {
            writer.Write(*step);
            writer.WriteValue(*unfusedWork);
            writer.Write(slots->Length);
            for (int i = 0; i < slots->Length; i++)
            {
                ref Slot slot = ref slots->Ptr[i];
                writer.Write(slot.Identity != NoSlot || slot.Created || slot.Forest != null);
                if (slot.Identity == NoSlot && !slot.Created && slot.Forest == null)
                {
                    continue;
                }

                writer.Write(slot.Key);
                writer.Write(slot.Identity);
                writer.Write(slot.LastUsed);
                writer.Write(slot.Created);
                if (slot.Created)
                {
                    slot.Solver.WriteRecording(writer);
                }

                WriteForest(writer, slot.Forest);
            }

            writer.Write(slotOfIdentity->Count);
            foreach (var pair in *slotOfIdentity)
            {
                writer.Write(pair.Key);
                writer.Write(pair.Value);
            }

            writer.Write(fusedSlots->Length);
            for (int i = 0; i < fusedSlots->Length; i++)
            {
                ref FusedSlot slot = ref fusedSlots->Ptr[i];
                writer.Write(slot.VesselId);
                writer.Write(slot.LastUsed);
                WriteForest(writer, slot.Forest);
            }

            writer.Write(vesselOwners->Count);
            foreach (var pair in *vesselOwners)
            {
                writer.Write(pair.Key);
                writer.Write(pair.Value.Step);
                writer.Write((ulong)pair.Value.Forest);
            }
        }

        internal void ReadRecording(BinaryReader reader)
        {
            *step = reader.ReadInt64();
            *unfusedWork = reader.ReadValue<IslandWorkCounters>();
            var forests = new Dictionary<ulong, nint>();
            slots->Resize(reader.ReadCount(MaxSlots));
            for (int i = 0; i < slots->Length; i++)
            {
                ref Slot slot = ref slots->Ptr[i];
                slot = EmptySlot;
                if (!reader.ReadBoolean())
                {
                    continue;
                }

                slot.Key = reader.ReadInt32();
                slot.Identity = reader.ReadUInt64();
                slot.LastUsed = reader.ReadInt64();
                slot.Created = reader.ReadBoolean();
                if (slot.Created)
                {
                    slot.Solver = new SparseConstraintSolver(Allocator.Persistent);
                    slot.Solver.ReadRecording(reader);
                }

                ReadForest(reader, forests, ref slot.Forest);
            }

            int identities = reader.ReadCount(MaxSlots);
            for (int i = 0; i < identities; i++)
            {
                slotOfIdentity->Add(reader.ReadUInt64(), reader.ReadInt32());
            }

            fusedSlots->Resize(reader.ReadCount(1 << 20));
            for (int i = 0; i < fusedSlots->Length; i++)
            {
                ref FusedSlot slot = ref fusedSlots->Ptr[i];
                slot.VesselId = reader.ReadUInt64();
                slot.LastUsed = reader.ReadInt64();
                ReadForest(reader, forests, ref slot.Forest);
                fusedSlotOfVessel->Add(slot.VesselId, i);
            }

            int owners = reader.ReadCount(1 << 20);
            for (int i = 0; i < owners; i++)
            {
                ulong id = reader.ReadUInt64();
                long used = reader.ReadInt64();
                var forest = (ForestIsland*)forests[reader.ReadUInt64()];
                vesselOwners->Add(id, new VesselOwner { Forest = forest, Step = used });
            }
        }

        private static void WriteForest(RecordingWriter writer, ForestIsland* forest)
        {
            writer.Write((ulong)forest);
            if (forest != null)
            {
                forest->WriteRecording(writer);
            }
        }

        private static void ReadForest(BinaryReader reader, Dictionary<ulong, nint> forests, ref ForestIsland* forest)
        {
            ulong saved = reader.ReadUInt64();
            if (saved != 0)
            {
                forest = ForestIsland.Create();
                forests.Add(saved, (nint)forest);
                forest->ReadRecording(reader);
            }
        }
    }
}
