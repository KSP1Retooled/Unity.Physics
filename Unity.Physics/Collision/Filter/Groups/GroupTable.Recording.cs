using System.IO;
using Unity.Physics.Stepping.Recording;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;

namespace Unity.Physics.Collision.Filter.Groups
{
    internal unsafe partial struct GroupTable
    {
        internal static void WriteRecording(RecordingWriter writer, GroupTable* table)
        {
            writer.Write(table != null);
            if (table == null)
            {
                return;
            }

            writer.Write(table->recordsUsed);
            writer.WriteBlock(table->records, checked((int)table->recordsUsed * sizeof(Record)));
            // An entry is written when it holds a group record or a crash speed. The crash speed limits the first impulse
            // of a fast contact, so a replay without it would solve crashes differently.
            int entryCount = 0;
            for (uint i = 0; i < table->entryCount; i++)
            {
                ref Entry entry = ref table->entries[i];
                if (entry.Key != EmptyKey && entry.Key != DeletedKey && entry.Kept)
                {
                    entryCount++;
                }
            }

            writer.Write(entryCount);
            for (uint i = 0; i < table->entryCount; i++)
            {
                ref Entry entry = ref table->entries[i];
                if (entry.Key != EmptyKey && entry.Key != DeletedKey && entry.Kept)
                {
                    writer.Write(entry.Key);
                    writer.Write(entry.Record);
                    writer.Write(entry.CrashSpeed);
                }
            }

            writer.Write(table->pairsLive);
            for (uint i = 0; i < table->pairCount; i++)
            {
                ulong key = table->pairs[i];
                if (key != EmptyKey && key != DeletedKey)
                {
                    writer.Write(key);
                }
            }
        }

        internal static void ReadRecording(BinaryReader reader, StoreGlobals* globals)
        {
            if (!reader.ReadBoolean())
            {
                return;
            }

            GroupTable* table = TableOf(globals);
            int records = reader.ReadCount(1 << 24);
            if (records != 0)
            {
                int bytes = checked(records * sizeof(Record));
                table->records = (Record*)StoreMemory.Reallocate(globals, null, (ulong)bytes);
                table->recordsUsed = table->recordCapacity = (uint)records;
                reader.ReadBlock(table->records, bytes);
            }

            int entries = reader.ReadCount(1 << 24);
            for (int i = 0; i < entries; i++)
            {
                ulong key = reader.ReadUInt64();
                uint record = reader.ReadUInt32();
                float crashSpeed = reader.ReadSingle();
                uint slot = table->TakeEntry(globals, key);
                table->entries[slot].Record = record;
                table->entries[slot].CrashSpeed = crashSpeed;
            }

            int pairs = reader.ReadCount(1 << 24);
            uint version = globals->ignoreVersion;
            for (int i = 0; i < pairs; i++)
            {
                ulong key = reader.ReadUInt64();
                SetPair(globals, (uint)(key >> 32), (uint)key, true);
            }

            globals->ignoreVersion = version;
        }

        internal static void FreeRecording(StoreGlobals* globals, GroupTable* table)
        {
            if (table == null)
            {
                return;
            }

            StoreMemory.Release(globals, table->records);
            StoreMemory.Release(globals, table->pairs);
            StoreMemory.Release(globals, table->entries);
            StoreMemory.Release(globals, table);
        }
    }
}
