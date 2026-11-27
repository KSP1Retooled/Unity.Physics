using System.IO;
using Unity.Physics.Stepping.Recording;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;

namespace Unity.Physics.Collision.Filter.Groups
{
    internal unsafe partial struct IgnoreTable
    {
        internal static void WriteRecording(RecordingWriter writer, StoreGlobals* globals)
        {
            IgnoreTable* table = globals->ignoreTable;
            writer.Write(checked((int)PairCount(globals)));
            for (uint i = 0; table != null && i < table->slotCount; i++)
            {
                // A slot holds the index of its pair's edge plus one; the edge holds the pair's key.
                uint content = table->slots[i];
                if (content != EmptySlot && content != DeletedSlot)
                {
                    writer.Write(table->edges[content - 1].Key);
                }
            }

            GroupTable.WriteRecording(writer, GroupsOf(globals));
        }

        internal static void ReadRecording(BinaryReader reader, StoreGlobals* globals)
        {
            int count = reader.ReadCount(1 << 24);
            if (count > 0)
            {
                IgnoreTable* table = TableOf(globals);
                for (int i = 0; i < count; i++)
                {
                    uint key = reader.ReadUInt32();
                    table->Add(globals, key >> 16, key & 0xffff);
                }
            }

            GroupTable.ReadRecording(reader, globals);
        }

        internal static void FreeRecording(StoreGlobals* globals)
        {
            IgnoreTable* table = globals->ignoreTable;
            if (table == null)
            {
                return;
            }

            GroupTable.FreeRecording(globals, table->Groups);
            StoreMemory.Release(globals, table->edges);
            StoreMemory.Release(globals, table->slots);
            StoreMemory.Release(globals, table->firstEdge);
            StoreMemory.Release(globals, table->freeIds);
            StoreMemory.Release(globals, table);
            globals->ignoreTable = null;
        }
    }
}
