using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Which collider pairs Physics.IgnoreCollision switched off, by the 16-bit ignore id each collider holds. Unity keeps the
    /// id on the Collider and starts it at 0, meaning none; the backend hands out real ids when a pair is first ignored and
    /// Unity sends them to the shapes in their filter data. Ids stay with a collider until it is destroyed, across restores.
    /// A step looks up every contact between shapes with ids, and a large craft can ignore a few hundred thousand pairs, so a
    /// lookup is a hash probe and each id chains its edges, which keeps destroying a collider to the cost of its own pairs.
    /// All of it lives on the store's heap, because ignored pairs have to survive a domain reload.
    /// </summary>
    internal unsafe struct IgnoreTable
    {
        private const uint IdCount = 1u << 16;
        private const uint InitialSlots = 1024;
        private const uint EmptySlot = 0;
        private const uint DeletedSlot = uint.MaxValue;

        // Chain links and slot contents are an edge's index plus one, so that zero can mean none.
        private struct Edge
        {
            public uint Key;
            public uint PreviousOfLow;
            public uint NextOfLow;
            public uint PreviousOfHigh;
            public uint NextOfHigh;
        }

        private Edge* edges;
        private uint edgeCapacity;
        private uint edgesUsed;
        private uint freeEdge;

        private uint* slots;
        private uint slotCount;
        private uint slotsTaken;

        private uint* firstEdge;
        private ushort* freeIds;
        private uint freeIdCount;
        private uint lastId;

        internal static bool Contains(NativeStore.StoreGlobals* globals, uint first, uint second)
        {
            IgnoreTable* table = globals->ignoreTable;
            if (table == null || first == 0 || second == 0)
            {
                return false;
            }

            return table->FindSlot(KeyOf(first, second), out _);
        }

        /// <summary>Ignores or restores a pair. An id of 0 gets a real one when ignoring, and the caller passes it back to Unity.</summary>
        internal static void Set(NativeStore.StoreGlobals* globals, ref uint first, ref uint second, bool ignored)
        {
            if (!ignored)
            {
                if (globals->ignoreTable != null)
                {
                    globals->ignoreTable->Remove(first, second);
                }

                return;
            }

            IgnoreTable* table = TableOf(globals);
            first = first != 0 ? first : table->AllocateId();
            second = second != 0 ? second : table->AllocateId();
            if (first != 0 && second != 0)
            {
                table->Add(globals, first, second);
            }
        }

        /// <summary>Drops every pair of a destroyed collider and frees its id.</summary>
        internal static void Forget(NativeStore.StoreGlobals* globals, uint id)
        {
            IgnoreTable* table = globals->ignoreTable;
            if (table == null || id == 0)
            {
                return;
            }

            while (table->firstEdge[id] != 0)
            {
                uint key = table->edges[table->firstEdge[id] - 1].Key;
                table->Remove(key >> 16, key & 0xffff);
            }

            table->freeIds[table->freeIdCount++] = (ushort)id;
        }

        private static IgnoreTable* TableOf(NativeStore.StoreGlobals* globals)
        {
            if (globals->ignoreTable == null)
            {
                IgnoreTable* table = (IgnoreTable*)StoreOps.Reallocate(globals, null, (ulong)sizeof(IgnoreTable));
                *table = default;
                table->firstEdge = (uint*)StoreOps.Reallocate(globals, null, IdCount * sizeof(uint));
                table->freeIds = (ushort*)StoreOps.Reallocate(globals, null, IdCount * sizeof(ushort));
                for (uint i = 0; i < IdCount; i++)
                {
                    table->firstEdge[i] = 0;
                }

                globals->ignoreTable = table;
            }

            return globals->ignoreTable;
        }

        private static uint KeyOf(uint first, uint second)
        {
            return (math.min(first, second) << 16) | math.max(first, second);
        }

        private uint AllocateId()
        {
            if (freeIdCount != 0)
            {
                return freeIds[--freeIdCount];
            }

            if (lastId == IdCount - 1)
            {
                Debug.LogError("[RetooledPhysics] out of ignore ids, so a collider pair stays colliding");
                return 0;
            }

            return ++lastId;
        }

        private readonly bool FindSlot(uint key, out uint index)
        {
            index = 0;
            if (slotCount == 0)
            {
                return false;
            }

            uint mask = slotCount - 1;
            uint firstDeleted = uint.MaxValue;
            for (uint slot = (key * 0x9E3779B1u) & mask; ; slot = (slot + 1) & mask)
            {
                uint content = slots[slot];
                if (content == EmptySlot)
                {
                    index = firstDeleted != uint.MaxValue ? firstDeleted : slot;
                    return false;
                }

                if (content == DeletedSlot)
                {
                    firstDeleted = math.min(firstDeleted, slot);
                }
                else if (edges[content - 1].Key == key)
                {
                    index = slot;
                    return true;
                }
            }
        }

        private void Add(NativeStore.StoreGlobals* globals, uint first, uint second)
        {
            uint key = KeyOf(first, second);
            if (FindSlot(key, out _))
            {
                return;
            }

            if ((slotsTaken + 1) * 4 >= slotCount * 3)
            {
                Rehash(globals);
            }

            FindSlot(key, out uint slot);
            slotsTaken += slots[slot] == EmptySlot ? 1u : 0u;

            uint edge = TakeEdge(globals);
            slots[slot] = edge;

            uint low = key >> 16;
            uint high = key & 0xffff;
            ref Edge added = ref edges[edge - 1];
            added = new Edge { Key = key, NextOfLow = firstEdge[low], NextOfHigh = firstEdge[high] };
            SetPrevious(added.NextOfLow, low, edge);
            SetPrevious(added.NextOfHigh, high, edge);
            firstEdge[low] = edge;
            firstEdge[high] = edge;
        }

        private void Remove(uint first, uint second)
        {
            uint key = KeyOf(first, second);
            if (!FindSlot(key, out uint slot))
            {
                return;
            }

            uint edge = slots[slot];
            slots[slot] = DeletedSlot;

            uint low = key >> 16;
            uint high = key & 0xffff;
            Edge removed = edges[edge - 1];
            Unlink(removed.PreviousOfLow, removed.NextOfLow, low);
            Unlink(removed.PreviousOfHigh, removed.NextOfHigh, high);

            edges[edge - 1].NextOfLow = freeEdge;
            freeEdge = edge;
        }

        private uint TakeEdge(NativeStore.StoreGlobals* globals)
        {
            if (freeEdge != 0)
            {
                uint reused = freeEdge;
                freeEdge = edges[reused - 1].NextOfLow;
                return reused;
            }

            if (edgesUsed == edgeCapacity)
            {
                edgeCapacity = StoreOps.Grown(edgeCapacity);
                edges = (Edge*)StoreOps.Reallocate(globals, edges, edgeCapacity * (ulong)sizeof(Edge));
            }

            return ++edgesUsed;
        }

        private void SetPrevious(uint edge, uint id, uint previous)
        {
            if (edge != 0)
            {
                ref Edge next = ref edges[edge - 1];
                if (next.Key >> 16 == id)
                {
                    next.PreviousOfLow = previous;
                }
                else
                {
                    next.PreviousOfHigh = previous;
                }
            }
        }

        private void Unlink(uint previous, uint next, uint id)
        {
            if (previous == 0)
            {
                firstEdge[id] = next;
            }
            else if (edges[previous - 1].Key >> 16 == id)
            {
                edges[previous - 1].NextOfLow = next;
            }
            else
            {
                edges[previous - 1].NextOfHigh = next;
            }

            SetPrevious(next, id, previous);
        }

        // Every live pair moves to a clean table, which also clears the deleted slots a restore leaves.
        private void Rehash(NativeStore.StoreGlobals* globals)
        {
            uint live = 0;
            for (uint i = 0; i < slotCount; i++)
            {
                live += slots[i] != EmptySlot && slots[i] != DeletedSlot ? 1u : 0u;
            }

            uint newCount = math.max(InitialSlots, slotCount);
            newCount = (live + 1) * 2 >= newCount ? newCount * 2 : newCount;

            uint* oldSlots = slots;
            uint oldCount = slotCount;
            slots = (uint*)StoreOps.Reallocate(globals, null, newCount * sizeof(uint));
            slotCount = newCount;
            slotsTaken = live;
            uint mask = newCount - 1;
            for (uint i = 0; i < newCount; i++)
            {
                slots[i] = EmptySlot;
            }

            for (uint i = 0; i < oldCount; i++)
            {
                uint content = oldSlots[i];
                if (content == EmptySlot || content == DeletedSlot)
                {
                    continue;
                }

                uint slot = (edges[content - 1].Key * 0x9E3779B1u) & mask;
                while (slots[slot] != EmptySlot)
                {
                    slot = (slot + 1) & mask;
                }

                slots[slot] = content;
            }

            StoreOps.Release(globals, oldSlots);
        }
    }
}
