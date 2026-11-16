using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Stores the collider pairs disabled by Physics.IgnoreCollision, using each collider's 16-bit ignore id. Unity
    /// stores the id on the Collider and initially sets it to 0, meaning no id. The backend assigns ids when a pair is
    /// first ignored, and Unity sends them to the shapes through their filter data. A collider keeps its id until it is
    /// destroyed, even when collisions are restored. During a step, contacts between shapes with ids require lookups. A
    /// large craft can ignore a few hundred thousand pairs, so lookups use a hash table. Each id also has a linked list
    /// of its edges, so destroying a collider only requires processing its own pairs. Everything is allocated on the
    /// store's heap so that ignored pairs survive a domain reload.
    /// </summary>
    internal unsafe struct IgnoreTable
    {
        private const uint IdCount = 1u << 16;
        private const uint InitialSlots = 1024;
        private const uint EmptySlot = 0;
        private const uint DeletedSlot = uint.MaxValue;

        // Chain links and occupied slots store an edge's index plus one, so zero can represent no edge.
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

        /// <summary>
        /// Marks a table that has the fields after <see cref="lastId"/>. The field sits in the four bytes of padding
        /// before <see cref="slots"/>, so it doesn't move any field, and every table that older code made has it, with a
        /// value of zero. See <see cref="Upgrade"/>.
        /// </summary>
        private uint layoutMark;

        private uint* slots;
        private uint slotCount;
        private uint slotsTaken;

        private uint* firstEdge;
        private ushort* freeIds;
        private uint freeIdCount;
        private uint lastId;

        /// <summary>
        /// The collision groups, or null until the first shape or group is entered. See <see cref="GroupTable"/>. Read
        /// it through <see cref="GroupsOf"/> unless a shape's group record shows that the groups exist.
        /// </summary>
        internal GroupTable* Groups;

        private const uint CurrentLayout = 0x33505247; // "GRP3"

        /// <summary>The size of a table made by code older than the groups.</summary>
        internal const int OldSize = 64;

        /// <summary>Returns the collision groups, or null if there are none yet.</summary>
        internal static GroupTable* GroupsOf(NativeStore.StoreGlobals* globals)
        {
            IgnoreTable* table = globals->ignoreTable;
            return table != null && table->layoutMark == CurrentLayout ? table->Groups : null;
        }

        /// <summary>
        /// Returns whether two shapes must not collide, either because Physics.IgnoreCollision ignored their pair or
        /// because the group rule stops them.
        /// </summary>
        internal static bool Ignores(NativeStore.StoreGlobals* globals, NativeStore.ShapeData* first, NativeStore.ShapeData* second)
        {
            return Contains(globals, first->ignorePairId, second->ignorePairId) || GroupTable.Ignores(globals, first, second);
        }

        internal static bool Contains(NativeStore.StoreGlobals* globals, uint first, uint second)
        {
            IgnoreTable* table = globals->ignoreTable;
            if (table == null || first == 0 || second == 0)
            {
                return false;
            }

            return table->FindSlot(KeyOf(first, second), out _);
        }

        /// <summary>
        /// Ignores a pair or restores its collisions. When ignoring a pair, assigns an id to either collider whose id
        /// is 0. The caller passes the ids back to Unity.
        /// </summary>
        internal static void Set(NativeStore.StoreGlobals* globals, ref uint first, ref uint second, bool ignored)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.IgnoreCollision.Auto();

            globals->ignoreVersion++;
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

        /// <summary>
        /// Returns the number of bytes that the ignore table holds on the store's heap, without the group table. Used
        /// by benchmarks.
        /// </summary>
        internal static ulong Bytes(NativeStore.StoreGlobals* globals)
        {
            IgnoreTable* table = globals->ignoreTable;
            if (table == null)
            {
                return 0;
            }

            return (ulong)sizeof(IgnoreTable) + IdCount * (sizeof(uint) + sizeof(ushort))
                + table->edgeCapacity * (ulong)sizeof(Edge) + table->slotCount * (ulong)sizeof(uint);
        }

        /// <summary>Returns the number of ignored pairs.</summary>
        internal static ulong PairCount(NativeStore.StoreGlobals* globals)
        {
            IgnoreTable* table = globals->ignoreTable;
            ulong count = 0;
            for (uint i = 0; table != null && i < table->slotCount; i++)
            {
                count += table->slots[i] != EmptySlot && table->slots[i] != DeletedSlot ? 1u : 0u;
            }

            return count;
        }

        /// <summary>
        /// Removes every ignored pair involving a destroyed collider and frees its id.
        /// </summary>
        internal static void Forget(NativeStore.StoreGlobals* globals, uint id)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ForgetIgnoredCollisions.Auto();

            globals->ignoreVersion++;
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

        /// <summary>
        /// Places every pair in the table again using the current domain's hash function. The table survives a domain
        /// reload. If the previous code used a different hash function, lookups would otherwise fail to find those
        /// pairs.
        /// </summary>
        internal static void Reslot(NativeStore.StoreGlobals* globals)
        {
            Upgrade(globals);
            IgnoreTable* table = globals->ignoreTable;
            if (table != null && table->slotCount != 0)
            {
                table->Rehash(globals);
            }

            GroupTable* groups = GroupsOf(globals);
            if (groups != null)
            {
                groups->Reslot(globals);
            }
        }

        internal static IgnoreTable* TableOf(NativeStore.StoreGlobals* globals)
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

                table->layoutMark = CurrentLayout;
                globals->ignoreTable = table;
            }

            return globals->ignoreTable;
        }

        /// <summary>
        /// Gives a table that older code made the fields that it lacks. <see cref="BackendHost.Attach"/> calls this
        /// through <see cref="Reslot"/>, after a domain reload and before anything else reads the table.
        ///
        /// A table made by code older than the groups is 64 bytes long. The code before this one put its layout mark
        /// after those 64 bytes, so to check the mark it read 8 bytes past an older table. Those bytes can hold anything
        /// that the heap left there, including the mark and a groups pointer of a table that was freed. This code keeps
        /// its mark inside the first 64 bytes, where every older table has zero. A table without the mark is moved to a
        /// block of the current size, which keeps its first 64 bytes, and it starts with no groups. A table that the
        /// code before this one made loses its groups once, when the editor first reloads into this code; the game sets
        /// them again when it next loads a vessel. Its old group table stays on the heap.
        /// </summary>
        internal static void Upgrade(NativeStore.StoreGlobals* globals)
        {
            IgnoreTable* table = globals->ignoreTable;
            if (table == null || table->layoutMark == CurrentLayout)
            {
                return;
            }

            table = (IgnoreTable*)StoreOps.Reallocate(globals, table, (ulong)sizeof(IgnoreTable));
            table->layoutMark = CurrentLayout;
            table->Groups = null;
            globals->ignoreTable = table;
        }

        /// <summary>Removes the layout mark, so that a test can make a table that looks like an older one.</summary>
        internal static void ClearLayoutMark(IgnoreTable* table)
        {
            table->layoutMark = 0;
        }

        private static uint KeyOf(uint first, uint second)
        {
            return (math.min(first, second) << 16) | math.max(first, second);
        }

        /// <summary>
        /// Mixes all bits of the key into the slot index. Ids are assigned in order, so keys for pairs in one craft
        /// differ in only a few bits of each half. Masking a simple product preserves only the low bits. Pairs with the
        /// same higher id would then start at just a few slots, requiring long sequences of probes.
        /// </summary>
        private static uint Mix(uint key)
        {
            key ^= key >> 16;
            key *= 0x7FEB352Du;
            key ^= key >> 15;
            key *= 0x846CA68Bu;
            key ^= key >> 16;
            return key;
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
            for (uint slot = Mix(key) & mask; ; slot = (slot + 1) & mask)
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

        // Move every live pair to a clean table. This also removes the deleted slots left when collisions were
        // restored.
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

                uint slot = Mix(edges[content - 1].Key) & mask;
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
