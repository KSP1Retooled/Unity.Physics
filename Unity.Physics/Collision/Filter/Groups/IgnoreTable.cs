using Unity.Mathematics;
using UnityEngine;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Profiling;

namespace Unity.Physics.Collision.Filter.Groups
{
    /// <summary>
    /// Store collider pairs disabled by Physics.IgnoreCollision, keyed by persistent 16-bit ignore IDs.
    /// Use a hash table for lookups and per-ID edge lists for removal. Native allocations survive domain reloads.
    /// </summary>
    internal unsafe partial struct IgnoreTable
    {
        private static readonly ProfilerMarker ForgetIgnoredCollisions = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Filter.ForgetIgnoredPairs");
        private static readonly ProfilerMarker IgnoreCollision = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Filter.IgnoreCollision");

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
        /// Layout marker in the original table's padding. A zero value identifies an older layout; see Upgrade.
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
        /// Collision groups, or null before the first entry. Access through GroupsOf unless a shape record proves
        /// they exist.
        /// </summary>
        internal GroupTable* Groups;

        private const uint CurrentLayout = 0x33505247; // "GRP3"

        /// <summary>The size of a table made by code older than the groups.</summary>
        internal const int OldSize = 64;

        /// <summary>Returns the collision groups, or null if there are none yet.</summary>
        internal static GroupTable* GroupsOf(StoreGlobals* globals)
        {
            IgnoreTable* table = globals->ignoreTable;
            return table != null && table->layoutMark == CurrentLayout ? table->Groups : null;
        }

        /// <summary>
        /// Returns whether two shapes must not collide, either because Physics.IgnoreCollision ignored their pair or
        /// because the group rule stops them.
        /// </summary>
        internal static bool Ignores(StoreGlobals* globals, ShapeData* first, ShapeData* second)
        {
            return Contains(globals, first->ignorePairId, second->ignorePairId) || GroupTable.Ignores(globals, first, second);
        }

        internal static bool Contains(StoreGlobals* globals, uint first, uint second)
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
        internal static void Set(StoreGlobals* globals, ref uint first, ref uint second, bool ignored)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = IgnoreCollision.Auto();

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
        internal static ulong Bytes(StoreGlobals* globals)
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
        internal static ulong PairCount(StoreGlobals* globals)
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
        internal static void Forget(StoreGlobals* globals, uint id)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = ForgetIgnoredCollisions.Auto();

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
        /// Rehash stored pairs after domain reload so that lookups use the current hash function.
        /// </summary>
        internal static void Reslot(StoreGlobals* globals)
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

        internal static IgnoreTable* TableOf(StoreGlobals* globals)
        {
            if (globals->ignoreTable == null)
            {
                IgnoreTable* table = (IgnoreTable*)StoreMemory.Reallocate(globals, null, (ulong)sizeof(IgnoreTable));
                *table = default;
                table->firstEdge = (uint*)StoreMemory.Reallocate(globals, null, IdCount * sizeof(uint));
                table->freeIds = (ushort*)StoreMemory.Reallocate(globals, null, IdCount * sizeof(ushort));
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
        /// Upgrade tables created before collision groups. Check the marker within the original 64 bytes before
        /// reading new fields.
        /// Copy those bytes into a table of the current size and initialize it without groups when the marker is
        /// absent.
        /// </summary>
        internal static void Upgrade(StoreGlobals* globals)
        {
            IgnoreTable* table = globals->ignoreTable;
            if (table == null || table->layoutMark == CurrentLayout)
            {
                return;
            }

            table = (IgnoreTable*)StoreMemory.Reallocate(globals, table, (ulong)sizeof(IgnoreTable));
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
        /// Mix both halves of pair keys to avoid clustering sequential collider IDs in the same probe chains.
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

        private void Add(StoreGlobals* globals, uint first, uint second)
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

        private uint TakeEdge(StoreGlobals* globals)
        {
            if (freeEdge != 0)
            {
                uint reused = freeEdge;
                freeEdge = edges[reused - 1].NextOfLow;
                return reused;
            }

            if (edgesUsed == edgeCapacity)
            {
                edgeCapacity = StoreMemory.Grown(edgeCapacity);
                edges = (Edge*)StoreMemory.Reallocate(globals, edges, edgeCapacity * (ulong)sizeof(Edge));
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
        private void Rehash(StoreGlobals* globals)
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
            slots = (uint*)StoreMemory.Reallocate(globals, null, newCount * sizeof(uint));
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

            StoreMemory.Release(globals, oldSlots);
        }
    }
}
