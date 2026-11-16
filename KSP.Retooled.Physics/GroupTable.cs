using Unity.Mathematics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Stores the collision groups that the game assigns through <see cref="CollisionGroups"/>. Each collider can have
    /// a group record with three values: the group, the member and a flag that lets the member collide with other
    /// members of its group. The game would use the vessel as the group and the part as the member. Two colliders in
    /// the same group don't collide, with two exceptions. They collide if they belong to different members and both
    /// have the flag set. They also collide if the game listed their two members as a pair that collides.
    ///
    /// The game names a collider by its engine object id, and the table finds the shape through a map from that id.
    /// Every shape with a collider is entered in the map when the shape is created, so that the game can set a group
    /// before or after the shape exists. A shape stores the index of its group record in <see
    /// cref="NativeStore.Shape.groupRecord"/>, so that the step reads the group without a lookup.
    ///
    /// A group record stays with the collider's engine id after its shape is destroyed, because Unity may destroy a
    /// shape and create a new one for the same collider. <see cref="CollisionGroups"/> removes records of colliders
    /// that no longer exist. Like <see cref="IgnoreTable"/>, everything is allocated on the store's heap so that the
    /// groups survive a domain reload. The ignore table holds the pointer to this table.
    /// </summary>
    internal unsafe struct GroupTable
    {
        /// <summary>The flag that lets a member collide with other members of its group that have the flag.</summary>
        internal const uint SelfCollides = 1;

        private const uint InitialSlots = 1024;
        private const ulong EmptyKey = 0;
        private const ulong DeletedKey = ulong.MaxValue;

        internal struct Record
        {
            public uint Group;
            public uint Member;
            public uint Flags;

            /// <summary>The next free record plus one, while this record is free.</summary>
            public uint NextFree;
        }

        private struct Entry
        {
            public ulong Key;
            public NativeStore.Shape* Shape;

            /// <summary>The index of the collider's group record plus one, or zero if it has no group.</summary>
            public uint Record;
            private readonly uint reserved0;
        }

        private Entry* entries;
        private uint entryCount;
        private uint entriesTaken;

        private Record* records;
        private uint recordCapacity;
        private uint recordsUsed;
        private uint freeRecord;

        // The member pairs that collide although they share a group. Each slot holds a key made of the two members.
        private ulong* pairs;
        private uint pairCount;
        private uint pairsTaken;
        private uint pairsLive;

        /// <summary>
        /// The number of entries that have a group record but no shape. <see cref="CollisionGroups"/> checks those
        /// colliders when this count grows.
        /// </summary>
        internal uint Orphans;

        /// <summary>
        /// Returns whether the group rule stops two shapes from colliding. The ignore table is checked separately.
        /// </summary>
        internal static bool Ignores(NativeStore.StoreGlobals* globals, NativeStore.ShapeData* first, NativeStore.ShapeData* second)
        {
            uint recordA = ((NativeStore.Shape*)first)->groupRecord;
            uint recordB = ((NativeStore.Shape*)second)->groupRecord;
            if (recordA == 0 || recordB == 0)
            {
                return false;
            }

            return globals->ignoreTable->Groups->IgnoresRecords(recordA, recordB);
        }

        /// <summary>
        /// Returns whether the group rule stops a wheel from touching a shape. The wheel's record comes from <see
        /// cref="RecordOfCollider"/>.
        /// </summary>
        internal static bool Ignores(NativeStore.StoreGlobals* globals, uint wheelRecord, NativeStore.ShapeData* shape)
        {
            uint shapeRecord = ((NativeStore.Shape*)shape)->groupRecord;
            if (wheelRecord == 0 || shapeRecord == 0)
            {
                return false;
            }

            return globals->ignoreTable->Groups->IgnoresRecords(wheelRecord, shapeRecord);
        }

        /// <summary>
        /// Returns the group record of a collider that has no shape of its own, such as a wheel, plus one. Returns zero
        /// if the collider has no group.
        /// </summary>
        internal static uint RecordOfCollider(NativeStore.StoreGlobals* globals, ulong engineId)
        {
            GroupTable* table = IgnoreTable.GroupsOf(globals);
            if (table == null || engineId == EmptyKey || !table->FindEntry(engineId, out uint slot))
            {
                return 0;
            }

            return table->entries[slot].Record;
        }

        /// <summary>
        /// Enters a new shape in the map, and gives it the group record of its collider if the game has already set
        /// one.
        /// </summary>
        internal static void AddShape(NativeStore.StoreGlobals* globals, NativeStore.Shape* shape)
        {
            ulong key = EngineIdOf(globals, shape);
            if (key == EmptyKey)
            {
                return;
            }

            GroupTable* table = TableOf(globals);

            // Take the slot before reading the entries pointer, because taking a slot may move the entries.
            uint slot = table->TakeEntry(globals, key);
            ref Entry entry = ref table->entries[slot];
            if (entry.Shape == null && entry.Record != 0)
            {
                table->Orphans--;
                globals->ignoreVersion++;
            }

            entry.Shape = shape;
            shape->groupRecord = entry.Record;
        }

        /// <summary>
        /// Removes a destroyed shape from the map. The collider keeps its group record in case Unity creates a new shape
        /// for it.
        /// </summary>
        internal static void RemoveShape(NativeStore.StoreGlobals* globals, NativeStore.Shape* shape)
        {
            ulong key = EngineIdOf(globals, shape);
            GroupTable* table = IgnoreTable.GroupsOf(globals);
            if (key == EmptyKey || table == null || !table->FindEntry(key, out uint slot))
            {
                return;
            }

            ref Entry entry = ref table->entries[slot];
            if (entry.Shape != shape)
            {
                return;
            }

            entry.Shape = null;
            if (entry.Record == 0)
            {
                entry.Key = DeletedKey;
            }
            else
            {
                table->Orphans++;
            }
        }

        /// <summary>
        /// Sets a collider's group, or removes its group when <paramref name="group"/> is zero. Returns the collider's
        /// shape so that the caller can wake its body, or null if the collider has no shape.
        /// </summary>
        internal static NativeStore.Shape* Set(NativeStore.StoreGlobals* globals, ulong engineId, uint group, uint member, uint flags)
        {
            if (engineId == EmptyKey)
            {
                return null;
            }

            GroupTable* table = TableOf(globals);
            globals->ignoreVersion++;
            if (group == 0)
            {
                if (!table->FindEntry(engineId, out uint found))
                {
                    return null;
                }

                ref Entry cleared = ref table->entries[found];
                table->FreeRecord(cleared.Record);
                cleared.Record = 0;
                NativeStore.Shape* shape = cleared.Shape;
                if (shape == null)
                {
                    table->Orphans--;
                    cleared.Key = DeletedKey;
                }
                else
                {
                    shape->groupRecord = 0;
                }

                return shape;
            }

            uint slot = table->TakeEntry(globals, engineId);
            ref Entry entry = ref table->entries[slot];
            if (entry.Record == 0)
            {
                entry.Record = table->TakeRecord(globals);
                table->Orphans += entry.Shape == null ? 1u : 0u;
            }

            table->records[entry.Record - 1] = new Record { Group = group, Member = member, Flags = flags };
            if (entry.Shape != null)
            {
                entry.Shape->groupRecord = entry.Record;
            }

            return entry.Shape;
        }

        /// <summary>Returns a collider's group record, or false if the collider has no group.</summary>
        internal static bool TryGet(NativeStore.StoreGlobals* globals, ulong engineId, out Record record)
        {
            record = default;
            uint index = RecordOfCollider(globals, engineId);
            if (index == 0)
            {
                return false;
            }

            record = IgnoreTable.GroupsOf(globals)->records[index - 1];
            return true;
        }

        /// <summary>Returns the shape of a collider, or null if the collider has no shape.</summary>
        internal static NativeStore.Shape* ShapeOf(NativeStore.StoreGlobals* globals, ulong engineId)
        {
            GroupTable* table = IgnoreTable.GroupsOf(globals);
            if (table == null || engineId == EmptyKey || !table->FindEntry(engineId, out uint slot))
            {
                return null;
            }

            return table->entries[slot].Shape;
        }

        /// <summary>
        /// Returns the number of bytes that the group table holds on the store's heap. Used by benchmarks.
        /// </summary>
        internal static ulong Bytes(NativeStore.StoreGlobals* globals)
        {
            GroupTable* table = IgnoreTable.GroupsOf(globals);
            if (table == null)
            {
                return 0;
            }

            return (ulong)sizeof(GroupTable) + table->entryCount * (ulong)sizeof(Entry)
                + table->recordCapacity * (ulong)sizeof(Record) + table->pairCount * (ulong)sizeof(ulong);
        }

        /// <summary>Lists the engine ids of colliders that have a group record but no shape.</summary>
        internal static int ListOrphans(NativeStore.StoreGlobals* globals, ulong* ids, int capacity)
        {
            GroupTable* table = IgnoreTable.GroupsOf(globals);
            int count = 0;
            for (uint i = 0; table != null && i < table->entryCount && count < capacity; i++)
            {
                Entry entry = table->entries[i];
                if (entry.Key != EmptyKey && entry.Key != DeletedKey && entry.Shape == null && entry.Record != 0)
                {
                    ids[count++] = entry.Key;
                }
            }

            return count;
        }

        /// <summary>Lets two members of one group collide, or stops them again.</summary>
        internal static void SetPair(NativeStore.StoreGlobals* globals, uint first, uint second, bool collides)
        {
            if (first == second)
            {
                return;
            }

            GroupTable* table = TableOf(globals);
            globals->ignoreVersion++;
            ulong key = PairKey(first, second);
            bool found = table->FindPair(key, out uint slot);
            if (!collides)
            {
                if (found)
                {
                    table->pairs[slot] = DeletedKey;
                    table->pairsLive--;
                }

                return;
            }

            if (found)
            {
                return;
            }

            if ((table->pairsTaken + 1) * 4 >= table->pairCount * 3)
            {
                table->RehashPairs(globals);
                table->FindPair(key, out slot);
            }

            table->pairsTaken += table->pairs[slot] == EmptyKey ? 1u : 0u;
            table->pairs[slot] = key;
            table->pairsLive++;
        }

        internal static bool HasPair(NativeStore.StoreGlobals* globals, uint first, uint second)
        {
            GroupTable* table = IgnoreTable.GroupsOf(globals);
            return table != null && table->pairsLive != 0 && table->FindPair(PairKey(first, second), out _);
        }

        /// <summary>
        /// Places every entry again with the current hash function, for the same reason as <see
        /// cref="IgnoreTable.Reslot"/>.
        /// </summary>
        internal void Reslot(NativeStore.StoreGlobals* globals)
        {
            if (entryCount != 0)
            {
                RehashEntries(globals);
            }

            if (pairCount != 0)
            {
                RehashPairs(globals);
            }
        }

        private readonly bool IgnoresRecords(uint first, uint second)
        {
            Record a = records[first - 1];
            Record b = records[second - 1];
            if (a.Group != b.Group)
            {
                return false;
            }

            if (a.Member == b.Member)
            {
                return true;
            }

            if ((a.Flags & b.Flags & SelfCollides) != 0)
            {
                return false;
            }

            return pairsLive == 0 || !FindPair(PairKey(a.Member, b.Member), out _);
        }

        private static ulong EngineIdOf(NativeStore.StoreGlobals* globals, NativeStore.Shape* shape)
        {
            void* userData = shape->data.userData;
            return userData != null ? *(ulong*)((byte*)userData + globals->engineObjectIdOffset) : EmptyKey;
        }

        private static GroupTable* TableOf(NativeStore.StoreGlobals* globals)
        {
            IgnoreTable* ignored = IgnoreTable.TableOf(globals);
            if (ignored->Groups == null)
            {
                GroupTable* table = StoreOps.Allocate<GroupTable>(globals);
                ignored->Groups = table;
            }

            return ignored->Groups;
        }

        private static ulong PairKey(uint first, uint second)
        {
            return ((ulong)math.min(first, second) << 32) | math.max(first, second);
        }

        private static uint Mix(ulong key)
        {
            key ^= key >> 33;
            key *= 0xFF51AFD7ED558CCDul;
            key ^= key >> 33;
            key *= 0xC4CEB9FE1A85EC53ul;
            key ^= key >> 33;
            return (uint)key;
        }

        private readonly bool FindEntry(ulong key, out uint index)
        {
            index = 0;
            if (entryCount == 0)
            {
                return false;
            }

            uint mask = entryCount - 1;
            uint firstDeleted = uint.MaxValue;
            for (uint slot = Mix(key) & mask; ; slot = (slot + 1) & mask)
            {
                ulong content = entries[slot].Key;
                if (content == EmptyKey)
                {
                    index = firstDeleted != uint.MaxValue ? firstDeleted : slot;
                    return false;
                }

                if (content == DeletedKey)
                {
                    firstDeleted = math.min(firstDeleted, slot);
                }
                else if (content == key)
                {
                    index = slot;
                    return true;
                }
            }
        }

        /// <summary>Returns the slot of the entry for a key, and adds an empty entry if there is none.</summary>
        private uint TakeEntry(NativeStore.StoreGlobals* globals, ulong key)
        {
            if (FindEntry(key, out uint slot))
            {
                return slot;
            }

            if ((entriesTaken + 1) * 4 >= entryCount * 3)
            {
                RehashEntries(globals);
                FindEntry(key, out slot);
            }

            entriesTaken += entries[slot].Key == EmptyKey ? 1u : 0u;
            entries[slot] = new Entry { Key = key };
            return slot;
        }

        private uint TakeRecord(NativeStore.StoreGlobals* globals)
        {
            if (freeRecord != 0)
            {
                uint reused = freeRecord;
                freeRecord = records[reused - 1].NextFree;
                return reused;
            }

            if (recordsUsed == recordCapacity)
            {
                recordCapacity = StoreOps.Grown(recordCapacity);
                records = (Record*)StoreOps.Reallocate(globals, records, recordCapacity * (ulong)sizeof(Record));
            }

            return ++recordsUsed;
        }

        private void FreeRecord(uint record)
        {
            if (record != 0)
            {
                records[record - 1] = new Record { NextFree = freeRecord };
                freeRecord = record;
            }
        }

        private readonly bool FindPair(ulong key, out uint index)
        {
            index = 0;
            if (pairCount == 0)
            {
                return false;
            }

            uint mask = pairCount - 1;
            uint firstDeleted = uint.MaxValue;
            for (uint slot = Mix(key) & mask; ; slot = (slot + 1) & mask)
            {
                ulong content = pairs[slot];
                if (content == EmptyKey)
                {
                    index = firstDeleted != uint.MaxValue ? firstDeleted : slot;
                    return false;
                }

                if (content == DeletedKey)
                {
                    firstDeleted = math.min(firstDeleted, slot);
                }
                else if (content == key)
                {
                    index = slot;
                    return true;
                }
            }
        }

        // Move every live entry to a clean array. This also removes the deleted slots.
        private void RehashEntries(NativeStore.StoreGlobals* globals)
        {
            uint live = 0;
            for (uint i = 0; i < entryCount; i++)
            {
                live += entries[i].Key != EmptyKey && entries[i].Key != DeletedKey ? 1u : 0u;
            }

            uint newCount = math.max(InitialSlots, entryCount);
            newCount = (live + 1) * 2 >= newCount ? newCount * 2 : newCount;

            Entry* oldEntries = entries;
            uint oldCount = entryCount;
            entries = (Entry*)StoreOps.Reallocate(globals, null, newCount * (ulong)sizeof(Entry));
            entryCount = newCount;
            entriesTaken = live;
            uint mask = newCount - 1;
            for (uint i = 0; i < newCount; i++)
            {
                entries[i] = default;
            }

            for (uint i = 0; i < oldCount; i++)
            {
                Entry entry = oldEntries[i];
                if (entry.Key == EmptyKey || entry.Key == DeletedKey)
                {
                    continue;
                }

                uint slot = Mix(entry.Key) & mask;
                while (entries[slot].Key != EmptyKey)
                {
                    slot = (slot + 1) & mask;
                }

                entries[slot] = entry;
            }

            if (oldEntries != null)
            {
                StoreOps.Release(globals, oldEntries);
            }
        }

        private void RehashPairs(NativeStore.StoreGlobals* globals)
        {
            uint newCount = math.max(InitialSlots, pairCount);
            newCount = (pairsLive + 1) * 2 >= newCount ? newCount * 2 : newCount;

            ulong* oldPairs = pairs;
            uint oldCount = pairCount;
            pairs = (ulong*)StoreOps.Reallocate(globals, null, newCount * (ulong)sizeof(ulong));
            pairCount = newCount;
            pairsTaken = pairsLive;
            uint mask = newCount - 1;
            for (uint i = 0; i < newCount; i++)
            {
                pairs[i] = EmptyKey;
            }

            for (uint i = 0; i < oldCount; i++)
            {
                ulong key = oldPairs[i];
                if (key == EmptyKey || key == DeletedKey)
                {
                    continue;
                }

                uint slot = Mix(key) & mask;
                while (pairs[slot] != EmptyKey)
                {
                    slot = (slot + 1) & mask;
                }

                pairs[slot] = key;
            }

            if (oldPairs != null)
            {
                StoreOps.Release(globals, oldPairs);
            }
        }
    }
}
