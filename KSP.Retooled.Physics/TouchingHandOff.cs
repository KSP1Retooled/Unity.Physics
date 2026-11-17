using Unity.Collections;
using Unity.Mathematics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Keeps the touching pairs of each world while the managed core is detached, for example during a domain reload in
    /// the editor, and gives them to the world's next step state.
    /// </summary>
    /// <remarks>
    /// Detaching discards each world's step state, but Unity keeps its scenes, and it keeps the trigger pairs that
    /// entered in the cache that it sends OnTriggerStay from. When the next step state started with no touching pairs,
    /// it didn't know the pairs that Unity held. A shape that was destroyed just before the detach then got no event that
    /// ended its trigger pair, because that event is sent in the next step, and Unity kept the pair with the ids of the
    /// destroyed colliders. A new pair whose shapes later took the same two addresses got one OnTriggerStay and then no
    /// more. A trigger pair that still touched entered a second time.
    ///
    /// The managed core's memory doesn't outlast a domain reload, and neither do shared statics. So the pairs wait in
    /// the world's array of trigger events, which belongs to the store. Unity has already read the events of the last
    /// step when the core detaches. Each pair takes one entry, with a type that Unity doesn't know and zero ids, so
    /// Unity would ignore such an entry if it read one. The world's next step state takes the pairs back and empties
    /// the array before its first step. All calls run on the main thread.
    /// </remarks>
    internal static unsafe class TouchingHandOff
    {
        // The type of an entry that holds a touching pair. The two low bits hold whether the pair is a trigger pair
        // and whether the first shape of the key is the trigger.
        private const uint PairType = 0x54500000;
        private const uint TypeMask = 0xFFFFFFFC;
        private const uint TriggerBit = 1;
        private const uint FirstIsTriggerBit = 2;

        /// <summary>Keeps the touching pairs of a world whose step state is about to be discarded.</summary>
        internal static void Save(NativeStore.WorldData* world, NativeStore.StoreGlobals* globals, NativeHashMap<ShapePairKey, ShapePairState> touching)
        {
            uint count = (uint)touching.Count;
            if (count > world->triggerCapacity)
            {
                world->triggerCapacity = math.max(count, StoreOps.Grown(world->triggerCapacity));
                world->triggers = (NativeStore.TriggerEvent*)StoreOps.Reallocate(globals, world->triggers, world->triggerCapacity * (ulong)sizeof(NativeStore.TriggerEvent));
            }

            uint i = 0;
            foreach (KVPair<ShapePairKey, ShapePairState> pair in touching)
            {
                ShapePairState state = pair.Value;
                world->triggers[i++] = new NativeStore.TriggerEvent
                {
                    triggerShape = (NativeStore.Shape*)pair.Key.First,
                    colliderShape = (NativeStore.Shape*)pair.Key.Second,
                    type = PairType | (state.Trigger ? TriggerBit : 0) | (state.FirstIsTrigger ? FirstIsTriggerBit : 0),
                };
            }

            world->triggerCount = count;
        }

        /// <summary>
        /// Puts the touching pairs that were kept for a world into its new step state's map, and empties the world's
        /// array of trigger events. A restored pair counts as last seen before the new state's first step, so a pair
        /// that doesn't touch in that step ends there, unless both of its bodies rest.
        /// </summary>
        internal static void Restore(NativeStore.WorldData* world, NativeHashMap<ShapePairKey, ShapePairState> touching)
        {
            // Without a detach, the array holds the events of the last step, or nothing.
            for (uint i = 0; i < world->triggerCount; i++)
            {
                if ((world->triggers[i].type & TypeMask) != PairType)
                {
                    return;
                }
            }

            for (uint i = 0; i < world->triggerCount; i++)
            {
                NativeStore.TriggerEvent entry = world->triggers[i];
                var key = new ShapePairKey { First = (System.IntPtr)entry.triggerShape, Second = (System.IntPtr)entry.colliderShape };
                touching[key] = new ShapePairState
                {
                    LastSeen = 0,
                    Trigger = (entry.type & TriggerBit) != 0,
                    FirstIsTrigger = (entry.type & FirstIsTriggerBit) != 0,
                };
            }

            world->triggerCount = 0;
        }
    }
}
