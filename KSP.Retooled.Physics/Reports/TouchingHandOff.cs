using Unity.Collections;
using Unity.Mathematics;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Reports;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Dynamics.Contacts;

namespace KSP.Retooled.Physics.Reports
{
    /// <summary>
    /// Preserve touching pairs across managed detach and domain reloads so that Unity's trigger cache stays
    /// consistent.
    /// Temporarily store pairs in the native trigger event array with unknown event types and zero IDs.
    /// The next StepState restores them before stepping. Main thread only.
    /// </summary>
    internal static unsafe class TouchingHandOff
    {
        // The type of an entry that holds a touching pair. The two low bits hold whether the pair is a trigger pair
        // and whether the first shape of the key is the trigger, and the next three bits hold the pair's
        // ShapePairState.Kinematic.
        private const uint PairType = 0x54500000;
        private const uint TypeMask = 0xFFFFFF00;
        private const uint TriggerBit = 1;
        private const uint FirstIsTriggerBit = 2;
        private const int KinematicShift = 2;

        /// <summary>Keeps the touching pairs of a world whose step state is about to be discarded.</summary>
        internal static void Save(WorldData* world, StoreGlobals* globals, NativeHashMap<ShapePairKey, ShapePairState> touching)
        {
            uint count = (uint)touching.Count;
            if (count > world->triggerCapacity)
            {
                world->triggerCapacity = math.max(count, StoreMemory.Grown(world->triggerCapacity));
                world->triggers = (TriggerEvent*)StoreMemory.Reallocate(globals, world->triggers, world->triggerCapacity * (ulong)sizeof(TriggerEvent));
            }

            uint i = 0;
            foreach (KVPair<ShapePairKey, ShapePairState> pair in touching)
            {
                ShapePairState state = pair.Value;
                world->triggers[i++] = new TriggerEvent
                {
                    triggerShape = (Shape*)pair.Key.First,
                    colliderShape = (Shape*)pair.Key.Second,
                    type = PairType | (state.Trigger ? TriggerBit : 0) | (state.FirstIsTrigger ? FirstIsTriggerBit : 0) | ((uint)state.Kinematic << KinematicShift),
                };
            }

            world->triggerCount = count;
        }

        /// <summary>
        /// Restore saved touching pairs and clear the temporary trigger array.
        /// Pairs count as seen before the first step and end if absent, unless both bodies are asleep.
        /// </summary>
        internal static void Restore(WorldData* world, NativeHashMap<ShapePairKey, ShapePairState> touching)
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
                TriggerEvent entry = world->triggers[i];
                var key = new ShapePairKey { First = (System.IntPtr)entry.triggerShape, Second = (System.IntPtr)entry.colliderShape };
                touching[key] = new ShapePairState
                {
                    LastSeen = 0,
                    Trigger = (entry.type & TriggerBit) != 0,
                    FirstIsTrigger = (entry.type & FirstIsTriggerBit) != 0,
                    Kinematic = (byte)((entry.type >> KinematicShift) & 7),
                };
            }

            world->triggerCount = 0;
        }
    }
}
