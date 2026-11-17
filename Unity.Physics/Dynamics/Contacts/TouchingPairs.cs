using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Reports;
using StoreContactPoint = Unity.Physics.Store.Reports.ContactPoint;
using TriggerEvent = Unity.Physics.Store.Reports.TriggerEvent;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Collectors;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Vessels.Detection;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Dynamics.Contacts
{
    internal static unsafe partial class TouchingPairs
    {
        internal static bool ShouldRestart(NativeHashMap<ShapePairKey, ShapePairState> touching, ShapePairKey key,
            bool trigger, out ShapePairState state, out byte kinematic, out bool found)
        {
            found = !touching.TryGetValue(key, out state);
            kinematic = ShapePairState.KinematicOf(key);
            return !found && (state.Trigger != trigger || state.KinematicChanged(kinematic));
        }

        internal static ShapeData* Remember(NativeHashMap<ShapePairKey, ShapePairState> touching,
            ShapePairKey key, ShapePairState state, uint step, ref ManifoldRecord record, byte kinematic, bool found)
        {
            state.LastSeen = step;
            state.Trigger = record.Trigger;
            state.Kinematic = kinematic;
            ShapeData* trigger = (record.ShapeA->flags & StoreConstants.ShapeFlagTrigger) != 0 ? record.ShapeA : record.ShapeB;
            if (found)
            {
                // Only a new pair sets the order: when a shape of a trigger pair becomes a trigger too, the flags
                // can pick the other shape, and the exit must still name the shapes as the enter did.
                state.FirstIsTrigger = (IntPtr)trigger == key.First;
            }

            touching[key] = state;

            return trigger;
        }

        /// <summary>Remove pairs that disappeared this step, keeping absent pairs while both bodies rest.</summary>
        internal static NativeList<LostPair> RemoveLost(WorldData* world,
            NativeHashMap<ShapePairKey, ShapePairState> touching, uint step)
        {
            var lost = new NativeList<LostPair>(Allocator.Temp);
            NativeArray<ShapePairKey> keys = touching.GetKeyArray(Allocator.Temp);
            foreach (ShapePairKey key in keys)
            {
                ShapePairState state = touching[key];
                if (state.LastSeen == step)
                {
                    continue;
                }

                ShapeData* first = (ShapeData*)key.First;
                ShapeData* second = (ShapeData*)key.Second;
                int rowFirst = (int)((Shape*)first)->body->index;
                int rowSecond = (int)((Shape*)second)->body->index;

                // Kinematic changes restart touching pairs even when both bodies are asleep.
                if (!IsAwake(world, rowFirst) && !IsAwake(world, rowSecond) && !state.KinematicChanged(ShapePairState.KinematicOf(key)))
                {
                    continue;
                }

                touching.Remove(key);
                long placeFirst = PlaceOf(world, first);
                long placeSecond = PlaceOf(world, second);
                lost.Add(new LostPair
                {
                    Key = key,
                    State = state,
                    First = math.min(placeFirst, placeSecond),
                    Second = math.max(placeFirst, placeSecond),
                });
            }

            lost.Sort();
            return lost;
        }

        internal static bool IsAwake(WorldData* world, int row)
        {
            const BodyFlags resting = BodyFlags.Static | BodyFlags.Sleeping;
            BodyFlags flags = world->bodies[row].flags;
            if ((flags & resting) != 0)
            {
                return false;
            }

            return (flags & BodyFlags.Kinematic) == 0 || (flags & BodyFlags.HasKinematicTarget) != 0;
        }

        /// <summary>
        /// A lost touching pair with body rows and shape positions for deterministic exit event ordering.
        /// </summary>
        internal struct LostPair : IComparable<LostPair>
        {
            public ShapePairKey Key;
            public ShapePairState State;
            public long First;
            public long Second;

            public int CompareTo(LostPair other)
            {
                int order = First.CompareTo(other.First);
                return order != 0 ? order : Second.CompareTo(other.Second);
            }
        }

        /// <summary>The place of a shape in the store, as one number: its body's row, then its position in the body.</summary>
        private static long PlaceOf(WorldData* world, ShapeData* shape)
        {
            Body* body = ((Shape*)shape)->body;
            BodyData* row = world->bodies + body->index;
            int index = 0;
            while (index < row->shapeCount && row->shapes[index] != shape)
            {
                index++;
            }

            return ((long)body->index << 32) | (uint)index;
        }
    }
}
