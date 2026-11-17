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

namespace Unity.Physics.Dynamics.Contacts
{
    internal static partial class TouchingPairs
    {
        /// <summary>
        /// Remove touching pairs for detached shapes and wake their remaining bodies.
        /// Keep the shape addresses of removed trigger pairs so the contact report can clear the cached pairs.
        /// </summary>
        [BurstCompile]
        internal unsafe struct ForgetRemovedShapesJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            public NativeHashMap<ShapePairKey, ShapePairState> Touching;

            /// <summary>The removed trigger pairs, in the shape order that the contact report needs.</summary>
            public NativeList<ShapePairKey> RemovedTriggerPairs;

            public void Execute()
            {
                if (World->removedShapeCount == 0)
                {
                    return;
                }

                NativeHashSet<IntPtr> removed = new NativeHashSet<IntPtr>((int)World->removedShapeCount, Allocator.Temp);
                for (uint i = 0; i < World->removedShapeCount; i++)
                {
                    removed.Add((IntPtr)World->removedShapes[i]);
                }

                World->removedShapeCount = 0;
                NativeArray<ShapePairKey> keys = Touching.GetKeyArray(Allocator.Temp);
                foreach (ShapePairKey key in keys)
                {
                    bool firstGone = removed.Contains(key.First);
                    bool secondGone = removed.Contains(key.Second);
                    if (firstGone || secondGone)
                    {
                        ShapePairState state = Touching[key];
                        if (state.Trigger)
                        {
                            // The removed shape may already have been replaced. Save its address without reading it.
                            var pair = new ShapePairKey
                            {
                                First = state.FirstIsTrigger ? key.First : key.Second,
                                Second = state.FirstIsTrigger ? key.Second : key.First,
                            };
                            RemovedTriggerPairs.Add(pair);
                        }

                        Touching.Remove(key);
                        WakeOwner(firstGone ? key.Second : key.First, firstGone && secondGone);
                    }
                }
            }

            private static void WakeOwner(IntPtr shape, bool gone)
            {
                Shape* left = (Shape*)shape;
                if (!gone && left->body != null)
                {
                    BodyKinematics.Wake(StoreMemory.Row(left->body));
                }
            }
        }
    }
}
