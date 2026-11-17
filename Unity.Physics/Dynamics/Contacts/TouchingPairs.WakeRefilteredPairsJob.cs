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
        /// Wake bodies whose touching pairs changed because of filters, trigger flags or kinematic state.
        /// Run before applying forces so that the narrow phase sees the changes in the same step.
        /// </summary>
        [BurstCompile]
        internal unsafe struct WakeRefilteredPairsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [ReadOnly]
            public NativeHashMap<ShapePairKey, ShapePairState> Touching;

            public void Execute()
            {
                // Removed shapes remain in touching pairs until later cleanup, but their memory may already be
                // freed. Do not dereference them.
                var removed = new NativeHashSet<IntPtr>((int)World->removedShapeCount, Allocator.Temp);
                for (uint i = 0; i < World->removedShapeCount; i++)
                {
                    removed.Add((IntPtr)World->removedShapes[i]);
                }

                foreach (KVPair<ShapePairKey, ShapePairState> pair in Touching)
                {
                    if (removed.Contains(pair.Key.First) || removed.Contains(pair.Key.Second))
                    {
                        continue;
                    }

                    ShapeData* first = (ShapeData*)pair.Key.First;
                    ShapeData* second = (ShapeData*)pair.Key.Second;
                    bool trigger = ((first->flags | second->flags) & StoreConstants.ShapeFlagTrigger) != 0;
                    if (trigger == pair.Value.Trigger && Collide(first, second) && !pair.Value.KinematicChanged(ShapePairState.KinematicOf(pair.Key)))
                    {
                        continue;
                    }

                    Wake((Shape*)first);
                    Wake((Shape*)second);
                }
            }

            /// <summary>
            /// Return whether shape filters permit contact or triggering. Explicit ignored pairs are handled
            /// separately.
            /// </summary>
            private static bool Collide(ShapeData* a, ShapeData* b)
            {
                return Responds(a) && Responds(b) && (a->belongsTo & b->collidesWith) != 0 && (b->belongsTo & a->collidesWith) != 0;
            }

            private static bool Responds(ShapeData* shape)
            {
                return shape->collidesWith != 0 && ((shape->flags & StoreConstants.ShapeFlagTrigger) != 0 || shape->collisionDisabled == 0);
            }

            private static void Wake(Shape* shape)
            {
                if (shape->body != null && (StoreMemory.Row(shape->body)->flags & BodyFlags.Sleeping) != 0)
                {
                    BodyKinematics.Wake(StoreMemory.Row(shape->body));
                }
            }
        }
    }
}
