using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Reports;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Profiling;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;
using TriggerEvent = Unity.Physics.Store.Reports.TriggerEvent;
using StoreContactPoint = Unity.Physics.Store.Reports.ContactPoint;

namespace KSP.Retooled.Physics.Reports
{
    internal static unsafe partial class ContactReport
    {
        /// <summary>
        /// Write pre-solve velocities, contact points and requested pair events to the plugin report.
        /// </summary>
        [BurstCompile]
        private unsafe struct PublishContactsJob : IJob
        {
            // Physics::ShapePairEventFlags.
            private const ushort NotifyTouchFound = 1 << 2;
            private const ushort NotifyTouchPersists = 1 << 3;
            private const ushort NotifyTouchLost = 1 << 4;

            private const uint NoFace = uint.MaxValue;

            private static readonly ProfilerMarker SortMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Contacts.Sort");
            private static readonly ProfilerMarker ReportMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Contacts.Report");
            private static readonly ProfilerMarker LostMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Contacts.Lost");

            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [NativeDisableUnsafePtrRestriction]
            public StoreGlobals* Globals;

            public NativeList<ManifoldRecord> Records;

            [ReadOnly]
            public NativeList<ManifoldPoint> Points;

            public NativeHashMap<ShapePairKey, ShapePairState> Touching;

            /// <summary>
            /// Removal events for trigger pairs. Report before current trigger events, then clear the list.
            /// </summary>
            public NativeList<ShapePairKey> RemovedTriggerPairs;

            public uint Step;

            public void Execute()
            {
                World->contactBodyCount = 0;
                World->contactShapeCount = 0;
                World->contactPointCount = 0;
                World->triggerCount = 0;

                using (SortMarker.Auto())
                {
                    Records.AsArray().Sort(new ManifoldOrder());
                }

                using (ReportMarker.Auto())
                {
                    ReportCurrentContacts();
                }

                using (LostMarker.Auto())
                {
                    ReportLostTouches();
                }
            }

            private void ReportCurrentContacts()
            {
                Reserve();

                // Removal events clear Unity's cached pair before any replacement pair is added at the same
                // addresses.
                for (int i = 0; i < RemovedTriggerPairs.Length; i++)
                {
                    World->triggers[World->triggerCount++] = RemovedEvent(RemovedTriggerPairs[i]);
                }

                RemovedTriggerPairs.Clear();

                int rowA = -1;
                int rowB = -1;
                ContactBodyPair* bodyPair = null;
                int index = 0;
                while (index < Records.Length)
                {
                    int first = index;
                    ref ManifoldRecord record = ref Records.ElementAt(first);
                    bool touching = false;
                    do
                    {
                        touching |= Records[index].Touching;
                        index++;
                    }
                    while (index < Records.Length && SameShapes(ref record, ref Records.ElementAt(index)));

                    if (!touching)
                    {
                        continue;
                    }

                    ShapePairKey key = ShapePairKey.Of(record.ShapeA, record.ShapeB);
                    if (TouchingPairs.ShouldRestart(Touching, key, ref record,
                        out ShapePairState state, out byte kinematic, out bool found))
                    {
                        // Restart the pair when trigger or kinematic flags change, matching PhysX enter/exit
                        // behavior.
                        EndPair(key, state);
                        found = true;

                        // The lost contact may have started a body pair of its own, so the shape pairs that follow start
                        // another one.
                        bodyPair = null;
                    }

                    ShapeData* trigger = TouchingPairs.Remember(Touching, key, state, Step, ref record, kinematic, found);

                    if (record.Trigger)
                    {
                        if (found)
                        {
                            AddTrigger(trigger, trigger == record.ShapeA ? record.ShapeB : record.ShapeA, StoreConstants.TriggerEventEnter);
                        }

                        continue;
                    }

                    ushort wanted = Messages(record.ShapeA, record.ShapeB);
                    ushort events = (ushort)(wanted & (found ? NotifyTouchFound : NotifyTouchPersists));
                    if (!found && !TouchingPairs.IsAwake(World, record.RowA) && !TouchingPairs.IsAwake(World, record.RowB))
                    {
                        events = 0;
                    }

                    if (events == 0)
                    {
                        continue;
                    }

                    if (record.RowA != rowA || record.RowB != rowB || bodyPair == null)
                    {
                        rowA = record.RowA;
                        rowB = record.RowB;
                        bodyPair = AddBodyPair(rowA, rowB, record.LinearVelocityA, record.AngularVelocityA, record.LinearVelocityB, record.AngularVelocityB);
                    }

                    AddShapePair(bodyPair, first, index, events);
                }
            }

            private static TriggerEvent RemovedEvent(ShapePairKey pair)
            {
                return new TriggerEvent
                {
                    triggerShape = (Shape*)pair.First,
                    colliderShape = (Shape*)pair.Second,
                    type = StoreConstants.TriggerEventShapeRemoved,
                };
            }

            private bool SameShapes(ref ManifoldRecord a, ref ManifoldRecord b)
            {
                return a.ShapeA == b.ShapeA && a.ShapeB == b.ShapeB;
            }

            /// <summary>
            /// Reserves all required space before building the report because its pairs contain pointers into these arrays.
            /// </summary>
            private void Reserve()
            {
                uint pairs = (uint)(Records.Length + Touching.Count + RemovedTriggerPairs.Length);
                Grow(ref World->contactBodies, ref World->contactBodyCapacity, pairs);
                Grow(ref World->contactShapes, ref World->contactShapeCapacity, pairs);
                Grow(ref World->contactPoints, ref World->contactPointCapacity, (uint)Points.Length);
                Grow(ref World->triggers, ref World->triggerCapacity, pairs);
            }

            private void Grow<T>(ref T* array, ref uint capacity, uint needed)
                where T : unmanaged
            {
                if (needed > capacity)
                {
                    capacity = math.max(needed, StoreMemory.Grown(capacity));
                    array = (T*)StoreMemory.Reallocate(Globals, array, capacity * (ulong)sizeof(T));
                }
            }

            private ulong EngineId(void* userData)
            {
                return userData != null ? *(ulong*)((byte*)userData + Globals->engineObjectIdOffset) : 0;
            }

            private ushort Messages(ShapeData* a, ShapeData* b)
            {
                return (ushort)(((Shape*)a)->messages | ((Shape*)b)->messages);
            }

            private ulong BodyId(int row)
            {
                BodyData* body = World->bodies + row;
                return (body->flags & BodyFlags.Static) != 0 ? 0 : EngineId(body->owner->userData);
            }

            private ContactBodyPair* AddBodyPair(int rowA, int rowB, float3 linearA, float3 angularA, float3 linearB, float3 angularB)
            {
                ContactBodyPair* pair = World->contactBodies + World->contactBodyCount++;
                *pair = new ContactBodyPair
                {
                    thisBody = BodyId(rowA),
                    otherBody = BodyId(rowB),
                    shapes = World->contactShapes + World->contactShapeCount,
                    thisLinearVelocity = linearA,
                    thisAngularVelocity = angularA,
                    otherLinearVelocity = linearB,
                    otherAngularVelocity = angularB,
                };

                return pair;
            }

            private void AddShapePair(ContactBodyPair* bodyPair, int first, int end, ushort events)
            {
                ref ManifoldRecord record = ref Records.ElementAt(first);
                StoreContactPoint* points = World->contactPoints + World->contactPointCount;
                uint pointCount = 0;
                float3 impulseSum = float3.zero;
                for (int i = first; i < end; i++)
                {
                    ref ManifoldRecord manifold = ref Records.ElementAt(i);
                    float3 pointImpulse = manifold.Normal * (manifold.Impulse / math.max(manifold.PointCount, 1));
                    for (int p = 0; p < manifold.PointCount; p++)
                    {
                        ManifoldPoint point = Points[manifold.FirstPoint + p];
                        points[pointCount++] = new StoreContactPoint
                        {
                            position = point.Position,
                            separation = point.Distance,
                            normal = manifold.Normal,
                            faceIndex0 = NoFace,
                            impulse = pointImpulse,
                            faceIndex1 = NoFace,
                        };

                        impulseSum += pointImpulse;
                    }
                }

                World->contactPointCount += pointCount;
                World->contactShapes[World->contactShapeCount++] = new ContactShapePair
                {
                    thisCollider = EngineId(record.ShapeA->userData),
                    otherCollider = EngineId(record.ShapeB->userData),
                    points = points,
                    pointCount = pointCount,
                    events = events,
                    impulseSum = impulseSum,
                };

                bodyPair->shapeCount++;
            }

            /// <summary>
            /// Add a trigger event in the supplied shape order. Do not infer order from flags that may have changed
            /// since entry.
            /// </summary>
            private void AddTrigger(ShapeData* trigger, ShapeData* other, uint type)
            {
                World->triggers[World->triggerCount++] = new TriggerEvent
                {
                    triggerShape = trigger->owner,
                    colliderShape = other->owner,
                    trigger = EngineId(trigger->userData),
                    collider = EngineId(other->userData),
                    type = type,
                };
            }

            /// <summary>
            /// Reports lost touches for pairs absent from this step. Retains pairs while both bodies are at rest, until
            /// either wakes.
            /// </summary>
            private void ReportLostTouches()
            {
                NativeList<TouchingPairs.LostPair> lost = TouchingPairs.RemoveLost(World, Touching, Step);
                for (int i = 0; i < lost.Length; i++)
                {
                    EndPair(lost[i].Key, lost[i].State);
                }
            }

            /// <summary>
            /// Report a trigger exit or requested collision exit for a pair whose shapes still exist.
            /// </summary>
            private void EndPair(ShapePairKey key, ShapePairState state)
            {
                ShapeData* first = (ShapeData*)key.First;
                ShapeData* second = (ShapeData*)key.Second;
                if (state.Trigger)
                {
                    // Name the shapes in the order of the pair's enter event, which Unity keeps the pair by. The flags can't
                    // tell that order any more when a shape stopped being a trigger, or when both shapes are triggers.
                    if (state.FirstIsTrigger)
                    {
                        AddTrigger(first, second, StoreConstants.TriggerEventExit);
                    }
                    else
                    {
                        AddTrigger(second, first, StoreConstants.TriggerEventExit);
                    }

                    return;
                }

                ushort events = (ushort)(Messages(first, second) & NotifyTouchLost);
                if (events != 0)
                {
                    // Preserve enter-event order: static body second, otherwise lower body row first. Shape
                    // addresses are not deterministic.
                    int rowFirst = (int)((Shape*)first)->body->index;
                    int rowSecond = (int)((Shape*)second)->body->index;
                    bool staticFirst = (World->bodies[rowFirst].flags & BodyFlags.Static) != 0;
                    bool staticSecond = (World->bodies[rowSecond].flags & BodyFlags.Static) != 0;
                    if (staticFirst != staticSecond ? staticFirst : rowFirst > rowSecond)
                    {
                        ShapeData* shape = first;
                        first = second;
                        second = shape;
                        int row = rowFirst;
                        rowFirst = rowSecond;
                        rowSecond = row;
                    }

                    BodyData* a = World->bodies + rowFirst;
                    BodyData* b = World->bodies + rowSecond;
                    ContactBodyPair* pair = AddBodyPair(rowFirst, rowSecond, a->velocity->Linear, BodyKinematics.WorldAngularVelocity(*a), b->velocity->Linear, BodyKinematics.WorldAngularVelocity(*b));
                    World->contactShapes[World->contactShapeCount++] = new ContactShapePair
                    {
                        thisCollider = EngineId(first->userData),
                        otherCollider = EngineId(second->userData),
                        events = events,
                    };

                    pair->shapeCount = 1;
                }
            }

            /// <summary>
            /// Order manifolds by body rows, shape positions and collider keys for deterministic contact messages.
            /// </summary>
            private struct ManifoldOrder : IComparer<ManifoldRecord>
            {
                public int Compare(ManifoldRecord x, ManifoldRecord y)
                {
                    int order = x.RowA.CompareTo(y.RowA);
                    order = order != 0 ? order : x.RowB.CompareTo(y.RowB);
                    order = order != 0 ? order : x.ShapeIndexA.CompareTo(y.ShapeIndexA);
                    order = order != 0 ? order : x.ShapeIndexB.CompareTo(y.ShapeIndexB);
                    order = order != 0 ? order : x.BuiltKeys.ColliderKeyA.Value.CompareTo(y.BuiltKeys.ColliderKeyA.Value);
                    return order != 0 ? order : x.BuiltKeys.ColliderKeyB.Value.CompareTo(y.BuiltKeys.ColliderKeyB.Value);
                }
            }
        }
    }
}
