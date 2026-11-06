using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// One manifold of the narrow phase that survived filtering, ordered so that a pair reads the same however Unity Physics
    /// built it: body A has the lower store row, except that a static is always body B, because Collision.impulse is one
    /// vector for both sides and PhysX has it pushing the body. Unity Physics' own keys stay for finding the manifold's solver
    /// event, and one pair of shapes can have many manifolds, one per triangle of a mesh.
    /// </summary>
    internal unsafe struct ManifoldRecord
    {
        public NativeStore.ShapeData* ShapeA;
        public NativeStore.ShapeData* ShapeB;
        public int RowA;
        public int RowB;

        public BodyIndexPair BuiltPair;
        public ColliderKeyPair BuiltKeys;

        public int FirstPoint;
        public int PointCount;

        // From B towards A, as a contact's normal runs from the other shape to the one that hears about it.
        public float3 Normal;
        public float3 LinearVelocityA;
        public float3 AngularVelocityA;
        public float3 LinearVelocityB;
        public float3 AngularVelocityB;

        public float Impulse;
        public bool Touching;
        public bool Trigger;
    }

    /// <summary>A contact point of a <see cref="ManifoldRecord"/>, in the shape the narrow phase gives it.</summary>
    internal struct ManifoldPoint
    {
        public float3 Position;
        public float Distance;
    }

    /// <summary>What finds a manifold's record from its solver event.</summary>
    internal struct ManifoldKey : IEquatable<ManifoldKey>
    {
        public int BodyA;
        public int BodyB;
        public uint KeyA;
        public uint KeyB;

        public bool Equals(ManifoldKey other)
        {
            return BodyA == other.BodyA && BodyB == other.BodyB && KeyA == other.KeyA && KeyB == other.KeyB;
        }

        public override int GetHashCode()
        {
            return (int)math.hash(new int4(BodyA, BodyB, (int)KeyA, (int)KeyB));
        }
    }

    /// <summary>A pair of touching shapes, in pointer order so that it stays one key as bodies come and go.</summary>
    internal struct ShapePairKey : IEquatable<ShapePairKey>
    {
        public IntPtr First;
        public IntPtr Second;

        internal static unsafe ShapePairKey Of(NativeStore.ShapeData* a, NativeStore.ShapeData* b)
        {
            bool ordered = a < b;
            return new ShapePairKey { First = (IntPtr)(ordered ? a : b), Second = (IntPtr)(ordered ? b : a) };
        }

        public bool Equals(ShapePairKey other)
        {
            return First == other.First && Second == other.Second;
        }

        public override int GetHashCode()
        {
            return First.GetHashCode() * 397 ^ Second.GetHashCode();
        }
    }

    /// <summary>What a touching pair remembers from one step to the next.</summary>
    internal struct ShapePairState
    {
        public uint LastSeen;
        public bool Trigger;
    }

    /// <summary>
    /// Runs between the narrow phase and the solver over every point of every manifold. Physics.IgnoreCollision acts here,
    /// because a manifold names the leaves that touched, so the pairs of a compound can be told apart, which a filter on body
    /// pairs can't do. The manifolds it keeps are what the step's events and sleeping work from.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ReadContactsJob : IContactsJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        [NativeDisableUnsafePtrRestriction]
        public RigidBody* Bodies;

        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldRecord> Records;

        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldPoint> Points;

        public float DeltaTime;

        private bool skipping;
        private float touchDistance;

        public void Execute(ref ModifiableContactHeader header, ref ModifiableContactPoint contact)
        {
            if (contact.Index == 0)
            {
                skipping = !Begin(ref header);
            }

            if (!skipping)
            {
                Add(contact.Position, contact.Distance);
            }
        }

        /// <summary>Starts a manifold's record, or disables or passes over the manifold and returns false.</summary>
        private bool Begin(ref ModifiableContactHeader header)
        {
            RigidBody bodyA = Bodies[header.BodyIndexA];
            RigidBody bodyB = Bodies[header.BodyIndexB];
            NativeStore.ShapeData* shapeA = ShapeOf(bodyA, header.ColliderKeyA);
            NativeStore.ShapeData* shapeB = ShapeOf(bodyB, header.ColliderKeyB);
            if (IgnoreTable.Contains(Globals, shapeA->ignorePairId, shapeB->ignorePairId))
            {
                header.JacobianFlags |= JacobianFlags.Disabled;
                return false;
            }

            bool trigger = (header.JacobianFlags & JacobianFlags.IsTrigger) != 0;
            int rowA = bodyA.Entity.Index;
            int rowB = bodyB.Entity.Index;
            if (!CanReport(Store[rowA].flags, Store[rowB].flags, trigger))
            {
                return false;
            }

            bool staticA = (Store[rowA].flags & NativeStore.BodyFlags.Static) != 0;
            bool staticB = (Store[rowB].flags & NativeStore.BodyFlags.Static) != 0;
            bool swapped = staticA != staticB ? staticA : rowA > rowB;
            float3 normal = swapped ? -header.Normal : header.Normal;
            ref NativeStore.BodyData first = ref Store[swapped ? rowB : rowA];
            ref NativeStore.BodyData second = ref Store[swapped ? rowA : rowB];

            // Triggers go by true overlap, and a solid pair by PhysX's rule: within both contact offsets.
            touchDistance = trigger ? 0.0f : ((NativeStore.Shape*)shapeA)->contactOffset + ((NativeStore.Shape*)shapeB)->contactOffset;
            Records.Add(new ManifoldRecord
            {
                ShapeA = swapped ? shapeB : shapeA,
                ShapeB = swapped ? shapeA : shapeB,
                RowA = swapped ? rowB : rowA,
                RowB = swapped ? rowA : rowB,
                BuiltPair = new BodyIndexPair { BodyIndexA = header.BodyIndexA, BodyIndexB = header.BodyIndexB },
                BuiltKeys = new ColliderKeyPair { ColliderKeyA = header.ColliderKeyA, ColliderKeyB = header.ColliderKeyB },
                FirstPoint = Points.Length,
                Normal = normal,
                LinearVelocityA = first.linearVelocity,
                AngularVelocityA = first.angularVelocity,
                LinearVelocityB = second.linearVelocity,
                AngularVelocityB = second.angularVelocity,
                Trigger = trigger,
            });

            return true;
        }

        private void Add(float3 position, float distance)
        {
            ref ManifoldRecord record = ref Records.ElementAt(Records.Length - 1);
            Points.Add(new ManifoldPoint { Position = position, Distance = distance });
            record.PointCount++;

            // Unity Physics stops a fast body at the surface with a contact that looks well apart, where PhysX lets it in and
            // reports next step. The pair touches in the step that spends the impact, so that its event carries the impulse.
            float closing = record.Trigger ? 0.0f : math.max(ClosingSpeed(ref record, position), 0.0f);
            record.Touching |= distance - closing * DeltaTime <= touchDistance;
        }

        private float ClosingSpeed(ref ManifoldRecord record, float3 position)
        {
            float3 velocityA = PointVelocity(ref Store[record.RowA], record.LinearVelocityA, record.AngularVelocityA, position);
            float3 velocityB = PointVelocity(ref Store[record.RowB], record.LinearVelocityB, record.AngularVelocityB, position);
            return math.dot(velocityB - velocityA, record.Normal);
        }

        private static float3 PointVelocity(ref NativeStore.BodyData body, float3 linear, float3 angular, float3 position)
        {
            float3 center = (float3)body.position + math.rotate(body.rotation, body.localCenterOfMass);
            return linear + math.cross(angular, position - center);
        }

        /// <summary>
        /// PhysX pairs a kinematic with a static or another kinematic only for triggers, under the scene's default contact
        /// mode, and never two statics. Unity Physics solves nothing there either, but it does generate the contacts.
        /// </summary>
        private static bool CanReport(NativeStore.BodyFlags a, NativeStore.BodyFlags b, bool trigger)
        {
            const NativeStore.BodyFlags immobile = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic;
            if (trigger)
            {
                return ((a & b) & NativeStore.BodyFlags.Static) == 0;
            }

            return (a & immobile) == 0 || (b & immobile) == 0;
        }

        /// <summary>The shape behind a key: a compound child knows it, and any other collider stands for one shape.</summary>
        private NativeStore.ShapeData* ShapeOf(RigidBody body, ColliderKey key)
        {
            NativeStore.BodyData* row = Store + body.Entity.Index;
            if (body.Collider.Value.Type != ColliderType.Compound)
            {
                return row->shapes[ColliderBuilder.FirstSupported(row)];
            }

            body.Collider.Value.GetChild(ref key, out ChildCollider child);
            return row->shapes[child.Entity.Index];
        }
    }

    /// <summary>
    /// Gives each manifold the impulse the solver spent on it. Without dynamic bodies there's no solve and no event stream,
    /// so a step without them doesn't run this.
    /// </summary>
    [BurstCompile]
    internal struct ReadImpulsesJob : IJob
    {
        public NativeList<ManifoldRecord> Records;

        public CollisionEvents Events;

        [ReadOnly]
        public PhysicsWorld Built;

        public void Execute()
        {
            NativeHashMap<ManifoldKey, int> byKey = new NativeHashMap<ManifoldKey, int>(Records.Length, Allocator.Temp);
            for (int i = 0; i < Records.Length; i++)
            {
                ref ManifoldRecord record = ref Records.ElementAt(i);
                byKey.TryAdd(KeyOf(record.BuiltPair, record.BuiltKeys), i);
            }

            foreach (CollisionEvent collision in Events)
            {
                BodyIndexPair pair = new BodyIndexPair { BodyIndexA = collision.BodyIndexA, BodyIndexB = collision.BodyIndexB };
                ColliderKeyPair keys = new ColliderKeyPair { ColliderKeyA = collision.ColliderKeyA, ColliderKeyB = collision.ColliderKeyB };
                if (byKey.TryGetValue(KeyOf(pair, keys), out int index))
                {
                    float impulse = collision.CalculateDetails(ref Built).EstimatedImpulse;
                    Records.ElementAt(index).Impulse = math.isfinite(impulse) ? impulse : 0.0f;
                }
            }
        }

        private static ManifoldKey KeyOf(BodyIndexPair pair, ColliderKeyPair keys)
        {
            return new ManifoldKey
            {
                BodyA = pair.BodyIndexA,
                BodyB = pair.BodyIndexB,
                KeyA = keys.ColliderKeyA.Value,
                KeyB = keys.ColliderKeyB.Value,
            };
        }
    }

    /// <summary>
    /// Turns the step's manifolds into what Unity reads back from GetLastSimulationStepResults, which is PhysX's report: each
    /// pair of bodies with their velocities before the solve, each pair of shapes under it, and their contact points. Touches
    /// found and lost are worked out here against the pairs of the step before, and a pair of bodies at rest stays touching
    /// without a word, as an island asleep does under PhysX. Triggers report only enters and exits, because Unity repeats
    /// OnTriggerStay itself between them.
    /// </summary>
    [BurstCompile]
    internal unsafe struct PublishContactsJob : IJob
    {
        // Physics::ShapePairEventFlags.
        private const ushort NotifyTouchFound = 1 << 2;
        private const ushort NotifyTouchPersists = 1 << 3;
        private const ushort NotifyTouchLost = 1 << 4;

        private const uint NoFace = uint.MaxValue;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        public NativeList<ManifoldRecord> Records;

        [ReadOnly]
        public NativeList<ManifoldPoint> Points;

        public NativeHashMap<ShapePairKey, ShapePairState> Touching;

        public uint Step;

        public void Execute()
        {
            World->contactBodyCount = 0;
            World->contactShapeCount = 0;
            World->contactPointCount = 0;
            World->triggerCount = 0;

            ForgetRemovedShapes();
            Records.AsArray().Sort(new ManifoldOrder());
            Reserve();

            int rowA = -1;
            int rowB = -1;
            NativeStore.ContactBodyPair* bodyPair = null;
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
                bool found = !Touching.TryGetValue(key, out ShapePairState state);
                state.LastSeen = Step;
                state.Trigger = record.Trigger;
                Touching[key] = state;

                if (record.Trigger)
                {
                    if (found)
                    {
                        AddTrigger(record.ShapeA, record.ShapeB, NativeStore.TriggerEventEnter);
                    }

                    continue;
                }

                ushort wanted = Messages(record.ShapeA, record.ShapeB);
                ushort events = (ushort)(wanted & (found ? NotifyTouchFound : NotifyTouchPersists));
                if (!found && !IsAwake(record.RowA) && !IsAwake(record.RowB))
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

            ReportLostTouches();
        }

        private bool SameShapes(ref ManifoldRecord a, ref ManifoldRecord b)
        {
            return a.ShapeA == b.ShapeA && a.ShapeB == b.ShapeB;
        }

        /// <summary>A shape taken off its body leaves its pairs behind with no exit, and its address can be reused by the next one.</summary>
        private void ForgetRemovedShapes()
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
                    Touching.Remove(key);
                    WakeOwner(firstGone ? key.Second : key.First, firstGone && secondGone);
                }
            }
        }

        /// <summary>What rested on a shape that went away has nothing holding it now, so PhysX wakes it.</summary>
        private void WakeOwner(IntPtr shape, bool gone)
        {
            NativeStore.Shape* left = (NativeStore.Shape*)shape;
            if (!gone && left->body != null)
            {
                JointKernel.WakeRow(StoreOps.Hot(left->body));
            }
        }

        /// <summary>Everything is sized up front, because a report's pairs point into these arrays.</summary>
        private void Reserve()
        {
            uint pairs = (uint)(Records.Length + Touching.Count);
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
                capacity = math.max(needed, StoreOps.Grown(capacity));
                array = (T*)StoreOps.Reallocate(Globals, array, capacity * (ulong)sizeof(T));
            }
        }

        private ulong EngineId(void* userData)
        {
            return userData != null ? *(ulong*)((byte*)userData + Globals->engineObjectIdOffset) : 0;
        }

        private ushort Messages(NativeStore.ShapeData* a, NativeStore.ShapeData* b)
        {
            return (ushort)(((NativeStore.Shape*)a)->messages | ((NativeStore.Shape*)b)->messages);
        }

        private bool IsAwake(int row)
        {
            const NativeStore.BodyFlags resting = NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Sleeping;
            NativeStore.BodyFlags flags = World->bodies[row].flags;
            if ((flags & resting) != 0)
            {
                return false;
            }

            return (flags & NativeStore.BodyFlags.Kinematic) == 0 || (flags & NativeStore.BodyFlags.HasKinematicTarget) != 0;
        }

        private ulong BodyId(int row)
        {
            NativeStore.BodyData* body = World->bodies + row;
            return (body->flags & NativeStore.BodyFlags.Static) != 0 ? 0 : EngineId(body->owner->userData);
        }

        private NativeStore.ContactBodyPair* AddBodyPair(int rowA, int rowB, float3 linearA, float3 angularA, float3 linearB, float3 angularB)
        {
            NativeStore.ContactBodyPair* pair = World->contactBodies + World->contactBodyCount++;
            *pair = new NativeStore.ContactBodyPair
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

        private void AddShapePair(NativeStore.ContactBodyPair* bodyPair, int first, int end, ushort events)
        {
            ref ManifoldRecord record = ref Records.ElementAt(first);
            NativeStore.ContactPoint* points = World->contactPoints + World->contactPointCount;
            uint pointCount = 0;
            float3 impulseSum = float3.zero;
            for (int i = first; i < end; i++)
            {
                ref ManifoldRecord manifold = ref Records.ElementAt(i);
                float3 pointImpulse = manifold.Normal * (manifold.Impulse / math.max(manifold.PointCount, 1));
                for (int p = 0; p < manifold.PointCount; p++)
                {
                    ManifoldPoint point = Points[manifold.FirstPoint + p];
                    points[pointCount++] = new NativeStore.ContactPoint
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
            World->contactShapes[World->contactShapeCount++] = new NativeStore.ContactShapePair
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

        private void AddTrigger(NativeStore.ShapeData* a, NativeStore.ShapeData* b, uint type)
        {
            bool firstIsTrigger = (a->flags & NativeStore.ShapeFlagTrigger) != 0;
            NativeStore.ShapeData* trigger = firstIsTrigger ? a : b;
            NativeStore.ShapeData* other = firstIsTrigger ? b : a;
            World->triggers[World->triggerCount++] = new NativeStore.TriggerEvent
            {
                triggerShape = trigger->owner,
                colliderShape = other->owner,
                trigger = EngineId(trigger->userData),
                collider = EngineId(other->userData),
                type = type,
            };
        }

        /// <summary>Pairs the step didn't touch: the ones whose bodies both rest wait for them to wake, and the rest are over.</summary>
        private void ReportLostTouches()
        {
            NativeArray<ShapePairKey> keys = Touching.GetKeyArray(Allocator.Temp);
            foreach (ShapePairKey key in keys)
            {
                ShapePairState state = Touching[key];
                if (state.LastSeen == Step)
                {
                    continue;
                }

                NativeStore.ShapeData* first = (NativeStore.ShapeData*)key.First;
                NativeStore.ShapeData* second = (NativeStore.ShapeData*)key.Second;
                int rowFirst = (int)((NativeStore.Shape*)first)->body->index;
                int rowSecond = (int)((NativeStore.Shape*)second)->body->index;
                if (!IsAwake(rowFirst) && !IsAwake(rowSecond))
                {
                    continue;
                }

                Touching.Remove(key);
                if (state.Trigger)
                {
                    AddTrigger(first, second, NativeStore.TriggerEventExit);
                    continue;
                }

                ushort events = (ushort)(Messages(first, second) & NotifyTouchLost);
                if (events != 0)
                {
                    NativeStore.BodyData* a = World->bodies + rowFirst;
                    NativeStore.BodyData* b = World->bodies + rowSecond;
                    NativeStore.ContactBodyPair* pair = AddBodyPair(rowFirst, rowSecond, a->linearVelocity, a->angularVelocity, b->linearVelocity, b->angularVelocity);
                    World->contactShapes[World->contactShapeCount++] = new NativeStore.ContactShapePair
                    {
                        thisCollider = EngineId(first->userData),
                        otherCollider = EngineId(second->userData),
                        events = events,
                    };

                    pair->shapeCount = 1;
                }
            }
        }

        /// <summary>Puts a report's pairs next to each other: body pairs, then shape pairs within them.</summary>
        private struct ManifoldOrder : IComparer<ManifoldRecord>
        {
            public int Compare(ManifoldRecord x, ManifoldRecord y)
            {
                int order = x.RowA.CompareTo(y.RowA);
                order = order != 0 ? order : x.RowB.CompareTo(y.RowB);
                order = order != 0 ? order : ((long)x.ShapeA).CompareTo((long)y.ShapeA);
                return order != 0 ? order : ((long)x.ShapeB).CompareTo((long)y.ShapeB);
            }
        }
    }
}
