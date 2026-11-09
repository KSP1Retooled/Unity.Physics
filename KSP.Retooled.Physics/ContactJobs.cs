using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
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

        // As the narrow phase gave it, which is how the solver's jacobian knows it.
        public float3 BuiltNormal;

        // What the last step left to start from, of which the solve is given WarmStartJob.Fraction, and what it ended on.
        public float WarmImpulse;
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

    /// <summary>What identifies a manifold from one step to the next, when bodies have moved in the built world.</summary>
    internal unsafe struct PersistentManifoldKey : IEquatable<PersistentManifoldKey>
    {
        public NativeStore.ShapeData* ShapeA;
        public NativeStore.ShapeData* ShapeB;
        public uint KeyA;
        public uint KeyB;

        internal static PersistentManifoldKey Of(ref ManifoldRecord record)
        {
            return new PersistentManifoldKey
            {
                ShapeA = record.ShapeA,
                ShapeB = record.ShapeB,
                KeyA = record.BuiltKeys.ColliderKeyA.Value,
                KeyB = record.BuiltKeys.ColliderKeyB.Value,
            };
        }

        public bool Equals(PersistentManifoldKey other)
        {
            return ShapeA == other.ShapeA && ShapeB == other.ShapeB && KeyA == other.KeyA && KeyB == other.KeyB;
        }

        public override int GetHashCode()
        {
            return ((IntPtr)ShapeA).GetHashCode() * 397 ^ ((IntPtr)ShapeB).GetHashCode() * 31 ^ (int)math.hash(new uint2(KeyA, KeyB));
        }
    }

    /// <summary>
    /// What finds a manifold from its jacobian, which names its bodies but not its colliders. Manifolds of one body pair with
    /// the same normal, such as a box on two coplanar triangles, come to one key and share out their impulse.
    /// </summary>
    internal struct JacobianKey : IEquatable<JacobianKey>
    {
        public int BodyA;
        public int BodyB;
        public float3 Normal;

        public bool Equals(JacobianKey other)
        {
            return BodyA == other.BodyA && BodyB == other.BodyB && Normal.Equals(other.Normal);
        }

        public override int GetHashCode()
        {
            return (int)math.hash(new int2(BodyA, BodyB)) ^ Normal.GetHashCode();
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
    /// Whether every collider pair between two bodies is ignored, as worked out when the ignored pairs and the bodies' shapes
    /// were at the versions it holds.
    /// </summary>
    internal struct KnownBodyPair
    {
        public uint IgnoreVersion;
        public uint WholeBuilds;
        public bool Ignored;
    }

    /// <summary>
    /// Takes out the overlaps whose every collider pair is under Physics.IgnoreCollision, which is what the parts of one craft
    /// are. There are many times more of those than of joints, and left in they would make a craft one crowded island and
    /// send them all through the narrow phase for nothing. Body pairs with some colliders still in contact stay, and
    /// <see cref="ReadContactsJob"/> tells those colliders apart.
    ///
    /// Finding that out takes a lookup per collider pair, each a likely cache miss in a table of a craft's every ignored pair,
    /// so the answer is kept per body pair. It holds until the ignored pairs change or a body's shapes may have: a body
    /// whose shapes change, or that is made, gets a whole collider build, which bumps the count it was kept with.
    /// </summary>
    [BurstCompile]
    internal unsafe struct DropIgnoredPairsJob : IJobParallelForDefer
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<DispatchPairSequencer.DispatchPair> Pairs;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        [NativeDisableUnsafePtrRestriction]
        public RigidBody* Bodies;

        [ReadOnly]
        public NativeHashMap<ulong, KnownBodyPair> Known;

        [ReadOnly]
        public NativeReference<uint> WholeBuilds;

        /// <summary>What this step worked out afresh, for <see cref="RememberBodyPairsJob"/> to keep.</summary>
        public NativeQueue<KeyValue>.ParallelWriter Learned;

        internal struct KeyValue
        {
            public ulong Key;
            public KnownBodyPair Value;
        }

        public void Execute(int index)
        {
            DispatchPairSequencer.DispatchPair pair = Pairs[index];
            if (pair.IsValid && pair.IsContact && pair.IsCollisionEnabled && IsIgnored(pair))
            {
                pair.DisableCollision();
                Pairs[index] = pair;
            }
        }

        private bool IsIgnored(DispatchPairSequencer.DispatchPair pair)
        {
            uint rowA = (uint)Bodies[pair.BodyIndexA].Entity.Index;
            uint rowB = (uint)Bodies[pair.BodyIndexB].Entity.Index;
            ulong key = ((ulong)math.min(rowA, rowB) << 32) | math.max(rowA, rowB);
            uint ignoreVersion = Globals->ignoreVersion;
            uint wholeBuilds = WholeBuilds.Value;
            if (Known.TryGetValue(key, out KnownBodyPair known) && known.IgnoreVersion == ignoreVersion && known.WholeBuilds == wholeBuilds)
            {
                return known.Ignored;
            }

            bool ignored = IsIgnored(Store + rowA, Store + rowB);
            Learned.Enqueue(new KeyValue
            {
                Key = key,
                Value = new KnownBodyPair { IgnoreVersion = ignoreVersion, WholeBuilds = wholeBuilds, Ignored = ignored },
            });
            return ignored;
        }

        private bool IsIgnored(NativeStore.BodyData* rowA, NativeStore.BodyData* rowB)
        {
            for (int a = 0; a < rowA->shapeCount; a++)
            {
                NativeStore.ShapeData* shapeA = rowA->shapes[a];
                if (!ColliderBuilder.IsSupported(shapeA))
                {
                    continue;
                }

                for (int b = 0; b < rowB->shapeCount; b++)
                {
                    NativeStore.ShapeData* shapeB = rowB->shapes[b];
                    if (ColliderBuilder.IsSupported(shapeB) && !IgnoreTable.Contains(Globals, shapeA->ignorePairId, shapeB->ignorePairId))
                    {
                        return false;
                    }
                }
            }

            return true;
        }
    }

    /// <summary>
    /// Keeps what <see cref="DropIgnoredPairsJob"/> worked out afresh. Pairs that stopped overlapping leave their answers
    /// behind, so past several times the overlaps a craft's parts have, everything is dropped and the next step works out what
    /// it needs again.
    /// </summary>
    [BurstCompile]
    internal struct RememberBodyPairsJob : IJob
    {
        private const int KeptPerBody = 32;

        public NativeHashMap<ulong, KnownBodyPair> Known;

        public NativeQueue<DropIgnoredPairsJob.KeyValue> Learned;

        public int Bodies;

        public void Execute()
        {
            if (Known.Count + Learned.Count > KeptPerBody * Bodies + 1024)
            {
                Known.Clear();
            }

            while (Learned.TryDequeue(out DropIgnoredPairsJob.KeyValue learned))
            {
                Known[learned.Key] = learned.Value;
            }
        }
    }

    /// <summary>
    /// Runs between the narrow phase and the solver over every point of every manifold. Physics.IgnoreCollision acts here for
    /// the body pairs <see cref="DropIgnoredPairsJob"/> left in, because a manifold names the leaves that touched, so the pairs
    /// of a compound can be told apart. So do friction and restitution, which need both shapes' materials and the points that
    /// touched. The manifolds it keeps are what the step's events and sleeping work from.
    ///
    /// Each of the narrow phase's buckets is read on its own, into lists of its own, and
    /// <see cref="MergeContactBucketsJob"/> joins them in bucket order, so the records come out in the order a single pass
    /// over the stream would give them.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ReadContactsJob : IJobParallelForDefer
    {
        // Measured against PhysX: a 0.015 m box slides on one anchor and a 0.02 m box on two, and their diagonals are 0.021 m
        // and 0.028 m. Points nearer each other than this share an anchor.
        private const float FrictionOffsetThreshold = 0.025f;

        public NativeStream.Reader Contacts;

        [NativeDisableContainerSafetyRestriction]
        public NativeList<ContactBucket> Buckets;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        [NativeDisableUnsafePtrRestriction]
        public RigidBody* Bodies;

        [ReadOnly]
        public NativeHashMap<PersistentManifoldKey, float> LastImpulses;

        public float DeltaTime;
        public float BounceThreshold;

        public void Execute(int bucket)
        {
            ref ContactBucket output = ref Buckets.ElementAt(bucket);
            output.Records.Clear();
            output.Points.Clear();

            // The narrow phase writes each manifold as its header followed by its points.
            NativeStream.Reader reader = Contacts;
            reader.BeginForEachIndex(bucket);
            while (reader.RemainingItemCount > 0)
            {
                ContactHeader* header = (ContactHeader*)reader.ReadUnsafePtr(sizeof(ContactHeader));
                bool kept = Begin(ref *header, ref output.Records, output.Points.Length, out float touchDistance);
                for (int i = 0; i < header->NumContacts; i++)
                {
                    ContactPoint* point = (ContactPoint*)reader.ReadUnsafePtr(sizeof(ContactPoint));
                    if (kept)
                    {
                        Add(ref output, point->Position, point->Distance, touchDistance);
                    }
                }

                if (kept)
                {
                    SetCoefficients(ref *header, ref output.Records.ElementAt(output.Records.Length - 1), ref output.Points);
                }
            }

            reader.EndForEachIndex();
        }

        /// <summary>Starts a manifold's record, or disables or passes over the manifold and returns false.</summary>
        private bool Begin(ref ContactHeader header, ref UnsafeList<ManifoldRecord> records, int firstPoint, out float touchDistance)
        {
            touchDistance = 0.0f;
            RigidBody bodyA = Bodies[header.BodyPair.BodyIndexA];
            RigidBody bodyB = Bodies[header.BodyPair.BodyIndexB];
            ColliderKey keyA = header.ColliderKeys.ColliderKeyA;
            ColliderKey keyB = header.ColliderKeys.ColliderKeyB;
            NativeStore.ShapeData* shapeA = ShapeOf(bodyA, keyA);
            NativeStore.ShapeData* shapeB = ShapeOf(bodyB, keyB);
            if (IgnoreTable.Contains(Globals, shapeA->ignorePairId, shapeB->ignorePairId))
            {
                header.JacobianFlags |= JacobianFlags.Disabled;
                return false;
            }

            if (IsBehindField(bodyA, keyA, shapeA, bodyB, shapeB) || IsBehindField(bodyB, keyB, shapeB, bodyA, shapeA))
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
            records.Add(new ManifoldRecord
            {
                ShapeA = swapped ? shapeB : shapeA,
                ShapeB = swapped ? shapeA : shapeB,
                RowA = swapped ? rowB : rowA,
                RowB = swapped ? rowA : rowB,
                BuiltPair = header.BodyPair,
                BuiltKeys = header.ColliderKeys,
                FirstPoint = firstPoint,
                Normal = normal,
                BuiltNormal = header.Normal,
                LinearVelocityA = first.velocity->Linear,
                AngularVelocityA = StoreOps.WorldAngularVelocity(first),
                LinearVelocityB = second.velocity->Linear,
                AngularVelocityB = StoreOps.WorldAngularVelocity(second),
                Trigger = trigger,
            });

            return true;
        }

        private void Add(ref ContactBucket output, float3 position, float distance, float touchDistance)
        {
            ref ManifoldRecord record = ref output.Records.ElementAt(output.Records.Length - 1);
            output.Points.Add(new ManifoldPoint { Position = position, Distance = distance });
            record.PointCount++;

            // Unity Physics stops a fast body at the surface with a contact that looks well apart, where PhysX lets it in and
            // reports next step. The pair touches in the step that spends the impact, so that its event carries the impulse.
            float closing = record.Trigger ? 0.0f : math.max(ClosingSpeed(ref record, position), 0.0f);
            record.Touching |= distance - closing * DeltaTime <= touchDistance;
        }

        /// <summary>
        /// Gives a manifold the coefficients PhysX would solve it with. A material pair can't say this: Unity Physics has no
        /// multiply, ranks the combine modes differently and keeps one friction, and only here is the number of anchors known.
        /// </summary>
        private void SetCoefficients(ref ContactHeader header, ref ManifoldRecord record, ref UnsafeList<ManifoldPoint> points)
        {
            if (record.Trigger)
            {
                return;
            }

            NativeStore.MaterialData* a = record.ShapeA->material;
            NativeStore.MaterialData* b = record.ShapeB->material;
            float anchors = FrictionAnchors(ref record, ref points);
            float dynamicFriction = Combine(a->dynamicFriction, b->dynamicFriction, a->frictionCombine, b->frictionCombine);
            float staticFriction = Combine(a->staticFriction, b->staticFriction, a->frictionCombine, b->frictionCombine);

            float3 center = PatchCenter(ref record, ref points);
            bool sticks = InFrictionCone(ref record, center, anchors * staticFriction);
            header.CoefficientOfFriction = anchors * (sticks ? staticFriction : dynamicFriction);

            // The solver's own resting test measures against gravity, and the step gives it none, so a body at rest would
            // bounce on a step's worth of gravity. PhysX's test is the threshold.
            bool bounces = ClosingSpeed(ref record, center) > BounceThreshold;
            header.CoefficientOfRestitution = bounces ? Combine(a->bounciness, b->bounciness, a->bounceCombine, b->bounceCombine) : 0.0f;

            // With substeps, a bouncy contact's rows stay shut until it bounces, so they couldn't take back a warm start.
            if (header.CoefficientOfRestitution <= 0.0f)
            {
                LastImpulses.TryGetValue(PersistentManifoldKey.Of(ref record), out record.WarmImpulse);
            }
        }

        /// <summary>
        /// PhysX anchors a patch's friction at its first point and the point farthest from it, and each anchor may spend the
        /// coefficient times the patch's whole normal impulse. So a patch with two anchors holds twice what the coefficient
        /// says, a sphere's single point holds what it says, and Unity Physics' one row at the manifold's centre is given the
        /// sum of the two.
        /// </summary>
        private static float FrictionAnchors(ref ManifoldRecord record, ref UnsafeList<ManifoldPoint> points)
        {
            float3 first = points[record.FirstPoint].Position;
            float farthest = 0.0f;
            for (int i = 1; i < record.PointCount; i++)
            {
                farthest = math.max(farthest, math.distancesq(first, points[record.FirstPoint + i].Position));
            }

            return farthest > FrictionOffsetThreshold * FrictionOffsetThreshold ? 2.0f : 1.0f;
        }

        /// <summary>
        /// Whether this step's velocities stay inside the cone of a coefficient, which decides between static and dynamic
        /// friction. PhysX decides it by the impulse the solve asks for. Both impulses would stop their own velocity through
        /// the same effective mass, so comparing the velocities at the patch's centre comes to the same thing.
        /// </summary>
        private bool InFrictionCone(ref ManifoldRecord record, float3 center, float coefficient)
        {
            float3 velocityA = PointVelocity(ref Store[record.RowA], record.LinearVelocityA, record.AngularVelocityA, center);
            float3 velocityB = PointVelocity(ref Store[record.RowB], record.LinearVelocityB, record.AngularVelocityB, center);
            float3 relative = velocityB - velocityA;
            float closing = math.dot(relative, record.Normal);
            float3 sliding = relative - closing * record.Normal;
            return math.lengthsq(sliding) <= math.square(coefficient * math.max(closing, 0.0f));
        }

        private static float3 PatchCenter(ref ManifoldRecord record, ref UnsafeList<ManifoldPoint> points)
        {
            float3 center = float3.zero;
            for (int i = 0; i < record.PointCount; i++)
            {
                center += points[record.FirstPoint + i].Position;
            }

            return center / record.PointCount;
        }

        /// <summary>PhysX's rule for a pair: the mode that ranks higher wins, and maximum outranks multiply, minimum, average.</summary>
        private static float Combine(float a, float b, uint modeA, uint modeB)
        {
            switch (CombineRank(modeA) >= CombineRank(modeB) ? modeA : modeB)
            {
                case NativeStore.CombineMaximum:
                    return math.max(a, b);
                case NativeStore.CombineMultiply:
                    return a * b;
                case NativeStore.CombineMinimum:
                    return math.min(a, b);
                default:
                    return (a + b) * 0.5f;
            }
        }

        private static int CombineRank(uint mode)
        {
            switch (mode)
            {
                case NativeStore.CombineMaximum:
                    return 3;
                case NativeStore.CombineMultiply:
                    return 2;
                case NativeStore.CombineMinimum:
                    return 1;
                default:
                    return 0;
            }
        }

        private float ClosingSpeed(ref ManifoldRecord record, float3 position)
        {
            float3 velocityA = PointVelocity(ref Store[record.RowA], record.LinearVelocityA, record.AngularVelocityA, position);
            float3 velocityB = PointVelocity(ref Store[record.RowB], record.LinearVelocityB, record.AngularVelocityB, position);
            return math.dot(velocityB - velocityA, record.Normal);
        }

        private static float3 PointVelocity(ref NativeStore.BodyData body, float3 linear, float3 angular, float3 position)
        {
            float3 center = (float3)body.pose->Position + math.rotate(body.pose->Rotation, body.mass->Transform.pos);
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

        /// <summary>
        /// PhysX culls a height field's triangle against the centre of the shape meeting it, so a shape centred below the
        /// surface has no contact, while Unity Physics would drive it on through.
        /// </summary>
        private bool IsBehindField(RigidBody field, ColliderKey key, NativeStore.ShapeData* fieldShape, RigidBody other, NativeStore.ShapeData* otherShape)
        {
            if (fieldShape->geometryType != NativeStore.GeometryHeightField)
            {
                return false;
            }

            ref NativeStore.BodyData row = ref Store[other.Entity.Index];
            ShapeMath.WorldBounds(otherShape, otherShape->localPosition, otherShape->localRotation, row.pose->Position, row.pose->Rotation, out float3 lower, out float3 upper);
            float3 centre = math.transform(math.inverse(field.WorldFromBody), (lower + upper) * 0.5f);
            return QueryCollectors.IsBehind((Collider*)field.Collider.GetUnsafePtr(), key, centre);
        }

        private NativeStore.ShapeData* ShapeOf(RigidBody body, ColliderKey key)
        {
            NativeStore.BodyData* row = Store + body.Entity.Index;
            return row->shapes[ColliderBuilder.ShapeIndexOf(body.Collider, key, ColliderBuilder.FirstSupported(row))];
        }
    }

    /// <summary>
    /// What <see cref="ReadContactsJob"/> kept from one bucket of the narrow phase's stream. A record's first point counts
    /// from the start of its bucket's points.
    /// </summary>
    internal struct ContactBucket
    {
        public UnsafeList<ManifoldRecord> Records;
        public UnsafeList<ManifoldPoint> Points;

        internal void Dispose()
        {
            Records.Dispose();
            Points.Dispose();
        }
    }

    /// <summary>
    /// Gives every bucket of the narrow phase's stream its lists, whose number is only known once the stream is made. The
    /// lists stay from step to step, so this seldom has anything to do.
    /// </summary>
    [BurstCompile]
    internal struct PrepareContactBucketsJob : IJob
    {
        [ReadOnly]
        public NativeStream.Reader Contacts;

        public NativeList<ContactBucket> Buckets;

        public void Execute()
        {
            while (Buckets.Length < Contacts.ForEachCount)
            {
                Buckets.Add(new ContactBucket
                {
                    Records = new UnsafeList<ManifoldRecord>(16, Allocator.Persistent),
                    Points = new UnsafeList<ManifoldPoint>(64, Allocator.Persistent),
                });
            }
        }
    }

    /// <summary>Joins what <see cref="ReadContactsJob"/> kept from each bucket into the step's manifolds, in bucket order.</summary>
    [BurstCompile]
    internal unsafe struct MergeContactBucketsJob : IJob
    {
        [ReadOnly]
        public NativeStream.Reader Contacts;

        [ReadOnly]
        public NativeList<ContactBucket> Buckets;

        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldRecord> Records;

        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldPoint> Points;

        public void Execute()
        {
            for (int bucket = 0; bucket < Contacts.ForEachCount; bucket++)
            {
                ContactBucket kept = Buckets[bucket];
                int firstRecord = Records.Length;
                int pointOffset = Points.Length;
                Records.AddRange(kept.Records.Ptr, kept.Records.Length);
                Points.AddRange(kept.Points.Ptr, kept.Points.Length);
                for (int i = firstRecord; i < Records.Length; i++)
                {
                    Records.ElementAt(i).FirstPoint += pointOffset;
                }
            }
        }
    }

    /// <summary>
    /// Finds each warm start its jacobian's key, once the step's manifolds are known. The value holds the impulse and the
    /// number of contact points that share it.
    /// </summary>
    [BurstCompile]
    internal struct IndexWarmStartsJob : IJob
    {
        [ReadOnly]
        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldRecord> Records;

        public NativeHashMap<JacobianKey, float2> WarmStarts;

        public void Execute()
        {
            WarmStarts.Clear();
            for (int i = 0; i < Records.Length; i++)
            {
                ManifoldRecord record = Records[i];
                if (record.WarmImpulse <= 0.0f)
                {
                    continue;
                }

                JacobianKey key = new JacobianKey
                {
                    BodyA = record.BuiltPair.BodyIndexA,
                    BodyB = record.BuiltPair.BodyIndexB,
                    Normal = record.BuiltNormal,
                };

                WarmStarts.TryGetValue(key, out float2 shared);
                WarmStarts[key] = shared + new float2(record.WarmImpulse, record.PointCount);
            }
        }
    }

    /// <summary>
    /// Starts each contact from most of the normal impulse it ended the last step on, spread evenly over its points, as the
    /// events only give a manifold's total. Unity Physics solves friction in every iteration from the first, and a cold start
    /// has the points of a manifold pulling the bodies about for most of the iterations, which friction takes for slipping
    /// and spends itself on. A body on another dynamic body then keeps whatever slip that leaves, step after step.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WarmStartJob : IJacobiansJob
    {
        // Less than all of it, because a manifold with nothing left to add raises no event and would start cold next step.
        internal const float Fraction = 0.95f;

        [ReadOnly]
        public NativeHashMap<JacobianKey, float2> WarmStarts;

        [NativeDisableUnsafePtrRestriction]
        public MotionVelocity* Motions;

        public int MotionCount;

        public void Execute(ref ModifiableJacobianHeader header, ref ModifiableContactJacobian jacobian)
        {
            JacobianKey key = new JacobianKey { BodyA = header.BodyIndexA, BodyB = header.BodyIndexB, Normal = jacobian.Normal };
            if ((header.Flags & JacobianFlags.Disabled) != 0 || !WarmStarts.TryGetValue(key, out float2 shared))
            {
                return;
            }

            float impulse = Fraction * shared.x / shared.y;
            for (int i = 0; i < jacobian.NumContacts; i++)
            {
                ContactJacAngAndVelToReachCp point = header.GetAngularJacobian(i);
                point.Jac.Impulse = impulse;
                header.SetAngularJacobian(i, point);

                Push(header.BodyIndexA, impulse * jacobian.Normal, impulse * point.Jac.AngularA);
                Push(header.BodyIndexB, -impulse * jacobian.Normal, impulse * point.Jac.AngularB);
            }
        }

        public void Execute(ref ModifiableJacobianHeader header, ref ModifiableTriggerJacobian jacobian)
        {
        }

        private void Push(int body, float3 linearImpulse, float3 angularImpulse)
        {
            if (body < MotionCount)
            {
                ref MotionVelocity motion = ref Motions[body];
                motion.LinearVelocity += linearImpulse * motion.InverseMass;
                motion.AngularVelocity += angularImpulse * motion.InverseInertia;
            }
        }
    }

    /// <summary>Keeps each manifold's impulse for the warm start of the next step.</summary>
    [BurstCompile]
    internal struct RememberImpulsesJob : IJob
    {
        // A heavy body on a light one rings for steps on end, and a start that follows every swing feeds the ringing.
        private const float Smoothing = 0.25f;

        [ReadOnly]
        public NativeList<ManifoldRecord> Records;

        public NativeHashMap<PersistentManifoldKey, float> LastImpulses;

        public void Execute()
        {
            LastImpulses.Clear();
            for (int i = 0; i < Records.Length; i++)
            {
                ManifoldRecord record = Records[i];
                if (!record.Trigger && record.Impulse > 0.0f)
                {
                    float last = record.WarmImpulse > 0.0f ? record.WarmImpulse : record.Impulse;
                    LastImpulses[PersistentManifoldKey.Of(ref record)] = math.lerp(last, record.Impulse, Smoothing);
                }
            }
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
                    ref ManifoldRecord record = ref Records.ElementAt(index);
                    record.Impulse = math.isfinite(impulse) ? impulse + WarmStartJob.Fraction * record.WarmImpulse : 0.0f;
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
    /// Drops the touching pairs of shapes taken off their bodies, which leave with no exit and whose addresses the next
    /// shapes can reuse. What rested on such a shape has nothing holding it now, so PhysX wakes it. That decides what the
    /// step's sleep and contact report see, so this runs before both.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ForgetRemovedShapesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        public NativeHashMap<ShapePairKey, ShapePairState> Touching;

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
                    Touching.Remove(key);
                    WakeOwner(firstGone ? key.Second : key.First, firstGone && secondGone);
                }
            }
        }

        private static void WakeOwner(IntPtr shape, bool gone)
        {
            NativeStore.Shape* left = (NativeStore.Shape*)shape;
            if (!gone && left->body != null)
            {
                JointKernel.WakeRow(StoreOps.Hot(left->body));
            }
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
                    NativeStore.ContactBodyPair* pair = AddBodyPair(rowFirst, rowSecond, a->velocity->Linear, StoreOps.WorldAngularVelocity(*a), b->velocity->Linear, StoreOps.WorldAngularVelocity(*b));
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
