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
    /// Stores a manifold that survived filtering in the narrow phase. Normalize the order of its shapes regardless of
    /// how Unity Physics built the pair: body A has the lower store row, except that a static body is always body B.
    /// Collision.impulse is one vector shared by both sides, and PhysX defines it as pushing the body. Retain Unity
    /// Physics' original keys to find the manifold's solver event. A pair of shapes can have several manifolds, one for
    /// each mesh triangle.
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

        // Points from B toward A. A reported contact normal points from the other shape toward the shape receiving the
        // event.
        public float3 Normal;
        public float3 LinearVelocityA;
        public float3 AngularVelocityA;
        public float3 LinearVelocityB;
        public float3 AngularVelocityB;

        // The original normal from the narrow phase, as used by the solver's jacobian.
        public float3 BuiltNormal;

        // The cached impulse from the previous step and the final impulse for the current step. Apply
        // WarmStartJob.Fraction of the cached impulse during the warm start.
        public float WarmImpulse;
        public float Impulse;
        public bool Touching;
        public bool Trigger;
    }

    /// <summary>
    /// Stores a contact point of a <see cref="ManifoldRecord"/> in the form produced by the narrow phase.
    /// </summary>
    internal struct ManifoldPoint
    {
        public float3 Position;
        public float Distance;
    }

    /// <summary>
    /// Identifies a manifold's record from its solver event.
    /// </summary>
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

    /// <summary>
    /// Identifies the same manifold across steps, even when body indices in the built world change.
    /// </summary>
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
    /// Identifies a manifold from its jacobian, which contains body indices but no collider keys. Manifolds with the
    /// same body pair and normal use the same key and share their impulse. For example, this occurs when a box rests on
    /// two coplanar triangles.
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

    /// <summary>
    /// Identifies a pair of touching shapes in pointer order, so that the key stays the same as bodies are added or
    /// removed.
    /// </summary>
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

    /// <summary>
    /// Stores the state of a touching pair between steps.
    /// </summary>
    internal struct ShapePairState
    {
        public uint LastSeen;
        public bool Trigger;
    }

    /// <summary>
    /// Caches whether every collider pair between two bodies is ignored. Records the versions of the ignored pairs and
    /// collider builds used to calculate that result.
    /// </summary>
    internal struct KnownBodyPair
    {
        public uint IgnoreVersion;
        public uint WholeBuilds;
        public bool Ignored;
    }

    /// <summary>
    /// Disables collisions for overlapping body pairs when every collider pair between them is ignored through
    /// Physics.IgnoreCollision. This is common among parts of one craft. Such overlaps greatly outnumber joints.
    /// Leaving them enabled would put the craft in one crowded island and send all of them through the narrow phase
    /// unnecessarily. Keep body pairs that still have colliders allowed to contact each other. <see
    /// cref="ReadContactsJob"/> filters the individual collider pairs.
    ///
    /// Checking a body pair requires a lookup for each collider pair. Each lookup is likely to miss the cache because
    /// the table contains all ignored pairs for the craft. Cache the result for each body pair until the ignored pairs
    /// change or any body's shapes may have changed. Creating a body or changing its shapes requires a full collider
    /// build, which increments the build counter recorded with the cached result.
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

        /// <summary>
        /// Stores results calculated during this step for <see cref="RememberBodyPairsJob"/> to cache.
        /// </summary>
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
    /// Caches the results calculated by <see cref="DropIgnoredPairsJob"/>. Cached results remain when pairs stop
    /// overlapping. If the number of cached and newly calculated results exceeds 32 per body plus 1024, clear the cache
    /// before storing the new results. Later lookups recalculate any results that are needed.
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
    /// Processes every point of every manifold between the narrow phase and the solver. Applies Physics.IgnoreCollision
    /// to collider pairs within body pairs retained by <see cref="DropIgnoredPairsJob"/>. Manifolds identify the leaf
    /// colliders that touched, so individual pairs within a compound can be filtered here. Also sets friction and
    /// restitution using both shapes' materials and the contact points. The retained manifolds are used for events and
    /// sleeping during this step.
    ///
    /// Unity Physics generates contacts with static meshes up to one step's travel ahead of a body. PhysX generates
    /// them only within the shapes' contact offsets. At a polygon edge, the normal of a distant contact can point back
    /// along the body's motion. Treating that contact as a plane would stop a body sliding along the mesh. For a point
    /// beyond the contact offsets, test whether its path during the step intersects its polygon. If the path is no
    /// longer than the separation, leave the contact as Unity Physics generated it. The point cannot close by more
    /// than its separation, regardless of the normal's direction.
    ///
    /// Read each bucket from the narrow phase into separate lists. <see cref="MergeContactBucketsJob"/> then merges
    /// them in bucket order, preserving the order that a single pass over the stream would produce.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ReadContactsJob : IJobParallelForDefer
    {
        // Measurements against PhysX: a 0.015 m box slides on one anchor and a 0.02 m box slides on two. Their
        // diagonals are 0.021 m and 0.028 m. Points separated by at most this threshold share an anchor.
        private const float FrictionOffsetThreshold = 0.025f;

        // Set this separation for solver points that cannot reach their polygons during this step. Unity Physics then
        // allows the bodies to approach by more than they can move. The direct solver creates no row for the point
        // because this separation exceeds Solver.ContactRowReach.
        private const float OutOfReach = 1.0e6f;

        /// <summary>
        /// Stores a static mesh polygon in world space and the motion of the body in contact with it.
        /// </summary>
        private struct MeshFace
        {
            public bool Valid;
            public bool MovingIsA;
            public bool Quad;
            public float3 V0;
            public float3 V1;
            public float3 V2;
            public float3 V3;
            public float3 Linear;
            public float3 Angular;
            public float3 CentreOfMass;
        }

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
            output.Wakes = false;

            // The narrow phase writes each manifold as a header followed by its points.
            NativeStream.Reader reader = Contacts;
            reader.BeginForEachIndex(bucket);
            while (reader.RemainingItemCount > 0)
            {
                ContactHeader* header = (ContactHeader*)reader.ReadUnsafePtr(sizeof(ContactHeader));
                bool kept = Begin(ref *header, ref output.Records, output.Points.Length, out float touchDistance,
                    out MeshFace face);
                for (int i = 0; i < header->NumContacts; i++)
                {
                    ContactPoint* point = (ContactPoint*)reader.ReadUnsafePtr(sizeof(ContactPoint));
                    if (!kept)
                    {
                        continue;
                    }

                    // Preserve the original point in the report. Change only the separation sent to the solver to put
                    // it out of reach.
                    float distance = point->Distance;
                    float3 moving = face.MovingIsA ? point->Position + header->Normal * distance : point->Position;
                    bool reachable = !face.Valid || distance <= touchDistance || Reaches(face, moving, distance);
                    if (!reachable)
                    {
                        point->Distance = OutOfReach;
                    }

                    Add(ref output, point->Position, distance, touchDistance, reachable);
                }

                if (kept)
                {
                    ref ManifoldRecord record = ref output.Records.ElementAt(output.Records.Length - 1);
                    SetCoefficients(ref *header, ref record, ref output.Points);
                    output.Wakes |= record.Touching && !record.Trigger
                        && WakeTouchedBodiesJob.Wakes(Store + record.RowA, Store + record.RowB);
                }
            }

            reader.EndForEachIndex();
        }

        /// <summary>
        /// Starts a manifold record. Returns false if the manifold is disabled or skipped.
        /// </summary>
        private bool Begin(ref ContactHeader header, ref UnsafeList<ManifoldRecord> records, int firstPoint,
            out float touchDistance, out MeshFace face)
        {
            touchDistance = 0.0f;
            face = default;
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
            if (!trigger && staticA != staticB)
            {
                face = staticB ? FaceOf(bodyB, keyB, Store[rowA], true) : FaceOf(bodyA, keyA, Store[rowB], false);
            }

            bool swapped = staticA != staticB ? staticA : rowA > rowB;
            float3 normal = swapped ? -header.Normal : header.Normal;
            ref NativeStore.BodyData first = ref Store[swapped ? rowB : rowA];
            ref NativeStore.BodyData second = ref Store[swapped ? rowA : rowB];

            // Triggers require actual overlap. Solid pairs follow PhysX's rule and may touch within the sum of their
            // contact offsets.
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

        private void Add(ref ContactBucket output, float3 position, float distance, float touchDistance, bool reachable)
        {
            ref ManifoldRecord record = ref output.Records.ElementAt(output.Records.Length - 1);
            output.Points.Add(new ManifoldPoint { Position = position, Distance = distance });
            record.PointCount++;

            // Unity Physics stops a fast body at the surface using a contact with a large separation. PhysX allows the
            // body to enter and reports the contact next step. Mark the pair as touching during the step that applies
            // the impact impulse so that the event includes that impulse.
            float closing = record.Trigger ? 0.0f : math.max(ClosingSpeed(ref record, position), 0.0f);
            record.Touching |= reachable && distance - closing * DeltaTime <= touchDistance;
        }

        /// <summary>
        /// Returns the static mesh polygon for a manifold in world space, along with the motion of the body in contact
        /// with it. Returns an invalid face for any other leaf collider.
        /// </summary>
        private MeshFace FaceOf(RigidBody mesh, ColliderKey key, in NativeStore.BodyData moving, bool movingIsA)
        {
            var root = (Collider*)mesh.Collider.GetUnsafePtr();
            if (!Collider.GetLeafCollider(out ChildCollider leaf, root, key, mesh.WorldFromBody))
            {
                return default;
            }

            Collider* collider = leaf.Collider;
            if (collider->Type != ColliderType.Triangle && collider->Type != ColliderType.Quad)
            {
                return default;
            }

            BlobArray.Accessor<float3> vertices = ((PolygonCollider*)collider)->Vertices;
            bool quad = collider->Type == ColliderType.Quad;
            return new MeshFace
            {
                Valid = true,
                MovingIsA = movingIsA,
                Quad = quad,
                V0 = math.transform(leaf.TransformFromChild, vertices[0]),
                V1 = math.transform(leaf.TransformFromChild, vertices[1]),
                V2 = math.transform(leaf.TransformFromChild, vertices[2]),
                V3 = quad ? math.transform(leaf.TransformFromChild, vertices[3]) : float3.zero,
                Linear = moving.velocity->Linear,
                Angular = StoreOps.WorldAngularVelocity(moving),
                CentreOfMass = moving.pose->Position + math.rotate(moving.pose->Rotation, moving.mass->Transform.pos),
            };
        }

        /// <summary>
        /// Tests whether a moving point's path during this step intersects the polygon. If the path is no longer than
        /// the separation, return true to retain Unity Physics' contact. The point cannot close by more than its
        /// separation, regardless of the normal's direction.
        /// </summary>
        private bool Reaches(in MeshFace face, float3 point, float distance)
        {
            float3 path = (face.Linear + math.cross(face.Angular, point - face.CentreOfMass)) * DeltaTime;
            if (math.length(path) <= distance)
            {
                return true;
            }

            float fraction = 1.0f;
            if (RaycastQueries.RayTriangle(point, path, face.V0, face.V1, face.V2, ref fraction, out _))
            {
                return true;
            }

            fraction = 1.0f;
            return face.Quad && RaycastQueries.RayTriangle(point, path, face.V0, face.V2, face.V3, ref fraction, out _);
        }

        /// <summary>
        /// Sets the coefficients that PhysX would use to solve this manifold. Unity Physics material pairs cannot
        /// express these rules: they lack a multiply mode, rank the combine modes differently, and store only one
        /// friction coefficient. The number of friction anchors is also known only here.
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

            // The solver's resting test depends on gravity, but no gravity is supplied to it during this step. A
            // resting body would therefore bounce after gaining one step's velocity from gravity. Use PhysX's speed
            // threshold to decide whether it bounces.
            bool bounces = ClosingSpeed(ref record, center) > BounceThreshold;
            header.CoefficientOfRestitution = bounces ? Combine(a->bounciness, b->bounciness, a->bounceCombine, b->bounceCombine) : 0.0f;

            // With substeps, rows for a bouncy contact remain inactive until the bounce, so they cannot remove an
            // applied warm start.
            if (header.CoefficientOfRestitution <= 0.0f)
            {
                LastImpulses.TryGetValue(PersistentManifoldKey.Of(ref record), out record.WarmImpulse);
            }
        }

        /// <summary>
        /// PhysX places friction anchors at a patch's first point and at the point farthest from it. Each anchor can
        /// apply friction up to the coefficient times the patch's total normal impulse. Two anchors therefore provide
        /// twice the friction implied by the coefficient alone. A sphere's single contact point provides that amount
        /// once. Scale the coefficient for Unity Physics' single friction row at the manifold's centre to account for
        /// both anchors.
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
        /// Chooses static or dynamic friction by testing velocities against the cone for the given coefficient.
        /// PhysX uses the impulse required by the solver to make this choice. This method approximates that decision
        /// using relative velocity at the patch's centre. Compare tangential speed with the coefficient times the
        /// nonnegative closing speed.
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

        /// <summary>
        /// Combines a pair using PhysX's higher-ranked mode. The ranking is maximum, multiply, minimum, then average.
        /// </summary>
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
        /// Under the scene's default contact mode, PhysX reports a kinematic body paired with a static or another
        /// kinematic body only for triggers. It never reports two static bodies. Unity Physics also applies no solver
        /// response to these pairs, but it does generate their contacts.
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
        /// PhysX culls a height field triangle using the centre of the shape in contact with it. A shape centred below
        /// the surface therefore has no contact, whereas Unity Physics would push it farther through the surface.
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
    /// Stores the records and points retained by <see cref="ReadContactsJob"/> from one bucket of the narrow phase
    /// stream. A record's first point is indexed relative to the start of that bucket's points.
    /// </summary>
    internal struct ContactBucket
    {
        public UnsafeList<ManifoldRecord> Records;
        public UnsafeList<ManifoldPoint> Points;

        /// <summary>
        /// Indicates whether a manifold contains a moving body touching a sleeping body and should wake it during
        /// contact processing.
        /// </summary>
        public bool Wakes;

        internal void Dispose()
        {
            Records.Dispose();
            Points.Dispose();
        }
    }

    /// <summary>
    /// Allocates lists for each bucket of the narrow phase stream. The number of buckets is known only after the stream
    /// is created. Reuse the lists between steps, so allocation is rarely needed. Also records how many buckets are
    /// used this step for jobs that read them after the stream has been released.
    /// </summary>
    [BurstCompile]
    internal struct PrepareContactBucketsJob : IJob
    {
        [ReadOnly]
        public NativeStream.Reader Contacts;

        public NativeList<ContactBucket> Buckets;

        public NativeReference<int> Filled;

        public void Execute()
        {
            Filled.Value = Contacts.ForEachCount;
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

    /// <summary>
    /// Merges the records retained by <see cref="ReadContactsJob"/> into this step's manifolds, in bucket order.
    /// </summary>
    [BurstCompile]
    internal unsafe struct MergeContactBucketsJob : IJob
    {
        [ReadOnly]
        public NativeReference<int> Filled;

        [ReadOnly]
        public NativeList<ContactBucket> Buckets;

        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldRecord> Records;

        [NativeDisableContainerSafetyRestriction]
        public NativeList<ManifoldPoint> Points;

        public void Execute()
        {
            for (int bucket = 0; bucket < Filled.Value; bucket++)
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
    /// Indexes warm starts by their jacobian keys after the manifolds for this step are known. Each value contains the
    /// total impulse and the number of contact points that share it.
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
    /// Starts each contact with most of its final normal impulse from the previous step. Divide the impulse evenly
    /// among contact points, applying it only to points close enough to touch, because events report only the
    /// manifold's total impulse. Unity Physics solves friction on every iteration, including the first. With a cold
    /// start, the contact points move the bodies around during most iterations. The friction calculation treats this
    /// motion as slipping and uses friction to oppose it. A body resting on another dynamic body can then retain the
    /// remaining slip from step to step.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WarmStartJob : IJacobiansJob
    {
        // Use less than the full impulse. A manifold that needs no additional impulse produces no event and would start
        // cold next step.
        internal const float Fraction = 0.95f;

        // Apply a share of the warm start only to points within this distance, in metres, because the impulse came from
        // touching points. Every such point must have a solver row so that the solver can remove any warm start impulse
        // that is no longer needed. Without a row, that impulse would push the bodies apart and increase the next
        // step's warm start, making crash debris accelerate after each step. The direct solver creates rows up to
        // Solver.ContactRowReach, which must be at least this distance.
        private const float FarthestPoint = 0.1f;

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
                if (point.ContactDistance > FarthestPoint)
                {
                    continue;
                }

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

    /// <summary>
    /// Stores each manifold's impulse for the next step's warm start.
    /// </summary>
    [BurstCompile]
    internal struct RememberImpulsesJob : IJob
    {
        // A heavy body on a light one can oscillate for many steps. Following each oscillation in the warm start would
        // reinforce it.
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
    /// Records the solver impulse for each manifold. A step with no dynamic bodies has no solver execution or event
    /// stream, so this job does not run for such a step.
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
    /// Removes touching pairs that involve shapes detached from their bodies. These shapes produce no exit event, and
    /// new shapes may reuse their addresses. PhysX wakes a body when the shape supporting it is removed. Do the same
    /// here before sleeping and contact reporting, since both depend on whether the body is awake.
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
    /// Converts this step's manifolds into the report that Unity reads through GetLastSimulationStepResults. Matches
    /// the PhysX report: each body pair includes its velocities before solving, its shape pairs, and their contact
    /// points. Compare against the previous step's pairs to detect new and lost touches. Bodies at rest remain touching
    /// without further reports, as a sleeping island does in PhysX. Report only trigger enters and exits because Unity
    /// repeats OnTriggerStay between them.
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

        /// <summary>
        /// Reserves all required space before building the report because its pairs contain pointers into these arrays.
        /// </summary>
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

        /// <summary>
        /// Reports lost touches for pairs absent from this step. Retains pairs while both bodies are at rest, until
        /// either wakes.
        /// </summary>
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

        /// <summary>
        /// Orders manifolds by body pair, then by shape pair within each body pair, so that related entries are
        /// adjacent.
        /// </summary>
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
