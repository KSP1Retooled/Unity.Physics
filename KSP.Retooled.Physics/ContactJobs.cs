using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
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

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Contact manifold retained after filtering, with shapes ordered by body row and statics assigned to body B.
    /// Keep original collider keys to match solver events. Several manifolds may belong to one shape pair.
    /// </summary>
    internal unsafe struct ManifoldRecord
    {
        public ShapeData* ShapeA;
        public ShapeData* ShapeB;
        public int RowA;
        public int RowB;

        // The positions of ShapeA and ShapeB in their bodies' lists of shapes. The contact report sorts the shape pairs of
        // a body pair by these, because the shapes' addresses differ from run to run.
        public int ShapeIndexA;
        public int ShapeIndexB;

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
        public ShapeData* ShapeA;
        public ShapeData* ShapeB;
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
    /// Identify a manifold by body pair and normal when the Jacobian has no collider keys.
    /// Coplanar manifolds with the same key share their impulse.
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

        internal static unsafe ShapePairKey Of(ShapeData* a, ShapeData* b)
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

        /// <summary>
        /// Whether the first shape was the trigger. Removal events use this cached value after shape memory is
        /// freed.
        /// </summary>
        public bool FirstIsTrigger;

        /// <summary>
        /// Cached kinematic flags for both bodies, plus KinematicKnown when initialized.
        /// Changes restart the touching pair to match PhysX.
        /// </summary>
        public byte Kinematic;

        internal const byte FirstKinematic = 1;
        internal const byte SecondKinematic = 2;
        internal const byte KinematicKnown = 4;

        /// <summary>
        /// Return current kinematic flags for the pair's bodies. A shape without a body counts as nonkinematic.
        /// </summary>
        internal static unsafe byte KinematicOf(ShapePairKey key)
        {
            return (byte)(KinematicKnown
                | (IsKinematic((Shape*)key.First) ? FirstKinematic : 0)
                | (IsKinematic((Shape*)key.Second) ? SecondKinematic : 0));
        }

        /// <summary>
        /// Return whether either body's kinematic state changed. Treat an unknown previous state as unchanged.
        /// </summary>
        internal readonly bool KinematicChanged(byte now)
        {
            return (Kinematic & KinematicKnown) != 0 && Kinematic != now;
        }

        private static unsafe bool IsKinematic(Shape* shape)
        {
            return shape->body != null && (StoreOps.Hot(shape->body)->flags & BodyFlags.Kinematic) != 0;
        }
    }

    /// <summary>
    /// Cached collision filtering result for a body pair, with filter/build versions and body serials.
    /// Serials distinguish bodies that reused the same store rows.
    /// </summary>
    internal struct KnownBodyPair
    {
        public uint IgnoreVersion;
        public uint WholeBuilds;
        public uint SerialLow;
        public uint SerialHigh;
        public bool Ignored;
    }

    /// <summary>
    /// Remove overlapping body pairs when all collider pairs are ignored by Physics.IgnoreCollision or collision
    /// groups.
    /// Cache results until filters or collider builds change. ReadContactsJob filters individual collider pairs.
    /// </summary>
    [BurstCompile]
    internal unsafe struct DropIgnoredPairsJob : IJobParallelForDefer
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<DispatchPairSequencer.DispatchPair> Pairs;

        [NativeDisableUnsafePtrRestriction]
        public BodyData* Store;

        [NativeDisableUnsafePtrRestriction]
        public StoreGlobals* Globals;

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

        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        /// <summary>The joint units, to find the leading joint of a vessel's edge or closure.</summary>
        [ReadOnly]
        public NativeArray<JointUnit> Units;

        /// <summary>See <see cref="VesselBuild.BodyVessels"/>. Empty when vessels are off.</summary>
        [ReadOnly]
        public NativeArray<int> BodyVessels;

        /// <summary>For each vessel, 1 when it takes part in this step.</summary>
        [ReadOnly]
        public NativeArray<byte> VesselActive;

        /// <summary>See <see cref="VesselBuild.PairUnits"/>.</summary>
        [ReadOnly]
        public NativeParallelMultiHashMap<ulong, int> VesselPairUnits;

        internal struct KeyValue
        {
            public ulong Key;
            public KnownBodyPair Value;
        }

        public void Execute(int index)
        {
            DispatchPairSequencer.DispatchPair pair = Pairs[index];
            if (pair.IsValid && pair.IsContact && pair.IsCollisionEnabled && (IsVesselJointed(pair, index) || IsIgnored(pair)))
            {
                pair.DisableCollision();
                Pairs[index] = pair;
            }
        }

        /// <summary>
        /// Return whether an active vessel joint disables collision for this body pair.
        /// Preserve the pair when another joint between the bodies allows collision.
        /// </summary>
        private bool IsVesselJointed(DispatchPairSequencer.DispatchPair pair, int index)
        {
            if (BodyVessels.Length == 0)
            {
                return false;
            }

            Entity entityA = Bodies[pair.BodyIndexA].Entity;
            Entity entityB = Bodies[pair.BodyIndexB].Entity;
            int rowA = entityA.Index;
            int rowB = entityB.Index;
            if (entityA.Version == 0 || entityB.Version == 0 || rowA >= BodyVessels.Length || rowB >= BodyVessels.Length)
            {
                return false;
            }

            int vessel = BodyVessels[rowA];
            if (vessel < 0 || vessel != BodyVessels[rowB] || VesselActive[vessel] == 0)
            {
                return false;
            }

            if (!VesselPairUnits.TryGetFirstValue(VesselBuild.PairKey(rowA, rowB), out int unit, out NativeParallelMultiHashMapIterator<ulong> iterator))
            {
                return false;
            }

            bool disabled = false;
            do
            {
                disabled |= (World->joints[Units[unit].Joint]->flags & StoreConstants.JointFlagCollisionEnabled) == 0;
            }
            while (!disabled && VesselPairUnits.TryGetNextValue(out unit, ref iterator));

            if (!disabled || index == 0)
            {
                return disabled;
            }

            DispatchPairSequencer.DispatchPair previous = Pairs[index - 1];
            bool permittedByJoint = previous.IsJoint && previous.BodyIndexA == pair.BodyIndexA && previous.BodyIndexB == pair.BodyIndexB;
            return !permittedByJoint;
        }

        private bool IsIgnored(DispatchPairSequencer.DispatchPair pair)
        {
            uint rowA = (uint)Bodies[pair.BodyIndexA].Entity.Index;
            uint rowB = (uint)Bodies[pair.BodyIndexB].Entity.Index;

            // When one of the bodies has no collider with an ignore id, only the collision groups can stop the pair.
            // Reading the groups costs less than the cache lookup, so the result isn't cached.
            if (!HasIgnoreIds(Store + rowA) || !HasIgnoreIds(Store + rowB))
            {
                return IsIgnored(Store + rowA, Store + rowB);
            }

            uint low = math.min(rowA, rowB);
            uint high = math.max(rowA, rowB);
            ulong key = ((ulong)low << 32) | high;
            uint ignoreVersion = Globals->ignoreVersion;
            uint wholeBuilds = WholeBuilds.Value;

            // When a body is destroyed, the store moves its last body into the destroyed body's row. That changes neither
            // version, so the serial numbers tell whether the rows still hold the bodies that the entry was made for.
            uint serialLow = Store[low].owner->serial;
            uint serialHigh = Store[high].owner->serial;
            if (Known.TryGetValue(key, out KnownBodyPair known) && known.IgnoreVersion == ignoreVersion && known.WholeBuilds == wholeBuilds
                && known.SerialLow == serialLow && known.SerialHigh == serialHigh)
            {
                return known.Ignored;
            }

            bool ignored = IsIgnored(Store + rowA, Store + rowB);
            Learned.Enqueue(new KeyValue
            {
                Key = key,
                Value = new KnownBodyPair
                {
                    IgnoreVersion = ignoreVersion,
                    WholeBuilds = wholeBuilds,
                    SerialLow = serialLow,
                    SerialHigh = serialHigh,
                    Ignored = ignored,
                },
            });
            return ignored;
        }

        private static bool HasIgnoreIds(BodyData* row)
        {
            for (int i = 0; i < row->shapeCount; i++)
            {
                if (row->shapes[i]->ignorePairId != 0)
                {
                    return true;
                }
            }

            return false;
        }

        private bool IsIgnored(BodyData* rowA, BodyData* rowB)
        {
            for (int a = 0; a < rowA->shapeCount; a++)
            {
                ShapeData* shapeA = rowA->shapes[a];
                if (!ColliderBuilder.IsSupported(shapeA))
                {
                    continue;
                }

                for (int b = 0; b < rowB->shapeCount; b++)
                {
                    ShapeData* shapeB = rowB->shapes[b];
                    if (ColliderBuilder.IsSupported(shapeB) && !IgnoreTable.Ignores(Globals, shapeA, shapeB))
                    {
                        return false;
                    }
                }
            }

            return true;
        }
    }

    /// <summary>
    /// Store newly computed body-pair filtering results. Clear the cache if it exceeds 32 entries per body plus
    /// 1024.
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
    /// Filter contact manifolds, set material coefficients, and retain contacts for events and sleeping.
    /// Reject speculative mesh contacts whose paths cannot reach their polygons this step.
    /// Read buckets independently; MergeContactBucketsJob restores stream order.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ReadContactsJob : IJobParallelForDefer
    {
        // Measurements against PhysX: a 0.015 m box slides on one anchor and a 0.02 m box slides on two. Their
        // diagonals are 0.021 m and 0.028 m. Points separated by at most this threshold share an anchor.
        private const float FrictionOffsetThreshold = 0.025f;

        // Separation assigned to unreachable mesh contacts. Must exceed Solver.ContactRowReach so that no solver
        // row is created.
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
        public BodyData* Store;

        [NativeDisableUnsafePtrRestriction]
        public StoreGlobals* Globals;

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
                    var point = (Unity.Physics.ContactPoint*)reader.ReadUnsafePtr(sizeof(Unity.Physics.ContactPoint));
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
            ShapeData* shapeA = ShapeOf(bodyA, keyA, out int indexA);
            ShapeData* shapeB = ShapeOf(bodyB, keyB, out int indexB);
            if (IgnoreTable.Ignores(Globals, shapeA, shapeB))
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

            bool staticA = (Store[rowA].flags & BodyFlags.Static) != 0;
            bool staticB = (Store[rowB].flags & BodyFlags.Static) != 0;
            if (!trigger && staticA != staticB)
            {
                face = staticB ? FaceOf(bodyB, keyB, Store[rowA], true) : FaceOf(bodyA, keyA, Store[rowB], false);
            }

            bool swapped = staticA != staticB ? staticA : rowA > rowB;
            float3 normal = swapped ? -header.Normal : header.Normal;
            ref BodyData first = ref Store[swapped ? rowB : rowA];
            ref BodyData second = ref Store[swapped ? rowA : rowB];

            // Triggers require actual overlap. Solid pairs follow PhysX's rule and may touch within the sum of their
            // contact offsets.
            touchDistance = trigger ? 0.0f : ((Shape*)shapeA)->contactOffset + ((Shape*)shapeB)->contactOffset;
            records.Add(new ManifoldRecord
            {
                ShapeA = swapped ? shapeB : shapeA,
                ShapeB = swapped ? shapeA : shapeB,
                RowA = swapped ? rowB : rowA,
                RowB = swapped ? rowA : rowB,
                ShapeIndexA = swapped ? indexB : indexA,
                ShapeIndexB = swapped ? indexA : indexB,
                BuiltPair = header.BodyPair,
                BuiltKeys = header.ColliderKeys,
                FirstPoint = firstPoint,
                Normal = normal,
                BuiltNormal = header.Normal,
                LinearVelocityA = first.Velocity()->Linear,
                AngularVelocityA = BodyKinematics.WorldAngularVelocity(first),
                LinearVelocityB = second.Velocity()->Linear,
                AngularVelocityB = BodyKinematics.WorldAngularVelocity(second),
                Trigger = trigger,
            });

            return true;
        }

        private void Add(ref ContactBucket output, float3 position, float distance, float touchDistance, bool reachable)
        {
            ref ManifoldRecord record = ref output.Records.ElementAt(output.Records.Length - 1);
            output.Points.Add(new ManifoldPoint { Position = position, Distance = distance });
            record.PointCount++;

            // Report speculative impacts in the step that applies their impulse so that the event includes that
            // impulse.
            float closing = record.Trigger ? 0.0f : math.max(ClosingSpeed(ref record, position), 0.0f);
            record.Touching |= reachable && distance - closing * DeltaTime <= touchDistance;
        }

        /// <summary>
        /// Return a static mesh contact polygon in world space and the other body's motion.
        /// Return an invalid face for other leaf types.
        /// </summary>
        private MeshFace FaceOf(RigidBody mesh, ColliderKey key, in BodyData moving, bool movingIsA)
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
                Linear = moving.Velocity()->Linear,
                Angular = BodyKinematics.WorldAngularVelocity(moving),
                CentreOfMass = moving.Pose()->Position + math.rotate(moving.Pose()->Rotation, moving.Mass()->Transform.pos),
            };
        }

        /// <summary>
        /// Test whether the point can reach its polygon this step.
        /// Retain Unity Physics contacts when travel is no greater than separation.
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
        /// Set PhysX-compatible friction and restitution, including material combine modes and friction anchor
        /// counts.
        /// </summary>
        private void SetCoefficients(ref ContactHeader header, ref ManifoldRecord record, ref UnsafeList<ManifoldPoint> points)
        {
            if (record.Trigger)
            {
                return;
            }

            MaterialData* a = record.ShapeA->material;
            MaterialData* b = record.ShapeB->material;
            float anchors = FrictionAnchors(ref record, ref points);
            float dynamicFriction = Combine(a->dynamicFriction, b->dynamicFriction, a->frictionCombine, b->frictionCombine);
            float staticFriction = Combine(a->staticFriction, b->staticFriction, a->frictionCombine, b->frictionCombine);

            float3 center = PatchCenter(ref record, ref points);
            bool sticks = InFrictionCone(ref record, center, anchors * staticFriction);
            header.CoefficientOfFriction = anchors * (sticks ? staticFriction : dynamicFriction);

            // Use the PhysX bounce speed threshold. The solver's gravity-based resting test has no gravity
            // input here.
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
        /// Scale friction for PhysX's two anchors, using the first and farthest patch points.
        /// A single-point contact, such as a sphere, has one anchor.
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
        /// Approximate the static friction cone using relative tangential and closing speeds at the patch centre.
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
                case StoreConstants.CombineMaximum:
                    return math.max(a, b);
                case StoreConstants.CombineMultiply:
                    return a * b;
                case StoreConstants.CombineMinimum:
                    return math.min(a, b);
                default:
                    return (a + b) * 0.5f;
            }
        }

        private static int CombineRank(uint mode)
        {
            switch (mode)
            {
                case StoreConstants.CombineMaximum:
                    return 3;
                case StoreConstants.CombineMultiply:
                    return 2;
                case StoreConstants.CombineMinimum:
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

        private static float3 PointVelocity(ref BodyData body, float3 linear, float3 angular, float3 position)
        {
            float3 center = (float3)body.Pose()->Position + math.rotate(body.Pose()->Rotation, body.Mass()->Transform.pos);
            return linear + math.cross(angular, position - center);
        }

        /// <summary>
        /// Match PhysX contact reporting: exclude static-static pairs, and report kinematic-static or
        /// kinematic-kinematic pairs only for triggers.
        /// </summary>
        private static bool CanReport(BodyFlags a, BodyFlags b, bool trigger)
        {
            const BodyFlags immobile = BodyFlags.Static | BodyFlags.Kinematic;
            if (trigger)
            {
                return ((a & b) & BodyFlags.Static) == 0;
            }

            return (a & immobile) == 0 || (b & immobile) == 0;
        }

        /// <summary>
        /// Cull height field contacts when the other shape's centre lies below the surface, matching PhysX.
        /// </summary>
        private bool IsBehindField(RigidBody field, ColliderKey key, ShapeData* fieldShape, RigidBody other, ShapeData* otherShape)
        {
            if (fieldShape->geometryType != StoreConstants.GeometryHeightField)
            {
                return false;
            }

            ref BodyData row = ref Store[other.Entity.Index];
            ShapeMassProperties.WorldBounds(otherShape, otherShape->localPosition, otherShape->localRotation, row.Pose()->Position, row.Pose()->Rotation, out float3 lower, out float3 upper);
            float3 centre = math.transform(math.inverse(field.WorldFromBody), (lower + upper) * 0.5f);
            return QueryCollectors.IsBehind((Collider*)field.Collider.GetUnsafePtr(), key, centre);
        }

        private ShapeData* ShapeOf(RigidBody body, ColliderKey key, out int index)
        {
            BodyData* row = Store + body.Entity.Index;
            index = ColliderBuilder.ShapeIndexOf(body.Collider, key, ColliderBuilder.FirstSupported(row));
            return row->shapes[index];
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
    /// Prepare reusable lists for the narrow phase stream's buckets and record the count for later jobs.
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
    /// Index previous total impulses and shared contact counts by Jacobian key after this step's manifolds are
    /// known.
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
    /// Warm start nearby contact points with an equal share of the previous manifold's normal impulse.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WarmStartJob : IJacobiansJob
    {
        // Use less than the full impulse. A manifold that needs no additional impulse produces no event and would start
        // cold next step.
        internal const float Fraction = 0.95f;

        // Maximum separation, in metres, for warm starting. Must not exceed Solver.ContactRowReach:
        // each warmed point needs a row that can remove excess impulse.
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

        // The job reads only the motions. The safety system would otherwise stop it while the dynamic tree is rebuilt,
        // which writes the built bodies.
        [ReadOnly]
        [NativeDisableContainerSafetyRestriction]
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
    /// Discard cached impulses for detached shapes before their addresses can be reused.
    /// Leave the removal list for ForgetRemovedShapesJob.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ForgetRemovedImpulsesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        public NativeHashMap<PersistentManifoldKey, float> LastImpulses;

        public void Execute()
        {
            var removed = new NativeHashSet<IntPtr>((int)World->removedShapeCount, Allocator.Temp);
            for (uint i = 0; i < World->removedShapeCount; i++)
            {
                removed.Add((IntPtr)World->removedShapes[i]);
            }

            NativeArray<PersistentManifoldKey> keys = LastImpulses.GetKeyArray(Allocator.Temp);
            foreach (PersistentManifoldKey key in keys)
            {
                if (removed.Contains((IntPtr)key.ShapeA) || removed.Contains((IntPtr)key.ShapeB))
                {
                    LastImpulses.Remove(key);
                }
            }
        }
    }

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
            if (shape->body != null && (StoreOps.Hot(shape->body)->flags & BodyFlags.Sleeping) != 0)
            {
                JointKernel.WakeRow(StoreOps.Hot(shape->body));
            }
        }
    }

    /// <summary>
    /// Remove touching pairs for detached shapes and wake their remaining bodies.
    /// For trigger pairs, emit TriggerEventShapeRemoved with a zero ID to clear Unity's cached pair;
    /// ordinary exit events are ignored when a collider has already been destroyed or disabled.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ForgetRemovedShapesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        public NativeHashMap<ShapePairKey, ShapePairState> Touching;

        /// <summary>The events that end the trigger pairs that this job removes, which the next contact report sends.</summary>
        public NativeList<TriggerEvent> RemovedTriggerPairs;

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
                        // The removed shape's memory may already hold a new shape, so this code reads only the
                        // addresses. The ids stay zero, which tells Unity that the pair ended without a message.
                        RemovedTriggerPairs.Add(new TriggerEvent
                        {
                            triggerShape = (Shape*)(state.FirstIsTrigger ? key.First : key.Second),
                            colliderShape = (Shape*)(state.FirstIsTrigger ? key.Second : key.First),
                            type = StoreConstants.TriggerEventShapeRemoved,
                        });
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
                JointKernel.WakeRow(StoreOps.Hot(left->body));
            }
        }
    }

    /// <summary>
    /// Build Unity's simulation report from this step's manifolds, including pre-solve velocities and contact
    /// points.
    /// Compare with previous pairs for enter/exit events. Keep sleeping pairs and report only trigger enters and
    /// exits.
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
        public NativeList<TriggerEvent> RemovedTriggerPairs;

        public uint Step;

        public void Execute()
        {
            World->contactBodyCount = 0;
            World->contactShapeCount = 0;
            World->contactPointCount = 0;
            World->triggerCount = 0;

            Records.AsArray().Sort(new ManifoldOrder());
            Reserve();

            // Removal events clear Unity's cached pair before any replacement pair is added at the same
            // addresses.
            for (int i = 0; i < RemovedTriggerPairs.Length; i++)
            {
                World->triggers[World->triggerCount++] = RemovedTriggerPairs[i];
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
                bool found = !Touching.TryGetValue(key, out ShapePairState state);
                byte kinematic = ShapePairState.KinematicOf(key);
                if (!found && (state.Trigger != record.Trigger || state.KinematicChanged(kinematic)))
                {
                    // Restart the pair when trigger or kinematic flags change, matching PhysX enter/exit
                    // behavior.
                    EndPair(key, state);
                    found = true;

                    // The lost contact may have started a body pair of its own, so the shape pairs that follow start
                    // another one.
                    bodyPair = null;
                }

                state.LastSeen = Step;
                state.Trigger = record.Trigger;
                state.Kinematic = kinematic;
                ShapeData* trigger = TriggerOf(record.ShapeA, record.ShapeB);
                if (found)
                {
                    // Only a new pair sets the order: when a shape of a trigger pair becomes a trigger too, the flags
                    // can pick the other shape, and the exit must still name the shapes as the enter did.
                    state.FirstIsTrigger = (IntPtr)trigger == key.First;
                }

                Touching[key] = state;

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

        private bool IsAwake(int row)
        {
            const BodyFlags resting = BodyFlags.Static | BodyFlags.Sleeping;
            BodyFlags flags = World->bodies[row].flags;
            if ((flags & resting) != 0)
            {
                return false;
            }

            return (flags & BodyFlags.Kinematic) == 0 || (flags & BodyFlags.HasKinematicTarget) != 0;
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
        /// Select the first trigger shape, or the second if only it is a trigger.
        /// Use the same shape order for enter, exit and removal events to match Unity's pair cache.
        /// </summary>
        private static ShapeData* TriggerOf(ShapeData* a, ShapeData* b)
        {
            return (a->flags & StoreConstants.ShapeFlagTrigger) != 0 ? a : b;
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
            var lost = new NativeList<LostPair>(Allocator.Temp);
            NativeArray<ShapePairKey> keys = Touching.GetKeyArray(Allocator.Temp);
            foreach (ShapePairKey key in keys)
            {
                ShapePairState state = Touching[key];
                if (state.LastSeen == Step)
                {
                    continue;
                }

                ShapeData* first = (ShapeData*)key.First;
                ShapeData* second = (ShapeData*)key.Second;
                int rowFirst = (int)((Shape*)first)->body->index;
                int rowSecond = (int)((Shape*)second)->body->index;

                // Kinematic changes restart touching pairs even when both bodies are asleep.
                if (!IsAwake(rowFirst) && !IsAwake(rowSecond) && !state.KinematicChanged(ShapePairState.KinematicOf(key)))
                {
                    continue;
                }

                Touching.Remove(key);
                long placeFirst = PlaceOf(first);
                long placeSecond = PlaceOf(second);
                lost.Add(new LostPair
                {
                    Key = key,
                    State = state,
                    First = math.min(placeFirst, placeSecond),
                    Second = math.max(placeFirst, placeSecond),
                });
            }

            lost.Sort();
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
                ContactBodyPair* pair = AddBodyPair(rowFirst, rowSecond, (*a).Velocity()->Linear, BodyKinematics.WorldAngularVelocity(*a), (*b).Velocity()->Linear, BodyKinematics.WorldAngularVelocity(*b));
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

        /// <summary>
        /// A lost touching pair with body rows and shape positions for deterministic exit event ordering.
        /// </summary>
        private struct LostPair : IComparable<LostPair>
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
        private long PlaceOf(ShapeData* shape)
        {
            Body* body = ((Shape*)shape)->body;
            BodyData* row = World->bodies + body->index;
            int index = 0;
            while (index < row->shapeCount && row->shapes[index] != shape)
            {
                index++;
            }

            return ((long)body->index << 32) | (uint)index;
        }
    }
}
