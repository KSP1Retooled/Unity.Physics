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
using Unity.Physics.Dynamics.Sleep;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Dynamics.Contacts
{
    internal static partial class ContactMaterials
    {
        /// <summary>
        /// Filter contact manifolds, set material coefficients, and retain contacts for events and sleeping.
        /// Reject speculative mesh contacts whose paths cannot reach their polygons this step.
        /// Read buckets independently; ContactBuckets.MergeContactBucketsJob restores stream order.
        /// </summary>
        [BurstCompile]
        internal unsafe struct ReadContactsJob : IJobParallelForDefer
        {
            // Measurements against PhysX: a 0.015 m box slides on one anchor and a 0.02 m box slides on two. Their
            // diagonals are 0.021 m and 0.028 m. Points separated by at most this threshold share an anchor.
            private const float FrictionOffsetThreshold = 0.025f;

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
            public NativeList<ContactBuckets.ContactBucket> Buckets;

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
                ref ContactBuckets.ContactBucket output = ref Buckets.ElementAt(bucket);
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

                        // Keep the original geometry for the report. The Jacobian builder skips rejected points.
                        float distance = point->Distance;
                        float3 moving = face.MovingIsA ? point->Position + header->Normal * distance : point->Position;
                        bool reachable = !face.Valid || distance <= touchDistance || Reaches(face, moving, distance);
                        if (!reachable)
                        {
                            header->RejectedPoints |= 1u << i;
                        }

                        Add(ref output, point->Position, distance, touchDistance, reachable);
                    }

                    if (kept)
                    {
                        ref ManifoldRecord record = ref output.Records.ElementAt(output.Records.Length - 1);
                        SetCoefficients(ref *header, ref record, ref output.Points);
                        output.Wakes |= record.Touching && !record.Trigger
                            && WakePropagation.WakeTouchedBodiesJob.Wakes(Store + record.RowA, Store + record.RowB);
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
                    LinearVelocityA = first.velocity->Linear,
                    AngularVelocityA = BodyKinematics.WorldAngularVelocity(first),
                    LinearVelocityB = second.velocity->Linear,
                    AngularVelocityB = BodyKinematics.WorldAngularVelocity(second),
                    Trigger = trigger,
                });

                return true;
            }

            private void Add(ref ContactBuckets.ContactBucket output, float3 position, float distance, float touchDistance, bool reachable)
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
                    Linear = moving.velocity->Linear,
                    Angular = BodyKinematics.WorldAngularVelocity(moving),
                    CentreOfMass = moving.pose->Position + math.rotate(moving.pose->Rotation, moving.mass->Transform.pos),
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
                float3 center = (float3)body.pose->Position + math.rotate(body.pose->Rotation, body.mass->Transform.pos);
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
                ShapeMassProperties.WorldBounds(otherShape, otherShape->localPosition, otherShape->localRotation, row.pose->Position, row.pose->Rotation, out float3 lower, out float3 upper);
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
    }
}
