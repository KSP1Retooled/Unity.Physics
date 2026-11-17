using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics
{
    /// <summary>Receives the hits that a <see cref="QueryCollector{T,S}"/> accepts.</summary>
    internal unsafe interface IQuerySink<T>
        where T : unmanaged, IQueryResult
    {
        bool EarlyOut { get; }

        float MaxFraction { get; }

        int NumHits { get; }

        /// <summary>
        /// Return false to stop sending hits. serial is the body's Entity.Version, used for distance tie-breaking.
        /// </summary>
        bool Add(in T hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart);
    }

    /// <summary>
    /// Retain the closest hit, breaking distance ties with QueryCollectors.Precedes regardless of tree traversal
    /// order.
    /// </summary>
    internal unsafe struct ClosestSink<T> : IQuerySink<T>
        where T : unmanaged, IQueryResult
    {
        internal T Hit;
        internal QueryShape* Shape;
        internal uint Serial;
        internal bool Backface;

        public bool EarlyOut => false;

        /// <summary>
        /// Initial query length, then the next float above the nearest hit fraction to admit equal-distance ties.
        /// </summary>
        public float MaxFraction { get; set; }

        public int NumHits => Shape != null ? 1 : 0;

        public bool Add(in T hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            bool closer = Shape == null
                ? hit.Fraction < MaxFraction
                : QueryCollectors.Precedes(hit.Fraction, serial, hit.ColliderKey, Hit.Fraction, Serial, Hit.ColliderKey);

            if (closer)
            {
                Hit = hit;
                Shape = shape;
                Serial = serial;
                Backface = backface;
                MaxFraction = QueryCollectors.JustAbove(hit.Fraction);
            }

            return true;
        }
    }

    /// <summary>Stores the first shape and stops the query.</summary>
    internal unsafe struct AnyShapeSink : IQuerySink<DistanceHit>
    {
        internal QueryShape* Shape;

        public bool EarlyOut => true;

        public float MaxFraction { get; set; }

        public int NumHits => Shape != null ? 1 : 0;

        public bool Add(in DistanceHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            Shape = shape;
            return false;
        }
    }

    /// <summary>Sends each shape to the plugin API's overlap collector.</summary>
    internal unsafe struct ShapeListSink : IQuerySink<DistanceHit>
    {
        internal FunctionPointer<NativeStore.ShapeCollectorFunction> Collector;
        internal void* Context;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in DistanceHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            void* userData = shape->UserData;
            NumHits++;
            return Collector.Invoke(Context, &userData, 1) != 0;
        }
    }

    /// <summary>Sends each shape to the plugin API's broadphase collector.</summary>
    internal unsafe struct BroadphaseListSink : IQuerySink<DistanceHit>
    {
        internal FunctionPointer<NativeStore.BroadphaseCollectorFunction> Collector;
        internal void* Context;
        internal QueryBody* Bodies;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in DistanceHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            NumHits++;
            return Collector.Invoke(Context, shape->Handle, Bodies[hit.RigidBodyIndex].Owner, shape->UserData, (ushort)shape->GeometryType) != 0;
        }
    }

    /// <summary>Sends each ray hit to the plugin API's hit collector.</summary>
    internal unsafe struct RayListSink : IQuerySink<RaycastHit>
    {
        internal FunctionPointer<NativeStore.HitCollectorFunction> Collector;
        internal void* Context;
        internal float Distance;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in RaycastHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            RayCastHit record = QueryCollectors.Record(hit.Position, hit.SurfaceNormal, hit.Fraction * Distance, shape, backface);
            NumHits++;
            return Collector.Invoke(Context, &record, 1) != 0;
        }
    }

    /// <summary>Sends each cast hit to the plugin API's hit collector.</summary>
    internal unsafe struct CastListSink : IQuerySink<ColliderCastHit>
    {
        internal FunctionPointer<NativeStore.HitCollectorFunction> Collector;
        internal void* Context;
        internal float Distance;
        internal float3 Direction;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in ColliderCastHit hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart)
        {
            RayCastHit record = overlappedAtStart
                ? QueryCollectors.Record(float3.zero, -Direction, 0.0f, shape, false)
                : QueryCollectors.Record(hit.Position, hit.SurfaceNormal, hit.Fraction * Distance, shape, backface);

            NumHits++;
            return Collector.Invoke(Context, &record, 1) != 0;
        }
    }

    /// <summary>Specifies how to handle hits for each kind of query.</summary>
    internal enum QueryHitRules
    {
        /// <summary>A ray that starts inside a shape doesn't hit it.</summary>
        Ray,

        /// <summary>
        /// Treats hits at the start of a sweep as overlaps. If any triangle in a mesh overlaps, the whole mesh counts
        /// as overlapped.
        /// </summary>
        Sweep,

        /// <summary>Treats every hit as an overlap and accepts either side of a triangle.</summary>
        Overlap,
    }

    /// <summary>
    /// Filter hits by body type, trigger/query flags and triangle side.
    /// When grouping by shape, combine triangle hits into one shape hit. The caller must call Finish after the
    /// query.
    /// </summary>
    internal unsafe struct QueryCollector<T, S> : ICollector<T>
        where T : unmanaged, IQueryResult
        where S : struct, IQuerySink<T>
    {
        internal S Sink;

        private readonly NativeArray<RigidBody> bodies;
        private readonly QueryBody* queryBodies;
        private readonly QueryShape* queryShapes;
        private readonly Body* ignoredBody;
        private readonly uint filterOptions;
        private readonly float3 direction;
        private readonly QueryHitRules rules;
        private readonly bool groupByShape;
        private readonly bool keepInitialOverlaps;

        private T run;
        private QueryShape* runShape;
        private uint runSerial;
        private bool runBackface;
        private bool runOverlapped;
        private bool stopped;

        internal QueryCollector(
            QueryWorld* world,
            S sink,
            uint filterOptions,
            float3 direction,
            QueryHitRules rules,
            bool groupByShape,
            bool keepInitialOverlaps = false,
            Body* ignoredBody = null)
        {
            Sink = sink;
            bodies = world->Collision.Bodies;
            queryBodies = world->Bodies;
            queryShapes = world->Shapes;
            this.ignoredBody = ignoredBody;
            this.filterOptions = filterOptions;
            this.direction = direction;
            this.rules = rules;
            this.groupByShape = groupByShape;
            this.keepInitialOverlaps = keepInitialOverlaps;
            run = default;
            runShape = null;
            runSerial = 0;
            runBackface = false;
            runOverlapped = false;
            stopped = false;
        }

        public bool EarlyOutOnFirstHit => Sink.EarlyOut || stopped;

        public float MaxFraction => Sink.MaxFraction;

        public int NumHits => Sink.NumHits + (runShape != null ? 1 : 0);

        public bool AddHit(T hit)
        {
            if (stopped)
            {
                return false;
            }

            RigidBody rigidBody = bodies[hit.RigidBodyIndex];
            if (rigidBody.Entity.Version == 0)
            {
                return false;
            }

            QueryBody* body = queryBodies + hit.RigidBodyIndex;
            if (ignoredBody != null && body->Owner == ignoredBody)
            {
                return false;
            }

            uint wanted = body->IsStatic != 0 ? StoreConstants.QueryUseStaticBodies : StoreConstants.QueryUseDynamicBodies;
            if ((filterOptions & wanted) == 0)
            {
                return false;
            }

            QueryShape* shape = queryShapes + body->FirstShape + ColliderBuilder.ShapeIndexOf(rigidBody.Collider, hit.ColliderKey, body->SingleShape);
            if ((shape->Flags & StoreConstants.ShapeFlagSceneQuery) == 0)
            {
                return false;
            }

            bool trigger = (shape->Flags & StoreConstants.ShapeFlagTrigger) != 0;
            if (trigger && (filterOptions & StoreConstants.QueryUseTriggerShapes) == 0)
            {
                return false;
            }

            if (rules == QueryHitRules.Ray && hit.Fraction <= 0.0f)
            {
                return false;
            }

            bool backface = false;
            if (rules != QueryHitRules.Overlap)
            {
                backface = QueryCollectors.IsBackface((Collider*)rigidBody.Collider.GetUnsafePtr(), hit.ColliderKey, rigidBody.WorldFromBody, direction);
                if (backface && (filterOptions & StoreConstants.QueryAllowBackfaces) == 0)
                {
                    return false;
                }
            }

            bool overlapped = rules == QueryHitRules.Sweep && hit.Fraction <= 0.0f;
            if (!groupByShape)
            {
                stopped = !Sink.Add(hit, shape, (uint)rigidBody.Entity.Version, backface, overlapped);
                return true;
            }

            if (shape != runShape)
            {
                FinishRun();
                run = hit;
                runShape = shape;
                runSerial = (uint)rigidBody.Entity.Version;
                runBackface = backface;
                runOverlapped = overlapped;
                return true;
            }

            runOverlapped |= overlapped;
            if (hit.Fraction < run.Fraction)
            {
                run = hit;
                runBackface = backface;
            }

            return true;
        }

        internal void Finish()
        {
            FinishRun();
        }

        private void FinishRun()
        {
            if (runShape != null && !stopped && (!runOverlapped || keepInitialOverlaps))
            {
                stopped = !Sink.Add(run, runShape, runSerial, runBackface, runOverlapped);
            }

            runShape = null;
        }
    }

    /// <summary>
    /// Refine cast hits that Unity Physics reports before reaching contact, rejecting casts that pass the shape.
    /// Keep the latest hit after at most four additional passes, even if contact is still unconfirmed.
    /// </summary>
    internal unsafe struct RefinedCastCollector<C> : ICollector<ColliderCastHit>
        where C : struct, ICollector<ColliderCastHit>
    {
        // Each additional cast allows up to ten more advances.
        private const int MaxPasses = 4;

        internal C Inner;

        private readonly NativeArray<RigidBody> bodies;
        private readonly Collider* query;
        private readonly quaternion orientation;
        private readonly float3 start;
        private readonly float3 end;

        internal RefinedCastCollector(QueryWorld* world, C inner, ColliderCastInput input)
        {
            Inner = inner;
            bodies = world->Collision.Bodies;
            query = input.Collider;
            orientation = input.Orientation;
            start = input.Start;
            end = input.End;
        }

        public bool EarlyOutOnFirstHit => Inner.EarlyOutOnFirstHit;

        public float MaxFraction => Inner.MaxFraction;

        public int NumHits => Inner.NumHits;

        public bool AddHit(ColliderCastHit hit)
        {
            if (hit.Fraction > 0.0f && !Refine(ref hit))
            {
                return false;
            }

            return Inner.AddHit(hit);
        }

        private bool Refine(ref ColliderCastHit hit)
        {
            RigidBody body = bodies[hit.RigidBodyIndex];
            if (!Collider.GetLeafCollider(out ChildCollider leaf, (Collider*)body.Collider.GetUnsafePtr(), hit.ColliderKey, body.WorldFromBody))
            {
                return true;
            }

            RigidTransform worldFromLeaf = leaf.TransformFromChild;
            RigidTransform leafFromWorld = math.inverse(worldFromLeaf);
            float3 from = math.transform(leafFromWorld, start);
            float3 to = math.transform(leafFromWorld, end);
            quaternion rotation = math.mul(leafFromWorld.rot, orientation);
            float fraction = hit.Fraction;
            for (int pass = 0; pass < MaxPasses; pass++)
            {
                ColliderCastInput onward = new ColliderCastInput
                {
                    Collider = query,
                    Orientation = rotation,
                    Start = math.lerp(from, to, fraction),
                    End = to,
                };

                var closest = new ClosestHitCollector<ColliderCastHit>(1.0f);
                if (!leaf.Collider->CastCollider(onward, ref closest))
                {
                    return false;
                }

                ColliderCastHit found = closest.ClosestHit;
                hit.Position = math.transform(worldFromLeaf, found.Position);
                hit.SurfaceNormal = math.rotate(worldFromLeaf.rot, found.SurfaceNormal);
                if (found.Fraction <= 0.0f)
                {
                    break;
                }

                fraction += (1.0f - fraction) * found.Fraction;
                hit.Fraction = fraction;
            }

            return true;
        }
    }

    /// <summary>
    /// Collect ray or cast hits against one shape. Rays starting inside miss; casts report initial overlaps only
    /// when requested.
    /// </summary>
    internal unsafe struct SingleShapeCollector<T> : ICollector<T>
        where T : unmanaged, IQueryResult
    {
        internal T Hit;
        internal bool Backface;

        private readonly Collider* root;
        private readonly uint filterOptions;
        private readonly float3 direction;
        private readonly bool keepOverlaps;

        internal SingleShapeCollector(Collider* root, uint filterOptions, float3 direction, bool keepOverlaps = false)
        {
            this.root = root;
            this.filterOptions = filterOptions;
            this.direction = direction;
            this.keepOverlaps = keepOverlaps;
            Hit = default;
            Backface = false;
            MaxFraction = 1.0f;
            NumHits = 0;
        }

        public bool EarlyOutOnFirstHit => false;

        public float MaxFraction { get; private set; }

        public int NumHits { get; private set; }

        public bool AddHit(T hit)
        {
            if (hit.Fraction <= 0.0f && !keepOverlaps)
            {
                return false;
            }

            bool backface = QueryCollectors.IsBackface(root, hit.ColliderKey, RigidTransform.identity, direction);
            if (backface && (filterOptions & StoreConstants.QueryAllowBackfaces) == 0)
            {
                return false;
            }

            Hit = hit;
            Backface = backface;
            MaxFraction = hit.Fraction;
            NumHits = 1;
            return true;
        }
    }

    /// <summary>
    /// Find the deepest convex overlap, excluding mesh triangles facing away from the shape's centre as PhysX does.
    /// </summary>
    internal unsafe struct PenetrationCollector : ICollector<DistanceHit>
    {
        internal DistanceHit Hit;

        private readonly Collider* target;
        private readonly float3 queryCentre;

        internal PenetrationCollector(Collider* target, float3 queryCentre, float maxDistance)
        {
            this.target = target;
            this.queryCentre = queryCentre;
            Hit = default;
            MaxFraction = maxDistance;
            NumHits = 0;
        }

        public bool EarlyOutOnFirstHit => false;

        public float MaxFraction { get; private set; }

        public int NumHits { get; private set; }

        public bool AddHit(DistanceHit hit)
        {
            if (QueryCollectors.IsBehind(target, hit.ColliderKey, queryCentre))
            {
                return false;
            }

            Hit = hit;
            MaxFraction = hit.Fraction;
            NumHits = 1;
            return true;
        }
    }

    internal static unsafe class QueryCollectors
    {
        /// <summary>
        /// Order closest hits by distance, then higher body serial, then lower collider key.
        /// Higher serial matches PhysX's preference for later-created bodies; the leaf tie-break is deterministic.
        /// </summary>
        internal static bool Precedes(float fraction, uint serial, ColliderKey key, float bestFraction, uint bestSerial, ColliderKey bestKey)
        {
            if (fraction != bestFraction)
            {
                return fraction < bestFraction;
            }

            if (serial != bestSerial)
            {
                return serial > bestSerial;
            }

            return key.Value < bestKey.Value;
        }

        /// <summary>
        /// Return the next representable float above fraction so that a closest-hit collector also receives
        /// equal-distance hits.
        /// </summary>
        internal static float JustAbove(float fraction)
        {
            if (fraction > 0.0f)
            {
                return math.asfloat(math.asint(fraction) + 1);
            }

            if (fraction < 0.0f)
            {
                return math.asfloat(math.asint(fraction) - 1);
            }

            // The smallest normal float, rather than a denormal one, which a thread that flushes denormals to zero
            // would read as zero. Precedes still rejects a hit between zero and this value, because it is farther.
            return 1.17549435e-38f;
        }

        // Physics.Raycast uses an infinite distance by default. Unity Physics cannot represent that distance, so use a
        // finite limit.
        internal const float MaxQueryLength = 1e7f;

        // RaycastHit.triangleIndex interprets this value as -1.
        internal const uint NoFace = uint.MaxValue;

        /// <summary>
        /// Convert a hit to the plugin format and restore the polygon normal for backface hits.
        /// Report face zero for triangle meshes and no face for other geometry; source triangle mapping is not
        /// retained.
        /// </summary>
        internal static RayCastHit Record(float3 position, float3 normal, float distance, QueryShape* shape, bool backface)
        {
            return new RayCastHit
            {
                point = position,
                normal = backface ? -normal : normal,
                faceId = shape->GeometryType == StoreConstants.GeometryTriangleMesh ? 0u : NoFace,
                distance = distance,
                uv = float2.zero,
                engineObjectId = shape->EngineObjectId,
            };
        }

        /// <summary>
        /// Determine the polygon's hit side from winding, since Unity Physics turns hit normals toward the query.
        /// </summary>
        internal static bool IsBackface(Collider* root, ColliderKey key, RigidTransform worldFromRoot, float3 direction)
        {
            if (!Collider.GetLeafCollider(out ChildCollider leaf, root, key, worldFromRoot))
            {
                return false;
            }

            Collider* collider = leaf.Collider;
            if (collider->Type != ColliderType.Triangle && collider->Type != ColliderType.Quad)
            {
                return false;
            }

            return math.dot(math.rotate(leaf.TransformFromChild.rot, FrontOf(root, key, collider)), direction) > 0.0f;
        }

        /// <summary>
        /// Determines whether a point lies behind the polygon identified by a collider key. Returns false for geometry
        /// other than polygons.
        /// </summary>
        internal static bool IsBehind(Collider* root, ColliderKey key, float3 point)
        {
            if (!Collider.GetLeafCollider(out ChildCollider leaf, root, key, RigidTransform.identity))
            {
                return false;
            }

            Collider* collider = leaf.Collider;
            if (collider->Type != ColliderType.Triangle && collider->Type != ColliderType.Quad)
            {
                return false;
            }

            BlobArray.Accessor<float3> vertices = ((PolygonCollider*)collider)->Vertices;
            float3 inLeaf = math.transform(math.inverse(leaf.TransformFromChild), point);
            return math.dot(FrontOf(root, key, collider), inLeaf - vertices[0]) < 0.0f;
        }

        /// <summary>
        /// Return the leaf polygon's front direction. Reverse terrain winding so that its normal points above the
        /// surface.
        /// </summary>
        private static float3 FrontOf(Collider* root, ColliderKey key, Collider* polygon)
        {
            BlobArray.Accessor<float3> vertices = ((PolygonCollider*)polygon)->Vertices;
            float3 normal = math.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
            return IsTerrainLeaf(root, key) ? -normal : normal;
        }

        private static bool IsTerrainLeaf(Collider* root, ColliderKey key)
        {
            if (root->Type == ColliderType.Compound)
            {
                root->GetChild(ref key, out ChildCollider child);
                return child.Collider->Type == ColliderType.Terrain;
            }

            return root->Type == ColliderType.Terrain;
        }
    }
}
