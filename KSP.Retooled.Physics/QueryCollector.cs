using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>Where a <see cref="QueryCollector{T,S}"/> sends the hits it passes.</summary>
    internal unsafe interface IQuerySink<T>
        where T : unmanaged, IQueryResult
    {
        bool EarlyOut { get; }

        float MaxFraction { get; }

        int NumHits { get; }

        /// <summary>False once the sink wants no more hits.</summary>
        bool Add(in T hit, QueryShape* shape, bool backface, bool overlappedAtStart);
    }

    /// <summary>Keeps the closest hit.</summary>
    internal unsafe struct ClosestSink<T> : IQuerySink<T>
        where T : unmanaged, IQueryResult
    {
        internal T Hit;
        internal QueryShape* Shape;
        internal bool Backface;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits => Shape != null ? 1 : 0;

        public bool Add(in T hit, QueryShape* shape, bool backface, bool overlappedAtStart)
        {
            if (hit.Fraction < MaxFraction)
            {
                Hit = hit;
                Shape = shape;
                Backface = backface;
                MaxFraction = hit.Fraction;
            }

            return true;
        }
    }

    /// <summary>Keeps the first shape and stops looking.</summary>
    internal unsafe struct AnyShapeSink : IQuerySink<DistanceHit>
    {
        internal QueryShape* Shape;

        public bool EarlyOut => true;

        public float MaxFraction { get; set; }

        public int NumHits => Shape != null ? 1 : 0;

        public bool Add(in DistanceHit hit, QueryShape* shape, bool backface, bool overlappedAtStart)
        {
            Shape = shape;
            return false;
        }
    }

    /// <summary>Hands each shape to the plugin API's overlap collector.</summary>
    internal unsafe struct ShapeListSink : IQuerySink<DistanceHit>
    {
        internal FunctionPointer<NativeStore.ShapeCollectorFunction> Collector;
        internal void* Context;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in DistanceHit hit, QueryShape* shape, bool backface, bool overlappedAtStart)
        {
            void* userData = shape->UserData;
            NumHits++;
            return Collector.Invoke(Context, &userData, 1) != 0;
        }
    }

    /// <summary>Hands each shape to the plugin API's broadphase collector.</summary>
    internal unsafe struct BroadphaseListSink : IQuerySink<DistanceHit>
    {
        internal FunctionPointer<NativeStore.BroadphaseCollectorFunction> Collector;
        internal void* Context;
        internal QueryBody* Bodies;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in DistanceHit hit, QueryShape* shape, bool backface, bool overlappedAtStart)
        {
            NumHits++;
            return Collector.Invoke(Context, shape->Handle, Bodies[hit.RigidBodyIndex].Owner, shape->UserData, (ushort)shape->GeometryType) != 0;
        }
    }

    /// <summary>Hands each ray hit to the plugin API's hit collector.</summary>
    internal unsafe struct RayListSink : IQuerySink<RaycastHit>
    {
        internal FunctionPointer<NativeStore.HitCollectorFunction> Collector;
        internal void* Context;
        internal float Distance;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in RaycastHit hit, QueryShape* shape, bool backface, bool overlappedAtStart)
        {
            NativeStore.RayCastHit record = QueryCollectors.Record(hit.Position, hit.SurfaceNormal, hit.Fraction * Distance, shape, backface);
            NumHits++;
            return Collector.Invoke(Context, &record, 1) != 0;
        }
    }

    /// <summary>Hands each cast hit to the plugin API's hit collector.</summary>
    internal unsafe struct CastListSink : IQuerySink<ColliderCastHit>
    {
        internal FunctionPointer<NativeStore.HitCollectorFunction> Collector;
        internal void* Context;
        internal float Distance;
        internal float3 Direction;

        public bool EarlyOut => false;

        public float MaxFraction { get; set; }

        public int NumHits { get; private set; }

        public bool Add(in ColliderCastHit hit, QueryShape* shape, bool backface, bool overlappedAtStart)
        {
            NativeStore.RayCastHit record = overlappedAtStart
                ? QueryCollectors.Record(float3.zero, -Direction, 0.0f, shape, false)
                : QueryCollectors.Record(hit.Position, hit.SurfaceNormal, hit.Fraction * Distance, shape, backface);

            NumHits++;
            return Collector.Invoke(Context, &record, 1) != 0;
        }
    }

    /// <summary>What a collector does with hits, by kind of query.</summary>
    internal enum QueryHitRules
    {
        /// <summary>A ray that starts inside a shape doesn't hit it.</summary>
        Ray,

        /// <summary>Hits at the very start are overlaps, and a mesh in one counts as overlapped as a whole.</summary>
        Sweep,

        /// <summary>Every hit is an overlap, and a triangle has no front.</summary>
        Overlap,
    }

    /// <summary>
    /// Passes the hits that survive what a collision filter can't express: which kinds of shape the query wants, whether the
    /// shape takes part in queries at all, and which way a triangle faces. Unity Physics reports a mesh per triangle and PhysX
    /// per shape, so <paramref name="groupByShape"/> merges a shape's hits, which needs no memory because the package finishes
    /// one shape before it visits another. The caller ends with <see cref="Finish"/>.
    /// </summary>
    internal unsafe struct QueryCollector<T, S> : ICollector<T>
        where T : unmanaged, IQueryResult
        where S : struct, IQuerySink<T>
    {
        internal S Sink;

        private readonly NativeArray<RigidBody> bodies;
        private readonly QueryBody* queryBodies;
        private readonly QueryShape* queryShapes;
        private readonly NativeStore.Body* ignoredBody;
        private readonly uint filterOptions;
        private readonly float3 direction;
        private readonly QueryHitRules rules;
        private readonly bool groupByShape;
        private readonly bool keepInitialOverlaps;

        private T run;
        private QueryShape* runShape;
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
            NativeStore.Body* ignoredBody = null)
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

            uint wanted = body->IsStatic != 0 ? NativeStore.QueryUseStaticBodies : NativeStore.QueryUseDynamicBodies;
            if ((filterOptions & wanted) == 0)
            {
                return false;
            }

            QueryShape* shape = queryShapes + body->FirstShape + ColliderBuilder.ShapeIndexOf(rigidBody.Collider, hit.ColliderKey, body->SingleShape);
            if ((shape->Flags & NativeStore.ShapeFlagSceneQuery) == 0)
            {
                return false;
            }

            bool trigger = (shape->Flags & NativeStore.ShapeFlagTrigger) != 0;
            if (trigger && (filterOptions & NativeStore.QueryUseTriggerShapes) == 0)
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
                if (backface && (filterOptions & NativeStore.QueryAllowBackfaces) == 0)
                {
                    return false;
                }
            }

            bool overlapped = rules == QueryHitRules.Sweep && hit.Fraction <= 0.0f;
            if (!groupByShape)
            {
                stopped = !Sink.Add(hit, shape, backface, overlapped);
                return true;
            }

            if (shape != runShape)
            {
                FinishRun();
                run = hit;
                runShape = shape;
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
                stopped = !Sink.Add(run, runShape, runBackface, runOverlapped);
            }

            runShape = null;
        }
    }

    /// <summary>
    /// Holds a cast to the contact it reports. Unity Physics gives up on a cast after ten advances and calls wherever that left
    /// it a hit, which a cast grazing a shape can reach well short of the shape or without ever touching it. Casting on from
    /// there against that one leaf either arrives or finds that the query passes by, in which case there was no hit.
    /// </summary>
    internal unsafe struct RefinedCastCollector<C> : ICollector<ColliderCastHit>
        where C : struct, ICollector<ColliderCastHit>
    {
        // Each pass is another ten advances.
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
    /// A ray or a cast against one shape, which PhysX treats like a query of a scene that holds only that shape. A ray that
    /// starts inside the shape misses it, and a cast that does reports the overlap only if asked to keep overlaps.
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
            if (backface && (filterOptions & NativeStore.QueryAllowBackfaces) == 0)
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
    /// The deepest overlap of a convex collider with its target. PhysX pushes a shape out of a mesh on the triangle's front
    /// side only, and leaves a triangle whose back faces the shape's centre out of the question.
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
        // Physics.Raycast defaults to an infinite distance, which Unity Physics can't represent.
        internal const float MaxQueryLength = 1e7f;

        // RaycastHit.triangleIndex reads this as -1.
        internal const uint NoFace = uint.MaxValue;

        /// <summary>
        /// A hit as the plugin API reads it. A hit names a primitive of the blob, which merges and reorders triangles. Nothing
        /// in the game reads the source triangle or the barycentric coordinates, so mapping back isn't worth the memory: a
        /// triangle mesh reports face 0, and anything else no face. PhysX gives a back face's hit the triangle's own normal,
        /// and Unity Physics one that faces the query.
        /// </summary>
        internal static NativeStore.RayCastHit Record(float3 position, float3 normal, float distance, QueryShape* shape, bool backface)
        {
            return new NativeStore.RayCastHit
            {
                point = position,
                normal = backface ? -normal : normal,
                faceId = shape->GeometryType == NativeStore.GeometryTriangleMesh ? 0u : NoFace,
                distance = distance,
                uv = float2.zero,
                engineObjectId = shape->EngineObjectId,
            };
        }

        /// <summary>
        /// Whether something moving along a world direction met a triangle mesh from behind. Unity Physics hits polygons from
        /// both sides and turns the hit's normal to face the query, so only the polygon's own winding can say, and a hit
        /// refused here lets the query go on.
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

        /// <summary>Whether a point lies behind the polygon a key names. Anything but a polygon has no behind.</summary>
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
        /// The way a polygon faces, in its leaf's frame. A mesh's winding says, and a terrain's triangles wind the opposite way
        /// from a mesh's, with the surface up.
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
