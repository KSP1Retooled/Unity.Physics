using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// What a query function reads: a copy of the built collision world and a snapshot of the store's bodies and shapes. It
    /// lives in unmanaged memory and holds no managed reference, so a Burst function on any thread can use it. The plugin
    /// keeps it behind the world's query lock, and <see cref="WorldState"/> republishes it after every build. A worker
    /// reads the last one published even when the store has moved on, so it must not lead back into the store.
    /// </summary>
    internal unsafe struct QueryWorld
    {
        public CollisionWorld Collision;

        // One per body of Collision.
        [NativeDisableUnsafePtrRestriction]
        public QueryBody* Bodies;

        // Each body's shapes in store order, from QueryBody.FirstShape.
        [NativeDisableUnsafePtrRestriction]
        public QueryShape* Shapes;

        public int ShapeCount;
    }

    /// <summary>A built body as queries see it.</summary>
    internal unsafe struct QueryBody
    {
        // Compared, or handed to Unity. Never read, because the record may be gone by the time a worker has this.
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.Body* Owner;

        public int FirstShape;
        public int SingleShape;     // the shape a collider that isn't a compound stands for, relative to FirstShape
        public int IsStatic;
        private readonly int reserved0;
    }

    /// <summary>A shape as queries see it.</summary>
    internal unsafe struct QueryShape
    {
        public ulong EngineObjectId;

        // Handed to Unity, like QueryBody.Owner.
        [NativeDisableUnsafePtrRestriction]
        public void* UserData;

        [NativeDisableUnsafePtrRestriction]
        public void* Handle;

        public uint Flags;
        public uint GeometryType;
    }

    /// <summary>A sphere, box or capsule on the stack, which is what Unity Physics casts and measures.</summary>
    [StructLayout(LayoutKind.Explicit)]
    internal unsafe struct PrimitiveCollider
    {
        [FieldOffset(0)]
        private SphereCollider sphere;

        [FieldOffset(0)]
        private BoxCollider box;

        [FieldOffset(0)]
        private CapsuleCollider capsule;

        /// <summary>
        /// Fills this with a geometry that reads as a <see cref="NativeStore.ShapeData"/> does, centred on its own origin.
        /// Null for a geometry that isn't a primitive. The result points into this, so it lasts as long as this does.
        /// </summary>
        internal Collider* Initialize(uint geometryType, float3 extents, CollisionFilter filter)
        {
            switch (geometryType)
            {
                case NativeStore.GeometrySphere:
                    sphere.Initialize(ColliderBuilder.SphereOf(extents, float3.zero), filter, Material.Default);
                    return (Collider*)UnsafeUtility.AddressOf(ref sphere);
                case NativeStore.GeometryBox:
                    box.Initialize(ColliderBuilder.BoxOf(extents, float3.zero, quaternion.identity), filter, Material.Default);
                    return (Collider*)UnsafeUtility.AddressOf(ref box);
                case NativeStore.GeometryCapsule:
                    capsule.Initialize(ColliderBuilder.CapsuleOf(extents, float3.zero, quaternion.identity), filter, Material.Default);
                    return (Collider*)UnsafeUtility.AddressOf(ref capsule);
                default:
                    return null;
            }
        }
    }

    /// <summary>
    /// The queries the plugin calls without going through Mono: one Burst function pointer, and until the async compile
    /// delivers, that pointer is a stub running the same method in Mono, so a query always works. All geometry is Unity
    /// Physics'. This adapts what it reports to what PhysX would have.
    /// </summary>
    /// <remarks>
    /// Safety checks stay off because a Burst exception can't cross the native caller; the Mono form catches its own.
    /// </remarks>
    [BurstCompile]
    internal static unsafe class QueryKernel
    {
        // PhysX counts shapes that touch as overlapping, and Unity Physics only asks for distances below a limit.
        private const float TouchDistance = 1e-4f;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int QueryFunction(uint kind, QueryWorld* world, NativeStore.QueryInput* input, NativeStore.QueryOutput* output);

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(QueryFunction))]
        internal static int Run(uint kind, QueryWorld* world, NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.Query.Auto();

            int result = 0;
            bool ranInMono = false;
            RunInMono(kind, world, input, output, ref result, ref ranInMono);
            return ranInMono ? result : Execute((NativeStore.QueryKind)kind, world, input, output);
        }

        /// <summary>Burst drops this, so it only runs in the Mono form, which needs the exception ended here.</summary>
        [BurstDiscard]
        private static void RunInMono(
            uint kind,
            QueryWorld* world,
            NativeStore.QueryInput* input,
            NativeStore.QueryOutput* output,
            ref int result,
            ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedQuery();
            try
            {
                // Inside a job, which is where most of these calls run, the safety system only lets a native container
                // be touched through a job's own copy of it. Burst has no such checks, so only this form needs one.
                int found = 0;
                new QueryJob { Kind = kind, World = *world, Input = input, Output = output, Result = &found }.Run();
                result = found;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogException(e);
            }
        }

        private struct QueryJob : IJob
        {
            public uint Kind;

            [ReadOnly]
            public QueryWorld World;

            [NativeDisableUnsafePtrRestriction]
            public NativeStore.QueryInput* Input;

            [NativeDisableUnsafePtrRestriction]
            public NativeStore.QueryOutput* Output;

            [NativeDisableUnsafePtrRestriction]
            public int* Result;

            public void Execute()
            {
                *Result = QueryKernel.Execute((NativeStore.QueryKind)Kind, (QueryWorld*)UnsafeUtility.AddressOf(ref World), Input, Output);
            }
        }

        private static int Execute(NativeStore.QueryKind kind, QueryWorld* world, NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            switch (kind)
            {
                case NativeStore.QueryKind.Ray:
                    return CastRay(world, input, output);
                case NativeStore.QueryKind.RayAll:
                    CastRayAll(world, input);
                    return 0;
                case NativeStore.QueryKind.RayAgainstShape:
                    return CastRayAgainstShape(input, output);
                case NativeStore.QueryKind.OverlapAny:
                    return OverlapAny(world, input, output);
                case NativeStore.QueryKind.OverlapAll:
                    OverlapAll(world, input);
                    return 0;
                case NativeStore.QueryKind.OverlapBroadphase:
                    OverlapBroadphase(world, input);
                    return 0;
                case NativeStore.QueryKind.Cast:
                    return CastGeometry(world, input, output);
                case NativeStore.QueryKind.CastAll:
                    CastGeometryAll(world, input);
                    return 0;
                case NativeStore.QueryKind.BodyCast:
                    return CastBody(world, input, output);
                case NativeStore.QueryKind.BodyCastAll:
                    CastBodyAll(world, input);
                    return 0;
                case NativeStore.QueryKind.ClosestPoint:
                    return ClosestPoint(input, output);
                case NativeStore.QueryKind.Penetration:
                    return Penetration(input, output);
                case NativeStore.QueryKind.CastAgainstShape:
                    return CastAgainstShape(input, output);
                default:
                    return 0;
            }
        }

        private static CollisionFilter FilterOf(NativeStore.QueryInput* input)
        {
            return new CollisionFilter
            {
                BelongsTo = uint.MaxValue,
                CollidesWith = (uint)input->layerMask,
            };
        }

        private static float LengthOf(NativeStore.QueryInput* input)
        {
            return math.min(input->maxDistance, QueryCollectors.MaxQueryLength);
        }

        private static RaycastInput RayOf(NativeStore.QueryInput* input, float3 direction, float distance)
        {
            return new RaycastInput
            {
                Start = input->position,
                End = input->position + direction * distance,
                Filter = FilterOf(input),
            };
        }

        private static int CastRay(QueryWorld* world, NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);

            // A hit needs nothing done per shape, because the closest triangle is the only one of a mesh that can stay.
            var collector = new QueryCollector<RaycastHit, ClosestSink<RaycastHit>>(
                world,
                new ClosestSink<RaycastHit> { MaxFraction = 1.0f },
                input->filterOptions,
                direction,
                QueryHitRules.Ray,
                groupByShape: false);

            world->Collision.CastRay(RayOf(input, direction, distance), ref collector);
            ref ClosestSink<RaycastHit> sink = ref collector.Sink;
            if (sink.Shape == null)
            {
                return 0;
            }

            output->hit = QueryCollectors.Record(sink.Hit.Position, sink.Hit.SurfaceNormal, sink.Hit.Fraction * distance, sink.Shape, sink.Backface);
            return 1;
        }

        private static void CastRayAll(QueryWorld* world, NativeStore.QueryInput* input)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            var collector = new QueryCollector<RaycastHit, RayListSink>(
                world,
                new RayListSink
                {
                    Collector = new FunctionPointer<NativeStore.HitCollectorFunction>((IntPtr)input->collector),
                    Context = input->collectorContext,
                    Distance = distance,
                    MaxFraction = 1.0f,
                },
                input->filterOptions,
                direction,
                QueryHitRules.Ray,
                groupByShape: (input->filterOptions & NativeStore.QueryAllowMultipleMeshHits) == 0);

            world->Collision.CastRay(RayOf(input, direction, distance), ref collector);
            collector.Finish();
        }

        private static int CastRayAgainstShape(NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            float3 worldDirection = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            RigidTransform worldFromShape = WorldPose(input->shape);

            PrimitiveCollider primitive = default;
            Collider* collider = ColliderOf(input->shape, ref primitive, worldFromShape, out RigidTransform worldFromCollider);
            if (collider == null)
            {
                return 0;
            }

            RigidTransform colliderFromWorld = math.inverse(worldFromCollider);
            float3 direction = math.rotate(colliderFromWorld.rot, worldDirection);
            float3 origin = math.transform(colliderFromWorld, input->position);
            var collector = new SingleShapeCollector<RaycastHit>(collider, input->filterOptions, direction);
            collider->CastRay(new RaycastInput { Start = origin, End = origin + direction * distance, Filter = CollisionFilter.Default }, ref collector);
            if (collector.NumHits == 0)
            {
                return 0;
            }

            RaycastHit hit = collector.Hit;
            float3 normal = math.rotate(worldFromCollider.rot, hit.SurfaceNormal);
            output->hit = new NativeStore.RayCastHit
            {
                point = math.transform(worldFromCollider, hit.Position),
                normal = collector.Backface ? -normal : normal,
                faceId = input->shape->data.geometryType == NativeStore.GeometryTriangleMesh ? 0u : QueryCollectors.NoFace,
                distance = hit.Fraction * distance,
            };

            return 1;
        }

        private static ColliderDistanceInput OverlapInput(Collider* collider, NativeStore.QueryInput* input)
        {
            return new ColliderDistanceInput
            {
                Collider = collider,
                Transform = new RigidTransform(input->rotation, input->position),
                MaxDistance = TouchDistance,
                Scale = 1.0f,
            };
        }

        private static int OverlapAny(QueryWorld* world, NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            PrimitiveCollider primitive = default;
            Collider* collider = primitive.Initialize(input->geometryType, input->extents, FilterOf(input));
            if (collider == null)
            {
                return 0;
            }

            var collector = new QueryCollector<DistanceHit, AnyShapeSink>(
                world,
                new AnyShapeSink { MaxFraction = TouchDistance },
                input->filterOptions,
                float3.zero,
                QueryHitRules.Overlap,
                groupByShape: true);

            world->Collision.CalculateDistance(OverlapInput(collider, input), ref collector);
            collector.Finish();
            if (collector.Sink.Shape == null)
            {
                return 0;
            }

            output->shapeUserData = collector.Sink.Shape->UserData;
            return 1;
        }

        private static void OverlapAll(QueryWorld* world, NativeStore.QueryInput* input)
        {
            PrimitiveCollider primitive = default;
            Collider* collider = primitive.Initialize(input->geometryType, input->extents, FilterOf(input));
            if (collider == null)
            {
                return;
            }

            var collector = new QueryCollector<DistanceHit, ShapeListSink>(
                world,
                new ShapeListSink
                {
                    Collector = new FunctionPointer<NativeStore.ShapeCollectorFunction>((IntPtr)input->collector),
                    Context = input->collectorContext,
                    MaxFraction = TouchDistance,
                },
                input->filterOptions,
                float3.zero,
                QueryHitRules.Overlap,
                groupByShape: true);

            world->Collision.CalculateDistance(OverlapInput(collider, input), ref collector);
            collector.Finish();
        }

        private static void OverlapBroadphase(QueryWorld* world, NativeStore.QueryInput* input)
        {
            PrimitiveCollider primitive = default;
            Collider* collider = primitive.Initialize(input->geometryType, input->extents, FilterOf(input));
            if (collider == null)
            {
                return;
            }

            var collector = new QueryCollector<DistanceHit, BroadphaseListSink>(
                world,
                new BroadphaseListSink
                {
                    Collector = new FunctionPointer<NativeStore.BroadphaseCollectorFunction>((IntPtr)input->collector),
                    Context = input->collectorContext,
                    Bodies = world->Bodies,
                    MaxFraction = TouchDistance,
                },
                input->filterOptions,
                float3.zero,
                QueryHitRules.Overlap,
                groupByShape: true);

            world->Collision.CalculateDistance(OverlapInput(collider, input), ref collector);
            collector.Finish();
        }

        private static ColliderCastInput CastInput(Collider* collider, float3 position, quaternion rotation, float3 direction, float distance)
        {
            return new ColliderCastInput
            {
                Collider = collider,
                Orientation = rotation,
                Start = position,
                End = position + direction * distance,
            };
        }

        private static bool IsCast(NativeStore.QueryInput* input, float3 direction)
        {
            return math.any(direction != 0.0f) && input->maxDistance > 0.0f;
        }

        private static int CastGeometry(QueryWorld* world, NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            PrimitiveCollider primitive = default;
            Collider* collider = primitive.Initialize(input->geometryType, input->extents, FilterOf(input));
            if (collider == null || !IsCast(input, direction))
            {
                return 0;
            }

            // The sink only learns a shape's closest hit once the shape is done, because a later triangle can still turn
            // that shape into a start overlap, which takes it out of the running.
            var collector = new QueryCollector<ColliderCastHit, ClosestSink<ColliderCastHit>>(
                world,
                new ClosestSink<ColliderCastHit> { MaxFraction = 1.0f },
                input->filterOptions,
                direction,
                QueryHitRules.Sweep,
                groupByShape: true);

            ColliderCastInput cast = CastInput(collider, input->position, input->rotation, direction, distance);
            var refined = new RefinedCastCollector<QueryCollector<ColliderCastHit, ClosestSink<ColliderCastHit>>>(world, collector, cast);
            world->Collision.CastCollider(cast, ref refined);
            refined.Inner.Finish();
            ref ClosestSink<ColliderCastHit> sink = ref refined.Inner.Sink;
            if (sink.Shape == null)
            {
                return 0;
            }

            output->hit = QueryCollectors.Record(sink.Hit.Position, sink.Hit.SurfaceNormal, sink.Hit.Fraction * distance, sink.Shape, sink.Backface);
            return 1;
        }

        private static void CastGeometryAll(QueryWorld* world, NativeStore.QueryInput* input)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            PrimitiveCollider primitive = default;
            Collider* collider = primitive.Initialize(input->geometryType, input->extents, FilterOf(input));
            if (collider == null || !IsCast(input, direction))
            {
                return;
            }

            var collector = new QueryCollector<ColliderCastHit, CastListSink>(
                world,
                ListSink(input, direction, distance),
                input->filterOptions,
                direction,
                QueryHitRules.Sweep,
                groupByShape: true,
                keepInitialOverlaps: input->keepInitialOverlaps != 0);

            ColliderCastInput cast = CastInput(collider, input->position, input->rotation, direction, distance);
            var refined = new RefinedCastCollector<QueryCollector<ColliderCastHit, CastListSink>>(world, collector, cast);
            world->Collision.CastCollider(cast, ref refined);
            refined.Inner.Finish();
        }

        private static CastListSink ListSink(NativeStore.QueryInput* input, float3 direction, float distance)
        {
            return new CastListSink
            {
                Collector = new FunctionPointer<NativeStore.HitCollectorFunction>((IntPtr)input->collector),
                Context = input->collectorContext,
                Distance = distance,
                Direction = direction,
                MaxFraction = 1.0f,
            };
        }

        /// <summary>
        /// Sweeps one of a body's shapes. Only convex shapes sweep, and every one of those does, triggers too: PhysX has no
        /// sweep for a triangle mesh, and Unity's filter applies to what the sweep can hit.
        /// </summary>
        private static void CastShape<S>(QueryWorld* world, NativeStore.QueryInput* input, NativeStore.Shape* shape, ref S sink, float3 direction, float distance)
            where S : struct, IQuerySink<ColliderCastHit>
        {
            RigidTransform worldFromShape = WorldPose(&shape->data);
            PrimitiveCollider primitive = default;
            Collider* collider = ColliderOf(shape, ref primitive, worldFromShape, out RigidTransform worldFromCollider);
            if (collider == null || collider->CollisionType != CollisionType.Convex)
            {
                return;
            }

            // The filter a query goes by is the query's, and a convex leaf holds its shape's. A blob holds relative offsets
            // only, so it can move, which is what the copy on this thread's stack relies on.
            int blocks = collider->MemorySize / sizeof(float4) + 1;
            float4* copy = stackalloc float4[blocks];
            UnsafeUtility.MemCpy(copy, collider, collider->MemorySize);
            collider = (Collider*)copy;
            collider->SetCollisionFilter(FilterOf(input));

            var collector = new QueryCollector<ColliderCastHit, S>(
                world,
                sink,
                input->filterOptions,
                direction,
                QueryHitRules.Sweep,
                groupByShape: true,
                ignoredBody: input->body);

            ColliderCastInput cast = CastInput(collider, worldFromCollider.pos, worldFromCollider.rot, direction, distance);
            var refined = new RefinedCastCollector<QueryCollector<ColliderCastHit, S>>(world, collector, cast);
            world->Collision.CastCollider(cast, ref refined);
            refined.Inner.Finish();
            sink = refined.Inner.Sink;
        }

        private static int CastBody(QueryWorld* world, NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            if (!IsCast(input, direction))
            {
                return 0;
            }

            var sink = new ClosestSink<ColliderCastHit> { MaxFraction = 1.0f };
            NativeStore.BodyData* body = StoreOps.Hot(input->body);
            for (int i = 0; i < body->shapeCount; i++)
            {
                CastShape(world, input, (NativeStore.Shape*)body->shapes[i], ref sink, direction, distance);
            }

            if (sink.Shape == null)
            {
                return 0;
            }

            output->hit = QueryCollectors.Record(sink.Hit.Position, sink.Hit.SurfaceNormal, sink.Hit.Fraction * distance, sink.Shape, sink.Backface);
            return 1;
        }

        private static void CastBodyAll(QueryWorld* world, NativeStore.QueryInput* input)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            if (!IsCast(input, direction))
            {
                return;
            }

            CastListSink sink = ListSink(input, direction, distance);
            NativeStore.BodyData* body = StoreOps.Hot(input->body);
            for (int i = 0; i < body->shapeCount; i++)
            {
                CastShape(world, input, (NativeStore.Shape*)body->shapes[i], ref sink, direction, distance);
            }
        }

        /// <summary>
        /// Sweeps a geometry against one shape. A geometry already in the shape is an overlap, which the caller either doesn't
        /// want, or wants as PhysX's minimum translation: the normal of the shortest way out, and the depth as a negative distance.
        /// </summary>
        private static int CastAgainstShape(NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            float3 worldDirection = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            PrimitiveCollider swept = default;
            PrimitiveCollider primitive = default;
            Collider* query = swept.Initialize(input->geometryType, input->extents, CollisionFilter.Default);
            Collider* target = ColliderOf(input->shape, ref primitive, WorldPose(input->shape), out RigidTransform worldFromTarget);
            if (query == null || target == null || !IsCast(input, worldDirection))
            {
                return 0;
            }

            RigidTransform targetFromWorld = math.inverse(worldFromTarget);
            float3 direction = math.rotate(targetFromWorld.rot, worldDirection);
            float3 start = math.transform(targetFromWorld, input->position);
            quaternion rotation = math.mul(targetFromWorld.rot, input->rotation);
            var collector = new SingleShapeCollector<ColliderCastHit>(target, input->filterOptions, direction, keepOverlaps: input->keepInitialOverlaps != 0);
            target->CastCollider(CastInput(query, start, rotation, direction, distance), ref collector);
            if (collector.NumHits == 0)
            {
                return 0;
            }

            ColliderCastHit hit = collector.Hit;
            float3 point = hit.Position;
            float3 normal = collector.Backface ? -hit.SurfaceNormal : hit.SurfaceNormal;
            float reported = hit.Fraction * distance;
            if (hit.Fraction <= 0.0f)
            {
                var deepest = new ClosestHitCollector<DistanceHit>(TouchDistance);
                ColliderDistanceInput overlap = new ColliderDistanceInput
                {
                    Collider = query,
                    Transform = new RigidTransform(rotation, start),
                    MaxDistance = TouchDistance,
                    Scale = 1.0f,
                };

                target->CalculateDistance(overlap, ref deepest);
                point = deepest.ClosestHit.Position;
                normal = deepest.ClosestHit.SurfaceNormal;
                reported = math.min(deepest.ClosestHit.Fraction, 0.0f);
            }

            output->hit = new NativeStore.RayCastHit
            {
                point = math.transform(worldFromTarget, point),
                normal = math.rotate(worldFromTarget.rot, normal),
                faceId = input->shape->data.geometryType == NativeStore.GeometryTriangleMesh ? 0u : QueryCollectors.NoFace,
                distance = reported,
            };

            return 1;
        }

        private static int ClosestPoint(NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            PrimitiveCollider primitive = default;
            RigidTransform worldFromShape = new RigidTransform(input->rotation, input->position);
            Collider* collider = ColliderOf(input->shape, ref primitive, worldFromShape, out RigidTransform worldFromCollider);
            if (collider == null || collider->CollisionType != CollisionType.Convex)
            {
                return 0;
            }

            float3 point = math.transform(math.inverse(worldFromCollider), input->position2);
            var collector = new ClosestHitCollector<DistanceHit>(float.MaxValue);
            PointDistanceInput distanceInput = new PointDistanceInput
            {
                Position = point,
                MaxDistance = float.MaxValue,
                Filter = CollisionFilter.Default,
            };

            collider->CalculateDistance(distanceInput, ref collector);

            // A point inside has no nearer one: PhysX reports the point itself.
            if (collector.NumHits == 0 || collector.ClosestHit.Fraction <= 0.0f)
            {
                return 0;
            }

            output->hit.point = math.transform(worldFromCollider, collector.ClosestHit.Position);
            output->hit.distance = collector.ClosestHit.Fraction;
            return 1;
        }

        private static int Penetration(NativeStore.QueryInput* input, NativeStore.QueryOutput* output)
        {
            PrimitiveCollider primitiveA = default;
            PrimitiveCollider primitiveB = default;
            RigidTransform poseA = new RigidTransform(input->rotation, input->position);
            RigidTransform poseB = new RigidTransform(input->rotation2, input->position2);
            Collider* a = ColliderOf(input->shape, ref primitiveA, poseA, out RigidTransform worldFromA);
            Collider* b = ColliderOf(input->shape2, ref primitiveB, poseB, out RigidTransform worldFromB);
            if (a == null || b == null)
            {
                return 0;
            }

            // A triangle mesh is only ever the target, the one place Unity Physics tells which triangle it measured.
            bool swapped = a->CollisionType == CollisionType.Convex && b->CollisionType != CollisionType.Convex;
            Collider* target = swapped ? b : a;
            Collider* query = swapped ? a : b;
            RigidTransform worldFromTarget = swapped ? worldFromB : worldFromA;
            RigidTransform worldFromQuery = swapped ? worldFromA : worldFromB;
            if (query->CollisionType != CollisionType.Convex)
            {
                return 0;
            }

            // The filters of a shape's leaf hold its layers, which a penetration test doesn't go by.
            int blocks = query->MemorySize / sizeof(float4) + 1;
            float4* copy = stackalloc float4[blocks];
            UnsafeUtility.MemCpy(copy, query, query->MemorySize);
            query = (Collider*)copy;
            query->SetCollisionFilter(CollisionFilter.Default);

            RigidTransform targetFromQuery = math.mul(math.inverse(worldFromTarget), worldFromQuery);
            float3 centre = math.transform(targetFromQuery, query->MassProperties.MassDistribution.Transform.pos);
            var collector = new PenetrationCollector(target, centre, TouchDistance);
            ColliderDistanceInput distanceInput = new ColliderDistanceInput
            {
                Collider = query,
                Transform = targetFromQuery,
                MaxDistance = TouchDistance,
                Scale = 1.0f,
            };

            target->CalculateDistance(distanceInput, ref collector);
            if (collector.NumHits == 0)
            {
                return 0;
            }

            float3 normal = math.rotate(worldFromTarget.rot, collector.Hit.SurfaceNormal);
            output->hit.normal = swapped ? normal : -normal;
            output->hit.distance = math.max(-collector.Hit.Fraction, 0.0f);
            return 1;
        }

        private static RigidTransform WorldPose(NativeStore.Shape* shape)
        {
            return WorldPose(&shape->data, shape->body);
        }

        private static RigidTransform WorldPose(NativeStore.ShapeData* shape)
        {
            return WorldPose(shape, shape->owner->body);
        }

        private static RigidTransform WorldPose(NativeStore.ShapeData* shape, NativeStore.Body* owner)
        {
            NativeStore.BodyData* body = StoreOps.Hot(owner);
            return math.mul(new RigidTransform(body->pose->Rotation, body->pose->Position), new RigidTransform(shape->localRotation, shape->localPosition));
        }

        /// <summary>
        /// The collider a shape's geometry is, and the pose that puts it where <paramref name="worldFromShape"/> says the shape
        /// is. A primitive is built in <paramref name="primitive"/> with a filter that lets everything through. A mesh takes too
        /// long to build for one query, so it is the leaf of its body's collider, whose filter and pose are the shape's. A
        /// compound's child carries the pose outside it, as a static body's only shape does, but any other single shape's blob
        /// has it baked in. Null for a shape with no
        /// such collider, which is what a shape off its body looks like, and PhysX answers for those as if it wasn't there.
        /// </summary>
        private static Collider* ColliderOf(
            NativeStore.Shape* shape,
            ref PrimitiveCollider primitive,
            RigidTransform worldFromShape,
            out RigidTransform worldFromCollider)
        {
            worldFromCollider = worldFromShape;
            Collider* collider = primitive.Initialize(shape->data.geometryType, shape->data.extents, CollisionFilter.Default);
            if (collider != null || shape->body == null || !ColliderBuilder.IsSupported(&shape->data))
            {
                return collider;
            }

            NativeStore.BodyData* body = StoreOps.Hot(shape->body);
            Collider* root = (Collider*)body->collider;
            if (root == null)
            {
                return null;
            }

            if (root->Type != ColliderType.Compound)
            {
                if (ColliderBuilder.PosedShape(body) == null)
                {
                    RigidTransform local = new RigidTransform(shape->data.localRotation, shape->data.localPosition);
                    worldFromCollider = math.mul(worldFromShape, math.inverse(local));
                }

                return root;
            }

            int index = 0;
            while (index < body->shapeCount && body->shapes[index] != &shape->data)
            {
                index++;
            }

            CompoundCollider* compound = (CompoundCollider*)root;
            for (int i = 0; i < compound->NumChildren; i++)
            {
                ref CompoundCollider.Child child = ref compound->Children[i];
                if (child.Entity.Index == index)
                {
                    if (shape->data.geometryType == NativeStore.GeometryHeightField)
                    {
                        worldFromCollider = math.mul(worldFromShape, ColliderBuilder.ShapeFromGrid(&shape->data));
                    }

                    return child.Collider;
                }
            }

            return null;
        }
    }
}
