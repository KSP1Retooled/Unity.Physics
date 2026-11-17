using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Collision.Queries.Collectors;

namespace KSP.Retooled.Physics
{

    /// <summary>
    /// Execute plugin queries using Unity Physics geometry and convert results to PhysX conventions.
    /// Disable Burst safety exceptions across the native boundary; the Mono fallback catches exceptions.
    /// </summary>
    [BurstCompile]
    internal static unsafe class QueryKernel
    {
        // PhysX counts touching shapes as overlapping. Unity Physics only reports distances below the specified limit,
        // so use a small positive limit to include touching shapes.
        private const float TouchDistance = 1e-4f;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int QueryFunction(uint kind, QueryWorld* world, QueryInput* input, QueryOutput* output);

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(QueryFunction))]
        internal static int Run(uint kind, QueryWorld* world, QueryInput* input, QueryOutput* output)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.Query.Auto();

            int result = 0;
            bool ranInMono = false;
            RunInMono(kind, world, input, output, ref result, ref ranInMono);
            return ranInMono ? result : Execute((QueryKind)kind, world, input, output);
        }

        /// <summary>
        /// Burst omits this method. It runs only in Mono, where exceptions must be caught before returning to native
        /// code.
        /// </summary>
        [BurstDiscard]
        private static void RunInMono(
            uint kind,
            QueryWorld* world,
            QueryInput* input,
            QueryOutput* output,
            ref int result,
            ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedQuery();
            try
            {
                // Run through a job in Mono so that native containers have the job's safety handles.
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
            public QueryInput* Input;

            [NativeDisableUnsafePtrRestriction]
            public QueryOutput* Output;

            [NativeDisableUnsafePtrRestriction]
            public int* Result;

            public void Execute()
            {
                *Result = QueryKernel.Execute((QueryKind)Kind, (QueryWorld*)UnsafeUtility.AddressOf(ref World), Input, Output);
            }
        }

        private static int Execute(QueryKind kind, QueryWorld* world, QueryInput* input, QueryOutput* output)
        {
            switch (kind)
            {
                case QueryKind.Ray:
                    return CastRay(world, input, output);
                case QueryKind.RayAll:
                    CastRayAll(world, input);
                    return 0;
                case QueryKind.RayAgainstShape:
                    return CastRayAgainstShape(input, output);
                case QueryKind.OverlapAny:
                    return OverlapAny(world, input, output);
                case QueryKind.OverlapAll:
                    OverlapAll(world, input);
                    return 0;
                case QueryKind.OverlapBroadphase:
                    OverlapBroadphase(world, input);
                    return 0;
                case QueryKind.Cast:
                    return CastGeometry(world, input, output);
                case QueryKind.CastAll:
                    CastGeometryAll(world, input);
                    return 0;
                case QueryKind.BodyCast:
                    return CastBody(world, input, output);
                case QueryKind.BodyCastAll:
                    CastBodyAll(world, input);
                    return 0;
                case QueryKind.ClosestPoint:
                    return ClosestPoint(input, output);
                case QueryKind.Penetration:
                    return Penetration(input, output);
                case QueryKind.CastAgainstShape:
                    return CastAgainstShape(input, output);
                default:
                    return 0;
            }
        }

        private static CollisionFilter FilterOf(QueryInput* input)
        {
            return new CollisionFilter
            {
                BelongsTo = uint.MaxValue,
                CollidesWith = (uint)input->layerMask,
            };
        }

        private static float LengthOf(QueryInput* input)
        {
            return math.min(input->maxDistance, QueryCollectors.MaxQueryLength);
        }

        private static RaycastInput RayOf(QueryInput* input, float3 direction, float distance)
        {
            return new RaycastInput
            {
                Start = input->position,
                End = input->position + direction * distance,
                Filter = FilterOf(input),
            };
        }

        private static int CastRay(QueryWorld* world, QueryInput* input, QueryOutput* output)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);

            // Hits do not need to be grouped by shape here. Only the closest triangle hit in a mesh can remain as the
            // closest hit overall.
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

            output->hit = Record(sink.Hit.Position, sink.Hit.SurfaceNormal, sink.Hit.Fraction * distance, sink.Shape, sink.Backface);
            return 1;
        }

        private static void CastRayAll(QueryWorld* world, QueryInput* input)
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
                groupByShape: (input->filterOptions & StoreConstants.QueryAllowMultipleMeshHits) == 0);

            world->Collision.CastRay(RayOf(input, direction, distance), ref collector);
            collector.Finish();
        }

        private static int CastRayAgainstShape(QueryInput* input, QueryOutput* output)
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
            output->hit = new RayCastHit
            {
                point = math.transform(worldFromCollider, hit.Position),
                normal = collector.Backface ? -normal : normal,
                faceId = input->shape->data.geometryType == StoreConstants.GeometryTriangleMesh ? 0u : NoFace,
                distance = hit.Fraction * distance,
            };

            return 1;
        }

        private static ColliderDistanceInput OverlapInput(Collider* collider, QueryInput* input)
        {
            return new ColliderDistanceInput
            {
                Collider = collider,
                Transform = new RigidTransform(input->rotation, input->position),
                MaxDistance = TouchDistance,
                Scale = 1.0f,
            };
        }

        private static int OverlapAny(QueryWorld* world, QueryInput* input, QueryOutput* output)
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

        private static void OverlapAll(QueryWorld* world, QueryInput* input)
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

        private static void OverlapBroadphase(QueryWorld* world, QueryInput* input)
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

        private static bool IsCast(QueryInput* input, float3 direction)
        {
            return math.any(direction != 0.0f) && input->maxDistance > 0.0f;
        }

        private static int CastGeometry(QueryWorld* world, QueryInput* input, QueryOutput* output)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            PrimitiveCollider primitive = default;
            Collider* collider = primitive.Initialize(input->geometryType, input->extents, FilterOf(input));
            if (collider == null || !IsCast(input, direction))
            {
                return 0;
            }

            // Send a shape's closest hit to the sink only after all its triangles have been processed. A later triangle
            // may reveal an overlap at the start of the cast, which excludes the shape from the results.
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

            output->hit = Record(sink.Hit.Position, sink.Hit.SurfaceNormal, sink.Hit.Fraction * distance, sink.Shape, sink.Backface);
            return 1;
        }

        private static void CastGeometryAll(QueryWorld* world, QueryInput* input)
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

        private static CastListSink ListSink(QueryInput* input, float3 direction, float distance)
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
        /// Sweep a body's convex shapes, including triggers, against filtered targets. Triangle meshes cannot be
        /// swept.
        /// </summary>
        private static void CastShape<S>(QueryWorld* world, QueryInput* input, Shape* shape, ref S sink, float3 direction, float distance)
            where S : struct, IQuerySink<ColliderCastHit>
        {
            RigidTransform worldFromShape = WorldPose(&shape->data);
            PrimitiveCollider primitive = default;
            Collider* collider = ColliderOf(shape, ref primitive, worldFromShape, out RigidTransform worldFromCollider);
            if (collider == null || collider->CollisionType != CollisionType.Convex)
            {
                return;
            }

            // Copy the leaf to thread-local storage before replacing its filter. Blob offsets remain valid in
            // the copy.
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

        private static int CastBody(QueryWorld* world, QueryInput* input, QueryOutput* output)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            if (!IsCast(input, direction))
            {
                return 0;
            }

            var sink = new ClosestSink<ColliderCastHit> { MaxFraction = 1.0f };
            BodyData* body = StoreOps.Hot(input->body);
            for (int i = 0; i < body->shapeCount; i++)
            {
                CastShape(world, input, (Shape*)body->shapes[i], ref sink, direction, distance);
            }

            if (sink.Shape == null)
            {
                return 0;
            }

            output->hit = Record(sink.Hit.Position, sink.Hit.SurfaceNormal, sink.Hit.Fraction * distance, sink.Shape, sink.Backface);
            return 1;
        }

        private static void CastBodyAll(QueryWorld* world, QueryInput* input)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = LengthOf(input);
            if (!IsCast(input, direction))
            {
                return;
            }

            CastListSink sink = ListSink(input, direction, distance);
            BodyData* body = StoreOps.Hot(input->body);
            for (int i = 0; i < body->shapeCount; i++)
            {
                CastShape(world, input, (Shape*)body->shapes[i], ref sink, direction, distance);
            }
        }

        /// <summary>
        /// Sweep against one shape. When initial overlaps are requested, report the minimum translation normal and
        /// negative penetration depth.
        /// </summary>
        private static int CastAgainstShape(QueryInput* input, QueryOutput* output)
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

            output->hit = new RayCastHit
            {
                point = math.transform(worldFromTarget, point),
                normal = math.rotate(worldFromTarget.rot, normal),
                faceId = input->shape->data.geometryType == StoreConstants.GeometryTriangleMesh ? 0u : NoFace,
                distance = reported,
            };

            return 1;
        }

        private static int ClosestPoint(QueryInput* input, QueryOutput* output)
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

            // For a point inside the shape, PhysX reports that same point as the closest point.
            if (collector.NumHits == 0 || collector.ClosestHit.Fraction <= 0.0f)
            {
                return 0;
            }

            output->hit.point = math.transform(worldFromCollider, collector.ClosestHit.Position);
            output->hit.distance = collector.ClosestHit.Fraction;
            return 1;
        }

        private static int Penetration(QueryInput* input, QueryOutput* output)
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

            // Use a triangle mesh only as the target. Unity Physics identifies the triangle used in the distance
            // calculation only for the target.
            bool swapped = a->CollisionType == CollisionType.Convex && b->CollisionType != CollisionType.Convex;
            Collider* target = swapped ? b : a;
            Collider* query = swapped ? a : b;
            RigidTransform worldFromTarget = swapped ? worldFromB : worldFromA;
            RigidTransform worldFromQuery = swapped ? worldFromA : worldFromB;
            if (query->CollisionType != CollisionType.Convex)
            {
                return 0;
            }

            // A shape's leaf collider contains its layer filter. Copy the query collider and allow all layers because
            // penetration tests ignore these filters.
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

        private static RigidTransform WorldPose(Shape* shape)
        {
            return WorldPose(&shape->data, shape->body);
        }

        private static RigidTransform WorldPose(ShapeData* shape)
        {
            return WorldPose(shape, shape->owner->body);
        }

        private static RigidTransform WorldPose(ShapeData* shape, Body* owner)
        {
            BodyData* body = StoreOps.Hot(owner);
            return math.mul(new RigidTransform((*body).Pose()->Rotation, (*body).Pose()->Position), new RigidTransform(shape->localRotation, shape->localPosition));
        }

        /// <summary>
        /// Return a shape collider and its world pose, or null if unavailable.
        /// Build primitives in the supplied storage; reuse mesh leaves from the body's collider.
        /// Detached shapes can supply primitives but not mesh leaves.
        /// </summary>
        private static Collider* ColliderOf(
            Shape* shape,
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

            BodyData* body = StoreOps.Hot(shape->body);
            Collider* root = (Collider*)(*body).Collider()->Value.GetUnsafePtr();
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
                    if (shape->data.geometryType == StoreConstants.GeometryHeightField)
                    {
                        worldFromCollider = math.mul(worldFromShape, ColliderBuilder.ShapeFromGrid(&shape->data));
                    }

                    return child.Collider;
                }
            }

            return null;
        }

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
    }
}
