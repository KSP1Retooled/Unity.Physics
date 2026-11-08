using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Turns a body's shapes into one Unity Physics collider: the shape alone when there is one, else a compound whose
    /// children record their shape index in <see cref="CompoundCollider.ColliderBlobInstance.Entity"/>. Colliders are
    /// held by the store as raw blob pointers, which is why a domain reload can release and rebuild them.
    /// </summary>
    [BurstCompile]
    internal static unsafe class ColliderBuilder
    {
        // PhysX allows a degenerate extent, Unity Physics refuses one.
        private const float MinimumExtent = 1e-4f;

        // Well above the 252 vertices a convex collider keeps, and small enough for the hull builder's stack use and time.
        private const int MaximumHullInput = 1024;

        internal static BlobAssetReference<Collider> FromPointer(void* collider)
        {
            return UnsafeUtility.As<System.IntPtr, BlobAssetReference<Collider>>(ref *(System.IntPtr*)&collider);
        }

        internal static void Release(void* collider)
        {
            if (collider != null)
            {
                FromPointer(collider).Dispose();
            }
        }

        private const NativeStore.BodyFlags NeedsBuild = NativeStore.BodyFlags.ColliderDirty | NativeStore.BodyFlags.CollidersMoved;

        // What a build that threw leaves in its result.
        private const long Failed = -1;

        /// <summary>A body whose collider a refresh builds, and whether it needs a whole build or only a move of its children.</summary>
        internal struct Work
        {
            public int Body;
            public bool Whole;
        }

        /// <summary>
        /// Schedules a new collider for every body that asks for one; <paramref name="all"/> treats every body as asking. The
        /// builds overlap whatever the main thread does next. Nothing may read a body's collider or write its flags until the
        /// returned handle completes, and <paramref name="work"/> and <paramref name="results"/> must outlive it.
        /// </summary>
        internal static JobHandle ScheduleRefresh(
            NativeStore.WorldData* world,
            bool all,
            NativeList<Work> work,
            NativeList<long> results,
            JobHandle dependency)
        {
            if (work.Capacity < world->bodyCount)
                work.Capacity = (int)world->bodyCount;
            work.Clear();

            // This writes the flags but nothing else writes them until it
            // completes, so it is safe to do so.
            var aggregateWork = new AggregateColliderWorkJob
            {
                World = world,
                Work = work.AsParallelWriter(),
                All = all
            }.ScheduleBatch((int)world->bodyCount, 1024, dependency);

            dependency = new InitializeResultsJob { Work = work, Results = results }.Schedule(aggregateWork);

            // One body a work item: a mesh can take a thousand times as long as a box.
            JobHandle built = new BuildCollidersJob
            {
                Bodies = world->bodies,
                Work = work.AsDeferredJobArray(),
                Results = results.AsDeferredJobArray(),
            }.Schedule(work, 1, dependency);

            var handle = new SwapCollidersJob
            {
                Bodies = world->bodies,
                Work = work.AsDeferredJobArray(),
                Results = results.AsDeferredJobArray(),
            }.Schedule(built);

            JobHandle.ScheduleBatchedJobs();

            return handle;
        }

        /// <summary>
        /// Asks again for a whole build of every body whose last build threw, which left it with its last collider. Main
        /// thread only, once that refresh completed.
        /// </summary>
        internal static void AskAgainForFailed(NativeStore.WorldData* world, NativeList<Work> work, NativeList<long> results)
        {
            for (int i = 0; i < results.Length; i++)
            {
                if (results[i] == Failed && work[i].Body < world->bodyCount)
                {
                    world->bodies[work[i].Body].flags |= NativeStore.BodyFlags.ColliderDirty;
                }
            }

            results.Clear();
        }

        [BurstCompile]
        private struct AggregateColliderWorkJob : IJobParallelForBatch
        {
            [NoAlias]
            [NativeDisableUnsafePtrRestriction]
            public NativeStore.WorldData* World;

            public NativeList<Work>.ParallelWriter Work;
            public bool All;

            public void Execute(int start, int length)
            {
                var end = start + length;

                var chunk = new NativeList<Work>(1024, Allocator.Temp);

                for (int i = start; i < end; ++i)
                {
                    ref var body = ref World->bodies[i];

                    if (!All && (body.flags & NeedsBuild) == 0)
                        continue;

                    chunk.Add(new()
                    {
                        Body = i,
                        Whole = All || (body.flags & NativeStore.BodyFlags.ColliderDirty) != 0
                    });
                    body.flags &= ~NeedsBuild;
                }

                Work.AddRangeNoResize(chunk.AsArray());
            }
        }

        [BurstCompile]
        private struct InitializeResultsJob : IJob
        {
            [ReadOnly]
            public NativeList<Work> Work;

            public NativeList<long> Results;

            public void Execute()
            {
                Results.ResizeUninitialized(Work.Length);

                for (int i = 0; i < Results.Length; ++i)
                    Results[i] = Failed;
            }
        }

        [BurstCompile]
        private struct BuildCollidersJob : IJobParallelForDefer
        {
            [NativeDisableUnsafePtrRestriction]
            public NativeStore.BodyData* Bodies;

            [ReadOnly]
            public NativeArray<Work> Work;

            public NativeArray<long> Results;

            public void Execute(int index)
            {
                NativeStore.BodyData* body = Bodies + Work[index].Body;
                void* collider = Work[index].Whole ? null : Repose(body);
                Results[index] = (long)(collider != null ? collider : Build(body));
            }
        }

        /// <summary>Gives each body the collider built for it and releases the one it replaces.</summary>
        [BurstCompile]
        private struct SwapCollidersJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public NativeStore.BodyData* Bodies;

            [ReadOnly]
            public NativeArray<Work> Work;

            [ReadOnly]
            public NativeArray<long> Results;

            public void Execute()
            {
                for (int i = 0; i < Results.Length; i++)
                {
                    if (Results[i] == Failed)
                    {
                        continue;
                    }

                    NativeStore.BodyData* body = Bodies + Work[i].Body;
                    Release(body->collider->Value.GetUnsafePtr());
                    body->collider->Value = FromPointer((void*)Results[i]);
                }
            }
        }

        /// <summary>
        /// A compound again from the children of the one built before, at its shapes' new poses, or null if the body has none.
        /// Building a child again would mean building its hull or hierarchy, while this only copies it.
        /// </summary>
        private static void* Repose(NativeStore.BodyData* body)
        {
            BlobAssetReference<Collider> built = body->collider->Value;
            if (!built.IsCreated || built.Value.Type != ColliderType.Compound)
            {
                return null;
            }

            ref CompoundCollider compound = ref UnsafeUtility.As<Collider, CompoundCollider>(ref built.Value);
            var children = new NativeArray<CompoundCollider.ColliderBlobInstance>(compound.NumChildren, Allocator.Temp);
            try
            {
                for (int i = 0; i < compound.NumChildren; i++)
                {
                    ref CompoundCollider.Child child = ref compound.Children[i];
                    NativeStore.ShapeData* shape = body->shapes[child.Entity.Index];
                    RigidTransform compoundFromShape = new RigidTransform(shape->localRotation, shape->localPosition);
                    bool heightField = shape->geometryType == NativeStore.GeometryHeightField;
                    children[i] = new CompoundCollider.ColliderBlobInstance
                    {
                        Collider = BlobAssetReference<Collider>.Create(child.Collider, child.Collider->MemorySize),
                        CompoundFromChild = heightField ? math.mul(compoundFromShape, ShapeFromGrid(shape)) : compoundFromShape,
                        Entity = child.Entity,
                    };
                }

                return CompoundCollider.Create(children).GetUnsafePtr();
            }
            finally
            {
                foreach (CompoundCollider.ColliderBlobInstance child in children)
                {
                    if (child.Collider.IsCreated)
                    {
                        child.Collider.Dispose();
                    }
                }

                children.Dispose();
            }
        }

        private static void* Build(NativeStore.BodyData* body)
        {
            int usable = 0;
            int lastUsable = -1;
            for (int i = 0; i < body->shapeCount; i++)
            {
                if (IsSupported(body->shapes[i]))
                {
                    usable++;
                    lastUsable = i;
                }
            }

            if (usable == 0)
            {
                return null;
            }

            if (usable == 1 && body->shapes[lastUsable]->geometryType != NativeStore.GeometryHeightField)
            {
                NativeStore.ShapeData* shape = body->shapes[lastUsable];
                if (PosedShape(body) != null)
                {
                    return Create(shape, float3.zero, quaternion.identity).GetUnsafePtr();
                }

                // A mesh away from its body's origin is a compound of one, so that moving it again only moves the child:
                // with the pose baked in, each move would build its hull or hierarchy again.
                if (!ShapeMath.IsMeshGeometry(shape->geometryType) || IsAtOrigin(shape))
                {
                    return Create(shape, shape->localPosition, shape->localRotation).GetUnsafePtr();
                }
            }

            var children = new NativeArray<CompoundCollider.ColliderBlobInstance>(usable, Allocator.Temp);
            try
            {
                int next = 0;
                for (int i = 0; i < body->shapeCount; i++)
                {
                    NativeStore.ShapeData* shape = body->shapes[i];
                    if (!IsSupported(shape))
                    {
                        continue;
                    }

                    RigidTransform compoundFromShape = new RigidTransform(shape->localRotation, shape->localPosition);
                    bool heightField = shape->geometryType == NativeStore.GeometryHeightField;
                    children[next++] = new CompoundCollider.ColliderBlobInstance
                    {
                        Collider = Create(shape, float3.zero, quaternion.identity),
                        CompoundFromChild = heightField ? math.mul(compoundFromShape, ShapeFromGrid(shape)) : compoundFromShape,
                        Entity = new Entity { Index = i, Version = 1 },
                    };
                }

                return CompoundCollider.Create(children).GetUnsafePtr();
            }
            finally
            {
                foreach (CompoundCollider.ColliderBlobInstance child in children)
                {
                    if (child.Collider.IsCreated)
                    {
                        child.Collider.Dispose();
                    }
                }

                children.Dispose();
            }
        }

        /// <summary>
        /// The index among a body's shapes of what a key names. A compound's child knows it, one level down only, because a
        /// polygon inside a mesh child doesn't, and any other collider stands for <paramref name="singleShape"/>.
        /// </summary>
        internal static int ShapeIndexOf(BlobAssetReference<Collider> collider, ColliderKey key, int singleShape)
        {
            if (collider.Value.Type != ColliderType.Compound)
            {
                return singleShape;
            }

            collider.Value.GetChild(ref key, out ChildCollider child);
            return child.Entity.Index;
        }

        /// <summary>
        /// The shape whose pose the built body carries instead of its collider, or null. Unity moves a static collider by
        /// posing its shape, every frame for an animated one, and a pose baked into a mesh would mean rebuilding the mesh each
        /// time. Unity joints can't reach a static body, so nothing else depends on where its built transform sits.
        /// </summary>
        internal static NativeStore.ShapeData* PosedShape(NativeStore.BodyData* body)
        {
            if ((body->flags & NativeStore.BodyFlags.Static) == 0)
            {
                return null;
            }

            NativeStore.ShapeData* posed = null;
            for (int i = 0; i < body->shapeCount; i++)
            {
                if (!IsSupported(body->shapes[i]))
                {
                    continue;
                }

                if (posed != null)
                {
                    return null;
                }

                posed = body->shapes[i];
            }

            return posed != null && posed->geometryType != NativeStore.GeometryHeightField ? posed : null;
        }

        private static bool IsAtOrigin(NativeStore.ShapeData* shape)
        {
            return math.all(shape->localPosition == float3.zero) && math.all(shape->localRotation.value == quaternion.identity.value);
        }

        /// <summary>The index of the shape a body's collider stands for when it isn't a compound.</summary>
        internal static int FirstSupported(NativeStore.BodyData* body)
        {
            for (int i = 0; i < body->shapeCount; i++)
            {
                if (IsSupported(body->shapes[i]))
                {
                    return i;
                }
            }

            return 0;
        }

        internal static bool IsSupported(NativeStore.ShapeData* shape)
        {
            switch (shape->geometryType)
            {
                case NativeStore.GeometryBox:
                case NativeStore.GeometrySphere:
                case NativeStore.GeometryCapsule:
                    return true;
                case NativeStore.GeometryConvexMesh:
                case NativeStore.GeometryTriangleMesh:
                {
                    NativeStore.MeshData* mesh = NativeStore.MeshOf(shape);
                    return mesh != null && mesh->vertexCount != 0 && math.all(shape->extents != 0.0f);
                }

                case NativeStore.GeometryHeightField:
                {
                    NativeStore.HeightField* field = (NativeStore.HeightField*)shape->mesh;
                    if (field == null || !math.all(shape->extents > 0.0f))
                    {
                        return false;
                    }

                    long cells = field->resolution - 1L;
                    return field->holeCount < cells * cells;
                }

                default:
                    return false;
            }
        }

        private static BlobAssetReference<Collider> Create(NativeStore.ShapeData* shape, float3 center, quaternion orientation)
        {
            // Unity Physics applies a query's filter both ways, so a collider that collides with nothing would also be
            // invisible to every ray. Such a collider opens its filter here and gives up the contacts in MaterialOf.
            CollisionFilter filter = new CollisionFilter
            {
                BelongsTo = shape->belongsTo,
                CollidesWith = shape->collidesWith != 0 ? shape->collidesWith : uint.MaxValue,
                GroupIndex = 0,
            };

            Material material = MaterialOf(shape);
            switch (shape->geometryType)
            {
                case NativeStore.GeometryBox:
                    return BoxCollider.Create(BoxOf(shape->extents, center, orientation), filter, material);
                case NativeStore.GeometrySphere:
                    return SphereCollider.Create(SphereOf(shape->extents, center), filter, material);
                case NativeStore.GeometryConvexMesh:
                case NativeStore.GeometryTriangleMesh:
                    return CreateMesh(shape, center, orientation, filter, material);
                case NativeStore.GeometryHeightField:
                    return CreateHeightField(shape, filter, material);
                default:
                    return CapsuleCollider.Create(CapsuleOf(shape->extents, center, orientation), filter, material);
            }
        }

        /// <summary>
        /// Where a height field's grid lies in its shape: a quarter turn about Y, with the grid's rows along X and its columns
        /// running back along Z. PhysX splits every cell from (x, z) to (x + 1, z + 1), and Unity Physics' queries always from
        /// (i + 1, j) to (i, j + 1), whatever TerrainCollider.Create is told, so only turning the grid makes the two agree.
        /// </summary>
        internal static RigidTransform ShapeFromGrid(NativeStore.ShapeData* shape)
        {
            NativeStore.HeightField* field = (NativeStore.HeightField*)shape->mesh;
            return new RigidTransform(quaternion.RotateY(math.PI / 2.0f), new float3(0.0f, 0.0f, (field->resolution - 1) * shape->extents.z));
        }

        private static BlobAssetReference<Collider> CreateHeightField(NativeStore.ShapeData* shape, CollisionFilter filter, Material material)
        {
            void* collider = null;
            new HeightFieldColliderJob
            {
                Field = (NativeStore.HeightField*)shape->mesh,
                Scale = shape->extents.zyx,
                Filter = filter,
                Material = material,
                Result = &collider,
            }.Execute();

            return FromPointer(collider);
        }

        internal static BoxGeometry BoxOf(float3 extents, float3 center, quaternion orientation)
        {
            return new BoxGeometry
            {
                Center = center,
                Orientation = orientation,
                Size = math.max(extents * 2.0f, MinimumExtent),
                BevelRadius = 0.0f,
            };
        }

        internal static SphereGeometry SphereOf(float3 extents, float3 center)
        {
            return new SphereGeometry { Center = center, Radius = math.max(extents.x, MinimumExtent) };
        }

        /// <summary>The extents hold the radius and half of the cylinder's length, along local X as in PhysX.</summary>
        internal static CapsuleGeometry CapsuleOf(float3 extents, float3 center, quaternion orientation)
        {
            float3 halfAxis = math.rotate(orientation, new float3(math.max(extents.y, MinimumExtent), 0.0f, 0.0f));
            return new CapsuleGeometry
            {
                Vertex0 = center - halfAxis,
                Vertex1 = center + halfAxis,
                Radius = math.max(extents.x, MinimumExtent),
            };
        }

        private static BlobAssetReference<Collider> CreateMesh(
            NativeStore.ShapeData* shape,
            float3 center,
            quaternion orientation,
            CollisionFilter filter,
            Material material)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.BuildMeshCollider.Auto();

            void* collider = null;
            new MeshColliderJob
            {
                Mesh = NativeStore.MeshOf(shape),
                Scale = shape->extents,
                Center = center,
                Orientation = orientation,
                Filter = filter,
                Material = material,
                Result = &collider,
            }.Execute();

            return FromPointer(collider);
        }

        /// <summary>
        /// A convex collider of the points, as they stand. The hull builder stackallocs in proportion to its input, and the
        /// main thread has 1 MB, and it tests every point against every face, so a large input is cut down first. It keeps
        /// no more than a collider's 252 vertices in any case, so that loses next to nothing.
        /// </summary>
        private static BlobAssetReference<Collider> CreateConvex(NativeArray<float3> points, CollisionFilter filter, Material material)
        {
            // PhysX collides with the hull as it stands, so nothing is rounded off or merged.
            ConvexHullGenerationParameters exact = new ConvexHullGenerationParameters
            {
                BevelRadius = 0.0f,
                SimplificationTolerance = 0.0f,
                MinimumAngle = 0.0f,
            };

            NativeArray<float3> input = points.Length > MaximumHullInput ? SupportPoints(points) : points;
            return ConvexCollider.Create(input, exact, filter, material);
        }

        /// <summary>
        /// The points furthest along evenly spread directions, which all lie on the hull. The axes come first so the result
        /// keeps the bounds of the whole.
        /// </summary>
        private static NativeArray<float3> SupportPoints(NativeArray<float3> points)
        {
            const float GoldenAngle = 2.39996323f;
            const int AxisDirections = 6;
            int spread = MaximumHullInput - AxisDirections;

            var chosen = new NativeArray<bool>(points.Length, Allocator.Temp);
            for (int axis = 0; axis < AxisDirections; axis++)
            {
                float3 direction = float3.zero;
                direction[axis / 2] = axis % 2 == 0 ? 1.0f : -1.0f;
                chosen[FurthestAlong(points, direction)] = true;
            }

            for (int i = 0; i < spread; i++)
            {
                float z = 1.0f - (2.0f * i + 1.0f) / spread;
                float radius = math.sqrt(1.0f - z * z);
                float angle = i * GoldenAngle;
                chosen[FurthestAlong(points, new float3(radius * math.cos(angle), radius * math.sin(angle), z))] = true;
            }

            var result = new NativeList<float3>(MaximumHullInput, Allocator.Temp);
            for (int i = 0; i < points.Length; i++)
            {
                if (chosen[i])
                {
                    result.Add(points[i]);
                }
            }

            return result.AsArray();
        }

        private static int FurthestAlong(NativeArray<float3> points, float3 direction)
        {
            int furthest = 0;
            float best = float.MinValue;
            for (int i = 0; i < points.Length; i++)
            {
                float distance = math.dot(points[i], direction);
                if (distance > best)
                {
                    best = distance;
                    furthest = i;
                }
            }

            return furthest;
        }

        /// <summary>
        /// The solid of a convex mesh's hull, unscaled. Burst compiled, like <see cref="BuildCollidersJob"/>, so that it stays
        /// quick when a body command in Mono asks for it. A command runs on the main thread, which has a temp allocator.
        /// </summary>
        [BurstCompile]
        internal static void ConvexMoments(NativeStore.MeshData* mesh, NativeStore.MeshMoments* moments)
        {
            int vertexCount = (int)mesh->vertexCount;
            var vertices = new NativeArray<float3>(vertexCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            UnsafeUtility.MemCpy(vertices.GetUnsafePtr(), NativeStore.VerticesOf(mesh), vertexCount * (long)sizeof(float3));

            BlobAssetReference<Collider> hull = CreateConvex(vertices, default, default);
            MassProperties mass = hull.Value.MassProperties;
            hull.Dispose();

            // Back from the principal axes to the mesh's own, where a scale applies. C = trace(I) / 2 - I for a unit mass.
            float3x3 axes = new float3x3(mass.MassDistribution.Transform.rot);
            float3 principal = mass.MassDistribution.InertiaTensor;
            float3x3 inertia = math.mul(axes, math.mul(float3x3.Scale(principal), math.transpose(axes)));
            float halfTrace = math.csum(principal) * 0.5f;

            moments->volume = mass.Volume;
            moments->centroid = mass.MassDistribution.Transform.pos;
            moments->covarianceDiagonal = halfTrace - new float3(inertia.c0.x, inertia.c1.y, inertia.c2.z);
            moments->covarianceOffDiagonal = -new float3(inertia.c1.x, inertia.c2.x, inertia.c2.y);
        }

        /// <summary>
        /// Builds one mesh's blob, with the shape's scale and pose baked in, since Unity Physics has neither per collider.
        /// <see cref="BuildCollidersJob"/> calls it in place.
        /// </summary>
        [BurstCompile]
        private struct MeshColliderJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public NativeStore.MeshData* Mesh;

            public float3 Scale;
            public float3 Center;
            public quaternion Orientation;
            public CollisionFilter Filter;
            public Material Material;

            [NativeDisableUnsafePtrRestriction]
            public void** Result;

            public void Execute()
            {
                int vertexCount = (int)Mesh->vertexCount;
                var vertices = new NativeArray<float3>(vertexCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                float3* source = NativeStore.VerticesOf(Mesh);
                for (int i = 0; i < vertexCount; i++)
                {
                    vertices[i] = Center + math.rotate(Orientation, source[i] * Scale);
                }

                BlobAssetReference<Collider> collider;
                if (Mesh->convex != 0)
                {
                    collider = CreateConvex(vertices, Filter, Material);
                }
                else
                {
                    // A mirroring scale turns a triangle inside out unless its winding turns too.
                    bool mirrored = Scale.x * Scale.y * Scale.z < 0.0f;
                    int triangleCount = (int)Mesh->triangleCount;
                    var triangles = new NativeArray<int3>(triangleCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    int3* sourceTriangles = NativeStore.TrianglesOf(Mesh);
                    for (int i = 0; i < triangleCount; i++)
                    {
                        int3 triangle = sourceTriangles[i];
                        triangles[i] = mirrored ? triangle.xzy : triangle;
                    }

                    collider = MeshCollider.Create(vertices, triangles, Filter, Material);
                }

                *Result = collider.GetUnsafePtr();
            }
        }

        /// <summary>
        /// Builds a height field's blob over the grid that <see cref="ShapeFromGrid"/> places. Grid cell (i, j) is the source
        /// cell (j, last - i). Unity Physics' terrain has no holes, so a field with any is a triangle mesh of its solid cells,
        /// split the same way, whose triangles face up as a mesh's do. A terrain collides by its triangles as PhysX does:
        /// VertexSamples pushes up any vertex under the surface, which throws a body whose centre is below it into the air.
        /// </summary>
        [BurstCompile]
        private struct HeightFieldColliderJob : IJob
        {
            // Unity Physics stretches the tallest sample to the whole range of a short, which a field of zeros would divide
            // by, so one corner rises by this fraction of a sample.
            private const float FlatFieldRise = 1.0f / 1024.0f;

            [NativeDisableUnsafePtrRestriction]
            public NativeStore.HeightField* Field;

            public float3 Scale;
            public CollisionFilter Filter;
            public Material Material;

            [NativeDisableUnsafePtrRestriction]
            public void** Result;

            public void Execute()
            {
                int size = (int)Field->resolution;
                var heights = new NativeArray<float>(size * size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int j = 0; j < size; j++)
                {
                    for (int i = 0; i < size; i++)
                    {
                        heights[j * size + i] = Field->heights[(size - 1 - i) * size + j];
                    }
                }

                BlobAssetReference<Collider> collider;
                if (Field->holeCount != 0)
                {
                    collider = CreateMesh(heights, size);
                }
                else
                {
                    if (Field->lowest == 0 && Field->highest == 0)
                    {
                        heights[0] = FlatFieldRise;
                    }

                    collider = TerrainCollider.Create(heights, new int2(size), Scale, TerrainCollider.CollisionMethod.Triangles, Filter, Material);
                }

                *Result = collider.GetUnsafePtr();
            }

            private BlobAssetReference<Collider> CreateMesh(NativeArray<float> heights, int size)
            {
                var vertices = new NativeArray<float3>(size * size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int j = 0; j < size; j++)
                {
                    for (int i = 0; i < size; i++)
                    {
                        vertices[j * size + i] = new float3(i, heights[j * size + i], j) * Scale;
                    }
                }

                int cells = size - 1;
                var triangles = new NativeList<int3>(2 * cells * cells, Allocator.Temp);
                for (int j = 0; j < cells; j++)
                {
                    for (int i = 0; i < cells; i++)
                    {
                        if (Field->cells[(cells - 1 - i) * cells + j] != 0)
                        {
                            int corner = j * size + i;
                            triangles.Add(new int3(corner, corner + size, corner + 1));
                            triangles.Add(new int3(corner + 1, corner + size, corner + size + 1));
                        }
                    }
                }

                return MeshCollider.Create(vertices, triangles.AsArray(), Filter, Material);
            }
        }

        private static Material MaterialOf(NativeStore.ShapeData* shape)
        {
            // Friction and restitution are set per manifold by ReadContactsJob, which sees both shapes.
            Material material = default;
            if (shape->collidesWith == 0)
            {
                material.CollisionResponse = CollisionResponsePolicy.None;
            }
            else if ((shape->flags & NativeStore.ShapeFlagTrigger) != 0)
            {
                material.CollisionResponse = CollisionResponsePolicy.RaiseTriggerEvents;
            }
            else if (shape->collisionDisabled != 0)
            {
                material.CollisionResponse = CollisionResponsePolicy.None;
            }
            else
            {
                // The events are the only place a contact's solved impulse comes out of the solver.
                material.CollisionResponse = CollisionResponsePolicy.CollideRaiseCollisionEvents;
            }

            return material;
        }
    }
}
