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
    /// Builds one Unity Physics collider from a body's shapes. Uses the shape directly when possible, or a compound
    /// whose children store their shape indices in <see cref="CompoundCollider.ColliderBlobInstance.Entity"/>. The
    /// store keeps colliders as raw blob pointers so that they can be released and rebuilt during a domain reload.
    /// </summary>
    [BurstCompile]
    internal static unsafe class ColliderBuilder
    {
        // PhysX permits a degenerate extent, but Unity Physics requires a positive extent.
        private const float MinimumExtent = 1e-4f;

        // Allow well over the 252 vertices retained by a convex collider, while limiting the hull builder's stack usage
        // and time.
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

        // The result value left when building a collider throws an exception.
        private const long Failed = -1;

        /// <summary>
        /// Identifies a body to refresh and whether to rebuild its entire collider or only move the compound's
        /// children.
        /// </summary>
        internal struct Work
        {
            public int Body;
            public bool Whole;
        }

        /// <summary>
        /// Schedules a new collider for every body marked for a rebuild or a change in child poses. If <paramref
        /// name="all"/> is true, includes every body. Building runs concurrently with subsequent work on the main
        /// thread. Until the returned handle completes, nothing may read a body's collider or write its flags. Keep
        /// <paramref name="work"/> and <paramref name="results"/> alive until then. If any body requires a full build,
        /// increment <paramref name="wholeBuilds"/> because its shapes may have changed. If any body gets a new
        /// collider, increment <paramref name="changes"/>.
        /// </summary>
        internal static JobHandle ScheduleRefresh(
            NativeStore.WorldData* world,
            bool all,
            NativeList<Work> work,
            NativeList<long> results,
            NativeReference<uint> wholeBuilds,
            NativeReference<uint> changes,
            JobHandle dependency)
        {
            if (work.Capacity < world->bodyCount)
                work.Capacity = (int)world->bodyCount;
            work.Clear();

            // This job writes the flags. No other code writes them until the job completes, so these writes are safe.
            var aggregateWork = new AggregateColliderWorkJob
            {
                World = world,
                Work = work.AsParallelWriter(),
                All = all
            }.ScheduleBatch((int)world->bodyCount, 1024, dependency);

            dependency = new InitializeResultsJob { Work = work, Results = results, WholeBuilds = wholeBuilds, Changes = changes }.Schedule(aggregateWork);

            // Schedule one body per work item because building a mesh can take a thousand times as long as building a
            // box.
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
        /// Marks every body whose last build threw an exception for a full rebuild. Those bodies still have their
        /// previous colliders. Call only on the main thread after the refresh has completed.
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

            public NativeReference<uint> WholeBuilds;

            /// <summary>
            /// The number of refreshes that gave any body a new collider. The cache of the dynamic bodies' bounds
            /// knows a collider by its address, and a new collider can take the memory of one that was freed, so
            /// the cache holds nothing while this number differs from the one it recorded.
            /// </summary>
            public NativeReference<uint> Changes;

            public void Execute()
            {
                Results.ResizeUninitialized(Work.Length);
                if (Work.Length > 0)
                {
                    Changes.Value++;
                }

                bool whole = false;
                for (int i = 0; i < Results.Length; ++i)
                {
                    Results[i] = Failed;
                    whole |= Work[i].Whole;
                }

                if (whole)
                {
                    WholeBuilds.Value++;
                }
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

        /// <summary>
        /// Assigns each body's newly built collider and releases its previous collider.
        /// </summary>
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
        /// Builds a new compound by copying the previous compound's children and applying the shapes' new poses.
        /// Returns null if the body has no built collider or its collider is not a compound. Copying a child avoids
        /// rebuilding its hull or hierarchy.
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

                // Represent a mesh away from its body's origin as a compound with one child. Subsequent moves only
                // change the child's pose. Baking the pose into the mesh would require rebuilding its hull or hierarchy
                // on every move.
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
        /// Returns the index of the body's shape identified by the key. For a compound, the index is stored on its
        /// immediate child. Do not descend further, because polygons within a mesh child do not store the shape index.
        /// For any other collider, return <paramref name="singleShape"/>.
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
        /// Returns the shape whose pose is applied to the built body instead of being baked into its collider, or null.
        /// Unity moves static colliders by setting their shapes' poses, every frame for animated colliders. Baking a
        /// pose into a mesh would require rebuilding it after each move. Unity joints cannot connect to a static body,
        /// so no other behavior depends on the built body's transform.
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

        /// <summary>
        /// Returns the index of the shape that a body's collider represents when the collider is not a compound.
        /// </summary>
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
            // Unity Physics applies query filters in both directions. A collider that collides with nothing would
            // therefore be invisible to every ray. Allow all collision layers in its filter here, then disable its
            // contacts in MaterialOf.
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
        /// Places a height field's grid in its shape with a quarter turn about Y. The grid's rows run along X and its
        /// columns run backwards along Z. PhysX splits each cell from (x, z) to (x + 1, z + 1). Unity Physics queries
        /// always split from (i + 1, j) to (i, j + 1), regardless of the setting passed to TerrainCollider.Create.
        /// Rotate the grid so that the splits agree.
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

        /// <summary>
        /// Uses extents.x as the radius and extents.y as half the cylinder's length. The axis is local X, as in PhysX.
        /// </summary>
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
        /// Builds a convex collider from the supplied points. The hull builder allocates stack space in proportion to
        /// the input size, and the main thread has a 1 MB stack. It also tests every point against every face. Reduce
        /// large inputs first to limit both costs. This loses very little detail because the collider retains at most
        /// 252 vertices anyway.
        /// </summary>
        private static BlobAssetReference<Collider> CreateConvex(NativeArray<float3> points, CollisionFilter filter, Material material)
        {
            // PhysX uses the hull without rounding or merging, so disable those adjustments here too.
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
        /// Selects the points farthest along evenly distributed directions. All selected points lie on the hull.
        /// Include the axes first so that the selected points preserve the bounds of the full set.
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
        /// Calculates the volume and mass moments of an unscaled convex mesh's hull. Compiles with Burst, like <see
        /// cref="BuildCollidersJob"/>, so that the calculation stays fast when called by a body command running in
        /// Mono. Commands run on the main thread, where a temporary allocator is available.
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

            // Convert from the principal axes to the mesh's axes, where scaling applies. For unit mass,
            // C = (trace(I) / 2) * identity - I.
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
        /// Builds a mesh blob with the shape's scale and pose baked into its vertices, since Unity Physics does not
        /// store these separately for each collider. <see cref="BuildCollidersJob"/> executes this job directly.
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
                    // A scale that mirrors the mesh reverses triangle orientation, so reverse the winding as well.
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
        /// Builds a height field blob using the grid placed by <see cref="ShapeFromGrid"/>. Grid cell (i, j)
        /// corresponds to source cell (j, last - i). Unity Physics terrain cannot contain holes. If there are holes,
        /// build a triangle mesh from the solid cells with the same splits and with triangles facing up, as they do in
        /// a mesh. Use triangle collisions for terrain, as PhysX does. VertexSamples instead pushes up any vertex below
        /// the surface, which can throw a body into the air when its centre is below the surface.
        /// </summary>
        [BurstCompile]
        private struct HeightFieldColliderJob : IJob
        {
            // Unity Physics scales the tallest sample to the full range of a short. An all-zero field would cause
            // division by zero, so raise one corner by this fraction of a sample.
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
            // ReadContactsJob sets friction and restitution for each manifold using both shapes.
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
                // Collision events are the only way to read a contact's solved impulse from the solver.
                material.CollisionResponse = CollisionResponsePolicy.CollideRaiseCollisionEvents;
            }

            return material;
        }
    }
}
