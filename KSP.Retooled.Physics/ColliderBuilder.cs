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
    /// Build a Unity Physics collider from a body's shapes.
    /// Compound children store shape indices in ColliderBlobInstance.Entity.
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
        /// Body to refresh, with a choice of rebuilding the collider or updating compound child poses.
        /// </summary>
        internal struct Work
        {
            public int Body;
            public bool Whole;
        }

        /// <summary>
        /// Schedule collider refreshes for dirty bodies, or every body if all is true.
        /// Keep work and results alive until completion. Do not read body colliders or write flags in the
        /// meantime.
        /// Increment wholeBuilds if any body needs a full rebuild, and changes if any collider is replaced.
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
        /// Retry failed collider builds, keeping their previous colliders until replacement succeeds.
        /// Call on the main thread after refresh completes. Invalidate the static tree for affected static
        /// bodies.
        /// </summary>
        internal static void AskAgainForFailed(NativeStore.WorldData* world, NativeList<Work> work, NativeList<long> results)
        {
            for (int i = 0; i < results.Length; i++)
            {
                if (results[i] == Failed && work[i].Body < world->bodyCount)
                {
                    world->bodies[work[i].Body].flags |= NativeStore.BodyFlags.ColliderDirty;
                    if ((world->bodies[work[i].Body].flags & NativeStore.BodyFlags.Static) != 0)
                    {
                        world->staticsVersion++;
                    }
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
            /// Number of refreshes that replaced colliders. Bounds caches must invalidate when this changes,
            /// because collider addresses can be reused.
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
                    SetCollider(body, FromPointer((void*)Results[i]));
                }
            }
        }

        /// <summary>
        /// Assign a collider and update the row's cached expansion factor. An uncreated collider clears both.
        /// </summary>
        internal static void SetCollider(NativeStore.BodyData* body, BlobAssetReference<Collider> collider)
        {
            body->collider->Value = collider;
            body->colliderExpansion = collider.IsCreated ? collider.Value.MassProperties.AngularExpansionFactor : 0.0f;
        }

        /// <summary>
        /// Copy a compound collider with updated child poses, reusing child geometry.
        /// Return null if the existing collider is absent or is not a compound.
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

                // Use a single-child compound for an offset mesh so that pose changes do not rebuild its
                // geometry.
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
        /// Return the shape index stored on the immediate compound child, or singleShape for a non-compound
        /// collider.
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
        /// Return the shape whose pose is applied to the built body rather than baked into its collider, or
        /// null.
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
            // Permit queries on all layers. MaterialOf disables contacts for shapes that collide with nothing.
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
        /// Rotate the height field grid a quarter turn about Y so that Unity Physics and PhysX split cells
        /// along the same diagonal.
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
        /// Build a convex collider, reducing large inputs to limit hull construction time and stack usage.
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
        /// Select hull support points along evenly distributed directions. Include the axes to preserve the
        /// original bounds.
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
        /// Compute volume and mass moments of an unscaled convex hull.
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
        /// Build a mesh collider with the shape's scale and pose baked into its vertices.
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
        /// Build a height field collider with the grid orientation from ShapeFromGrid.
        /// Use a triangle mesh when cells contain holes, since Unity Physics terrain cannot represent them.
        /// Use triangle collisions to match PhysX.
        /// </summary>
        [BurstCompile]
        private struct HeightFieldColliderJob : IJob
        {
            // Raise one corner slightly to avoid division by zero when scaling an all-zero height field.
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
