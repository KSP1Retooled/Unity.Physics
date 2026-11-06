using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend
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

        /// <summary>Rebuilds the collider of every body that asks for it. <paramref name="all"/> treats every body as asking.</summary>
        internal static void Refresh(NativeStore.WorldData* world, bool all)
        {
            for (uint i = 0; i < world->bodyCount; i++)
            {
                NativeStore.BodyData* body = world->bodies + i;
                if (!all && (body->flags & NativeStore.BodyFlags.ColliderDirty) == 0)
                {
                    continue;
                }

                // A build that throws leaves the body with its last collider, and still asking for a new one.
                void* collider = Build(body);
                Release(body->collider);
                body->collider = collider;
                body->flags &= ~NativeStore.BodyFlags.ColliderDirty;
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

            if (usable == 1)
            {
                NativeStore.ShapeData* shape = body->shapes[lastUsable];
                return Create(shape, shape->localPosition, shape->localRotation).GetUnsafePtr();
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

                    children[next++] = new CompoundCollider.ColliderBlobInstance
                    {
                        Collider = Create(shape, float3.zero, quaternion.identity),
                        CompoundFromChild = new RigidTransform(shape->localRotation, shape->localPosition),
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
                {
                    BoxGeometry box = new BoxGeometry
                    {
                        Center = center,
                        Orientation = orientation,
                        Size = math.max(shape->extents * 2.0f, MinimumExtent),
                        BevelRadius = 0.0f,
                    };

                    return BoxCollider.Create(box, filter, material);
                }

                case NativeStore.GeometrySphere:
                {
                    SphereGeometry sphere = new SphereGeometry { Center = center, Radius = math.max(shape->extents.x, MinimumExtent) };
                    return SphereCollider.Create(sphere, filter, material);
                }

                case NativeStore.GeometryConvexMesh:
                case NativeStore.GeometryTriangleMesh:
                    return CreateMesh(shape, center, orientation, filter, material);

                default:
                {
                    // The extents hold the radius and half of the cylinder's length, along local X as in PhysX.
                    float3 halfAxis = math.rotate(orientation, new float3(math.max(shape->extents.y, MinimumExtent), 0.0f, 0.0f));
                    CapsuleGeometry capsule = new CapsuleGeometry
                    {
                        Vertex0 = center - halfAxis,
                        Vertex1 = center + halfAxis,
                        Radius = math.max(shape->extents.x, MinimumExtent),
                    };

                    return CapsuleCollider.Create(capsule, filter, material);
                }
            }
        }

        private static BlobAssetReference<Collider> CreateMesh(
            NativeStore.ShapeData* shape,
            float3 center,
            quaternion orientation,
            CollisionFilter filter,
            Material material)
        {
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
            }.Run();

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
        /// The solid of a convex mesh's hull, unscaled. Burst compiled, like <see cref="MeshColliderJob"/>, so that it stays
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
        /// A job so that Burst runs it: a mesh's hierarchy or hull is far slower to build in Mono.
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
