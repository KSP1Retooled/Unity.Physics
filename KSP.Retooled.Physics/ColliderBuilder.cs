using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Turns a body's shapes into one Unity Physics collider: the shape alone when there is one, else a compound whose
    /// children record their shape index in <see cref="CompoundCollider.ColliderBlobInstance.Entity"/>. Colliders are
    /// held by the store as raw blob pointers, which is why a domain reload can release and rebuild them.
    /// </summary>
    internal static unsafe class ColliderBuilder
    {
        // PhysX allows a degenerate extent, Unity Physics refuses one.
        private const float MinimumExtent = 1e-4f;

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

                Release(body->collider);
                body->collider = Build(body);
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

            BlobAssetReference<Collider> compound = CompoundCollider.Create(children);
            foreach (CompoundCollider.ColliderBlobInstance child in children)
            {
                child.Collider.Dispose();
            }

            children.Dispose();
            return compound.GetUnsafePtr();
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
            return shape->geometryType == NativeStore.GeometryBox
                || shape->geometryType == NativeStore.GeometrySphere
                || shape->geometryType == NativeStore.GeometryCapsule;
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

        private static Material MaterialOf(NativeStore.ShapeData* shape)
        {
            NativeStore.MaterialData* source = shape->material;
            Material material = new Material
            {
                Friction = source->dynamicFriction,
                Restitution = source->bounciness,
                FrictionCombinePolicy = CombinePolicy(source->frictionCombine),
                RestitutionCombinePolicy = CombinePolicy(source->bounceCombine),
            };

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
                material.CollisionResponse = CollisionResponsePolicy.Collide;
            }

            return material;
        }

        // Unity Physics has no multiply; the geometric mean is the closest of its policies.
        private static Material.CombinePolicy CombinePolicy(uint mode)
        {
            switch (mode)
            {
                case NativeStore.CombineMinimum:
                    return Material.CombinePolicy.Minimum;
                case NativeStore.CombineMaximum:
                    return Material.CombinePolicy.Maximum;
                case NativeStore.CombineMultiply:
                    return Material.CombinePolicy.GeometricMean;
                default:
                    return Material.CombinePolicy.ArithmeticMean;
            }
        }
    }
}
