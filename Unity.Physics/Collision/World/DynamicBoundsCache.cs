using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Cache dynamic collider bounds between refits and builds.
    /// Reuse only for exact body identity, collider and pose matches marked by View.Same.
    /// Collider version changes invalidate reuse even when an allocation keeps the same address.
    /// </summary>
    internal unsafe struct DynamicBoundsCache : IDisposable
    {
        /// <summary>
        /// The collider of an entry and the body that it belonged to, with the collider's filter, which the build would
        /// read from it.
        /// </summary>
        internal struct ColliderInfo
        {
            public long Collider;
            public Unity.Entities.Entity Body;
            public CollisionFilter Filter;
            public int RespondsToCollision;
        }

        /// <summary>
        /// Cached collider data for computing bounds at a new pose without reading collider memory.
        /// Require matching collider address, body identity and collider version. Results must match collider
        /// bounds bit for bit.
        /// </summary>
        internal struct ShapeBounds
        {
            /// <summary>The kinds of entry: the collider must compute its own bounds, which a convex hull does.</summary>
            public const int ReadCollider = 0;

            /// <summary>A box: <see cref="Local"/> holds its bounds in its own space.</summary>
            public const int Box = 1;

            /// <summary>A compound or mesh: its bounds in its own space, clipped to the sphere of <see cref="Radius"/>.</summary>
            public const int InSphere = 2;

            /// <summary>A sphere: the center in <see cref="Local"/>.Min, and <see cref="Radius"/>.</summary>
            public const int Sphere = 3;

            /// <summary>A capsule: the ends in <see cref="Local"/>.Min and Max, and <see cref="Radius"/>.</summary>
            public const int Capsule = 4;

            public long Collider;
            public Unity.Entities.Entity Body;
            public uint Version;
            public int Kind;
            public Aabb Local;
            public float Radius;

            /// <summary>Makes the entry for a collider. The collider is read here and not again while the entry holds.</summary>
            public static ShapeBounds Of(Collider* collider, long address, Unity.Entities.Entity body, uint version)
            {
                var entry = new ShapeBounds { Collider = address, Body = body, Version = version, Kind = ReadCollider };
                switch (collider->Type)
                {
                    case ColliderType.Box:
                        entry.Kind = Box;
                        entry.Local = ((BoxCollider*)collider)->LocalAabb();
                        break;
                    case ColliderType.Compound:
                        entry.Kind = InSphere;
                        entry.Local = ((CompoundCollider*)collider)->BoundingVolumeHierarchy.Domain;
                        entry.Radius = ((CompoundCollider*)collider)->m_BoundingRadius;
                        break;
                    case ColliderType.Mesh:
                        entry.Kind = InSphere;
                        entry.Local = ((MeshCollider*)collider)->LocalAabb;
                        entry.Radius = ((MeshCollider*)collider)->BoundingRadius;
                        break;
                    case ColliderType.Sphere:
                        entry.Kind = Sphere;
                        entry.Local.Min = ((SphereCollider*)collider)->Center;
                        entry.Radius = ((SphereCollider*)collider)->Radius;
                        break;
                    case ColliderType.Capsule:
                        entry.Kind = Capsule;
                        entry.Local.Min = ((CapsuleCollider*)collider)->Vertex0;
                        entry.Local.Max = ((CapsuleCollider*)collider)->Vertex1;
                        entry.Radius = ((CapsuleCollider*)collider)->Radius;
                        break;
                }

                return entry;
            }

            /// <summary>
            /// The bounds of the collider at the pose, with a scale of 1, as the collider's CalculateAabb gives them.
            /// The entry's kind must not be <see cref="ReadCollider"/>.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Aabb At(RigidTransform pose)
            {
                switch (Kind)
                {
                    case Box:
                        return Math.TransformAabb(Local, pose, 1.0f);
                    case InSphere:
                        return Math.TransformAabbInSphere(Local, Radius, pose, 1.0f);
                    case Sphere:
                        return SphereCollider.CalculateAabb(Local.Min, Radius, pose, 1.0f);
                    default:
                        return CapsuleCollider.CalculateAabb(Local.Min, Local.Max, Radius, pose, 1.0f);
                }
            }
        }

        /// <summary>The cache's entries and versions, for a job to read and write.</summary>
        internal struct View
        {
            [NativeDisableUnsafePtrRestriction]
            public Aabb* Bounds;
            [NativeDisableUnsafePtrRestriction]
            public ColliderInfo* Colliders;

            /// <summary>
            /// Cached geometry for the body last written at each index, or null when ShapesEnabled is false.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public ShapeBounds* Shapes;

            /// <summary>
            /// For each body of the world being built, 1 if the body has exactly the collider and pose that the
            /// world's body at the same index had before the build, and 0 otherwise.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public byte* Same;
            public int Length;
            [NativeDisableUnsafePtrRestriction]
            public uint* HeldVersion;
            [NativeDisableUnsafePtrRestriction]
            public uint* ColliderVersion;

            /// <summary>The counts of the check, while the check is on (<see cref="CheckEnabled"/>), or null.</summary>
            [NativeDisableUnsafePtrRestriction]
            public CheckCounts* Check;

            public bool IsCreated => Bounds != null;

            /// <summary>
            /// Validate cached bounds, filter and response against the collider and count mismatches.
            /// </summary>
            public void CheckEntry(int i, in RigidBody body)
            {
                Aabb computed = body.CalculateAabb();
                Aabb cached = Bounds[i];
                bool same = math.all(math.asuint(computed.Min) == math.asuint(cached.Min)) && math.all(math.asuint(computed.Max) == math.asuint(cached.Max))
                    && body.Collider.Value.GetCollisionFilter().Equals(Colliders[i].Filter)
                    && body.Collider.Value.RespondsToCollision == (Colliders[i].RespondsToCollision != 0);
                System.Threading.Interlocked.Increment(ref Check->Hits);
                if (!same)
                {
                    System.Threading.Interlocked.Increment(ref Check->Mismatches);
                    Check->LastIndex = i;
                    Check->LastEntity = body.Entity.Index;
                    Check->LastComputedMin = computed.Min;
                    Check->LastCachedMin = cached.Min;
                }
            }

            /// <summary>
            /// Compute collider bounds using cached shape geometry when available, otherwise refresh the cache.
            /// Convex hulls still read their vertices from the collider.
            /// </summary>
            public Aabb ColliderBounds(int i, RigidBody* body)
            {
                // The collider's address is the first field of the body. Reading it directly, and not through
                // BlobAssetReference, doesn't read the collider's header, which the editor's checks do.
                long address = *(long*)body;
                if (Shapes == null || address == 0 || body->Scale != 1.0f)
                {
                    return body->CalculateAabb();
                }

                ShapeBounds* entry = Shapes + i;
                uint version = *ColliderVersion;
                if (entry->Collider != address || entry->Body != body->Entity || entry->Version != version)
                {
                    *entry = ShapeBounds.Of((Collider*)address, address, body->Entity, version);
                    return body->CalculateAabb();
                }

                if (entry->Kind == ShapeBounds.ReadCollider)
                {
                    return body->CalculateAabb();
                }

                Aabb bounds = entry->At(body->WorldFromBody);
                if (Check != null)
                {
                    CheckShape(i, body, bounds);
                }

                return bounds;
            }

            /// <summary>
            /// Compare supplied bounds with collider-computed bounds and increment validation/mismatch counters.
            /// </summary>
            void CheckShape(int i, RigidBody* body, Aabb bounds)
            {
                Aabb computed = body->CalculateAabb();
                System.Threading.Interlocked.Increment(ref Check->ShapeHits);
                if (math.any(math.asuint(computed.Min) != math.asuint(bounds.Min)) || math.any(math.asuint(computed.Max) != math.asuint(bounds.Max)))
                {
                    System.Threading.Interlocked.Increment(ref Check->ShapeMismatches);
                    System.Threading.Interlocked.Increment(ref Check->Mismatches);
                    Check->LastIndex = i;
                    Check->LastEntity = body->Entity.Index;
                    Check->LastComputedMin = computed.Min;
                    Check->LastCachedMin = bounds.Min;
                }
            }

            /// <summary>
            /// Return whether entry i matches the body's pose, identity and current collider version.
            /// Non-unit body scale never matches.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool Holds(int i, in RigidBody body)
            {
                long collider = (long)body.Collider.GetUnsafePtr();
                return Same[i] != 0 && Colliders[i].Collider == collider && collider != 0 && body.Scale == 1.0f
                    && *HeldVersion == *ColliderVersion;
            }

            /// <summary>
            /// Write bounds, filter and response from a body's collider into entry i.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Write(int i, in RigidBody body, Aabb bounds)
            {
                Write(i, body, bounds, body.Collider.Value.GetCollisionFilter(), body.Collider.Value.RespondsToCollision);
            }

            /// <summary>
            /// Write collider bounds, filter and response for entry i.
            /// Always refresh filter and response: a replaced collider can reuse the same address on the same body.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Write(int i, in RigidBody body, Aabb bounds, CollisionFilter filter, bool respondsToCollision)
            {
                Bounds[i] = bounds;
                Colliders[i] = new ColliderInfo
                {
                    Collider = body.Scale == 1.0f ? (long)body.Collider.GetUnsafePtr() : 0,
                    Body = body.Entity,
                    Filter = filter,
                    RespondsToCollision = respondsToCollision ? 1 : 0,
                };
            }

            /// <summary>
            /// Return whether identity, collider and pose exactly match the previous body at index i.
            /// Call before overwriting the body and store the result in Same.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Matches(RigidBody before, RigidBody after)
            {
                // Compare stored addresses directly. The old collider may be freed, so GetUnsafePtr's header
                // check is unsafe.
                return *(void**)&before == *(void**)&after && before.Entity == after.Entity && before.Scale == after.Scale
                    && math.all(math.asuint(before.WorldFromBody.pos) == math.asuint(after.WorldFromBody.pos))
                    && math.all(math.asuint(before.WorldFromBody.rot.value) == math.asuint(after.WorldFromBody.rot.value));
            }
        }

        /// <summary>The counts of the check of the cache. Read them on the main thread between steps.</summary>
        internal struct CheckCounts
        {
            public long Steps;
            public long Hits;
            public long Mismatches;
            public long LoggedMismatches;

            /// <summary>
            /// The bodies whose collider bounds came from <see cref="View.Shapes"/>, and those whose bounds differed
            /// from the bounds that the collider computes.
            /// </summary>
            public long ShapeHits;
            public long ShapeMismatches;
            public int LastIndex;
            public int LastEntity;
            public float3 LastComputedMin;
            public float3 LastCachedMin;
        }

        sealed class CheckCountsKey
        {
        }

        sealed class CheckOnKey
        {
        }

        static readonly Unity.Burst.SharedStatic<CheckCounts> SharedCheckCounts = Unity.Burst.SharedStatic<CheckCounts>.GetOrCreate<DynamicBoundsCache, CheckCountsKey>();
        static readonly Unity.Burst.SharedStatic<int> SharedCheckOn = Unity.Burst.SharedStatic<int>.GetOrCreate<DynamicBoundsCache, CheckOnKey>();

        /// <summary>
        /// Validate cached bounds, filters and responses against recomputed values while still using cached
        /// results.
        /// Record mismatches in Counts.
        /// </summary>
        public static bool CheckEnabled
        {
            get => SharedCheckOn.Data != 0;
            set => SharedCheckOn.Data = value ? 1 : 0;
        }

        /// <summary>The counts of the check since the program started.</summary>
        internal static CheckCounts* Counts => (CheckCounts*)UnsafeUtility.AddressOf(ref SharedCheckCounts.Data);

        sealed class ShapesOffKey
        {
        }

        static readonly Unity.Burst.SharedStatic<int> SharedShapesOff = Unity.Burst.SharedStatic<int>.GetOrCreate<DynamicBoundsCache, ShapesOffKey>();

        /// <summary>
        /// Whether the cache keeps what each body's collider needs to compute its bounds (<see cref="View.Shapes"/>).
        /// On by default; RETOOLED_SHAPE_BOUNDS=0 turns it off. The views that the cache gives after a change take it.
        /// </summary>
        public static bool ShapesEnabled
        {
            get => SharedShapesOff.Data == 0;
            set => SharedShapesOff.Data = value ? 0 : 1;
        }

        NativeList<Aabb> bounds;
        NativeList<ColliderInfo> colliders;
        NativeList<ShapeBounds> shapes;
        NativeList<byte> same;
        NativeReference<uint> heldVersion;
        uint* colliderVersion;

        /// <summary>
        /// Makes an empty cache. The owner adds one to the number that <paramref name="colliderVersion"/> points to
        /// whenever a body's collider changes, and keeps that number alive as long as the cache.
        /// </summary>
        public DynamicBoundsCache(uint* colliderVersion, Allocator allocator)
        {
            bounds = new NativeList<Aabb>(allocator);
            colliders = new NativeList<ColliderInfo>(allocator);
            shapes = new NativeList<ShapeBounds>(allocator);
            same = new NativeList<byte>(allocator);
            heldVersion = new NativeReference<uint>(*colliderVersion - 1, allocator);
            this.colliderVersion = colliderVersion;
        }

        public bool IsCreated => bounds.IsCreated;

        /// <summary>
        /// Gives the cache one entry for each of the given number of bodies, and returns the view for the jobs. New
        /// entries hold no collider and no body is the same as before. Call it only on the main thread while no job
        /// uses the cache.
        /// </summary>
        internal View Resize(int count)
        {
            if (bounds.Length != count)
            {
                bounds.Resize(count, NativeArrayOptions.ClearMemory);
                colliders.Resize(count, NativeArrayOptions.ClearMemory);
                shapes.Resize(count, NativeArrayOptions.ClearMemory);
                same.Resize(count, NativeArrayOptions.ClearMemory);
            }

            return CurrentView;
        }

        /// <summary>The view of the entries as they are, for a job to read and write.</summary>
        internal View CurrentView => new View
        {
            Bounds = bounds.GetUnsafePtr(),
            Colliders = colliders.GetUnsafePtr(),
            Shapes = ShapesEnabled ? shapes.GetUnsafePtr() : null,
            Same = same.GetUnsafePtr(),
            Length = bounds.Length,
            HeldVersion = heldVersion.GetUnsafePtr(),
            ColliderVersion = colliderVersion,
            Check = CheckEnabled ? Counts : null,
        };

        /// <summary>
        /// Record the current collider version after every reusable entry has been refreshed by a build.
        /// Call on the main thread while colliders cannot change.
        /// </summary>
        public void MarkCurrent()
        {
            heldVersion.Value = *colliderVersion;
        }

        public void Dispose()
        {
            if (bounds.IsCreated)
            {
                bounds.Dispose();
                colliders.Dispose();
                shapes.Dispose();
                same.Dispose();
                heldVersion.Dispose();
            }
        }
    }
}
