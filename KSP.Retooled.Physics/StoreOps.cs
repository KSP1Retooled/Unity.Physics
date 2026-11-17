using System.Threading;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Shared version for non-pose data copied into query snapshots, including body/shape membership and shape
    /// settings.
    /// Snapshot reuse also requires the same built body row order.
    /// </summary>
    internal static class QuerySnapshotVersion
    {
        private readonly struct Key { }

        private static readonly SharedStatic<int> Shared = SharedStatic<int>.GetOrCreate<Key>();

        internal static uint Version => (uint)Shared.Data;

        /// <summary>Moves the version on.</summary>
        internal static unsafe void Changed()
        {
            Interlocked.Increment(ref UnsafeUtility.AsRef<int>(Shared.UnsafeDataPointer));
        }
    }

    /// <summary>
    /// Shared command helpers for native allocations, query invalidation and body shape lists.
    /// </summary>
    internal static unsafe class StoreOps
    {
        // The initial counter value when a body wakes. PhysX's Dy::sleepCheck uses 20 steps of 0.02 s, regardless of
        // the actual step duration.
        internal const float WakeCounterReset = 0.4f;

        private const uint MinimumCapacity = 32;

        /// <summary>
        /// Return the body's row with refreshed component pointers. Main thread only.
        /// The row may move when bodies are created or destroyed; entity pointers may change after structural
        /// changes.
        /// </summary>
        internal static NativeStore.BodyData* Hot(NativeStore.Body* body)
        {
            NativeStore.BodyData* hot = body->world->bodies + body->index;
            BodyEntities.Refresh(hot);
            return hot;
        }

        /// <summary>
        /// Return the row without refreshing entity component pointers. Workers may read row-owned flags, shapes
        /// and owner.
        /// The row may move when bodies are created or destroyed.
        /// </summary>
        internal static NativeStore.BodyData* Row(NativeStore.Body* body)
        {
            return body->world->bodies + body->index;
        }

        /// <summary>
        /// Points a row at the components stored in its body record. That component storage remains at the same
        /// address.
        /// </summary>
        internal static void PointAtHome(NativeStore.BodyData* hot, NativeStore.Body* body)
        {
            NativeStore.BodyHome* home = &body->home;
            hot->pose = &home->pose;
            hot->velocity = &home->velocity;
            hot->mass = &home->mass;
            hot->bodyMass = &home->bodyMass;
            hot->damping = &home->damping;
            hot->limits = &home->limits;
            hot->sleep = &home->sleep;
            hot->forces = &home->forces;
            hot->target = &home->target;
            hot->collider = &home->collider;
        }

        /// <summary>
        /// Returns a body's angular velocity in world space. PhysicsVelocity stores angular velocity relative to the
        /// principal axes of inertia, but Rigidbody callers use world space.
        /// </summary>
        internal static float3 WorldAngularVelocity(in NativeStore.BodyData body)
        {
            return math.rotate(math.mul(body.pose->Rotation, body.mass->Transform.rot), body.velocity->Angular);
        }

        internal static void SetWorldAngularVelocity(in NativeStore.BodyData body, float3 angular)
        {
            quaternion worldFromInertia = math.mul(body.pose->Rotation, body.mass->Transform.rot);
            body.velocity->Angular = math.rotate(math.conjugate(worldFromInertia), angular);
        }

        /// <summary>
        /// Changes a body's rotation while preserving its angular velocity in world space. The angular velocity stored
        /// relative to the principal axes is adjusted to match.
        /// </summary>
        internal static void SetRotation(in NativeStore.BodyData body, quaternion rotation)
        {
            float3 angular = WorldAngularVelocity(body);
            body.pose->Rotation = rotation;
            SetWorldAngularVelocity(body, angular);
        }

        /// <summary>
        /// Changes the orientation of the principal axes of inertia while preserving angular velocity in world space,
        /// as <see cref="SetRotation"/> does for the body's rotation.
        /// </summary>
        internal static void SetInertiaRotation(in NativeStore.BodyData body, quaternion rotation)
        {
            float3 angular = WorldAngularVelocity(body);
            body.mass->Transform.rot = rotation;
            SetWorldAngularVelocity(body, angular);
        }

        internal static void* Reallocate(NativeStore.StoreGlobals* globals, void* memory, ulong size)
        {
            return new FunctionPointer<NativeStore.ReallocateFunction>(globals->reallocate).Invoke(memory, size);
        }

        internal static void Release(NativeStore.StoreGlobals* globals, void* memory)
        {
            new FunctionPointer<NativeStore.PointerFunction>(globals->release).Invoke(memory);
        }

        /// <summary>Allocates a record on the store's heap and initializes it to zero.</summary>
        internal static T* Allocate<T>(NativeStore.StoreGlobals* globals)
            where T : unmanaged
        {
            T* record = (T*)Reallocate(globals, null, (ulong)sizeof(T));
            *record = default;
            return record;
        }

        /// <summary>
        /// Returns the new capacity for a full array that currently has <paramref name="capacity"/> entries.
        /// </summary>
        internal static uint Grown(uint capacity)
        {
            return math.max(MinimumCapacity, capacity * 2);
        }

        internal static void RetainMesh(NativeStore.StoreGlobals* globals, void* mesh)
        {
            new FunctionPointer<NativeStore.PointerFunction>(globals->retainMesh).Invoke(mesh);
        }

        internal static void ReleaseMesh(NativeStore.StoreGlobals* globals, void* mesh)
        {
            new FunctionPointer<NativeStore.PointerFunction>(globals->releaseMesh).Invoke(mesh);
        }

        internal static void RetainMaterial(NativeStore.StoreGlobals* globals, NativeStore.MaterialData* material)
        {
            new FunctionPointer<NativeStore.PointerFunction>(globals->retainMaterial).Invoke(material);
        }

        internal static void ReleaseMaterial(NativeStore.StoreGlobals* globals, NativeStore.MaterialData* material)
        {
            new FunctionPointer<NativeStore.PointerFunction>(globals->releaseMaterial).Invoke(material);
        }

        /// <summary>Marks a query world that needs to be rebuilt from the store.</summary>
        internal const uint QueryStale = 1;

        /// <summary>
        /// Query state indicating that only static poses and their tree need updating.
        /// </summary>
        internal const uint QueryStaticsMoved = 2;

        /// <summary>
        /// Query state indicating that the world is current after its pending dynamic tree refit.
        /// Any build finishes the refit before processing further changes.
        /// </summary>
        internal const uint QueryTreePending = 3;

        /// <summary>
        /// Marks the query world as stale. The write is atomic because worker threads read this flag.
        /// </summary>
        internal static void MarkQueryDirty(NativeStore.WorldData* world)
        {
            Interlocked.Exchange(ref *(int*)&world->queryDirty, (int)QueryStale);
        }

        /// <summary>
        /// Mark static pose changes without weakening an existing full-rebuild request.
        /// </summary>
        internal static void MarkStaticMoved(NativeStore.Body* body)
        {
            ref int dirty = ref *(int*)&body->world->queryDirty;
            if (Interlocked.CompareExchange(ref dirty, (int)QueryStaticsMoved, 0) == (int)QueryTreePending)
            {
                Interlocked.CompareExchange(ref dirty, (int)QueryStaticsMoved, (int)QueryTreePending);
            }

            body->world->staticsVersion++;
        }

        internal static void MarkPoseChanged(NativeStore.Body* body, NativeStore.BodyFlags flags)
        {
            MarkQueryDirty(body->world);
            if ((flags & NativeStore.BodyFlags.Static) != 0)
            {
                body->world->staticsVersion++;
            }
        }

        /// <summary>
        /// Mark collider geometry dirty and invalidate query snapshot data after shape changes.
        /// </summary>
        internal static void MarkColliderDirty(NativeStore.Body* body)
        {
            NativeStore.BodyData* hot = Hot(body);
            hot->flags |= NativeStore.BodyFlags.ColliderDirty;
            QuerySnapshotVersion.Changed();
            MarkPoseChanged(body, hot->flags);
        }

        /// <summary>
        /// Mark shape pose changes. Copy compounds with new child poses; rebuild colliders with baked poses.
        /// </summary>
        internal static void MarkCollidersMoved(NativeStore.Body* body)
        {
            NativeStore.BodyData* hot = Hot(body);
            BlobAssetReference<Collider> built = hot->collider->Value;
            if (built.IsCreated && built.Value.Type == ColliderType.Compound)
            {
                hot->flags |= NativeStore.BodyFlags.CollidersMoved;
                MarkPoseChanged(body, hot->flags);
            }
            else
            {
                MarkColliderDirty(body);
            }
        }

        private static void PublishShapes(NativeStore.Body* body)
        {
            NativeStore.BodyData* hot = Hot(body);
            hot->shapes = body->shapes;
            hot->shapeCount = body->shapeCount;
            MarkColliderDirty(body);
        }

        internal static void AddShape(NativeStore.Body* body, NativeStore.Shape* shape, NativeStore.StoreGlobals* globals)
        {
            if (body->shapeCount == body->shapeCapacity)
            {
                body->shapeCapacity = Grown(body->shapeCapacity);
                body->shapes = (NativeStore.ShapeData**)Reallocate(globals, body->shapes, (ulong)(body->shapeCapacity * sizeof(void*)));
            }

            shape->body = body;
            body->shapes[body->shapeCount++] = &shape->data;
            PublishShapes(body);
        }

        private static void NoteRemovedShape(NativeStore.WorldData* world, NativeStore.Shape* shape, NativeStore.StoreGlobals* globals)
        {
            if (world->removedShapeCount == world->removedShapeCapacity)
            {
                world->removedShapeCapacity = Grown(world->removedShapeCapacity);
                world->removedShapes = (void**)Reallocate(globals, world->removedShapes, world->removedShapeCapacity * (ulong)sizeof(void*));
            }

            world->removedShapes[world->removedShapeCount++] = shape;
            HideFromQueries(world, shape);
        }

        /// <summary>
        /// Hide a removed shape from the published query snapshot before its collider is destroyed.
        /// Verify that the snapshot slot still belongs to this shape.
        /// </summary>
        private static void HideFromQueries(NativeStore.WorldData* world, NativeStore.Shape* shape)
        {
            QueryWorld* query = world->queryWorld;
            if (query != null && shape->querySlot != 0 && shape->querySlot <= (uint)query->ShapeCount)
            {
                QueryShape* published = query->Shapes + (shape->querySlot - 1);
                if (published->Handle == shape)
                {
                    published->Flags = 0;
                    published->UserData = null;
                }
            }

            shape->querySlot = 0;
        }

        /// <summary>
        /// Removes a shape from its body and preserves the order of the remaining shapes. The child indices in a
        /// compound collider depend on that order.
        /// </summary>
        internal static void RemoveShape(NativeStore.Body* body, NativeStore.Shape* shape, NativeStore.StoreGlobals* globals)
        {
            NoteRemovedShape(body->world, shape, globals);
            uint kept = 0;
            for (uint i = 0; i < body->shapeCount; i++)
            {
                if (body->shapes[i] != &shape->data)
                {
                    body->shapes[kept++] = body->shapes[i];
                }
            }

            body->shapeCount = kept;
            shape->body = null;
            PublishShapes(body);
        }

        /// <summary>
        /// Detaches a body's shapes when the body is destroyed. The shapes remain alive until Unity destroys them.
        /// </summary>
        internal static void DetachAllShapes(NativeStore.Body* body, NativeStore.StoreGlobals* globals)
        {
            for (uint i = 0; i < body->shapeCount; i++)
            {
                NoteRemovedShape(body->world, (NativeStore.Shape*)body->shapes[i], globals);
                ((NativeStore.Shape*)body->shapes[i])->body = null;
            }

            body->shapeCount = 0;
            Release(globals, body->shapes);
            body->shapes = null;
            body->shapeCapacity = 0;
            PublishShapes(body);
        }
    }
}
