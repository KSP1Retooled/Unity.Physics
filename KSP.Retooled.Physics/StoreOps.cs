using System.Threading;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Provides operations that the command kernels share for store memory, flags that mark query worlds as dirty, and
    /// lists of shapes attached to bodies. A world's handle points to its <see cref="NativeStore.WorldData"/>, and a
    /// shape's handle points to its <see cref="NativeStore.ShapeData"/>.
    /// </summary>
    /// <summary>
    /// A version counter for the store data that the query snapshot copies (see <see cref="SnapshotQueryDataJob"/>).
    /// The snapshot holds no poses. For each built body it holds the body's record, its shapes and whether it is
    /// static. For each shape it holds the shape's record, user data, engine object id, flags and geometry type. The
    /// version moves when a body is created or destroyed, when a body's list of shapes changes, and when a command
    /// changes an attached shape's flags, filter, material or geometry. The step also compares the built body order
    /// with the order that the snapshot was taken in, so a body that moves to a new row or stops being simulated
    /// still causes a new snapshot. The plugin mirrors the store's records, so there is no space for this counter in
    /// them. All worlds share it, as they share <see cref="JointTopology"/>.
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

    internal static unsafe class StoreOps
    {
        // The initial counter value when a body wakes. PhysX's Dy::sleepCheck uses 20 steps of 0.02 s, regardless of
        // the actual step duration.
        internal const float WakeCounterReset = 0.4f;

        private const uint MinimumCapacity = 32;

        /// <summary>
        /// Returns the body's row in the dense array with current component pointers. The row remains valid until a
        /// body in the same world is created or destroyed. Its component pointers remain valid until the next
        /// structural change in the bound Entities world. Use this method on the main thread only. Refreshing the
        /// pointers requires looking up the entity through EntityManager, which worker threads must never do.
        /// </summary>
        internal static NativeStore.BodyData* Hot(NativeStore.Body* body)
        {
            NativeStore.BodyData* hot = body->world->bodies + body->index;
            BodyEntities.Refresh(hot);
            return hot;
        }

        /// <summary>
        /// Returns the body's row for access to fields stored directly in it: flags, shapes and owner. These fields do
        /// not depend on component storage in entities. Unity sends some commands from worker threads, which may read
        /// these fields even when the component pointers are stale. Only the main thread can refresh the component
        /// pointers. The row itself can move when bodies are created or destroyed.
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
        /// Marks a query world in which only static bodies have moved since the last build. The next build only needs
        /// to update their poses and update or rebuild the static tree. Unity moves static bodies every frame: KSC's
        /// tracking dishes turn, and the scaled spheres for planets orbit.
        /// </summary>
        internal const uint QueryStaticsMoved = 2;

        /// <summary>
        /// Marks the query world as stale. The write is atomic because worker threads read this flag.
        /// </summary>
        internal static void MarkQueryDirty(NativeStore.WorldData* world)
        {
            Interlocked.Exchange(ref *(int*)&world->queryDirty, (int)QueryStale);
        }

        /// <summary>
        /// Marks the query world after a static body has only moved. If the world is already marked as stale, it
        /// remains stale because a full rebuild will include this movement.
        /// </summary>
        internal static void MarkStaticMoved(NativeStore.Body* body)
        {
            Interlocked.CompareExchange(ref *(int*)&body->world->queryDirty, (int)QueryStaticsMoved, 0);
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
        /// Marks a body whose collider must be built again because its shapes changed. A change to the shapes can also
        /// change what the query snapshot holds, so this moves <see cref="QuerySnapshotVersion"/> as well.
        /// </summary>
        internal static void MarkColliderDirty(NativeStore.Body* body)
        {
            NativeStore.BodyData* hot = Hot(body);
            hot->flags |= NativeStore.BodyFlags.ColliderDirty;
            QuerySnapshotVersion.Changed();
            MarkPoseChanged(body, hot->flags);
        }

        /// <summary>
        /// Marks colliders after a shape moves within its body. A compound collider can be copied with updated child
        /// poses. Other colliders have the pose baked into their geometry and must be rebuilt.
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
        /// Hides a removed shape from the last published query world, which workers continue to read. Otherwise a hit
        /// could return a pointer to Unity for a collider that is about to be destroyed. The shape's slot may refer to
        /// an older query world, so the code checks that the slot still identifies this shape.
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
