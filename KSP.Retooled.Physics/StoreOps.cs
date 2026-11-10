using System.Threading;
using Unity.Burst;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// What the command kernels share: the store's memory, its dirty marks and the list of a body's shapes. A world's handle
    /// points at its <see cref="NativeStore.WorldData"/>, and a shape's at its <see cref="NativeStore.ShapeData"/>.
    /// </summary>
    internal static unsafe class StoreOps
    {
        // What a woken body's counter starts from. PhysX's Dy::sleepCheck: 20 steps of 0.02 s, whatever the step actually is.
        internal const float WakeCounterReset = 0.4f;

        private const uint MinimumCapacity = 32;

        /// <summary>
        /// The body's hot row, with its component pointers current. It stays valid until a body in the same world is created
        /// or destroyed, and its pointers until the next structural change in the bound Entities world. Main thread only:
        /// making the pointers current looks the entity up through the EntityManager, which a worker must never do.
        /// </summary>
        internal static NativeStore.BodyData* Hot(NativeStore.Body* body)
        {
            NativeStore.BodyData* hot = body->world->bodies + body->index;
            BodyEntities.Refresh(hot);
            return hot;
        }

        /// <summary>
        /// The body's hot row for the fields the row holds itself, which never move: its flags, shapes and owner. Unity sends
        /// some commands from job workers, and this is all they may read, because the component pointers may be stale and
        /// only the main thread can renew them.
        /// </summary>
        internal static NativeStore.BodyData* Row(NativeStore.Body* body)
        {
            return body->world->bodies + body->index;
        }

        /// <summary>Points a row at the components its body holds itself, which never move.</summary>
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
        /// A body's angular velocity in world space. PhysicsVelocity holds it about the principal axes, which is what a
        /// Rigidbody's callers never see.
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

        /// <summary>Moving a body keeps its angular velocity in world space, so the value held about its axes has to follow.</summary>
        internal static void SetRotation(in NativeStore.BodyData body, quaternion rotation)
        {
            float3 angular = WorldAngularVelocity(body);
            body.pose->Rotation = rotation;
            SetWorldAngularVelocity(body, angular);
        }

        /// <summary>As <see cref="SetRotation"/>, for the axes themselves.</summary>
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

        /// <summary>A zeroed record on the store's heap.</summary>
        internal static T* Allocate<T>(NativeStore.StoreGlobals* globals)
            where T : unmanaged
        {
            T* record = (T*)Reallocate(globals, null, (ulong)sizeof(T));
            *record = default;
            return record;
        }

        /// <summary>The capacity a full array of <paramref name="capacity"/> entries grows to.</summary>
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

        /// <summary>The query world's dirty mark when it needs building again from the store.</summary>
        internal const uint QueryStale = 1;

        /// <summary>
        /// The query world's dirty mark when only static bodies moved since it was built, so a build need only pose them and
        /// rebuild the static tree. Unity moves statics every frame: KSC's tracking dishes turn, and the planets' scaled
        /// spheres orbit.
        /// </summary>
        internal const uint QueryStaticsMoved = 2;

        /// <summary>A worker thread reads this flag to learn its query world is stale, so the write is an atomic one.</summary>
        internal static void MarkQueryDirty(NativeStore.WorldData* world)
        {
            Interlocked.Exchange(ref *(int*)&world->queryDirty, (int)QueryStale);
        }

        /// <summary>For a static body that only moved. A world already marked stale stays so, and that build covers this.</summary>
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

        internal static void MarkColliderDirty(NativeStore.Body* body)
        {
            NativeStore.BodyData* hot = Hot(body);
            hot->flags |= NativeStore.BodyFlags.ColliderDirty;
            MarkPoseChanged(body, hot->flags);
        }

        /// <summary>
        /// For a shape that moved within its body. A compound is copied about its children, where any other collider has the
        /// pose baked in and is built again.
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
        /// A worker keeps reading the query world last published, where a hit on this shape would hand Unity a pointer to a
        /// collider that may be about to go away. The slot can be left over from an older world, hence the check.
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

        /// <summary>Takes the shape off its body, keeping the order of the rest, which a compound's child indices follow.</summary>
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

        /// <summary>Lets go of a body's shapes when the body goes away. They outlive it until Unity destroys them.</summary>
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
