using System.Threading;
using Unity.Burst;
using Unity.Mathematics;

namespace KSP.Retooled.PhysicsBackend
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

        /// <summary>The body's hot row. It stays valid until a body in the same world is created or destroyed.</summary>
        internal static NativeStore.BodyData* Hot(NativeStore.Body* body)
        {
            return body->world->bodies + body->index;
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

        /// <summary>A worker thread reads this flag to learn its query world is stale, so the write is an atomic one.</summary>
        internal static void MarkQueryDirty(NativeStore.WorldData* world)
        {
            Interlocked.Exchange(ref *(int*)&world->queryDirty, 1);
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
