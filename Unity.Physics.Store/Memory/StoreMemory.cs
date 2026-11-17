using System.Threading;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;

namespace Unity.Physics.Store.Memory
{
    internal static unsafe class StoreMemory
    {
        // The initial counter value when a body wakes. PhysX's Dy::sleepCheck uses 20 steps of 0.02 s, regardless of
        // the actual step duration.
        internal const float WakeCounterReset = 0.4f;

        private const uint MinimumCapacity = 32;

        /// <summary>
        /// Return the row without refreshing entity component pointers. Workers may read row-owned flags, shapes
        /// and owner.
        /// The row may move when bodies are created or destroyed.
        /// </summary>
        internal static BodyData* Row(Body* body)
        {
            return body->world->bodies + body->index;
        }

        internal static void* Reallocate(StoreGlobals* globals, void* memory, ulong size)
        {
            return ((delegate* unmanaged[Cdecl]<void*, ulong, void*>)globals->reallocate)(memory, size);
        }

        internal static void Release(StoreGlobals* globals, void* memory)
        {
            ((delegate* unmanaged[Cdecl]<void*, void>)globals->release)(memory);
        }

        /// <summary>Allocates a record on the store's heap and initializes it to zero.</summary>
        internal static T* Allocate<T>(StoreGlobals* globals)
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

        internal static void RetainMesh(StoreGlobals* globals, void* mesh)
        {
            ((delegate* unmanaged[Cdecl]<void*, void>)globals->retainMesh)(mesh);
        }

        internal static void ReleaseMesh(StoreGlobals* globals, void* mesh)
        {
            ((delegate* unmanaged[Cdecl]<void*, void>)globals->releaseMesh)(mesh);
        }

        internal static void RetainMaterial(StoreGlobals* globals, MaterialData* material)
        {
            ((delegate* unmanaged[Cdecl]<void*, void>)globals->retainMaterial)(material);
        }

        internal static void ReleaseMaterial(StoreGlobals* globals, MaterialData* material)
        {
            ((delegate* unmanaged[Cdecl]<void*, void>)globals->releaseMaterial)(material);
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
        internal static void MarkQueryDirty(WorldData* world)
        {
            Interlocked.Exchange(ref *(int*)&world->queryDirty, (int)QueryStale);
        }
    }
}
