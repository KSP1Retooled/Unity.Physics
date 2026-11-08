using KSP.Retooled.Components.Physics;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Physics;
using Unity.Transforms;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Moves bodies' components between the bodies themselves and entities. A world can only exist once Entities has
    /// started, and bodies exist from the first scene, so until a world binds a body holds its components in its
    /// <see cref="NativeStore.BodyHome"/>. Nothing else cares which it is: a row's pointers lead to the components either way.
    /// </summary>
    /// <remarks>
    /// Rows go through the entity's chunk directly, because a craft sends tens of thousands of commands a frame, and only
    /// look the entity up again when something outside this file restructured body entities. The structural changes made
    /// here leave every row current themselves: a created entity moves nothing, and a destroyed one is replaced in its
    /// chunk by the chunk's last entity, whose row takes over the pointers. Those writes don't bump a chunk's change
    /// version. A job that writes these components has to finish before its system's update ends.
    ///
    /// Unity sends some commands from job workers, and only the main thread may use the EntityManager, so a worker that
    /// finds a row stale reads it as it is. That can only happen between a foreign restructure and the next step.
    /// </remarks>
    internal static unsafe class BodyEntities
    {
        private struct Binding
        {
            public EntityManager Manager;
            public EntityArchetype Archetype;
            public bool Bound;
        }

        private readonly struct BindingKey { }

        private static readonly SharedStatic<Binding> Current = SharedStatic<Binding>.GetOrCreate<BindingKey>();

        internal static bool Bound => Current.Data.Bound;

        /// <summary>Moves every body onto an entity in <paramref name="manager"/>'s world, as new bodies will be until <see cref="Unbind"/>.</summary>
        internal static void Bind(EntityManager manager)
        {
            ref Binding binding = ref Current.Data;
            binding.Manager = manager;
            binding.Archetype = manager.CreateArchetype(
                ComponentType.ReadWrite<LocalTransform>(),
                ComponentType.ReadWrite<PhysicsVelocity>(),
                ComponentType.ReadWrite<PhysicsMass>(),
                ComponentType.ReadWrite<BodyMass>(),
                ComponentType.ReadWrite<PhysicsDamping>(),
                ComponentType.ReadWrite<BodyLimits>(),
                ComponentType.ReadWrite<BodySleep>(),
                ComponentType.ReadWrite<BodyForces>(),
                ComponentType.ReadWrite<BodyTarget>(),
                ComponentType.ReadWrite<PhysicsCollider>());
            binding.Bound = true;

            int worlds = NativeStore.RetooledPhysics_WorldCount();
            for (int w = 0; w < worlds; w++)
            {
                NativeStore.WorldData* world = NativeStore.RetooledPhysics_GetWorld(w);
                for (uint i = 0; i < world->bodyCount; i++)
                {
                    Place(world->bodies + i);
                }
            }

            StampAll();
        }

        /// <summary>
        /// Moves every body back onto itself. The entities go, so that a later bind doesn't leave them behind. The world
        /// is usually being disposed, so whatever Entities refuses on the way, no row may still lead into it afterwards:
        /// Unity's shutdown sends shape and body commands after the world is gone, and a row reaching a disposed
        /// EntityManager from one of them takes the process down with the lock file held.
        /// </summary>
        internal static void Unbind()
        {
            ref Binding binding = ref Current.Data;
            int worlds = NativeStore.RetooledPhysics_WorldCount();
            try
            {
                for (int w = 0; w < worlds; w++)
                {
                    NativeStore.WorldData* world = NativeStore.RetooledPhysics_GetWorld(w);
                    for (uint i = 0; i < world->bodyCount; i++)
                    {
                        Release(world->bodies + i);
                    }
                }
            }
            finally
            {
                for (int w = 0; w < worlds; w++)
                {
                    NativeStore.WorldData* world = NativeStore.RetooledPhysics_GetWorld(w);
                    for (uint i = 0; i < world->bodyCount; i++)
                    {
                        Abandon(world->bodies + i);
                    }
                }

                binding.Bound = false;
            }
        }

        /// <summary>Puts a body on an entity if a world is bound.</summary>
        internal static void Adopt(NativeStore.BodyData* hot)
        {
            if (!Current.Data.Bound)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.AdoptBody.Auto();

            Place(hot);
            StampAll();
        }

        private static void Place(NativeStore.BodyData* hot)
        {
            NativeStore.BodyHome* home = &hot->owner->home;
            hot->entity = Current.Data.Manager.CreateEntity(Current.Data.Archetype);
            Resolve(hot);
            *hot->pose = home->pose;
            *hot->velocity = home->velocity;
            *hot->mass = home->mass;
            *hot->bodyMass = home->bodyMass;
            *hot->damping = home->damping;
            *hot->limits = home->limits;
            *hot->sleep = home->sleep;
            *hot->forces = home->forces;
            *hot->target = home->target;
            *hot->collider = home->collider;
        }

        /// <summary>Takes a body off its entity, if it has one, keeping its state.</summary>
        internal static void Release(NativeStore.BodyData* hot)
        {
            if (hot->entity == Entity.Null)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ReleaseBody.Auto();

            Refresh(hot);
            NativeStore.BodyHome* home = &hot->owner->home;
            home->pose = *hot->pose;
            home->velocity = *hot->velocity;
            home->mass = *hot->mass;
            home->bodyMass = *hot->bodyMass;
            home->damping = *hot->damping;
            home->limits = *hot->limits;
            home->sleep = *hot->sleep;
            home->forces = *hot->forces;
            home->target = *hot->target;
            home->collider = *hot->collider;

            ref EntityManager manager = ref Current.Data.Manager;
            EntityStorageInfo slot = manager.GetStorageInfo(hot->entity);
            bool refilled = slot.IndexInChunk != slot.Chunk.Count - 1;
            NativeStore.BodyData vacated = *hot;
            manager.DestroyEntity(hot->entity);
            hot->entity = Entity.Null;
            StoreOps.PointAtHome(hot, hot->owner);
            if (refilled)
            {
                HandOver(slot, &vacated);
            }

            StampAll();
        }

        /// <summary>Points a row home without its entity's state, for a world that can't give it back.</summary>
        private static void Abandon(NativeStore.BodyData* hot)
        {
            if (hot->entity == Entity.Null)
            {
                return;
            }

            hot->entity = Entity.Null;
            StoreOps.PointAtHome(hot, hot->owner);
        }

        /// <summary>
        /// Gives the pointers of a destroyed entity's slot to the row of the entity Entities moved into it, which is the one
        /// that was last in the chunk.
        /// </summary>
        private static void HandOver(EntityStorageInfo slot, NativeStore.BodyData* vacated)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.HandOverRow.Auto();

            ref EntityManager manager = ref Current.Data.Manager;
            int worlds = NativeStore.RetooledPhysics_WorldCount();
            for (int w = 0; w < worlds; w++)
            {
                NativeStore.WorldData* world = NativeStore.RetooledPhysics_GetWorld(w);
                for (uint i = 0; i < world->bodyCount; i++)
                {
                    NativeStore.BodyData* hot = world->bodies + i;
                    if (hot->entity == Entity.Null)
                    {
                        continue;
                    }

                    EntityStorageInfo moved = manager.GetStorageInfo(hot->entity);
                    if (moved.Chunk == slot.Chunk && moved.IndexInChunk == slot.IndexInChunk)
                    {
                        hot->pose = vacated->pose;
                        hot->velocity = vacated->velocity;
                        hot->mass = vacated->mass;
                        hot->bodyMass = vacated->bodyMass;
                        hot->damping = vacated->damping;
                        hot->limits = vacated->limits;
                        hot->sleep = vacated->sleep;
                        hot->forces = vacated->forces;
                        hot->target = vacated->target;
                        hot->collider = vacated->collider;
                        return;
                    }
                }
            }
        }

        /// <summary>Marks every row current, after a structural change here that left none of them behind.</summary>
        private static void StampAll()
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.StampRows.Auto();

            int version = Version();
            int worlds = NativeStore.RetooledPhysics_WorldCount();
            for (int w = 0; w < worlds; w++)
            {
                NativeStore.WorldData* world = NativeStore.RetooledPhysics_GetWorld(w);
                for (uint i = 0; i < world->bodyCount; i++)
                {
                    NativeStore.BodyData* hot = world->bodies + i;
                    if (hot->entity != Entity.Null)
                    {
                        hot->orderVersion = version;
                    }
                }
            }
        }

        /// <summary>
        /// The order version of the body archetype: it moves when body entities are created, destroyed or given other
        /// components, and other archetypes don't move it.
        /// </summary>
        private static int Version()
        {
            return Current.Data.Manager.GetComponentOrderVersion<BodyTarget>();
        }

        /// <summary>Makes a row's pointers current, if its body has an entity. Only the main thread can, as the remarks say.</summary>
        internal static void Refresh(NativeStore.BodyData* hot)
        {
            if (hot->entity == Entity.Null || hot->orderVersion == Version())
            {
                return;
            }

            if (NativeStore.RetooledPhysics_OnMainThread() == 0)
            {
                return;
            }

            Resolve(hot);
        }

        /// <summary>
        /// Readies a world's rows for jobs, which read through their pointers and can't refresh them. No structural change
        /// can happen until the jobs finish, because they run inside a step that blocks the main thread.
        /// </summary>
        internal static void PrepareForJobs(NativeStore.WorldData* world)
        {
            ref Binding binding = ref Current.Data;
            if (!binding.Bound)
            {
                return;
            }

            // Query builds arrive in the middle of system updates, so this can't be CompleteAllTrackedJobs, which also
            // invalidates the type handles those systems are holding. It leaves out the components other packages share,
            // whose dependencies can reach jobs that have nothing to do with bodies.
            ref EntityManager manager = ref binding.Manager;
            manager.CompleteDependencyBeforeRW<BodyMass>();
            manager.CompleteDependencyBeforeRW<BodyLimits>();
            manager.CompleteDependencyBeforeRW<BodySleep>();
            manager.CompleteDependencyBeforeRW<BodyForces>();
            manager.CompleteDependencyBeforeRW<BodyTarget>();
            for (uint i = 0; i < world->bodyCount; i++)
            {
                Refresh(world->bodies + i);
            }
        }

        private static void Resolve(NativeStore.BodyData* hot)
        {
            EntityManager manager = Current.Data.Manager;
            Entity entity = hot->entity;
            hot->pose = Pointer(manager.GetComponentDataRW<LocalTransform>(entity));
            hot->velocity = Pointer(manager.GetComponentDataRW<PhysicsVelocity>(entity));
            hot->mass = Pointer(manager.GetComponentDataRW<PhysicsMass>(entity));
            hot->bodyMass = Pointer(manager.GetComponentDataRW<BodyMass>(entity));
            hot->damping = Pointer(manager.GetComponentDataRW<PhysicsDamping>(entity));
            hot->limits = Pointer(manager.GetComponentDataRW<BodyLimits>(entity));
            hot->sleep = Pointer(manager.GetComponentDataRW<BodySleep>(entity));
            hot->forces = Pointer(manager.GetComponentDataRW<BodyForces>(entity));
            hot->target = Pointer(manager.GetComponentDataRW<BodyTarget>(entity));
            hot->collider = Pointer(manager.GetComponentDataRW<PhysicsCollider>(entity));

            // Last, so that a worker never sees a current version with old pointers.
            hot->orderVersion = Version();
        }

        private static T* Pointer<T>(RefRW<T> component)
            where T : unmanaged, IComponentData
        {
            return (T*)UnsafeUtility.AddressOf(ref component.ValueRW);
        }
    }
}
