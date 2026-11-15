using KSP.Retooled.Components.Physics;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Physics;
using Unity.Transforms;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Moves body components between storage in the body records and storage in entities. An Entities world cannot
    /// exist until Entities has started, but bodies exist from the first scene. Until a world is bound, each body
    /// stores its components in its <see cref="NativeStore.BodyHome"/>. Other code uses the row's component pointers to
    /// access the components in either location.
    /// </summary>
    /// <remarks>
    /// Commands access components directly in the entity's chunk through the row's pointers. A craft sends tens of
    /// thousands of commands per frame, so the entity is looked up again only after structural changes made outside
    /// this file. Structural changes made here preserve the validity of pointers that were already current. Creating an
    /// entity does not move other entities. Destroying one moves the chunk's last entity into the vacated slot, and
    /// this code updates that entity's row to use the pointers for that slot. Writes through these pointers do not
    /// increment the chunk's change version. A job that writes these components must finish before its system's update
    /// ends.
    ///
    /// Unity sends some commands from worker threads, but only the main thread may use EntityManager. If a worker
    /// encounters stale pointers in a row, it uses the row without refreshing them. This can happen only after an
    /// external structural change and before the next step.
    /// </remarks>
    internal static unsafe class BodyEntities
    {
        private struct Binding
        {
            public EntityManager Manager;
            public EntityArchetype Archetype;
            public bool Bound;

            /// <summary>
            /// Records the order version for which all rows were known to have current pointers. Structural changes
            /// made here update this version only if all rows were already current. If it matches the current order
            /// version, no row needs to be refreshed.
            /// </summary>
            public int CleanVersion;
        }

        private readonly struct BindingKey { }

        private static readonly SharedStatic<Binding> Current = SharedStatic<Binding>.GetOrCreate<BindingKey>();

        internal static bool Bound => Current.Data.Bound;

        /// <summary>
        /// Moves every body's components to an entity in <paramref name="manager"/>'s world. New bodies also use
        /// entities in that world until <see cref="Unbind"/> is called.
        /// </summary>
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
                ComponentType.ReadWrite<PhysicsCollider>(),
                ComponentType.ReadOnly<BodyOwner>());
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

            binding.CleanVersion = Version();
        }

        /// <summary>
        /// Moves every body's components back into its body record and destroys the entities so that a later bind does
        /// not leave them behind. The world is usually being disposed. Even if Entities throws during cleanup, every
        /// row must point outside that world afterwards. Unity sends shape and body commands during shutdown after the
        /// world is gone. If one of those commands accesses a disposed EntityManager through a row, the process can
        /// crash while still holding the lock file.
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

        /// <summary>Moves a body's components to an entity if an Entities world is bound.</summary>
        internal static void Adopt(NativeStore.BodyData* hot)
        {
            if (!Current.Data.Bound)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.AdoptBody.Auto();

            // Creating an entity does not move any other entity. Existing pointers remain current if they were already
            // current, but stale pointers still need to be refreshed.
            bool clean = Clean();
            Place(hot);
            if (clean)
            {
                Current.Data.CleanVersion = Version();
            }
        }

        private static void Place(NativeStore.BodyData* hot)
        {
            NativeStore.BodyHome* home = &hot->owner->home;
            ref EntityManager manager = ref Current.Data.Manager;
            hot->entity = manager.CreateEntity(Current.Data.Archetype);
            manager.SetComponentData(hot->entity, new BodyOwner { Body = hot->owner });
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

        /// <summary>Copies a body's state back to its body record and destroys its entity, if it has one.</summary>
        internal static void Release(NativeStore.BodyData* hot)
        {
            if (hot->entity == Entity.Null)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ReleaseBody.Auto();

            Refresh(hot);
            bool clean = Clean();
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

            if (clean)
            {
                Current.Data.CleanVersion = Version();
            }
        }

        /// <summary>
        /// Points a row at the body's own component storage without copying state from its entity. Use this when the
        /// Entities world can no longer provide that state.
        /// </summary>
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
        /// Updates the row for the entity that Entities moved into a destroyed entity's slot. That entity was the last
        /// one in the chunk. Its row now uses the component pointers for the vacated slot.
        /// </summary>
        private static void HandOver(EntityStorageInfo slot, NativeStore.BodyData* vacated)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.HandOverRow.Auto();

            ref EntityManager manager = ref Current.Data.Manager;
            ComponentTypeHandle<BodyOwner> owners = manager.GetComponentTypeHandle<BodyOwner>(true);
            NativeStore.Body* body = slot.Chunk.GetNativeArray(ref owners)[slot.IndexInChunk].Body;
            NativeStore.BodyData* hot = body->world->bodies + body->index;
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
            hot->orderVersion = Version();
        }

        private static bool Clean()
        {
            return Current.Data.CleanVersion == Version();
        }

        /// <summary>
        /// Returns the component order version for BodyTarget, which is used to track structural changes to body
        /// entities. Creating or destroying entities with BodyTarget, or changing their archetypes, changes this
        /// version. Structural changes to entities without BodyTarget do not affect it.
        /// </summary>
        private static int Version()
        {
            return Current.Data.Manager.GetComponentOrderVersion<BodyTarget>();
        }

        /// <summary>
        /// Refreshes a row's component pointers if its body has an entity. Only the main thread can refresh these
        /// pointers, as described in the class remarks.
        /// </summary>
        internal static void Refresh(NativeStore.BodyData* hot)
        {
            // A row without an entity always has current pointers. Check this first because jobs call this method
            // frequently.
            if (hot->entity == Entity.Null)
            {
                return;
            }

            int version = Version();
            if (hot->orderVersion == version || Current.Data.CleanVersion == version)
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
        /// Refreshes a world's rows before jobs use their component pointers. The jobs cannot refresh these pointers
        /// themselves. They run within a step that blocks the main thread, so no structural changes can occur until the
        /// jobs finish.
        /// </summary>
        internal static void PrepareForJobs(NativeStore.WorldData* world)
        {
            ref Binding binding = ref Current.Data;
            if (!binding.Bound)
            {
                return;
            }

            // Query worlds can be built during system updates. Do not use CompleteAllTrackedJobs here, because it also
            // invalidates the type handles that those systems are using. Complete dependencies only for components
            // specific to bodies. Dependencies for components shared with other packages may include jobs unrelated to
            // bodies.
            ref EntityManager manager = ref binding.Manager;
            manager.CompleteDependencyBeforeRW<BodyMass>();
            manager.CompleteDependencyBeforeRW<BodyLimits>();
            manager.CompleteDependencyBeforeRW<BodySleep>();
            manager.CompleteDependencyBeforeRW<BodyForces>();
            manager.CompleteDependencyBeforeRW<BodyTarget>();

            // Nothing in the loop below moves an entity, so the order version stays the same throughout the loop. If
            // all rows are already current for this version, none need to be refreshed.
            int version = Version();
            if (binding.CleanVersion == version)
            {
                return;
            }

            for (uint i = 0; i < world->bodyCount; i++)
            {
                NativeStore.BodyData* hot = world->bodies + i;
                if (hot->entity != Entity.Null && hot->orderVersion != version)
                {
                    Refresh(hot);
                }
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

            // Update the version last so that a worker cannot see a current version before the new pointers have been
            // stored.
            hot->orderVersion = Version();
        }

        private static T* Pointer<T>(RefRW<T> component)
            where T : unmanaged, IComponentData
        {
            return (T*)UnsafeUtility.AddressOf(ref component.ValueRW);
        }
    }

    /// <summary>
    /// Identifies the body that owns an entity. When Entities moves the entity, this component provides access to the
    /// body's row. The body record remains at the same address and stores the row's index.
    /// </summary>
    internal unsafe struct BodyOwner : IComponentData
    {
        public NativeStore.Body* Body;
    }
}
