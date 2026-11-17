using KSP.Retooled.Components.Physics;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Physics;
using Unity.Transforms;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Move body components between native body records and entities. Row pointers provide access to either
    /// location.
    /// Refresh pointers after external structural changes. Only the main thread may use EntityManager.
    /// Direct writes do not update chunk change versions; jobs that write must finish within their system
    /// update.
    /// </summary>
    internal static unsafe class BodyEntities
    {
        private struct Binding
        {
            public EntityManager Manager;
            public EntityArchetype Archetype;
            public bool Bound;

            /// <summary>
            /// Component order version for which all row pointers are current.
            /// </summary>
            public int CleanVersion;
        }

        private readonly struct BindingKey { }

        private static readonly SharedStatic<Binding> Current = SharedStatic<Binding>.GetOrCreate<BindingKey>();

        internal static bool Bound => Current.Data.Bound;

        /// <summary>
        /// Move every body's components to the given Entities world. New bodies use it until Unbind is called.
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
        /// Copy body components back to native records and destroy their entities.
        /// All row pointers must leave the Entities world, even if cleanup throws, because shutdown commands
        /// still access them.
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

            // Creating an entity preserves existing pointers, but does not refresh pointers that were already
            // stale.
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
        /// Point a row at native component storage without copying from its entity. Use when the Entities world
        /// is unavailable.
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
        /// Refresh the row for the entity moved into a destroyed entity's chunk slot.
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
        /// Component order version for BodyTarget. Changes when body entities are created, destroyed or moved
        /// between archetypes.
        /// </summary>
        private static int Version()
        {
            return Current.Data.Manager.GetComponentOrderVersion<BodyTarget>();
        }

        /// <summary>
        /// Refresh a body's entity component pointers on the main thread.
        /// </summary>
        internal static void Refresh(NativeStore.BodyData* hot)
        {
            // Rows in native storage need no pointer refresh.
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
        /// Refresh row pointers before scheduling jobs. The step blocks the main thread until those jobs
        /// finish.
        /// </summary>
        internal static void PrepareForJobs(NativeStore.WorldData* world)
        {
            ref Binding binding = ref Current.Data;
            if (!binding.Bound)
            {
                return;
            }

            // Complete body dependencies only. CompleteAllTrackedJobs would invalidate type handles in active
            // systems.
            ref EntityManager manager = ref binding.Manager;
            manager.CompleteDependencyBeforeRW<BodyMass>();
            manager.CompleteDependencyBeforeRW<BodyLimits>();
            manager.CompleteDependencyBeforeRW<BodySleep>();
            manager.CompleteDependencyBeforeRW<BodyForces>();
            manager.CompleteDependencyBeforeRW<BodyTarget>();

            // Skip the refresh when every row is current for this component order version.
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

            // Publish the version after the pointers so that workers cannot observe partially refreshed state.
            hot->orderVersion = Version();
        }

        private static T* Pointer<T>(RefRW<T> component)
            where T : unmanaged, IComponentData
        {
            return (T*)UnsafeUtility.AddressOf(ref component.ValueRW);
        }
    }

    /// <summary>
    /// Link an entity to its native body record, whose address remains stable when entities move.
    /// </summary>
    internal unsafe struct BodyOwner : IComponentData
    {
        public NativeStore.Body* Body;
    }
}
