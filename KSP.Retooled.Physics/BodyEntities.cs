using KSP.Retooled.Components.Physics;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Moves bodies' components between the bodies themselves and entities. A world can only exist once Entities has
    /// started, and bodies exist from the first scene, so until a world binds a body holds its components in its
    /// <see cref="NativeStore.BodyHome"/>. Nothing else cares which it is: a row's pointers lead to the components either way.
    /// </summary>
    /// <remarks>
    /// Rows go through the entity's chunk directly, and only look the entity up again when a structural change may have
    /// moved it, because a craft sends tens of thousands of commands a frame. Those writes don't bump a chunk's change
    /// version. A job that writes these components has to finish before its system's update ends.
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
                ComponentType.ReadWrite<BodyPose>(),
                ComponentType.ReadWrite<BodyVelocity>(),
                ComponentType.ReadWrite<BodyMass>(),
                ComponentType.ReadWrite<BodyLimits>(),
                ComponentType.ReadWrite<BodySleep>(),
                ComponentType.ReadWrite<BodyForces>(),
                ComponentType.ReadWrite<BodyTarget>());
            binding.Bound = true;

            int worlds = NativeStore.RetooledPhysics_WorldCount();
            for (int w = 0; w < worlds; w++)
            {
                NativeStore.WorldData* world = NativeStore.RetooledPhysics_GetWorld(w);
                for (uint i = 0; i < world->bodyCount; i++)
                {
                    Adopt(world->bodies + i);
                }
            }
        }

        /// <summary>Moves every body back onto itself. The entities go, so that a later bind doesn't leave them behind.</summary>
        internal static void Unbind()
        {
            ref Binding binding = ref Current.Data;
            int worlds = NativeStore.RetooledPhysics_WorldCount();
            for (int w = 0; w < worlds; w++)
            {
                NativeStore.WorldData* world = NativeStore.RetooledPhysics_GetWorld(w);
                for (uint i = 0; i < world->bodyCount; i++)
                {
                    Release(world->bodies + i);
                }
            }

            binding.Bound = false;
        }

        /// <summary>Puts a body on an entity if a world is bound.</summary>
        internal static void Adopt(NativeStore.BodyData* hot)
        {
            ref Binding binding = ref Current.Data;
            if (!binding.Bound)
            {
                return;
            }

            NativeStore.BodyHome* home = &hot->owner->home;
            hot->entity = binding.Manager.CreateEntity(binding.Archetype);
            Resolve(hot);
            *hot->pose = home->pose;
            *hot->velocity = home->velocity;
            *hot->mass = home->mass;
            *hot->limits = home->limits;
            *hot->sleep = home->sleep;
            *hot->forces = home->forces;
            *hot->target = home->target;
        }

        /// <summary>Takes a body off its entity, if it has one, keeping its state.</summary>
        internal static void Release(NativeStore.BodyData* hot)
        {
            if (hot->entity == Entity.Null)
            {
                return;
            }

            Refresh(hot);
            NativeStore.BodyHome* home = &hot->owner->home;
            home->pose = *hot->pose;
            home->velocity = *hot->velocity;
            home->mass = *hot->mass;
            home->limits = *hot->limits;
            home->sleep = *hot->sleep;
            home->forces = *hot->forces;
            home->target = *hot->target;

            Current.Data.Manager.DestroyEntity(hot->entity);
            hot->entity = Entity.Null;
            StoreOps.PointAtHome(hot, hot->owner);
        }

        /// <summary>Makes a row's pointers current, if its body has an entity.</summary>
        internal static void Refresh(NativeStore.BodyData* hot)
        {
            if (hot->entity != Entity.Null && hot->orderVersion != Current.Data.Manager.EntityOrderVersion)
            {
                Resolve(hot);
            }
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
            // invalidates the type handles those systems are holding.
            ref EntityManager manager = ref binding.Manager;
            manager.CompleteDependencyBeforeRW<BodyPose>();
            manager.CompleteDependencyBeforeRW<BodyVelocity>();
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
            hot->pose = Pointer(manager.GetComponentDataRW<BodyPose>(entity));
            hot->velocity = Pointer(manager.GetComponentDataRW<BodyVelocity>(entity));
            hot->mass = Pointer(manager.GetComponentDataRW<BodyMass>(entity));
            hot->limits = Pointer(manager.GetComponentDataRW<BodyLimits>(entity));
            hot->sleep = Pointer(manager.GetComponentDataRW<BodySleep>(entity));
            hot->forces = Pointer(manager.GetComponentDataRW<BodyForces>(entity));
            hot->target = Pointer(manager.GetComponentDataRW<BodyTarget>(entity));

            // Last, so that a query on a worker never sees a current version with old pointers.
            hot->orderVersion = manager.EntityOrderVersion;
        }

        private static T* Pointer<T>(RefRW<T> component)
            where T : unmanaged, IComponentData
        {
            return (T*)UnsafeUtility.AddressOf(ref component.ValueRW);
        }
    }
}
