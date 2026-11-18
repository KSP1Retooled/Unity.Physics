using Unity.Collections;
using Unity.Jobs;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.World;
using Unity.Profiling;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the work lists and version for collider refreshes.
    /// A material version change forces a refresh of every collider.
    /// </summary>
    internal unsafe struct ColliderState
    {
        internal NativeList<ColliderBuilder.Work> colliderWork;
        private NativeList<long> colliderResults;
        internal NativeReference<uint> wholeColliderBuilds;
        internal JobHandle collidersBuilt;
        internal uint builtMaterialVersion;
        internal static readonly ProfilerMarker RefreshCollidersMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.RefreshColliders");

        internal void Initialize()
        {
            colliderWork = new NativeList<ColliderBuilder.Work>(Allocator.Persistent);
            colliderResults = new NativeList<long>(Allocator.Persistent);
            wholeColliderBuilds = new NativeReference<uint>(0, Allocator.Persistent);
        }

        internal void Dispose()
        {
            colliderWork.Dispose();
            colliderResults.Dispose();
            wholeColliderBuilds.Dispose();
        }

        internal void ReleaseWorldColliders(WorldData* store)
        {
            ReleaseDeadColliders(store);
            for (uint i = 0; i < store->bodyCount; i++)
            {
                BodyData* body = store->bodies + i;
                ColliderBuilder.Release(body->collider->Value.GetUnsafePtr());
                ColliderBuilder.SetCollider(body, default);
                body->flags |= BodyFlags.ColliderDirty;
            }
        }

        /// <summary>
        /// Schedule requested collider updates. Consumers must wait for the returned handle before reading
        /// colliders.
        /// </summary>
        internal JobHandle RefreshColliders(WorldData* store, StoreGlobals* globals, NativeReference<uint> colliderChanges, JobHandle dependency)
        {
            using var span = RefreshCollidersMarker.Auto();

            // Every step and query build completes the previous build, so this handle has already completed.
            collidersBuilt.Complete();
            ColliderBuilder.AskAgainForFailed(store, colliderWork, colliderResults);
            ReleaseDeadColliders(store);

            // Material changes can replace colliders at the same addresses. Invalidate static rows and the
            // static tree.
            uint materialVersion = globals->materialVersion;
            bool all = materialVersion != builtMaterialVersion;
            if (all)
            {
                store->staticsVersion++;
            }

            collidersBuilt = ColliderBuilder.ScheduleRefresh(store, all, colliderWork, colliderResults,
                wholeColliderBuilds, colliderChanges, dependency);
            builtMaterialVersion = materialVersion;
            return collidersBuilt;
        }

        internal void ReleaseDeadColliders(WorldData* store)
        {
            for (uint i = 0; i < store->deadColliderCount; i++)
            {
                ColliderBuilder.Release(store->deadColliders[i]);
            }

            store->deadColliderCount = 0;
        }
    }
}
