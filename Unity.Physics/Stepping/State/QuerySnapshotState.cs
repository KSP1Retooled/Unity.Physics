using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the query snapshot and tracks publication of the built world.
    /// Reuses the snapshot when its version and both lists of body rows match.
    /// </summary>
    internal unsafe struct QuerySnapshotState
    {
        private NativeList<QueryBody> queryBodies;
        private NativeList<QueryShape> queryShapes;
        private NativeList<int> queryFirstShapes;

        // Reuse query snapshots while versions and body row order match.
        // Mark a scheduled snapshot incomplete until it finishes so that exceptions force a fresh copy.
        private bool snapshotTaken;
        private uint snapshotVersion;
        private NativeList<int> snapshotDynamic;
        private NativeList<int> snapshotStatic;
        private long keptSnapshots;
        // False from the start of a build until the query world is published. If a build fails, the next build rebuilds
        // the whole world.
        internal bool builtWhole;

        /// <summary>The number of steps and query builds that kept the last query snapshot. Used by tests.</summary>
        internal readonly long KeptSnapshots => keptSnapshots;

        internal void Initialize()
        {
            queryBodies = new NativeList<QueryBody>(Allocator.Persistent);
            queryShapes = new NativeList<QueryShape>(Allocator.Persistent);
            queryFirstShapes = new NativeList<int>(Allocator.Persistent);
            snapshotDynamic = new NativeList<int>(Allocator.Persistent);
            snapshotStatic = new NativeList<int>(Allocator.Persistent);
        }

        internal void Dispose()
        {
            queryBodies.Dispose();
            queryShapes.Dispose();
            queryFirstShapes.Dispose();
            snapshotDynamic.Dispose();
            snapshotStatic.Dispose();
        }

        /// <summary>
        /// Copy query data from the store, or return dependency unchanged if the snapshot is still current.
        /// Do not retain pointers into movable store memory. Call FinishQuerySnapshot after completion.
        /// </summary>
        internal JobHandle ScheduleQuerySnapshot(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref BodyPartition partition, JobHandle dependency)
        {
            uint version = QuerySnapshotVersion.Version;
            if (snapshotTaken && version == snapshotVersion && BodyPartition.SameRows(snapshotDynamic, partition.dynamicIndices) && BodyPartition.SameRows(snapshotStatic, partition.staticIndices))
            {
                keptSnapshots++;
                return dependency;
            }

            snapshotTaken = false;
            snapshotVersion = version;
            snapshotDynamic.CopyFrom(partition.dynamicIndices.AsArray());
            snapshotStatic.CopyFrom(partition.staticIndices.AsArray());
            queryBodies.ResizeUninitialized(world.NumBodies);
            queryFirstShapes.ResizeUninitialized(world.NumBodies);
            JobHandle found = new QuerySnapshot.FindFirstShapesJob
            {
                Store = store->bodies,
                DynamicIndices = partition.dynamicIndices.AsArray(),
                StaticIndices = partition.staticIndices.AsArray(),
                FirstShapes = queryFirstShapes.AsArray(),
                QueryShapes = queryShapes,
            }.Schedule(dependency);

            return new QuerySnapshot.SnapshotQueryDataJob
            {
                Store = store->bodies,
                DynamicIndices = partition.dynamicIndices.AsArray(),
                StaticIndices = partition.staticIndices.AsArray(),
                FirstShapes = queryFirstShapes.AsArray(),
                QueryBodies = queryBodies.AsArray(),
                QueryShapes = queryShapes.AsDeferredJobArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(world.NumBodies, 64, found);
        }

        /// <summary>Records that the snapshot scheduled by <see cref="ScheduleQuerySnapshot"/> has completed.</summary>
        internal void FinishQuerySnapshot()
        {
            snapshotTaken = true;
        }

        /// <summary>
        /// Publish the built world and completed snapshot for queries.
        /// Colliders remain valid until the next build acquires the exclusive query lock.
        /// </summary>
        internal void PublishQueryWorld(WorldData* store, QueryWorld* query, ref PhysicsWorld world)
        {
            query->Collision = world.CollisionWorld;
            query->Bodies = queryBodies.GetUnsafeReadOnlyPtr();
            query->Shapes = queryShapes.GetUnsafePtr();
            query->ShapeCount = queryShapes.Length;
            store->queryWorld = query;
            builtWhole = true;
        }

        internal bool BeginBuild(WorldData* store)
        {
            // The world that the last step published is current now that its refit has finished.
            if (store->queryDirty == StoreMemory.QueryTreePending && builtWhole)
            {
                return false;
            }

            store->queryWorld = null;
            return true;
        }

        internal readonly bool CanUpdateStaticPoses(WorldData* store, StoreGlobals* globals, ref ColliderState colliders)
        {
            return store->queryDirty == StoreMemory.QueryStaticsMoved && builtWhole
                && globals->materialVersion == colliders.builtMaterialVersion;
        }

        internal readonly bool OnlyRefitPending(WorldData* store)
        {
            return store->queryDirty == StoreMemory.QueryTreePending && builtWhole && store->queryWorld != null;
        }
    }
}
