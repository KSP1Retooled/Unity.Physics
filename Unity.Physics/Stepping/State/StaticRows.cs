using Unity.Physics.Configuration;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Reuses static body rows when the static version, row order, and body storage still match.
    /// </summary>
    internal unsafe struct StaticRows
    {
        internal bool staticsChanged;
        // Store rows used for the retained static bodies.
        internal NativeList<int> staticRows;
        internal long keptStaticRowsCount;
        private uint builtStaticsVersion;

        /// <summary>The builds that kept the static bodies' rows of the last build. Used by tests.</summary>
        internal readonly long KeptStaticRowsCount => keptStaticRowsCount;

        internal void NoteBuild(WorldData* store, ref bool builtWhole)
        {
            staticsChanged = !builtWhole || store->staticsVersion != builtStaticsVersion;
            builtStaticsVersion = store->staticsVersion;
        }

        internal void NotePoseUpdate(WorldData* store)
        {
            builtStaticsVersion = store->staticsVersion;
            staticsChanged = true;
        }

        internal void Initialize()
        {
            staticRows = new NativeList<int>(Allocator.Persistent);
        }

        internal void Dispose()
        {
            staticRows.Dispose();
        }

        /// <summary>
        /// Reuse static body rows when the previous build completed, staticsVersion is unchanged,
        /// and static row order matches. The caller must also verify that body storage was retained.
        /// </summary>
        internal readonly bool KeepsStaticRows(ref PhysicsWorld world, ref BodyPartition partition)
        {
            return StepCacheSwitches.StaticReuseEnabled && !staticsChanged && world.NumStaticBodies == partition.staticIndices.Length + 1
                && BodyPartition.SameRows(staticRows, partition.staticIndices);
        }

        /// <summary>
        /// Move retained static rows after the new dynamic body range when its size changes.
        /// Call only while the built world is inaccessible to queries and jobs.
        /// </summary>
        internal void MoveStaticRows(ref PhysicsWorld world, ref BodyPartition partition, int dynamicBefore)
        {
            if (dynamicBefore == partition.dynamicIndices.Length)
            {
                return;
            }

            RigidBody* bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies);
            UnsafeUtility.MemMove(bodies + partition.dynamicIndices.Length, bodies + dynamicBefore, (long)partition.staticIndices.Length * sizeof(RigidBody));
        }

        internal bool RetainOrRemember(ref PhysicsWorld world, ref BodyPartition partition,
            int dynamicBefore, bool keepStaticRows, bool keptBodies)
        {
            keepStaticRows &= keptBodies;
            if (keepStaticRows)
            {
                keptStaticRowsCount++;
                MoveStaticRows(ref world, ref partition, dynamicBefore);
                NativeArray<RigidBody> bodies = world.Bodies;
                bodies[bodies.Length - 1] = RigidBody.Zero;
            }
            else if (StepCacheSwitches.StaticReuseEnabled)
            {
                // With the reuse off, every build writes the rows, so the rows that they come from don't matter.
                staticRows.CopyFrom(partition.staticIndices.AsArray());
            }
            return keepStaticRows;
        }
    }
}
