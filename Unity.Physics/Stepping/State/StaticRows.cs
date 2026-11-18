using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Tracks the static body rows and the anchor slot.
    /// Reuses their data when the static version, row order, and body storage still match.
    /// </summary>
    internal unsafe struct StaticRows
    {
        internal bool staticsChanged;
        // Store rows used for the retained static bodies. Reuse them and the anchor when KeepsStaticRows
        // permits.
        internal NativeList<int> staticRows;
        internal long keptStaticRowsCount;
        private uint builtStaticsVersion;

        /// <summary>
        /// Reuse unchanged static body rows. RETOOLED_STATIC_REUSE=0 disables it.
        /// Stored in SharedStatic because Burst reads it.
        /// </summary>
        internal static bool StaticReuseEnabled
        {
            get => StaticReuseOff.Data == 0;
            set => StaticReuseOff.Data = value ? 0 : 1;
        }

        private sealed class StaticReuseOffKey
        {
        }

        private static readonly Unity.Burst.SharedStatic<int> StaticReuseOff = Unity.Burst.SharedStatic<int>.GetOrCreate<StaticRows, StaticReuseOffKey>();

        /// <summary>Read RETOOLED_STATIC_REUSE. A value of 0 disables static row reuse.</summary>
        internal static void ReadStaticReuseSwitch()
        {
            string reuse = Environment.GetEnvironmentVariable("RETOOLED_STATIC_REUSE");
            StaticReuseEnabled = reuse != "0";
            if (reuse == "0")
            {
                UnityEngine.Debug.Log("[StoreStepper] RETOOLED_STATIC_REUSE=0: every build writes the static bodies' rows from the store");
            }
        }

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
        /// Reuse static body rows and the anchor when the previous build completed, staticsVersion is unchanged,
        /// and static row order matches. The caller must also verify that body storage was retained.
        /// </summary>
        internal readonly bool KeepsStaticRows(ref PhysicsWorld world, ref BodyPartition partition)
        {
            return StaticReuseEnabled && !staticsChanged && world.NumStaticBodies == partition.staticIndices.Length + 1
                && BodyPartition.SameRows(staticRows, partition.staticIndices);
        }

        /// <summary>
        /// Move retained static rows and the anchor after the new dynamic body range when its size changes.
        /// Call only while the built world is inaccessible to queries and jobs.
        /// </summary>
        internal void MoveStaticRows(ref PhysicsWorld world, ref BodyPartition partition, int dynamicBefore)
        {
            if (dynamicBefore == partition.dynamicIndices.Length)
            {
                return;
            }

            RigidBody* bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies);
            UnsafeUtility.MemMove(bodies + partition.dynamicIndices.Length, bodies + dynamicBefore, (long)(partition.staticIndices.Length + 1) * sizeof(RigidBody));
        }

        internal bool RetainOrRemember(ref PhysicsWorld world, ref BodyPartition partition,
            int dynamicBefore, bool keepStaticRows, bool sameMemory)
        {
            keepStaticRows &= sameMemory;
            if (keepStaticRows)
            {
                keptStaticRowsCount++;
                MoveStaticRows(ref world, ref partition, dynamicBefore);
            }
            else if (StaticReuseEnabled)
            {
                // With the reuse off, every build writes the rows, so the rows that they come from don't matter.
                staticRows.CopyFrom(partition.staticIndices.AsArray());
            }
            return keepStaticRows;
        }
    }
}
