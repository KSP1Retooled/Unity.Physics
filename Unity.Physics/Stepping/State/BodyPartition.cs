using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the store rows in build order and the map from store rows to built indices.
    /// Rebuilds the partition from the store on each build.
    /// </summary>
    internal unsafe struct BodyPartition
    {
        internal NativeList<int> dynamicIndices;
        internal NativeList<int> staticIndices;
        internal NativeList<int> builtIndices;
        internal NativeReference<int> partitionResult;
        internal bool motionsMatchStore;

        internal void InitializeRows()
        {
            dynamicIndices = new NativeList<int>(Allocator.Persistent);
            staticIndices = new NativeList<int>(Allocator.Persistent);
        }

        internal void Initialize()
        {
            builtIndices = new NativeList<int>(Allocator.Persistent);
            partitionResult = new NativeReference<int>(1, Allocator.Persistent);
        }

        internal void DisposeRows()
        {
            dynamicIndices.Dispose();
            staticIndices.Dispose();
        }

        internal void Dispose()
        {
            builtIndices.Dispose();
            partitionResult.Dispose();
        }

        /// <summary>
        /// Return whether safety handles permit container writes before starting a build.
        /// A main-thread query inside a job must keep using the published world if access is denied.
        /// After checking access outside jobs, pass the containers to build jobs again to restore ownership.
        /// </summary>
        internal bool CanWriteContainers()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            try
            {
                // These calls check write access without changing the containers.
                staticIndices.GetUnsafePtr();
                dynamicIndices.GetUnsafePtr();
            }
            catch (InvalidOperationException)
            {
                return false;
            }
#endif
            return true;
        }

        // StoreBodies.WriteBackJob assigns these bodies a pose that differs from the result of integrating their solved motion.
        private const BodyFlags PosedByWriteBack = BodyFlags.Sleeping | BodyFlags.HasKinematicTarget;

        internal void Rebuild(WorldData* store)
        {
            new StoreBodies.PartitionBodiesJob
            {
                World = store,
                DynamicIndices = dynamicIndices,
                StaticIndices = staticIndices,
                BuiltIndices = builtIndices,
                PosedByWriteBack = PosedByWriteBack,
                MotionsMatchStore = partitionResult,
            }.Run();
            motionsMatchStore = partitionResult.Value != 0;
        }

        internal static bool SameRows(NativeList<int> kept, NativeList<int> built)
        {
            return kept.Length == built.Length
                && UnsafeUtility.MemCmp(kept.GetUnsafeReadOnlyPtr(), built.GetUnsafeReadOnlyPtr(), (long)kept.Length * sizeof(int)) == 0;
        }
    }
}
