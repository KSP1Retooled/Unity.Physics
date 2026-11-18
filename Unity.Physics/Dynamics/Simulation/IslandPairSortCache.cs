using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using DispatchPair = Unity.Physics.DispatchPairSequencer.DispatchPair;

namespace Unity.Physics
{
    /// <summary>Stores the input and output of the last serial island pair sort for one simulation.</summary>
    internal unsafe struct IslandPairSortCache : IDisposable
    {
        // The simulation's step dependency orders the jobs that use this cache.
        [NativeDisableUnsafePtrRestriction]
        private UnsafeList<DispatchPair>* unsorted;

        [NativeDisableUnsafePtrRestriction]
        private UnsafeList<DispatchPair>* sorted;

        internal bool IsCreated => unsorted != null;

        internal static IslandPairSortCache Create()
        {
            return new IslandPairSortCache
            {
                unsorted = UnsafeList<DispatchPair>.Create(0, Allocator.Persistent),
                sorted = UnsafeList<DispatchPair>.Create(0, Allocator.Persistent),
            };
        }

        internal bool TryCopySorted(NativeList<DispatchPair> input, NativeList<DispatchPair> output)
        {
            long bytes = (long)input.Length * sizeof(DispatchPair);
            if (unsorted->Length != input.Length || UnsafeUtility.MemCmp(unsorted->Ptr, input.GetUnsafeReadOnlyPtr(), bytes) != 0)
            {
                return false;
            }

            UnsafeUtility.MemCpy(output.GetUnsafePtr(), sorted->Ptr, bytes);
            return true;
        }

        internal void Remember(NativeList<DispatchPair> input, NativeList<DispatchPair> output)
        {
            unsorted->Resize(input.Length);
            sorted->Resize(output.Length);
            UnsafeUtility.MemCpy(unsorted->Ptr, input.GetUnsafeReadOnlyPtr(), (long)input.Length * sizeof(DispatchPair));
            UnsafeUtility.MemCpy(sorted->Ptr, output.GetUnsafeReadOnlyPtr(), (long)output.Length * sizeof(DispatchPair));
        }

        internal void Clear()
        {
            if (IsCreated)
            {
                unsorted->Clear();
                sorted->Clear();
            }
        }

        public void Dispose()
        {
            if (IsCreated)
            {
                UnsafeList<DispatchPair>.Destroy(unsorted);
                UnsafeList<DispatchPair>.Destroy(sorted);
                this = default;
            }
        }
    }
}
