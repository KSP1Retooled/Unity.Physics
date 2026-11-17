using System;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Vessels.Model;

namespace Unity.Physics.Dynamics.Simulation.Islands
{
    /// <summary>
    /// Cache islands while pairs, dynamic body count and vessels match.
    /// Retain vessel/body mappings until their inputs change. The caller owns the cache through
    /// SimulationStepInput.
    /// </summary>
    internal unsafe struct SolverIslandCache : IDisposable
    {
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<DispatchPairSequencer.DispatchPair>* pairs;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<DispatchPairSequencer.SolverIsland>* islands;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<DispatchPairSequencer.DispatchPair>* islandPairs;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* islandBodies;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* islandVessels;

        /// <summary>
        /// The number of dynamic bodies used to find the cached islands, or -1 if no islands have been cached.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        int* numDynamicBodies;

        /// <summary>The vessel records of the last step, as the step's vessel set gave them.</summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<VesselRecord>* vessels;

        /// <summary>The part bodies of the last step's vessels, one vessel after another.</summary>
        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* partBodies;

        /// <summary>
        /// The number of dynamic bodies for which the map was built, or -1 if it has not been built.
        /// </summary>
        [NativeDisableUnsafePtrRestriction]
        int* mapDynamicBodies;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* bodyVessel;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* dynamicParts;

        [NativeDisableUnsafePtrRestriction]
        UnsafeList<int>* keyBody;

        public bool IsCreated => pairs != null;

        public static SolverIslandCache Create()
        {
            var cache = new SolverIslandCache
            {
                pairs = UnsafeList<DispatchPairSequencer.DispatchPair>.Create(0, Allocator.Persistent),
                islands = UnsafeList<DispatchPairSequencer.SolverIsland>.Create(0, Allocator.Persistent),
                islandPairs = UnsafeList<DispatchPairSequencer.DispatchPair>.Create(0, Allocator.Persistent),
                islandBodies = UnsafeList<int>.Create(0, Allocator.Persistent),
                islandVessels = UnsafeList<int>.Create(0, Allocator.Persistent),
                numDynamicBodies = (int*)UnsafeUtility.MallocTracked(sizeof(int), UnsafeUtility.AlignOf<int>(), Allocator.Persistent, 0),
                vessels = UnsafeList<VesselRecord>.Create(0, Allocator.Persistent),
                partBodies = UnsafeList<int>.Create(0, Allocator.Persistent),
                mapDynamicBodies = (int*)UnsafeUtility.MallocTracked(sizeof(int), UnsafeUtility.AlignOf<int>(), Allocator.Persistent, 0),
                bodyVessel = UnsafeList<int>.Create(0, Allocator.Persistent),
                dynamicParts = UnsafeList<int>.Create(0, Allocator.Persistent),
                keyBody = UnsafeList<int>.Create(0, Allocator.Persistent),
            };
            *cache.numDynamicBodies = -1;
            *cache.mapDynamicBodies = -1;
            return cache;
        }

        public void Dispose()
        {
            if (!IsCreated)
            {
                return;
            }

            UnsafeList<DispatchPairSequencer.DispatchPair>.Destroy(pairs);
            UnsafeList<DispatchPairSequencer.SolverIsland>.Destroy(islands);
            UnsafeList<DispatchPairSequencer.DispatchPair>.Destroy(islandPairs);
            UnsafeList<int>.Destroy(islandBodies);
            UnsafeList<int>.Destroy(islandVessels);
            UnsafeUtility.FreeTracked(numDynamicBodies, Allocator.Persistent);
            UnsafeList<VesselRecord>.Destroy(vessels);
            UnsafeList<int>.Destroy(partBodies);
            UnsafeUtility.FreeTracked(mapDynamicBodies, Allocator.Persistent);
            UnsafeList<int>.Destroy(bodyVessel);
            UnsafeList<int>.Destroy(dynamicParts);
            UnsafeList<int>.Destroy(keyBody);
            this = default;
        }

        /// <summary>
        /// Compare vessel records and part bodies with the previous step, rebuilding the map if changed.
        /// Return true when they match and cached islands may remain valid.
        /// </summary>
        internal bool KeepVessels(in VesselSet set, int dynamicBodies, out VesselBodyMap map)
        {
            int vesselCount = set.IsCreated ? set.VesselCount : 0;
            bool same = *mapDynamicBodies == dynamicBodies && vessels->Length == vesselCount;
            int offset = 0;
            for (int v = 0; v < vesselCount && same; v++)
            {
                // Compare the fields one by one, because the padding bytes of a record can differ.
                VesselRecord vessel = set.Vessels[v];
                VesselRecord kept = (*vessels)[v];
                same = vessel.Id == kept.Id && vessel.StructureVersion == kept.StructureVersion && vessel.FirstPart == kept.FirstPart
                    && vessel.PartCount == kept.PartCount && vessel.FirstClosure == kept.FirstClosure && vessel.ClosureCount == kept.ClosureCount
                    && offset + vessel.PartCount <= partBodies->Length
                    && UnsafeUtility.MemCmp(partBodies->Ptr + offset, set.PartBodies + vessel.FirstPart, (long)vessel.PartCount * sizeof(int)) == 0;
                offset += vessel.PartCount;
            }

            same &= offset == partBodies->Length;
            if (!same)
            {
                *mapDynamicBodies = dynamicBodies;
                vessels->Resize(vesselCount);
                if (vesselCount > 0)
                {
                    UnsafeUtility.MemCpy(vessels->Ptr, set.Vessels, (long)vesselCount * sizeof(VesselRecord));
                }

                partBodies->Clear();
                for (int v = 0; v < vesselCount; v++)
                {
                    VesselRecord vessel = set.Vessels[v];
                    partBodies->AddRange(set.PartBodies + vessel.FirstPart, vessel.PartCount);
                }

                bodyVessel->Resize(math.max(1, dynamicBodies));
                dynamicParts->Resize(math.max(1, vesselCount));
                keyBody->Resize(math.max(1, vesselCount));
            }

            map = new VesselBodyMap { BodyVessel = bodyVessel->Ptr, DynamicParts = dynamicParts->Ptr, KeyBody = keyBody->Ptr };
            if (!same && vesselCount > 0)
            {
                map.Fill(set, dynamicBodies);
            }

            return same;
        }

        /// <summary>
        /// Reuse cached islands when solved pairs and dynamic body count match. The caller must first verify
        /// vessels with KeepVessels.
        /// </summary>
        internal bool TryReuse(NativeArray<DispatchPairSequencer.DispatchPair> solved, int dynamicBodies,
            NativeList<DispatchPairSequencer.SolverIsland> outIslands, NativeList<DispatchPairSequencer.DispatchPair> outPairs,
            NativeList<int> outBodies, NativeList<int> outVessels)
        {
            if (*numDynamicBodies != dynamicBodies || pairs->Length != solved.Length
                || UnsafeUtility.MemCmp(pairs->Ptr, solved.GetUnsafeReadOnlyPtr(), (long)solved.Length * sizeof(DispatchPairSequencer.DispatchPair)) != 0)
            {
                return false;
            }

            Copy(islands, outIslands);
            Copy(islandPairs, outPairs);
            Copy(islandBodies, outBodies);
            Copy(islandVessels, outVessels);
            return true;
        }

        internal void Keep(NativeArray<DispatchPairSequencer.DispatchPair> solved, int dynamicBodies,
            NativeList<DispatchPairSequencer.SolverIsland> found, NativeList<DispatchPairSequencer.DispatchPair> foundPairs,
            NativeList<int> foundBodies, NativeList<int> foundVessels)
        {
            *numDynamicBodies = dynamicBodies;
            Store(pairs, solved);
            Store(islands, found.AsArray());
            Store(islandPairs, foundPairs.AsArray());
            Store(islandBodies, foundBodies.AsArray());
            Store(islandVessels, foundVessels.AsArray());
        }

        static void Copy<T>(UnsafeList<T>* from, NativeList<T> to)
            where T : unmanaged
        {
            to.ResizeUninitialized(from->Length);
            UnsafeUtility.MemCpy(to.GetUnsafePtr(), from->Ptr, (long)from->Length * sizeof(T));
        }

        static void Store<T>(UnsafeList<T>* to, NativeArray<T> from)
            where T : unmanaged
        {
            to->Resize(from.Length);
            UnsafeUtility.MemCpy(to->Ptr, from.GetUnsafeReadOnlyPtr(), (long)from.Length * sizeof(T));
        }
    }
}
