using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the manifold buffers, touching pairs, impulses, warm starts, and known body pairs.
    /// Rechecks touching pairs when the filter version changes and drops entries for removed shapes.
    /// </summary>
    internal unsafe struct ContactState
    {
        internal NativeList<ManifoldRecord> manifolds;
        internal NativeList<ManifoldPoint> manifoldPoints;
        internal NativeList<ContactBuckets.ContactBucket> contactBuckets;
        internal NativeReference<int> filledContactBuckets;
        internal NativeHashMap<ShapePairKey, ShapePairState> touching;

        // The events that end trigger pairs whose shapes were removed, which the next contact report sends to Unity.
        internal NativeList<ShapePairKey> removedTriggerPairs;

        // The version of PairFilterVersion that the last step checked the touching pairs against.
        private uint seenPairFilterVersion;
        internal NativeHashMap<PersistentManifoldKey, float> lastImpulses;
        internal NativeHashMap<JacobianKey, float2> warmStarts;
        internal NativeHashMap<ulong, KnownBodyPairs.KnownBodyPair> knownBodyPairs;
        internal NativeQueue<PairFilter.DropIgnoredPairsJob.KeyValue> learnedBodyPairs;
        internal uint stepCount;

        internal void Initialize()
        {
            manifolds = new NativeList<ManifoldRecord>(Allocator.Persistent);
            manifoldPoints = new NativeList<ManifoldPoint>(Allocator.Persistent);
            contactBuckets = new NativeList<ContactBuckets.ContactBucket>(Allocator.Persistent);
            filledContactBuckets = new NativeReference<int>(0, Allocator.Persistent);
            touching = new NativeHashMap<ShapePairKey, ShapePairState>(64, Allocator.Persistent);
            removedTriggerPairs = new NativeList<ShapePairKey>(Allocator.Persistent);
        }

        internal void InitializeImpulseMemory()
        {
            lastImpulses = new NativeHashMap<PersistentManifoldKey, float>(64, Allocator.Persistent);
            warmStarts = new NativeHashMap<JacobianKey, float2>(64, Allocator.Persistent);
        }

        internal void InitializePairCache()
        {
            knownBodyPairs = new NativeHashMap<ulong, KnownBodyPairs.KnownBodyPair>(1024, Allocator.Persistent);
            learnedBodyPairs = new NativeQueue<PairFilter.DropIgnoredPairsJob.KeyValue>(Allocator.Persistent);
        }

        internal void Dispose()
        {
            manifolds.Dispose();
            manifoldPoints.Dispose();
            for (int i = 0; i < contactBuckets.Length; i++)
            {
                contactBuckets[i].Dispose();
            }

            contactBuckets.Dispose();
            filledContactBuckets.Dispose();
            touching.Dispose();
            removedTriggerPairs.Dispose();
        }

        internal void DisposeImpulseMemory()
        {
            lastImpulses.Dispose();
            warmStarts.Dispose();
        }

        internal void DisposePairCache()
        {
            knownBodyPairs.Dispose();
            learnedBodyPairs.Dispose();
        }

        internal void WakeRefilteredPairs(WorldData* store)
        {
            // Wake refiltered pairs before joint wake propagation and the narrow phase.
            uint filterVersion = PairFilterVersion.Version;
            if (filterVersion != seenPairFilterVersion)
            {
                seenPairFilterVersion = filterVersion;
                if (!touching.IsEmpty)
                {
                    new TouchingPairs.WakeRefilteredPairsJob { World = store, Touching = touching }.Run();
                }
            }
        }

        internal void ForgetRemovedImpulses(WorldData* store)
        {
            // Drop impulses for removed shapes before new contacts can reuse their addresses.
            if (store->removedShapeCount != 0 && lastImpulses.Count != 0)
            {
                new ImpulseMemory.ForgetRemovedImpulsesJob { World = store, LastImpulses = lastImpulses }.Run();
            }
        }

        internal void ForgetRemovedShapes(WorldData* store, ref JobHandle forces)
        {
            if (store->removedShapeCount != 0)
            {
                forces.Complete();
                new TouchingPairs.ForgetRemovedShapesJob { World = store, Touching = touching, RemovedTriggerPairs = removedTriggerPairs }.Run();
            }
        }
    }
}
