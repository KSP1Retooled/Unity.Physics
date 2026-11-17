using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Memory;

namespace Unity.Physics.Dynamics.Vessels.Detection
{
    /// <summary>
    /// Compile vessels when joint topology changes or tombstones expire, and prepare each step's vessel set.
    /// List parents before children; store tree joints as edges and loop joints as closures.
    /// Reuse constraints until body motion frames or joint settings change.
    /// </summary>
    internal unsafe struct VesselBuild : IDisposable
    {
        /// <summary>The compiled vessels, ordered by the store indices of their root bodies.</summary>
        public NativeList<VesselRecord> Vessels;

        /// <summary>For each part, the store index of its body.</summary>
        public NativeList<int> PartStoreBodies;

        /// <summary>
        /// For each part, the joint unit connecting it to its parent, or -1 for the root of a vessel.
        /// </summary>
        public NativeList<int> PartUnits;

        /// <summary>
        /// For each part, the index of its parent relative to the vessel's first part, or -1 for the root.
        /// </summary>
        public NativeList<int> PartParents;

        /// <summary>For each part, 1 when the part's body is the edge joint's body0, and 0 when it is body1.</summary>
        public NativeList<byte> PartIsBody0;

        /// <summary>For each part, the index of the first part in its vessel.</summary>
        public NativeList<int> PartVesselFirst;

        /// <summary>
        /// For each closure, these lists contain its joint unit and the indices of its two parts. Only the part indices
        /// are relative to the vessel's first part.
        /// </summary>
        public NativeList<int> ClosureUnits;

        public NativeList<int> ClosurePartA;
        public NativeList<int> ClosurePartB;

        /// <summary>
        /// Removal step for each part edge and closure, or -1 while its joint exists. Tombstones have unit -1.
        /// </summary>
        public NativeList<int> PartTombstones;

        public NativeList<int> ClosureTombstones;

        /// <summary>For each joint unit, the vessel that it belongs to, or -1.</summary>
        public NativeList<int> UnitVessels;

        /// <summary>
        /// For each store body, the vessel that it is a part of, or -1. A body added after the last compilation has no
        /// entry.
        /// </summary>
        public NativeList<int> BodyVessels;

        /// <summary>
        /// The edge and closure units of all vessels, keyed by the store indices of their two bodies (see
        /// <see cref="PairKey"/>). Two bodies can share more than one unit.
        /// </summary>
        public NativeParallelMultiHashMap<ulong, int> PairUnits;

        // Stable addresses of the body and joint records from the previous compilation. Compare them during the next
        // compilation to identify unchanged vessels and retain their Ids.
        public NativeList<IntPtr> PartBodyRecords;
        public NativeList<IntPtr> PartJointRecords;
        public NativeList<IntPtr> ClosureJointRecords;

        // The serial numbers of the bodies and joints at those records. The store reuses the memory of a destroyed body
        // or joint, so a body or joint is the same one only if its record and its serial number are the same.
        public NativeList<uint> PartBodySerials;
        public NativeList<uint> PartJointSerials;
        public NativeList<uint> ClosureJointSerials;

        /// <summary>For each compiled vessel, 1 when the vessel takes part in this step.</summary>
        public NativeList<byte> Active;

        /// <summary>The vessels participating in this step, included in the set passed to the solver.</summary>
        public NativeList<VesselRecord> StepVessels;

        // The set passed to the step contains pointers to these lists. Each list has one entry per compiled part or
        // closure, as appropriate.
        public NativeList<int> PartBodies;
        public NativeList<VesselEdge> Edges;
        public NativeList<VesselClosure> Closures;
        public NativeList<VesselEdgeLoad> EdgeLoads;
        public NativeList<VesselEdgeLoad> ClosureLoads;
        public NativeList<bool> EdgeLocked;

        // Cached edges and closures, with both body motion frames and the highest joint settings stamp.
        // Reuse entries while units, frames and stamps match.
        public NativeList<VesselEdge> EdgeCache;
        public NativeList<VesselClosure> ClosureCache;
        public NativeList<RigidTransform> EdgeCacheFrames;
        public NativeList<RigidTransform> ClosureCacheFrames;
        public NativeList<uint> EdgeCacheStamps;
        public NativeList<uint> ClosureCacheStamps;

        public static VesselBuild Create()
        {
            return new VesselBuild
            {
                Vessels = new NativeList<VesselRecord>(Allocator.Persistent),
                PartStoreBodies = new NativeList<int>(Allocator.Persistent),
                PartUnits = new NativeList<int>(Allocator.Persistent),
                PartParents = new NativeList<int>(Allocator.Persistent),
                PartIsBody0 = new NativeList<byte>(Allocator.Persistent),
                PartVesselFirst = new NativeList<int>(Allocator.Persistent),
                ClosureUnits = new NativeList<int>(Allocator.Persistent),
                ClosurePartA = new NativeList<int>(Allocator.Persistent),
                ClosurePartB = new NativeList<int>(Allocator.Persistent),
                PartTombstones = new NativeList<int>(Allocator.Persistent),
                ClosureTombstones = new NativeList<int>(Allocator.Persistent),
                UnitVessels = new NativeList<int>(Allocator.Persistent),
                BodyVessels = new NativeList<int>(Allocator.Persistent),
                PairUnits = new NativeParallelMultiHashMap<ulong, int>(64, Allocator.Persistent),
                PartBodyRecords = new NativeList<IntPtr>(Allocator.Persistent),
                PartJointRecords = new NativeList<IntPtr>(Allocator.Persistent),
                ClosureJointRecords = new NativeList<IntPtr>(Allocator.Persistent),
                PartBodySerials = new NativeList<uint>(Allocator.Persistent),
                PartJointSerials = new NativeList<uint>(Allocator.Persistent),
                ClosureJointSerials = new NativeList<uint>(Allocator.Persistent),
                Active = new NativeList<byte>(Allocator.Persistent),
                StepVessels = new NativeList<VesselRecord>(Allocator.Persistent),
                PartBodies = new NativeList<int>(Allocator.Persistent),
                Edges = new NativeList<VesselEdge>(Allocator.Persistent),
                Closures = new NativeList<VesselClosure>(Allocator.Persistent),
                EdgeLoads = new NativeList<VesselEdgeLoad>(Allocator.Persistent),
                ClosureLoads = new NativeList<VesselEdgeLoad>(Allocator.Persistent),
                EdgeLocked = new NativeList<bool>(Allocator.Persistent),
                EdgeCache = new NativeList<VesselEdge>(Allocator.Persistent),
                ClosureCache = new NativeList<VesselClosure>(Allocator.Persistent),
                EdgeCacheFrames = new NativeList<RigidTransform>(Allocator.Persistent),
                ClosureCacheFrames = new NativeList<RigidTransform>(Allocator.Persistent),
                EdgeCacheStamps = new NativeList<uint>(Allocator.Persistent),
                ClosureCacheStamps = new NativeList<uint>(Allocator.Persistent),
            };
        }

        public void Dispose()
        {
            Vessels.Dispose();
            PartStoreBodies.Dispose();
            PartUnits.Dispose();
            PartParents.Dispose();
            PartIsBody0.Dispose();
            PartVesselFirst.Dispose();
            ClosureUnits.Dispose();
            ClosurePartA.Dispose();
            ClosurePartB.Dispose();
            PartTombstones.Dispose();
            ClosureTombstones.Dispose();
            UnitVessels.Dispose();
            BodyVessels.Dispose();
            PairUnits.Dispose();
            PartBodyRecords.Dispose();
            PartJointRecords.Dispose();
            ClosureJointRecords.Dispose();
            PartBodySerials.Dispose();
            PartJointSerials.Dispose();
            ClosureJointSerials.Dispose();
            Active.Dispose();
            StepVessels.Dispose();
            PartBodies.Dispose();
            Edges.Dispose();
            Closures.Dispose();
            EdgeLoads.Dispose();
            ClosureLoads.Dispose();
            EdgeLocked.Dispose();
            EdgeCache.Dispose();
            ClosureCache.Dispose();
            EdgeCacheFrames.Dispose();
            ClosureCacheFrames.Dispose();
            EdgeCacheStamps.Dispose();
            ClosureCacheStamps.Dispose();
        }

        /// <summary>Clears all compiled vessels, as required when vessels are disabled.</summary>
        public void Clear(int unitCount)
        {
            Vessels.Clear();
            PartStoreBodies.Clear();
            PartUnits.Clear();
            PartParents.Clear();
            PartIsBody0.Clear();
            PartVesselFirst.Clear();
            ClosureUnits.Clear();
            ClosurePartA.Clear();
            ClosurePartB.Clear();
            PartTombstones.Clear();
            ClosureTombstones.Clear();
            PartBodyRecords.Clear();
            PartJointRecords.Clear();
            ClosureJointRecords.Clear();
            PartBodySerials.Clear();
            PartJointSerials.Clear();
            ClosureJointSerials.Clear();
            Active.Clear();
            StepVessels.Clear();
            BodyVessels.Clear();
            PairUnits.Clear();
            EdgeCache.Clear();
            ClosureCache.Clear();
            EdgeCacheFrames.Clear();
            ClosureCacheFrames.Clear();
            EdgeCacheStamps.Clear();
            ClosureCacheStamps.Clear();
            UnitVessels.ResizeUninitialized(unitCount);
            for (int i = 0; i < unitCount; i++)
            {
                UnitVessels[i] = -1;
            }
        }

        /// <summary>
        /// Returns the first step at which a tombstone expires, so that the vessels must be compiled again, or -1 if
        /// there are no tombstones.
        /// </summary>
        public int TombstoneExpiry(int tombstoneSteps)
        {
            int oldest = int.MaxValue;
            for (int i = 0; i < PartTombstones.Length; i++)
            {
                oldest = PartTombstones[i] >= 0 ? math.min(oldest, PartTombstones[i]) : oldest;
            }

            for (int i = 0; i < ClosureTombstones.Length; i++)
            {
                oldest = ClosureTombstones[i] >= 0 ? math.min(oldest, ClosureTombstones[i]) : oldest;
            }

            return oldest == int.MaxValue ? -1 : oldest + tombstoneSteps;
        }

        /// <summary>
        /// Resize edge and closure caches. Return true when a size change requires all entries to be recomputed.
        /// </summary>
        public bool SizeCaches()
        {
            int parts = PartStoreBodies.Length;
            int closures = ClosureUnits.Length;
            bool resized = EdgeCache.Length != parts || ClosureCache.Length != closures;
            EdgeCache.ResizeUninitialized(parts);
            EdgeCacheFrames.ResizeUninitialized(2 * parts);
            ClosureCache.ResizeUninitialized(closures);
            ClosureCacheFrames.ResizeUninitialized(2 * closures);
            EdgeCacheStamps.ResizeUninitialized(parts);
            ClosureCacheStamps.ResizeUninitialized(closures);
            return resized;
        }

        /// <summary>The key for two store bodies in <see cref="PairUnits"/>. The order of the bodies doesn't matter.</summary>
        public static ulong PairKey(int bodyA, int bodyB)
        {
            return ((ulong)(uint)math.min(bodyA, bodyB) << 32) | (uint)math.max(bodyA, bodyB);
        }

        /// <summary>Returns the vessel set for this step, with pointers to the lists in this build.</summary>
        public VesselSet Set()
        {
            if (StepVessels.Length == 0)
            {
                return default;
            }

            return new VesselSet
            {
                VesselCount = StepVessels.Length,
                Vessels = (VesselRecord*)StepVessels.GetUnsafePtr(),
                PartBodies = (int*)PartBodies.GetUnsafePtr(),
                Edges = (VesselEdge*)Edges.GetUnsafePtr(),
                Closures = (VesselClosure*)Closures.GetUnsafePtr(),
                EdgeLoads = (VesselEdgeLoad*)EdgeLoads.GetUnsafePtr(),
                ClosureLoads = (VesselEdgeLoad*)ClosureLoads.GetUnsafePtr(),
                EdgeLocked = (bool*)EdgeLocked.GetUnsafePtr(),
            };
        }

        /// <summary>
        /// Select vessels whose bodies are all built as awake moving bodies and resize step lists.
        /// Call on the main thread. Exclude sleeping vessels.
        /// </summary>
        public void ChooseStepVessels(WorldData* store, NativeArray<int> builtIndices, int dynamicCount)
        {
            // The job that finds islands reads the parts' bodies, so the job below fills them now. Only the solve waits
            // for the jobs that fill the edges and closures. Burst runs the loops over the parts much faster than C#.
            int parts = PartStoreBodies.Length;
            Active.ResizeUninitialized(Vessels.Length);
            PartBodies.ResizeUninitialized(parts);
            new StepVessels.ChooseStepVesselsJob
            {
                World = store,
                BuiltIndices = builtIndices,
                DynamicCount = dynamicCount,
                Vessels = Vessels.AsArray(),
                PartStoreBodies = PartStoreBodies.AsArray(),
                Active = Active.AsArray(),
                PartBodies = PartBodies.AsArray(),
                StepVessels = StepVessels,
            }.Run();

            Edges.ResizeUninitialized(parts);
            EdgeLoads.ResizeUninitialized(parts);
            EdgeLocked.ResizeUninitialized(parts);
            Closures.ResizeUninitialized(ClosureUnits.Length);
            ClosureLoads.ResizeUninitialized(ClosureUnits.Length);
        }

        /// <summary>
        /// Schedule clearing of vessel loads and lock flags. The solve must wait for the returned handle;
        /// vessels not solved this step retain zero loads.
        /// </summary>
        public JobHandle ScheduleClearLoads(JobHandle dependency)
        {
            return new VesselLoads.ClearVesselLoadsJob
            {
                EdgeLoads = EdgeLoads.AsArray(),
                ClosureLoads = ClosureLoads.AsArray(),
                EdgeLocked = EdgeLocked.AsArray(),
            }.Schedule(dependency);
        }
    }
}
