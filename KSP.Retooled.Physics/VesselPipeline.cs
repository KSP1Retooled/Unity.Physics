using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Controls persistent vessels. When enabled, the backend compiles part joints between moving bodies into vessels
    /// and passes them to the step through <see cref="SimulationStepInput.VesselSet"/> . The solver uses the vessel
    /// kernel's rows for those joints. The backend omits their Unity Physics constraints, but may still build joints
    /// without constraints to disable collisions. Enabling vessels also enables the forest path. The flag is stored in
    /// a SharedStatic so that Burst code can read it. Vessels are on by default. When the first world attaches, the
    /// backend reads RETOOLED_VESSEL: the value 0 or false turns vessels off, and any other value turns them on. The
    /// comparison with "true" and "false" ignores case.
    /// </summary>
    public static class VesselPipeline
    {
        private readonly struct Key { }

        private readonly struct IdKey { }

        private readonly struct MinimumPartsKey { }

        private static readonly SharedStatic<bool> Shared = SharedStatic<bool>.GetOrCreate<Key>();

        private static readonly SharedStatic<int> MinimumPartsShared = SharedStatic<int>.GetOrCreate<MinimumPartsKey>();

        /// <summary>The value of <see cref="MinimumParts"/> until something sets it.</summary>
        public const int DefaultMinimumParts = 8;

        /// <summary>
        /// The number of steps for which an edge or closure whose joint went away stays in its vessel as a tombstone.
        /// See <see cref="CompileVesselsJob"/>.
        /// </summary>
        public const int TombstoneSteps = 10;

        /// <summary>
        /// The smallest number of parts for which the backend compiles a vessel. The joints of a smaller craft stay Unity
        /// Physics joints, as the joints of loose bodies and debris do. A value below 1 restores the default. When the
        /// first world attaches, the backend reads RETOOLED_VESSEL_MIN_PARTS. A change to this value affects a vessel
        /// only when the vessels are compiled again.
        /// </summary>
        public static int MinimumParts
        {
            get => MinimumPartsShared.Data > 0 ? MinimumPartsShared.Data : DefaultMinimumParts;
            set => MinimumPartsShared.Data = value;
        }

        // The last vessel Id allocated. The counter is shared across the process, so vessels in different worlds also
        // receive distinct Ids.
        private static readonly SharedStatic<long> LastId = SharedStatic<long>.GetOrCreate<IdKey>();

        private static bool initialized;

        public static bool Enabled
        {
            get => Shared.Data;
            set => Shared.Data = value;
        }

        /// <summary>Allocates a vessel Id that has not been used before.</summary>
        internal static unsafe ulong NewId()
        {
            return (ulong)System.Threading.Interlocked.Increment(ref UnsafeUtility.AsRef<long>(LastId.UnsafeDataPointer));
        }

        /// <summary>Reads the environment once, on the main thread, before the first step.</summary>
        internal static void Initialize()
        {
            if (initialized)
            {
                return;
            }

            initialized = true;
            string minimum = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_MIN_PARTS");
            if (int.TryParse(minimum, out int parts))
            {
                MinimumParts = parts;
                UnityEngine.Debug.Log($"[VesselPipeline] RETOOLED_VESSEL_MIN_PARTS={minimum}: vessels have at least {MinimumParts} parts");
            }

            string fused = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_FUSED");
            Unity.Physics.Solver.VesselFused = fused != "0";
            string fusedParts = Environment.GetEnvironmentVariable("RETOOLED_VESSEL_FUSED_PARTS");
            if (int.TryParse(fusedParts, out int fusedMinimum) && fusedMinimum > 0)
            {
                Unity.Physics.Solver.FusedVesselParts = fusedMinimum;
                UnityEngine.Debug.Log($"[VesselPipeline] RETOOLED_VESSEL_FUSED_PARTS={fusedParts}: the fused job takes vessel islands of {fusedMinimum} parts or more");
            }

            string value = Environment.GetEnvironmentVariable("RETOOLED_VESSEL");
            if (string.IsNullOrEmpty(value))
            {
                Enabled = true;
                return;
            }

            Enabled = !(value == "0" || value.Equals("false", StringComparison.OrdinalIgnoreCase));
            UnityEngine.Debug.Log($"[VesselPipeline] RETOOLED_VESSEL={value}: vessels {(Enabled ? "on" : "off")}");
        }
    }

    /// <summary>
    /// Stores the vessels compiled from the store's joints and prepares the set passed to each step.
    /// <para>
    /// A vessel is a group of moving bodies connected by part joints. Vessels are compiled only when the joint topology
    /// changes ( <see cref="JointTopology"/> ) or when a tombstone expires. Parts are listed with parents before
    /// children. Each part has an edge connecting it to its parent, and joints that close loops are listed as closures.
    /// For every step, the fill jobs give the solver the edges and closures of the vessels that take part in the step.
    /// They reuse an edge or closure that they computed in an earlier step, and compute it again from the stored joints
    /// and bodies when the motion frame of one of its bodies changed or when the settings of one of its joints changed
    /// ( <see cref="JointSettings"/> ). Changes to a drive, anchor or mass therefore affect the solver in that same
    /// step.
    /// </para>
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
        /// For each part and for each closure, -1 when its joint exists, or the step at which its joint went away and it
        /// became a tombstone. A tombstone's unit is -1. See <see cref="CompileVesselsJob"/>.
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

        // The edges and closures that the fill jobs computed in an earlier step, with the motion frames of the two
        // bodies that each one used (the child's or part A's first, then the parent's or part B's), and the highest
        // settings stamp among the joints of its unit. The fill jobs reuse an entry while the units, both motion frames
        // and the stamp stay the same.
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
        /// Sizes the cached edges and closures to match the compiled parts and closures. Returns true if any cache list
        /// changed size, because then the fill jobs must compute every entry again.
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
        /// Selects the vessels that participate in this step and resizes the lists for the step. A vessel participates
        /// only if all of its bodies are built as moving bodies for this step. Before the build, connected sleeping
        /// bodies are woken through the joints, so a vessel's bodies are either all awake or all asleep. A sleeping
        /// vessel is excluded from the set. Its joints are built as Unity Physics joints without constraints, as they
        /// would be with vessels disabled. Runs on the main thread.
        /// </summary>
        public void ChooseStepVessels(NativeStore.WorldData* store, NativeArray<int> builtIndices, int dynamicCount)
        {
            const NativeStore.BodyFlags held = NativeStore.BodyFlags.Sleeping | NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic;
            StepVessels.Clear();
            Active.ResizeUninitialized(Vessels.Length);
            for (int v = 0; v < Vessels.Length; v++)
            {
                VesselRecord vessel = Vessels[v];
                bool takesPart = true;
                for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount && takesPart; part++)
                {
                    int storeBody = PartStoreBodies[part];
                    int built = builtIndices[storeBody];
                    takesPart = built >= 0 && built < dynamicCount && (store->bodies[storeBody].flags & held) == 0;
                }

                Active[v] = takesPart ? (byte)1 : (byte)0;
                if (takesPart)
                {
                    StepVessels.Add(vessel);
                }
            }

            int parts = PartStoreBodies.Length;
            PartBodies.ResizeUninitialized(parts);

            // The job that finds islands reads the parts' bodies, so fill them here. Only the solve waits for the jobs
            // that fill the edges and closures.
            for (int part = 0; part < parts; part++)
            {
                PartBodies[part] = builtIndices[PartStoreBodies[part]];
            }

            Edges.ResizeUninitialized(parts);
            EdgeLoads.ResizeUninitialized(parts);
            EdgeLocked.ResizeUninitialized(parts);
            Closures.ResizeUninitialized(ClosureUnits.Length);
            ClosureLoads.ResizeUninitialized(ClosureUnits.Length);

            // The solver writes loads for the vessels that it solves. Clear the arrays so that loads for all other
            // vessels are zero.
            UnsafeUtility.MemClear(EdgeLoads.GetUnsafePtr(), (long)parts * sizeof(VesselEdgeLoad));
            UnsafeUtility.MemClear(ClosureLoads.GetUnsafePtr(), (long)ClosureLoads.Length * sizeof(VesselEdgeLoad));
            UnsafeUtility.MemClear(EdgeLocked.GetUnsafePtr(), parts);
        }
    }

    /// <summary>
    /// Compiles vessels from the joint units found by <see cref="GroupJointsJob"/> . Each vessel is a connected group
    /// of forest units between moving bodies. Tree units become edges, and units that close loops become closures.
    /// Units connected to the world, a static body or a kinematic body remain Unity Physics joints. Joints that cannot
    /// be merged also remain Unity Physics joints; examples include hinges, servos with targets and wheels.
    /// <para>
    /// Before it compiles new vessels, the job tries to keep each vessel of the previous compilation as it was. A vessel
    /// is kept when all of its bodies still exist and move, and every forest unit that touches one of its bodies joins
    /// the two bodies of one of its edges or closures. The job knows a body by its record's address and its serial
    /// number. The store reuses the memory of a destroyed body for the next body that it makes, so without the serial
    /// number a vessel could be kept on the new bodies of a craft that was built again in the same memory. When no unit joins the two bodies of an edge or closure any more,
    /// because the game removed the joint or the joint broke, the edge or closure becomes a tombstone. A tombstone stays
    /// in its place in the vessel but has no constraints: an edge is marked broken, and a closure has no stiffness.
    /// When a joint between the same two bodies appears within <see cref="TombstoneSteps"/> steps, it takes the
    /// tombstone's place again. A kept vessel keeps its Id, so a strut that the game destroys and creates again doesn't
    /// make the solver start the vessel over. The vessel's structure version changes when one of its edges or closures
    /// becomes a tombstone or comes back. After <see cref="TombstoneSteps"/> steps, the vessel is compiled anew without
    /// the tombstone.
    /// </para>
    /// <para>
    /// A tombstone is meant for a joint that the game destroys and creates again, as it does with autostruts. A joint
    /// that still exists but can no longer be part of a vessel, such as a servo that was given a target, won't come back
    /// in that way. A vessel with such a joint is therefore not kept. It is compiled anew in the same step, and the
    /// joint becomes a Unity Physics joint between two vessels.
    /// </para>
    /// <para>
    /// Two units can join the same two bodies, for example when their joint frames differ. When the job keeps a vessel,
    /// each edge and closure first takes the unit that holds the joint that it had before, and only then any other unit
    /// between its bodies. An edge that takes a unit without the joint that it had before counts as a change to the
    /// vessel's structure, because the solver keeps a lock state for each edge.
    /// </para>
    /// <para>
    /// A new vessel has as its root the body with the lowest store index. Parts are listed in breadth-first order from
    /// the root, so every parent precedes its children. A new vessel with fewer than <see cref="MinimumParts"/> parts is
    /// not emitted: its joints stay Unity Physics joints, as the joints of loose bodies and debris do. A new vessel
    /// retains the Id of a vessel from the previous compilation if it has the same root, parts, connecting joints and
    /// closures in the same order. Otherwise, it receives a new Id.
    /// </para>
    /// </summary>
    [BurstCompile]
    internal unsafe struct CompileVesselsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<JointUnit> Units;

        [ReadOnly]
        public NativeArray<JointForestRole> Roles;

        [ReadOnly]
        public NativeArray<JointGroup> Groups;

        [ReadOnly]
        public NativeArray<JointGroupMember> Members;

        public VesselBuild Build;

        /// <summary>The number of the step that is being built. A new tombstone records it.</summary>
        public int Step;

        /// <summary>See <see cref="VesselPipeline.TombstoneSteps"/>.</summary>
        public int TombstoneSteps;

        /// <summary>See <see cref="VesselPipeline.MinimumParts"/>.</summary>
        public int MinimumParts;

        public void Execute()
        {
            new Compiler
            {
                World = World,
                Units = Units,
                Roles = Roles,
                Groups = Groups,
                Members = Members,
                Build = Build,
                Step = Step,
                TombstoneSteps = TombstoneSteps,
                MinimumParts = MinimumParts,
            }.Run();
        }

        /// <summary>The work of the job, with the temporary lists that it shares between its stages.</summary>
        private struct Compiler
        {
            public NativeStore.WorldData* World;
            public NativeArray<JointUnit> Units;
            public NativeArray<JointForestRole> Roles;
            public NativeArray<JointGroup> Groups;
            public NativeArray<JointGroupMember> Members;
            public VesselBuild Build;
            public int Step;
            public int TombstoneSteps;
            public int MinimumParts;

            // The results of the previous compilation, copied before the build is cleared.
            private NativeArray<VesselRecord> oldVessels;
            private NativeArray<IntPtr> oldBodies;
            private NativeArray<uint> oldBodySerials;
            private NativeArray<IntPtr> oldJoints;
            private NativeArray<IntPtr> oldClosures;
            private NativeArray<uint> oldJointSerials;
            private NativeArray<uint> oldClosureSerials;
            private NativeArray<int> oldParents;
            private NativeArray<byte> oldIsBody0;
            private NativeArray<int> oldPartTombstones;
            private NativeArray<int> oldClosurePartA;
            private NativeArray<int> oldClosurePartB;
            private NativeArray<int> oldClosureTombstones;

            // For each unit: whether it can be part of a vessel, and whether a kept vessel uses it.
            private NativeArray<bool> eligible;
            private NativeArray<bool> claimed;

            // For each store body, its part index in the new build, or -1.
            private NativeArray<int> partOf;

            // The records of all joints in the store.
            private NativeHashSet<IntPtr> liveJoints;

            public void Run()
            {
                int bodyCount = (int)World->bodyCount;
                int unitCount = Units.Length;

                // Keep the results of the previous compilation until they have been compared with the new vessels.
                oldVessels = new NativeArray<VesselRecord>(Build.Vessels.AsArray(), Allocator.Temp);
                oldBodies = new NativeArray<IntPtr>(Build.PartBodyRecords.AsArray(), Allocator.Temp);
                oldBodySerials = new NativeArray<uint>(Build.PartBodySerials.AsArray(), Allocator.Temp);
                oldJoints = new NativeArray<IntPtr>(Build.PartJointRecords.AsArray(), Allocator.Temp);
                oldClosures = new NativeArray<IntPtr>(Build.ClosureJointRecords.AsArray(), Allocator.Temp);
                oldJointSerials = new NativeArray<uint>(Build.PartJointSerials.AsArray(), Allocator.Temp);
                oldClosureSerials = new NativeArray<uint>(Build.ClosureJointSerials.AsArray(), Allocator.Temp);
                oldParents = new NativeArray<int>(Build.PartParents.AsArray(), Allocator.Temp);
                oldIsBody0 = new NativeArray<byte>(Build.PartIsBody0.AsArray(), Allocator.Temp);
                oldPartTombstones = new NativeArray<int>(Build.PartTombstones.AsArray(), Allocator.Temp);
                oldClosurePartA = new NativeArray<int>(Build.ClosurePartA.AsArray(), Allocator.Temp);
                oldClosurePartB = new NativeArray<int>(Build.ClosurePartB.AsArray(), Allocator.Temp);
                oldClosureTombstones = new NativeArray<int>(Build.ClosureTombstones.AsArray(), Allocator.Temp);
                Build.Clear(unitCount);

                // A unit can be part of a vessel if it is in the forest and both of its bodies move.
                eligible = new NativeArray<bool>(unitCount, Allocator.Temp);
                claimed = new NativeArray<bool>(unitCount, Allocator.Temp);
                var unitsByBodies = new NativeParallelMultiHashMap<ulong, int>(math.max(1, unitCount), Allocator.Temp);
                for (int u = 0; u < unitCount; u++)
                {
                    NativeStore.JointData* joint = World->joints[Units[u].Joint];
                    eligible[u] = Roles[u] != JointForestRole.None && Moving(joint->body0) && Moving(joint->body1);
                    if (eligible[u])
                    {
                        unitsByBodies.Add(VesselBuild.PairKey((int)joint->body0->index, (int)joint->body1->index), u);
                    }
                }

                partOf = new NativeArray<int>(bodyCount, Allocator.Temp);
                for (int i = 0; i < bodyCount; i++)
                {
                    partOf[i] = -1;
                }

                liveJoints = new NativeHashSet<IntPtr>(math.max(1, (int)World->jointCount), Allocator.Temp);
                for (int i = 0; i < World->jointCount; i++)
                {
                    liveJoints.Add((IntPtr)World->joints[i]);
                }

                KeepVessels(unitsByBodies);
                int firstNew = Build.Vessels.Length;
                int firstNewClosure = Build.ClosureUnits.Length;
                CompileNewVessels(firstNew, firstNewClosure);
                KeepOrRenewIds(firstNew);

                Build.BodyVessels.ResizeUninitialized(bodyCount);
                for (int body = 0; body < bodyCount; body++)
                {
                    Build.BodyVessels[body] = -1;
                }

                for (int v = 0; v < Build.Vessels.Length; v++)
                {
                    VesselRecord record = Build.Vessels[v];
                    for (int part = record.FirstPart; part < record.FirstPart + record.PartCount; part++)
                    {
                        Build.BodyVessels[Build.PartStoreBodies[part]] = v;
                    }
                }

                // List every edge and closure unit by its two bodies, so that DropIgnoredPairsJob can find them.
                Build.PairUnits.Capacity = math.max(Build.PairUnits.Capacity, Build.PartStoreBodies.Length + Build.ClosureUnits.Length);
                for (int u = 0; u < unitCount; u++)
                {
                    if (Build.UnitVessels[u] >= 0)
                    {
                        NativeStore.JointData* joint = World->joints[Units[u].Joint];
                        Build.PairUnits.Add(VesselBuild.PairKey((int)joint->body0->index, (int)joint->body1->index), u);
                    }
                }
            }

            /// <summary>
            /// Keeps the vessels of the previous compilation that can be kept, in their previous order, and adds them to the
            /// build. See the summary of this job.
            /// </summary>
            private void KeepVessels(NativeParallelMultiHashMap<ulong, int> unitsByBodies)
            {
                int bodyCount = (int)World->bodyCount;
                var bodyOfRecord = new NativeHashMap<IntPtr, int>(math.max(1, bodyCount), Allocator.Temp);
                for (int i = 0; i < bodyCount; i++)
                {
                    bodyOfRecord.TryAdd((IntPtr)World->bodies[i].owner, i);
                }

                // For each store body, the old vessel that would keep it, or -1.
                var keptBy = new NativeArray<int>(bodyCount, Allocator.Temp);
                for (int i = 0; i < bodyCount; i++)
                {
                    keptBy[i] = -1;
                }

                var kept = new NativeArray<bool>(oldVessels.Length, Allocator.Temp);
                var partBody = new NativeArray<int>(oldBodies.Length, Allocator.Temp);
                var partUnit = new NativeArray<int>(oldBodies.Length, Allocator.Temp);
                var closureUnit = new NativeArray<int>(oldClosures.Length, Allocator.Temp);
                for (int v = 0; v < oldVessels.Length; v++)
                {
                    kept[v] = TryKeep(v, unitsByBodies, bodyOfRecord, keptBy, partBody, partUnit, closureUnit);
                    if (!kept[v])
                    {
                        Release(v, keptBy, partBody, partUnit, closureUnit);
                    }
                }

                // A unit that joins a kept vessel's bodies but takes no place in it, or that joins the vessel to another
                // body, changes the vessel's structure. Such a vessel is compiled anew.
                for (int u = 0; u < Units.Length; u++)
                {
                    if (!eligible[u] || claimed[u])
                    {
                        continue;
                    }

                    NativeStore.JointData* joint = World->joints[Units[u].Joint];
                    int first = keptBy[(int)joint->body0->index];
                    int second = keptBy[(int)joint->body1->index];
                    if (first >= 0)
                    {
                        kept[first] = false;
                        Release(first, keptBy, partBody, partUnit, closureUnit);
                    }

                    if (second >= 0)
                    {
                        kept[second] = false;
                        Release(second, keptBy, partBody, partUnit, closureUnit);
                    }
                }

                for (int v = 0; v < oldVessels.Length; v++)
                {
                    if (kept[v])
                    {
                        AddKeptVessel(oldVessels[v], partBody, partUnit, closureUnit);
                    }
                }
            }

            /// <summary>
            /// Finds the current bodies and units of an old vessel and claims them. Returns false if the vessel can't be
            /// kept.
            /// </summary>
            private bool TryKeep(int v, NativeParallelMultiHashMap<ulong, int> unitsByBodies, NativeHashMap<IntPtr, int> bodyOfRecord,
                NativeArray<int> keptBy, NativeArray<int> partBody, NativeArray<int> partUnit, NativeArray<int> closureUnit)
            {
                VesselRecord vessel = oldVessels[v];
                for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    partBody[part] = -1;
                    partUnit[part] = -1;
                }

                for (int c = vessel.FirstClosure; c < vessel.FirstClosure + vessel.ClosureCount; c++)
                {
                    closureUnit[c] = -1;
                }

                for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    if (!bodyOfRecord.TryGetValue(oldBodies[part], out int body) || World->bodies[body].owner->serial != oldBodySerials[part]
                        || keptBy[body] >= 0 || !Moving(World->bodies + body))
                    {
                        return false;
                    }

                    partBody[part] = body;
                    keptBy[body] = v;
                }

                // First give every edge and closure the unit that still holds its old joint, so that no other edge or
                // closure between the same two bodies takes that unit.
                for (int part = vessel.FirstPart + 1; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    int parent = vessel.FirstPart + oldParents[part];
                    partUnit[part] = Claim(unitsByBodies, partBody[part], partBody[parent], oldJoints[part], oldJointSerials[part]);
                }

                for (int c = vessel.FirstClosure; c < vessel.FirstClosure + vessel.ClosureCount; c++)
                {
                    int a = partBody[vessel.FirstPart + oldClosurePartA[c]];
                    int b = partBody[vessel.FirstPart + oldClosurePartB[c]];
                    closureUnit[c] = Claim(unitsByBodies, a, b, oldClosures[c], oldClosureSerials[c]);
                }

                // Then give the others any unit between their bodies, or make them tombstones.
                for (int part = vessel.FirstPart + 1; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    if (partUnit[part] >= 0)
                    {
                        continue;
                    }

                    int parent = vessel.FirstPart + oldParents[part];
                    partUnit[part] = Claim(unitsByBodies, partBody[part], partBody[parent], IntPtr.Zero, 0);
                    if (partUnit[part] < 0
                        && (Expired(oldPartTombstones[part]) || StillJoins(oldJoints[part], oldJointSerials[part], partBody[part], partBody[parent])))
                    {
                        return false;
                    }
                }

                for (int c = vessel.FirstClosure; c < vessel.FirstClosure + vessel.ClosureCount; c++)
                {
                    if (closureUnit[c] >= 0)
                    {
                        continue;
                    }

                    int a = partBody[vessel.FirstPart + oldClosurePartA[c]];
                    int b = partBody[vessel.FirstPart + oldClosurePartB[c]];
                    closureUnit[c] = Claim(unitsByBodies, a, b, IntPtr.Zero, 0);
                    if (closureUnit[c] < 0 && (Expired(oldClosureTombstones[c]) || StillJoins(oldClosures[c], oldClosureSerials[c], a, b)))
                    {
                        return false;
                    }
                }

                return true;
            }

            /// <summary>
            /// Returns true if a joint from the previous compilation still exists, still joins the same two bodies and is
            /// not broken. Such a joint lost its unit because it can no longer be part of a vessel, not because it went
            /// away, so its edge or closure must not become a tombstone.
            /// </summary>
            private bool StillJoins(IntPtr record, uint serial, int bodyA, int bodyB)
            {
                if (record == IntPtr.Zero || !liveJoints.Contains(record))
                {
                    return false;
                }

                var joint = (NativeStore.JointData*)record;
                return joint->serial == serial && (joint->state & JointConstraints.Inactive) == 0 && joint->body0 != null && joint->body1 != null
                    && VesselBuild.PairKey((int)joint->body0->index, (int)joint->body1->index) == VesselBuild.PairKey(bodyA, bodyB);
            }

            /// <summary>Returns true if one of the unit's joints is the joint with the given record and serial number.</summary>
            private bool Holds(int unit, IntPtr record, uint serial)
            {
                JointUnit jointUnit = Units[unit];
                if (jointUnit.Group < 0)
                {
                    return Is(World->joints[jointUnit.Joint], record, serial);
                }

                JointGroup group = Groups[jointUnit.Group];
                for (int i = 0; i < group.MemberCount; i++)
                {
                    if (Is(World->joints[Members[group.FirstMember + i].Joint], record, serial))
                    {
                        return true;
                    }
                }

                return false;
            }

            private static bool Is(NativeStore.JointData* joint, IntPtr record, uint serial)
            {
                return (IntPtr)joint == record && joint->serial == serial;
            }

            /// <summary>Returns whether a tombstone made at the given step (or one made now, for -1) has expired.</summary>
            private bool Expired(int stamp)
            {
                return Step - (stamp >= 0 ? stamp : Step) >= TombstoneSteps;
            }

            /// <summary>
            /// Claims an unclaimed eligible unit between two bodies, or returns -1 if there is none. If a record is given,
            /// claims only a unit that holds the joint with that record and serial number. The map lists the units between two bodies in no
            /// useful order.
            /// </summary>
            private int Claim(NativeParallelMultiHashMap<ulong, int> unitsByBodies, int bodyA, int bodyB, IntPtr record, uint serial)
            {
                if (unitsByBodies.TryGetFirstValue(VesselBuild.PairKey(bodyA, bodyB), out int unit, out NativeParallelMultiHashMapIterator<ulong> iterator))
                {
                    do
                    {
                        if (!claimed[unit] && (record == IntPtr.Zero || Holds(unit, record, serial)))
                        {
                            claimed[unit] = true;
                            return unit;
                        }
                    }
                    while (unitsByBodies.TryGetNextValue(out unit, ref iterator));
                }

                return -1;
            }

            /// <summary>Gives back the bodies and units that an old vessel claimed.</summary>
            private void Release(int v, NativeArray<int> keptBy, NativeArray<int> partBody, NativeArray<int> partUnit, NativeArray<int> closureUnit)
            {
                VesselRecord vessel = oldVessels[v];
                for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    if (partBody[part] >= 0 && keptBy[partBody[part]] == v)
                    {
                        keptBy[partBody[part]] = -1;
                    }

                    if (partUnit[part] >= 0)
                    {
                        claimed[partUnit[part]] = false;
                        partUnit[part] = -1;
                    }
                }

                for (int c = vessel.FirstClosure; c < vessel.FirstClosure + vessel.ClosureCount; c++)
                {
                    if (closureUnit[c] >= 0)
                    {
                        claimed[closureUnit[c]] = false;
                        closureUnit[c] = -1;
                    }
                }
            }

            /// <summary>Adds a kept vessel to the build with its old layout and Id.</summary>
            private void AddKeptVessel(VesselRecord vessel, NativeArray<int> partBody, NativeArray<int> partUnit, NativeArray<int> closureUnit)
            {
                int index = Build.Vessels.Length;
                int firstPart = Build.PartStoreBodies.Length;
                bool changed = false;
                for (int k = 0; k < vessel.PartCount; k++)
                {
                    int old = vessel.FirstPart + k;
                    int unit = partUnit[old];
                    byte isBody0 = oldIsBody0[old];
                    int tombstone = -1;
                    if (unit >= 0)
                    {
                        NativeStore.JointData* joint = World->joints[Units[unit].Joint];
                        isBody0 = (int)joint->body0->index == partBody[old] ? (byte)1 : (byte)0;
                        Build.UnitVessels[unit] = index;

                        // A different joint between the same two bodies took the edge. The solver must not carry the
                        // old joint's lock state over to it. The old joint of a tombstone is no joint, and that case
                        // is counted below.
                        changed |= oldJoints[old] != IntPtr.Zero && !Holds(unit, oldJoints[old], oldJointSerials[old]);
                    }
                    else if (k > 0)
                    {
                        tombstone = oldPartTombstones[old] >= 0 ? oldPartTombstones[old] : Step;
                    }

                    changed |= k > 0 && (tombstone >= 0) != (oldPartTombstones[old] >= 0);
                    AddPart(partBody[old], unit, oldParents[old], isBody0, firstPart, tombstone);
                }

                int firstClosure = Build.ClosureUnits.Length;
                for (int c = vessel.FirstClosure; c < vessel.FirstClosure + vessel.ClosureCount; c++)
                {
                    int unit = closureUnit[c];
                    int a = oldClosurePartA[c];
                    int b = oldClosurePartB[c];
                    int tombstone = -1;
                    IntPtr record = IntPtr.Zero;
                    if (unit >= 0)
                    {
                        NativeStore.JointData* joint = World->joints[Units[unit].Joint];
                        a = partOf[(int)joint->body0->index] - firstPart;
                        b = partOf[(int)joint->body1->index] - firstPart;
                        record = (IntPtr)joint;
                        Build.UnitVessels[unit] = index;
                    }
                    else
                    {
                        tombstone = oldClosureTombstones[c] >= 0 ? oldClosureTombstones[c] : Step;
                    }

                    changed |= (tombstone >= 0) != (oldClosureTombstones[c] >= 0) || a != oldClosurePartA[c];
                    AddClosure(unit, a, b, record, tombstone);
                }

                vessel.FirstPart = firstPart;
                vessel.FirstClosure = firstClosure;
                if (changed)
                {
                    vessel.StructureVersion++;
                }

                Build.Vessels.Add(vessel);
            }

            /// <summary>
            /// Compiles new vessels from the units and bodies that no kept vessel uses. Uses Kruskal's algorithm on the
            /// eligible units, processing them in store order as in the forest. A unit connecting two trees becomes an
            /// edge; one between bodies already in the same tree becomes a closure.
            /// </summary>
            private void CompileNewVessels(int firstNew, int firstNewClosure)
            {
                int bodyCount = (int)World->bodyCount;
                int unitCount = Units.Length;
                var kindList = new NativeArray<JointForestRole>(unitCount, Allocator.Temp);
                kinds = (JointForestRole*)kindList.GetUnsafePtr();
                var root = new NativeArray<int>(bodyCount, Allocator.Temp);
                var degree = new NativeArray<int>(bodyCount + 1, Allocator.Temp);
                for (int i = 0; i < bodyCount; i++)
                {
                    root[i] = i;
                }

                for (int u = 0; u < unitCount; u++)
                {
                    kinds[u] = JointForestRole.None;
                    if (!eligible[u] || claimed[u])
                    {
                        continue;
                    }

                    NativeStore.JointData* joint = World->joints[Units[u].Joint];
                    int a = (int)joint->body0->index;
                    int b = (int)joint->body1->index;
                    if (partOf[a] >= 0 || partOf[b] >= 0)
                    {
                        continue;
                    }

                    a = Find(root, a);
                    b = Find(root, b);
                    if (a == b)
                    {
                        kinds[u] = JointForestRole.Closure;
                        continue;
                    }

                    root[a] = b;
                    kinds[u] = JointForestRole.Tree;
                    degree[(int)joint->body0->index]++;
                    degree[(int)joint->body1->index]++;
                }

                // Store the neighbours connected to each body by tree units in a single list. Record the start index for
                // each body.
                var firstNeighbour = new NativeArray<int>(bodyCount + 1, Allocator.Temp);
                for (int i = 0; i < bodyCount; i++)
                {
                    firstNeighbour[i + 1] = firstNeighbour[i] + degree[i];
                    degree[i] = firstNeighbour[i];
                }

                var neighbourUnit = new NativeArray<int>(firstNeighbour[bodyCount], Allocator.Temp);
                var neighbourBody = new NativeArray<int>(firstNeighbour[bodyCount], Allocator.Temp);
                for (int u = 0; u < unitCount; u++)
                {
                    if (!IsVesselUnit(u, JointForestRole.Tree, out int a, out int b))
                    {
                        continue;
                    }

                    neighbourUnit[degree[a]] = u;
                    neighbourBody[degree[a]++] = b;
                    neighbourUnit[degree[b]] = u;
                    neighbourBody[degree[b]++] = a;
                }

                // Traverse each connected component in breadth-first order, starting with the body that has the lowest
                // store index. A body with no tree units does not form a vessel, and neither does a component with fewer
                // parts than the minimum.
                var vesselOfRoot = new NativeArray<int>(bodyCount, Allocator.Temp);
                var visited = new NativeArray<bool>(bodyCount, Allocator.Temp);
                var order = new NativeList<int>(Allocator.Temp);
                var orderUnit = new NativeList<int>(Allocator.Temp);
                var orderParent = new NativeList<int>(Allocator.Temp);
                for (int i = 0; i < bodyCount; i++)
                {
                    vesselOfRoot[i] = -1;
                }

                for (int body = 0; body < bodyCount; body++)
                {
                    if (firstNeighbour[body + 1] == firstNeighbour[body] || visited[body] || partOf[body] >= 0)
                    {
                        continue;
                    }

                    order.Clear();
                    orderUnit.Clear();
                    orderParent.Clear();
                    order.Add(body);
                    orderUnit.Add(-1);
                    orderParent.Add(-1);
                    visited[body] = true;
                    for (int next = 0; next < order.Length; next++)
                    {
                        int parent = order[next];
                        for (int n = firstNeighbour[parent]; n < firstNeighbour[parent + 1]; n++)
                        {
                            int child = neighbourBody[n];
                            if (!visited[child])
                            {
                                visited[child] = true;
                                order.Add(child);
                                orderUnit.Add(neighbourUnit[n]);
                                orderParent.Add(next);
                            }
                        }
                    }

                    if (order.Length < MinimumParts)
                    {
                        continue;
                    }

                    int vessel = Build.Vessels.Length;
                    int firstPart = Build.PartStoreBodies.Length;
                    vesselOfRoot[Find(root, body)] = vessel;
                    for (int k = 0; k < order.Length; k++)
                    {
                        byte isBody0 = 0;
                        if (orderUnit[k] >= 0)
                        {
                            NativeStore.JointData* joint = World->joints[Units[orderUnit[k]].Joint];
                            isBody0 = joint->body0 != null && (int)joint->body0->index == order[k] ? (byte)1 : (byte)0;
                            Build.UnitVessels[orderUnit[k]] = vessel;
                        }

                        AddPart(order[k], orderUnit[k], orderParent[k], isBody0, firstPart, -1);
                    }

                    Build.Vessels.Add(new VesselRecord
                    {
                        FirstPart = firstPart,
                        PartCount = Build.PartStoreBodies.Length - firstPart,
                    });
                }

                // Count the closures for each new vessel, then place them in consecutive ranges grouped by vessel, after the
                // closures of the kept vessels.
                int newCount = Build.Vessels.Length - firstNew;
                var closureCount = new NativeArray<int>(newCount + 1, Allocator.Temp);
                for (int u = 0; u < unitCount; u++)
                {
                    if (IsVesselUnit(u, JointForestRole.Closure, out int a, out _) && partOf[a] >= 0)
                    {
                        closureCount[vesselOfRoot[Find(root, a)] - firstNew + 1]++;
                    }
                }

                for (int v = 0; v < newCount; v++)
                {
                    closureCount[v + 1] += closureCount[v];
                    VesselRecord record = Build.Vessels[firstNew + v];
                    record.FirstClosure = firstNewClosure + closureCount[v];
                    record.ClosureCount = closureCount[v + 1] - closureCount[v];
                    Build.Vessels[firstNew + v] = record;
                }

                int closures = firstNewClosure + closureCount[newCount];
                Build.ClosureUnits.ResizeUninitialized(closures);
                Build.ClosurePartA.ResizeUninitialized(closures);
                Build.ClosurePartB.ResizeUninitialized(closures);
                Build.ClosureJointRecords.ResizeUninitialized(closures);
                Build.ClosureJointSerials.ResizeUninitialized(closures);
                Build.ClosureTombstones.ResizeUninitialized(closures);
                for (int u = 0; u < unitCount; u++)
                {
                    if (!IsVesselUnit(u, JointForestRole.Closure, out int a, out int b) || partOf[a] < 0)
                    {
                        continue;
                    }

                    int vessel = vesselOfRoot[Find(root, a)];
                    int slot = firstNewClosure + closureCount[vessel - firstNew]++;
                    int firstPart = Build.Vessels[vessel].FirstPart;
                    Build.ClosureUnits[slot] = u;
                    Build.ClosurePartA[slot] = partOf[a] - firstPart;
                    Build.ClosurePartB[slot] = partOf[b] - firstPart;
                    Build.ClosureJointRecords[slot] = (IntPtr)World->joints[Units[u].Joint];
                    Build.ClosureJointSerials[slot] = World->joints[Units[u].Joint]->serial;
                    Build.ClosureTombstones[slot] = -1;
                    Build.UnitVessels[u] = vessel;
                }
            }

            // The role assigned to each unit when the new vessels are compiled: a vessel edge, a vessel closure, or neither.
            private JointForestRole* kinds;

            /// <summary>
            /// Returns whether the compilation of new vessels assigned the requested kind to the unit. Also returns the
            /// store indices of its bodies: a is the leading joint's body0, and b is its body1.
            /// </summary>
            private bool IsVesselUnit(int unit, JointForestRole kind, out int a, out int b)
            {
                a = b = -1;
                if (kinds[unit] != kind)
                {
                    return false;
                }

                NativeStore.JointData* joint = World->joints[Units[unit].Joint];
                a = (int)joint->body0->index;
                b = (int)joint->body1->index;
                return true;
            }

            /// <summary>
            /// Returns whether a body is simulated and is neither static nor kinematic, so it can be built as a moving
            /// body.
            /// </summary>
            private static bool Moving(NativeStore.Body* body)
            {
                return body != null && Moving(StoreOps.Hot(body));
            }

            private static bool Moving(NativeStore.BodyData* body)
            {
                NativeStore.BodyFlags flags = body->flags;
                return (flags & NativeStore.BodyFlags.Simulated) != 0
                    && (flags & (NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic)) == 0;
            }

            /// <summary>
            /// Adds a part. The unit is -1 for a root and for an edge that is a tombstone; the tombstone is the step at
            /// which the edge became one, or -1.
            /// </summary>
            private void AddPart(int body, int unit, int parent, byte isBody0, int firstPart, int tombstone)
            {
                partOf[body] = Build.PartStoreBodies.Length;
                Build.PartStoreBodies.Add(body);
                Build.PartUnits.Add(unit);
                Build.PartParents.Add(parent);
                Build.PartIsBody0.Add(isBody0);
                Build.PartVesselFirst.Add(firstPart);
                Build.PartTombstones.Add(tombstone);
                Build.PartBodyRecords.Add((IntPtr)World->bodies[body].owner);
                Build.PartBodySerials.Add(World->bodies[body].owner->serial);
                Build.PartJointRecords.Add(unit >= 0 ? (IntPtr)World->joints[Units[unit].Joint] : IntPtr.Zero);
                Build.PartJointSerials.Add(unit >= 0 ? World->joints[Units[unit].Joint]->serial : 0u);
            }

            private void AddClosure(int unit, int partA, int partB, IntPtr record, int tombstone)
            {
                Build.ClosureUnits.Add(unit);
                Build.ClosurePartA.Add(partA);
                Build.ClosurePartB.Add(partB);
                Build.ClosureJointRecords.Add(record);
                Build.ClosureJointSerials.Add(record != IntPtr.Zero ? ((NativeStore.JointData*)record)->serial : 0u);
                Build.ClosureTombstones.Add(tombstone);
            }

            private static int Find(NativeArray<int> root, int v)
            {
                while (root[v] != v)
                {
                    root[v] = root[root[v]];
                    v = root[v];
                }

                return v;
            }

            /// <summary>
            /// Gives each new vessel the Id of a previously compiled vessel when it has the same root body, the same parts
            /// in the same order, the same connecting joints, and the same closures. Assigns a new Id to every other new
            /// vessel. A kept vessel already has its Id, so another vessel can't take that Id.
            /// </summary>
            private void KeepOrRenewIds(int firstNew)
            {
                var usedIds = new NativeHashSet<ulong>(math.max(1, firstNew), Allocator.Temp);
                for (int v = 0; v < firstNew; v++)
                {
                    usedIds.Add(Build.Vessels[v].Id);
                }

                var oldByRoot = new NativeHashMap<IntPtr, int>(math.max(1, oldVessels.Length), Allocator.Temp);
                for (int v = 0; v < oldVessels.Length; v++)
                {
                    oldByRoot.TryAdd(oldBodies[oldVessels[v].FirstPart], v);
                }

                for (int v = firstNew; v < Build.Vessels.Length; v++)
                {
                    VesselRecord vessel = Build.Vessels[v];
                    if (oldByRoot.TryGetValue(Build.PartBodyRecords[vessel.FirstPart], out int old)
                        && !usedIds.Contains(oldVessels[old].Id) && Same(vessel, oldVessels[old]))
                    {
                        vessel.Id = oldVessels[old].Id;
                        vessel.StructureVersion = oldVessels[old].StructureVersion;
                    }
                    else
                    {
                        vessel.Id = VesselPipeline.NewId();
                        vessel.StructureVersion = (uint)vessel.Id;
                    }

                    usedIds.Add(vessel.Id);
                    Build.Vessels[v] = vessel;
                }
            }

            private bool Same(in VesselRecord vessel, in VesselRecord old)
            {
                if (vessel.PartCount != old.PartCount || vessel.ClosureCount != old.ClosureCount)
                {
                    return false;
                }

                for (int i = 0; i < vessel.PartCount; i++)
                {
                    if (Build.PartBodyRecords[vessel.FirstPart + i] != oldBodies[old.FirstPart + i]
                        || Build.PartBodySerials[vessel.FirstPart + i] != oldBodySerials[old.FirstPart + i]
                        || Build.PartJointRecords[vessel.FirstPart + i] != oldJoints[old.FirstPart + i]
                        || Build.PartJointSerials[vessel.FirstPart + i] != oldJointSerials[old.FirstPart + i])
                    {
                        return false;
                    }
                }

                for (int i = 0; i < vessel.ClosureCount; i++)
                {
                    if (Build.ClosureJointRecords[vessel.FirstClosure + i] != oldClosures[old.FirstClosure + i]
                        || Build.ClosureJointSerials[vessel.FirstClosure + i] != oldClosureSerials[old.FirstClosure + i])
                    {
                        return false;
                    }
                }

                return true;
            }
        }
    }

    /// <summary>
    /// Parameters for a vessel edge or closure, calculated from the joints in its unit: anchors and joint frames in
    /// each body's local frame, spring stiffness and damping for each angular axis, and break limits. A group of joints
    /// between the same two bodies is represented as one joint at the mean of its members' anchors.
    /// </summary>
    internal unsafe struct UnitParameters
    {
        public float3 Anchor0;
        public float3 Anchor1;
        public quaternion Rotation0;
        public quaternion Rotation1;

        /// <summary>
        /// Spring stiffness and damping for each angular axis of the joint frame: twist, followed by the two swing
        /// axes.
        /// </summary>
        public double3 Stiffness;

        public double3 Damping;
        public double BreakForce;
        public double BreakTorque;

        /// <summary>
        /// The threshold for the sum of squared perpendicular offsets from an angular axis, in square metres. If that
        /// sum exceeds this value, the members' linear constraints prevent rotation about that axis. Otherwise, their
        /// anchors are treated as coincident for that axis. These linear constraints are always rigid.
        /// </summary>
        private const float SpreadSquared = 1e-8f;

        public static UnitParameters Of(NativeStore.WorldData* world, JointUnit unit, NativeArray<JointGroup> groups,
            NativeArray<JointGroupMember> members)
        {
            var parameters = new UnitParameters();
            NativeStore.JointData* leader = world->joints[unit.Joint];
            parameters.Rotation0 = leader->rotation0;
            parameters.Rotation1 = leader->rotation1;
            int count = unit.Group < 0 ? 1 : groups[unit.Group].MemberCount;
            int firstMember = unit.Group < 0 ? 0 : groups[unit.Group].FirstMember;
            for (int i = 0; i < count; i++)
            {
                NativeStore.JointData* joint = unit.Group < 0 ? leader : world->joints[members[firstMember + i].Joint];
                parameters.Anchor0 += joint->position0;
                parameters.Anchor1 += joint->position1;
                parameters.BreakForce += joint->breakForce;
                parameters.BreakTorque += joint->breakTorque;
            }

            parameters.Anchor0 /= count;
            parameters.Anchor1 /= count;

            // Treat locked axes as rigid. For a driven axis, sum the spring stiffness and damping from the members'
            // drives.
            bool3 hard = false;
            for (int i = 0; i < count; i++)
            {
                NativeStore.JointData* joint = unit.Group < 0 ? leader : world->joints[members[firstMember + i].Joint];
                NativeStore.JointAxis* wholeRotation = JointConstraints.WholeRotationDrive(joint);
                for (int axis = 0; axis < 3; axis++)
                {
                    NativeStore.JointAxis* drive = wholeRotation != null ? wholeRotation : BuildJointsJob.AngularDrive(joint, axis);
                    if (NativeStore.AxisOf(joint, NativeStore.AxisAngularX + axis)->dofLock == NativeStore.DofLocked)
                    {
                        hard[axis] = true;
                    }
                    else if (BuildJointsJob.HasAxisDrive(drive))
                    {
                        parameters.Stiffness[axis] += drive->driveSpring;
                        parameters.Damping[axis] += drive->driveDamper;
                    }
                }
            }

            // Members may have anchors at different points, and their linear constraints are rigid. The group therefore
            // cannot rotate about an axis unless all anchor offsets from the mean lie along that axis, within the
            // spread threshold.
            if (count > 1)
            {
                quaternion toJoint = math.conjugate(leader->rotation0);
                float3 spread = float3.zero;
                for (int i = 0; i < count; i++)
                {
                    float3 offset = math.rotate(toJoint, world->joints[members[firstMember + i].Joint]->position0 - parameters.Anchor0);
                    spread += math.lengthsq(offset) - offset * offset;
                }

                hard |= spread > SpreadSquared;
            }

            for (int axis = 0; axis < 3; axis++)
            {
                if (hard[axis])
                {
                    parameters.Stiffness[axis] = double.PositiveInfinity;
                }
            }

            return parameters;
        }

        /// <summary>
        /// Converts a point from the body's local frame to its motion frame, whose origin is the centre of mass and
        /// whose axes are the principal inertia axes.
        /// </summary>
        public static double3 InMotionFrame(NativeStore.BodyData* body, float3 point)
        {
            RigidTransform bodyFromMotion = body->mass->Transform;
            return math.rotate(math.conjugate(bodyFromMotion.rot), point - bodyFromMotion.pos);
        }

        /// <summary>Converts a rotation from the body's local frame to its motion frame.</summary>
        public static quaternion InMotionFrame(NativeStore.BodyData* body, quaternion rotation)
        {
            return math.mul(math.conjugate(body->mass->Transform.rot), rotation);
        }

        /// <summary>
        /// The rotation that transforms coordinates from part B's motion frame into part A's motion frame when the
        /// joint is at rest. At rest, the joint frames align in world space: A's orientation multiplied by A's joint
        /// frame equals B's orientation multiplied by B's joint frame.
        /// </summary>
        public static double4 Rest(NativeStore.BodyData* bodyA, quaternion frameA, NativeStore.BodyData* bodyB, quaternion frameB)
        {
            quaternion rest = math.mul(math.mul(InMotionFrame(bodyA, frameA), math.conjugate(frameB)), bodyB->mass->Transform.rot);
            return math.normalize(rest).value;
        }

        /// <summary>
        /// Returns true if two motion frames are the same bit for bit, so that values computed from one hold for the
        /// other.
        /// </summary>
        public static bool Same(RigidTransform a, RigidTransform b)
        {
            return UnsafeUtility.MemCmp(&a, &b, sizeof(RigidTransform)) == 0;
        }

        /// <summary>
        /// Returns the highest settings stamp among the joints of a unit. A command that changes a joint gives the joint
        /// a stamp that is higher than all earlier stamps, so the result changes whenever one of the unit's joints
        /// changes.
        /// </summary>
        public static uint Stamp(NativeStore.WorldData* world, JointUnit unit, NativeArray<JointGroup> groups,
            NativeArray<JointGroupMember> members)
        {
            if (unit.Group < 0)
            {
                return world->joints[unit.Joint]->settingsStamp;
            }

            // The leader is one of the group's members.
            JointGroup group = groups[unit.Group];
            uint stamp = 0;
            for (int i = 0; i < group.MemberCount; i++)
            {
                stamp = math.max(stamp, world->joints[members[group.FirstMember + i].Joint]->settingsStamp);
            }

            return stamp;
        }

        public static double3x3 Frame(NativeStore.BodyData* body, quaternion frame)
        {
            return new double3x3(new float3x3(InMotionFrame(body, frame)));
        }
    }

    /// <summary>
    /// Fills the body index and edge for each part in this step. Converts the anchors, frame and rest orientation from
    /// the stored joints into the parts' motion frames. The springs and break limits also come from the stored joints.
    /// </summary>
    [BurstCompile]
    internal unsafe struct FillVesselEdgesJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<JointUnit> Units;

        // This job runs at the same time as BuildJointsJob and FillVesselClosuresJob. All three jobs write groups, but
        // never the same group: this job writes only the groups of the edges in active vessels, FillVesselClosuresJob
        // writes only those of the closures, and BuildJointsJob writes only the groups outside active vessels. This job
        // therefore reads no group of an inactive vessel, because BuildJointsJob may be writing it.
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<JointGroup> Groups;

        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<JointGroupMember> Members;

        [ReadOnly]
        public NativeArray<int> UnitVessels;

        [ReadOnly]
        public NativeArray<byte> Active;

        [ReadOnly]
        public NativeArray<int> PartStoreBodies;

        [ReadOnly]
        public NativeArray<int> PartUnits;

        [ReadOnly]
        public NativeArray<int> PartParents;

        [ReadOnly]
        public NativeArray<byte> PartIsBody0;

        public NativeArray<VesselEdge> Edges;

        /// <summary>
        /// For each part, the index of its vessel's first part. Add this to a parent's relative index to find its index
        /// in the full list.
        /// </summary>
        [ReadOnly]
        public NativeArray<int> PartVesselFirst;

        /// <summary>
        /// True when the cached edges can't be reused, because the units or the compiled vessels changed since the
        /// cache was filled.
        /// </summary>
        public bool RefillAll;

        /// <summary>
        /// True when some joint's settings changed since the cache was filled. Each entry then compares the settings
        /// stamp of its unit with the stamp that it recorded.
        /// </summary>
        public bool SettingsChanged;

        public NativeArray<VesselEdge> EdgeCache;

        [NativeDisableParallelForRestriction]
        public NativeArray<RigidTransform> EdgeCacheFrames;

        public NativeArray<uint> EdgeCacheStamps;

        public void Execute(int part)
        {
            int storeBody = PartStoreBodies[part];
            int unit = PartUnits[part];
            if (unit < 0)
            {
                // A root, or an edge whose joint went away. Such an edge is a tombstone: the solver treats it as broken
                // and never locks it.
                bool root = PartParents[part] < 0;
                Edges[part] = new VesselEdge
                {
                    Parent = PartParents[part],
                    State = root ? VesselEdgeState.Yielding : VesselEdgeState.Broken,
                    MayLock = root,
                    Frame = double3x3.identity,
                    Rest = new double4(0.0, 0.0, 0.0, 1.0),
                };
                return;
            }

            if (Active[UnitVessels[unit]] == 0)
            {
                // The vessel takes no part in this step, so the solver doesn't read its edges. Clear the cached frames,
                // so that the edge is computed again when the vessel takes part again. The joint's settings may change
                // in the meantime, and this job doesn't record that change.
                Edges[part] = new VesselEdge
                {
                    Parent = PartParents[part],
                    State = VesselEdgeState.Broken,
                    Frame = double3x3.identity,
                    Rest = new double4(0.0, 0.0, 0.0, 1.0),
                };
                EdgeCacheFrames[2 * part] = default;
                EdgeCacheFrames[2 * part + 1] = default;
                return;
            }

            BuildJointsJob.PrepareVesselGroup(World, Units[unit], Groups, Members);
            NativeStore.BodyData* child = World->bodies + storeBody;
            NativeStore.BodyData* parent = World->bodies + PartStoreBodies[PartVesselFirst[part] + PartParents[part]];
            RigidTransform childMotion = child->mass->Transform;
            RigidTransform parentMotion = parent->mass->Transform;
            uint stamp = RefillAll || SettingsChanged ? UnitParameters.Stamp(World, Units[unit], Groups, Members) : EdgeCacheStamps[part];
            if (!RefillAll && stamp == EdgeCacheStamps[part] && UnitParameters.Same(EdgeCacheFrames[2 * part], childMotion)
                && UnitParameters.Same(EdgeCacheFrames[2 * part + 1], parentMotion))
            {
                Edges[part] = EdgeCache[part];
                return;
            }

            UnitParameters parameters = UnitParameters.Of(World, Units[unit], Groups, Members);
            bool childIsBody0 = PartIsBody0[part] != 0;
            float3 childAnchor = childIsBody0 ? parameters.Anchor0 : parameters.Anchor1;
            float3 parentAnchor = childIsBody0 ? parameters.Anchor1 : parameters.Anchor0;
            quaternion childFrame = childIsBody0 ? parameters.Rotation0 : parameters.Rotation1;
            quaternion parentFrame = childIsBody0 ? parameters.Rotation1 : parameters.Rotation0;

            // Use the softer swing axis because the edge uses one spring for both swing axes. Use the damping from that
            // same axis.
            int swing = parameters.Stiffness.y <= parameters.Stiffness.z ? 1 : 2;
            var edge = new VesselEdge
            {
                Parent = PartParents[part],
                State = VesselEdgeState.Yielding,
                MayLock = true,
                ParentAnchor = UnitParameters.InMotionFrame(parent, parentAnchor),
                ChildAnchor = UnitParameters.InMotionFrame(child, childAnchor),
                Frame = UnitParameters.Frame(parent, parentFrame),
                Rest = UnitParameters.Rest(parent, parentFrame, child, childFrame),
                TwistStiffness = parameters.Stiffness.x,
                SwingStiffness = parameters.Stiffness[swing],
                TwistDamping = parameters.Damping.x,
                SwingDamping = parameters.Damping[swing],
                BreakForce = parameters.BreakForce,
                BreakTorque = parameters.BreakTorque,
            };
            Edges[part] = edge;
            EdgeCache[part] = edge;
            EdgeCacheFrames[2 * part] = childMotion;
            EdgeCacheFrames[2 * part + 1] = parentMotion;
            EdgeCacheStamps[part] = stamp;
        }
    }

    /// <summary>
    /// Fills each closure for this step. Part A is the leading joint's body0, and part B is its body1. Linear
    /// constraints are rigid. For all angular constraints, use the stiffness and damping of the joint's softest angular
    /// axis.
    /// </summary>
    [BurstCompile]
    internal unsafe struct FillVesselClosuresJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<JointUnit> Units;

        // See the comment on FillVesselEdgesJob.Groups.
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<JointGroup> Groups;

        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<JointGroupMember> Members;

        [ReadOnly]
        public NativeArray<int> UnitVessels;

        [ReadOnly]
        public NativeArray<byte> Active;

        [ReadOnly]
        public NativeArray<int> ClosureUnits;

        [ReadOnly]
        public NativeArray<int> ClosurePartA;

        [ReadOnly]
        public NativeArray<int> ClosurePartB;

        public NativeArray<VesselClosure> Closures;

        /// <summary>See <see cref="FillVesselEdgesJob.RefillAll"/>.</summary>
        public bool RefillAll;

        /// <summary>See <see cref="FillVesselEdgesJob.SettingsChanged"/>.</summary>
        public bool SettingsChanged;

        public NativeArray<VesselClosure> ClosureCache;

        [NativeDisableParallelForRestriction]
        public NativeArray<RigidTransform> ClosureCacheFrames;

        public NativeArray<uint> ClosureCacheStamps;

        public void Execute(int closure)
        {
            if (ClosureUnits[closure] < 0)
            {
                // A closure whose joint went away is a tombstone. With no stiffness and no damping, it has no effect.
                Closures[closure] = new VesselClosure
                {
                    PartA = ClosurePartA[closure],
                    PartB = ClosurePartB[closure],
                    Frame = double3x3.identity,
                    Rest = new double4(0.0, 0.0, 0.0, 1.0),
                };
                return;
            }

            JointUnit unit = Units[ClosureUnits[closure]];
            if (Active[UnitVessels[ClosureUnits[closure]]] == 0)
            {
                // The vessel takes no part in this step. See the same case in FillVesselEdgesJob.
                Closures[closure] = new VesselClosure
                {
                    PartA = ClosurePartA[closure],
                    PartB = ClosurePartB[closure],
                    Frame = double3x3.identity,
                    Rest = new double4(0.0, 0.0, 0.0, 1.0),
                };
                ClosureCacheFrames[2 * closure] = default;
                ClosureCacheFrames[2 * closure + 1] = default;
                return;
            }

            BuildJointsJob.PrepareVesselGroup(World, unit, Groups, Members);
            NativeStore.JointData* leader = World->joints[unit.Joint];
            NativeStore.BodyData* bodyA = StoreOps.Hot(leader->body0);
            NativeStore.BodyData* bodyB = StoreOps.Hot(leader->body1);
            RigidTransform motionA = bodyA->mass->Transform;
            RigidTransform motionB = bodyB->mass->Transform;
            uint stamp = RefillAll || SettingsChanged ? UnitParameters.Stamp(World, unit, Groups, Members) : ClosureCacheStamps[closure];
            if (!RefillAll && stamp == ClosureCacheStamps[closure] && UnitParameters.Same(ClosureCacheFrames[2 * closure], motionA)
                && UnitParameters.Same(ClosureCacheFrames[2 * closure + 1], motionB))
            {
                Closures[closure] = ClosureCache[closure];
                return;
            }

            UnitParameters parameters = UnitParameters.Of(World, unit, Groups, Members);
            int softest = 0;
            for (int axis = 1; axis < 3; axis++)
            {
                softest = parameters.Stiffness[axis] < parameters.Stiffness[softest] ? axis : softest;
            }

            var filled = new VesselClosure
            {
                PartA = ClosurePartA[closure],
                PartB = ClosurePartB[closure],
                AnchorA = UnitParameters.InMotionFrame(bodyA, parameters.Anchor0),
                AnchorB = UnitParameters.InMotionFrame(bodyB, parameters.Anchor1),
                Frame = UnitParameters.Frame(bodyA, parameters.Rotation0),
                Rest = UnitParameters.Rest(bodyA, parameters.Rotation0, bodyB, parameters.Rotation1),
                LinearStiffness = double.PositiveInfinity,
                AngularStiffness = parameters.Stiffness[softest],
                AngularDamping = parameters.Damping[softest],
            };
            Closures[closure] = filled;
            ClosureCache[closure] = filled;
            ClosureCacheFrames[2 * closure] = motionA;
            ClosureCacheFrames[2 * closure + 1] = motionB;
            ClosureCacheStamps[closure] = stamp;
        }
    }

    /// <summary>
    /// Converts the vessel loads from the solver into loads for their joint units, then settles the loads of the
    /// units' joints. As with Unity Physics impulse events, the load of a unit is the force and torque acting on the
    /// leading joint's body0. Edge loads describe what the parent applied to the child. Closure loads describe what
    /// part A applied to part B. The load of a group is shared among its members as <see cref="ShareGroupLoadsJob"/>
    /// does. Each member's load starts at zero because the bodies of an active vessel are awake. Each joint's load is
    /// then settled as <see cref="SettleJointLoadsJob"/> does. The joints of active vessels have no impulse events, so
    /// this job doesn't need to wait for the jobs that read the events, and those jobs skip these joints.
    /// </summary>
    [BurstCompile]
    internal unsafe struct VesselLoadsJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<JointUnit> Units;

        [ReadOnly]
        public NativeArray<JointGroup> Groups;

        [ReadOnly]
        public NativeArray<JointGroupMember> Members;

        public float DeltaTime;

        /// <summary>
        /// For each part, the joint unit for its edge, or -1 for a vessel root. Closure units are stored separately in
        /// <see cref="ClosureUnits"/> .
        /// </summary>
        [ReadOnly]
        public NativeArray<int> PartUnits;

        [ReadOnly]
        public NativeArray<int> ClosureUnits;

        [ReadOnly]
        public NativeArray<byte> PartIsBody0;

        [ReadOnly]
        public NativeArray<int> UnitVessels;

        [ReadOnly]
        public NativeArray<byte> Active;

        [ReadOnly]
        public NativeArray<VesselEdgeLoad> EdgeLoads;

        [ReadOnly]
        public NativeArray<VesselEdgeLoad> ClosureLoads;

        // The jobs for the other joints write their loads at the same time. No joint is in both sets.
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<JointLoad> JointLoads;

        /// <summary>Processes part indices first, followed by closure indices.</summary>
        public void Execute(int index)
        {
            int parts = PartUnits.Length;
            int unit;
            VesselEdgeLoad load;
            float sign;
            if (index < parts)
            {
                unit = PartUnits[index];
                if (unit < 0)
                {
                    return;
                }

                // The load acts on the child. If the child is body1, reverse the sign to obtain the load on body0.
                load = EdgeLoads[index];
                sign = PartIsBody0[index] != 0 ? 1.0f : -1.0f;
            }
            else
            {
                unit = ClosureUnits[index - parts];
                if (unit < 0)
                {
                    return;
                }

                // The load acts on part B, which is body1. Reverse the sign to obtain the load on body0.
                load = ClosureLoads[index - parts];
                sign = -1.0f;
            }

            if (Active[UnitVessels[unit]] == 0)
            {
                return;
            }

            var jointLoad = new JointLoad { Force = sign * (float3)load.Force, Torque = sign * (float3)load.Torque };
            JointUnit jointUnit = Units[unit];
            if (jointUnit.Group < 0)
            {
                SettleJointLoadsJob.Settle(World->joints[jointUnit.Joint], ref jointLoad, DeltaTime);
                JointLoads[jointUnit.Joint] = jointLoad;
                return;
            }

            JointGroup group = Groups[jointUnit.Group];
            for (int i = 0; i < group.MemberCount; i++)
            {
                JointLoads[Members[group.FirstMember + i].Joint] = default;
            }

            ShareGroupLoadsJob.Share(World, group, jointLoad, Members, JointLoads);
            for (int i = 0; i < group.MemberCount; i++)
            {
                int joint = Members[group.FirstMember + i].Joint;
                JointLoad memberLoad = JointLoads[joint];
                SettleJointLoadsJob.Settle(World->joints[joint], ref memberLoad, DeltaTime);
                JointLoads[joint] = memberLoad;
            }
        }
    }

    /// <summary>
    /// Lists the joints and groups that are not in an active vessel. The jobs that read the impulse events and settle
    /// the loads handle only these joints and groups, and <see cref="VesselLoadsJob"/> handles the others.
    /// </summary>
    [BurstCompile]
    internal struct ListOtherJointsJob : IJob
    {
        [ReadOnly]
        public NativeArray<JointUnit> Units;

        [ReadOnly]
        public NativeArray<JointGroup> Groups;

        [ReadOnly]
        public NativeArray<JointGroupMember> Members;

        [ReadOnly]
        public NativeArray<int> UnitVessels;

        [ReadOnly]
        public NativeArray<byte> Active;

        public NativeList<int> OtherJoints;
        public NativeList<int> OtherGroups;

        public void Execute()
        {
            OtherJoints.Clear();
            OtherGroups.Clear();
            for (int u = 0; u < Units.Length; u++)
            {
                if (UnitVessels[u] >= 0 && Active[UnitVessels[u]] != 0)
                {
                    continue;
                }

                JointUnit unit = Units[u];
                if (unit.Group < 0)
                {
                    OtherJoints.Add(unit.Joint);
                    continue;
                }

                OtherGroups.Add(unit.Group);
                JointGroup group = Groups[unit.Group];
                for (int i = 0; i < group.MemberCount; i++)
                {
                    OtherJoints.Add(Members[group.FirstMember + i].Joint);
                }
            }
        }
    }
}
