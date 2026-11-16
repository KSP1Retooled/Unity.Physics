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

        private static readonly SharedStatic<bool> Shared = SharedStatic<bool>.GetOrCreate<Key>();

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
    /// changes ( <see cref="JointTopology"/> ). Parts are listed with parents before children. Each part has an edge
    /// connecting it to its parent, and joints that close loops are listed as closures. For every step, the backend
    /// recalculates the edge parameters from the stored joints and bodies. Changes to a drive, anchor or mass therefore
    /// affect the solver in that same step.
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
                UnitVessels = new NativeList<int>(Allocator.Persistent),
                BodyVessels = new NativeList<int>(Allocator.Persistent),
                PairUnits = new NativeParallelMultiHashMap<ulong, int>(64, Allocator.Persistent),
                PartBodyRecords = new NativeList<IntPtr>(Allocator.Persistent),
                PartJointRecords = new NativeList<IntPtr>(Allocator.Persistent),
                ClosureJointRecords = new NativeList<IntPtr>(Allocator.Persistent),
                Active = new NativeList<byte>(Allocator.Persistent),
                StepVessels = new NativeList<VesselRecord>(Allocator.Persistent),
                PartBodies = new NativeList<int>(Allocator.Persistent),
                Edges = new NativeList<VesselEdge>(Allocator.Persistent),
                Closures = new NativeList<VesselClosure>(Allocator.Persistent),
                EdgeLoads = new NativeList<VesselEdgeLoad>(Allocator.Persistent),
                ClosureLoads = new NativeList<VesselEdgeLoad>(Allocator.Persistent),
                EdgeLocked = new NativeList<bool>(Allocator.Persistent),
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
            UnitVessels.Dispose();
            BodyVessels.Dispose();
            PairUnits.Dispose();
            PartBodyRecords.Dispose();
            PartJointRecords.Dispose();
            ClosureJointRecords.Dispose();
            Active.Dispose();
            StepVessels.Dispose();
            PartBodies.Dispose();
            Edges.Dispose();
            Closures.Dispose();
            EdgeLoads.Dispose();
            ClosureLoads.Dispose();
            EdgeLocked.Dispose();
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
            PartBodyRecords.Clear();
            PartJointRecords.Clear();
            ClosureJointRecords.Clear();
            Active.Clear();
            StepVessels.Clear();
            BodyVessels.Clear();
            PairUnits.Clear();
            UnitVessels.ResizeUninitialized(unitCount);
            for (int i = 0; i < unitCount; i++)
            {
                UnitVessels[i] = -1;
            }
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
    /// The root of a vessel is the body with the lowest store index. Parts are listed in breadth-first order from the
    /// root, so every parent precedes its children. A vessel retains its Id if compilation finds the same root, parts,
    /// connecting joints and closures in the same order as before. Otherwise, it receives a new Id.
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

        public VesselBuild Build;

        public void Execute()
        {
            int bodyCount = (int)World->bodyCount;
            int unitCount = Units.Length;

            // Keep the results of the previous compilation until they have been compared with the newly compiled
            // vessels.
            var oldVessels = new NativeArray<VesselRecord>(Build.Vessels.AsArray(), Allocator.Temp);
            var oldBodies = new NativeArray<IntPtr>(Build.PartBodyRecords.AsArray(), Allocator.Temp);
            var oldJoints = new NativeArray<IntPtr>(Build.PartJointRecords.AsArray(), Allocator.Temp);
            var oldClosures = new NativeArray<IntPtr>(Build.ClosureJointRecords.AsArray(), Allocator.Temp);
            Build.Clear(unitCount);

            // Use Kruskal's algorithm on forest units between two moving bodies, processing them in store order as in
            // the forest. A unit connecting two trees becomes an edge; one between bodies already in the same tree
            // becomes a closure. The forest classifies bodies as moving even when they are not simulated. Reclassify
            // the units here to exclude those bodies.
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
                if (Roles[u] == JointForestRole.None)
                {
                    continue;
                }

                NativeStore.JointData* joint = World->joints[Units[u].Joint];
                if (!Moving(joint->body0) || !Moving(joint->body1))
                {
                    continue;
                }

                int a = Find(root, (int)joint->body0->index);
                int b = Find(root, (int)joint->body1->index);
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
            // store index. A body with no tree units does not form a vessel.
            var partOf = new NativeArray<int>(bodyCount, Allocator.Temp);
            var vesselOfRoot = new NativeArray<int>(bodyCount, Allocator.Temp);
            for (int i = 0; i < bodyCount; i++)
            {
                partOf[i] = -1;
                vesselOfRoot[i] = -1;
            }

            for (int body = 0; body < bodyCount; body++)
            {
                if (firstNeighbour[body + 1] == firstNeighbour[body] || partOf[body] >= 0)
                {
                    continue;
                }

                int vessel = Build.Vessels.Length;
                int firstPart = Build.PartStoreBodies.Length;
                vesselOfRoot[Find(root, body)] = vessel;
                AddPart(body, -1, -1, 0, firstPart, partOf);
                for (int next = firstPart; next < Build.PartStoreBodies.Length; next++)
                {
                    int parent = Build.PartStoreBodies[next];
                    for (int n = firstNeighbour[parent]; n < firstNeighbour[parent + 1]; n++)
                    {
                        int child = neighbourBody[n];
                        if (partOf[child] < 0)
                        {
                            NativeStore.JointData* joint = World->joints[Units[neighbourUnit[n]].Joint];
                            byte isBody0 = joint->body0 != null && (int)joint->body0->index == child ? (byte)1 : (byte)0;
                            AddPart(child, neighbourUnit[n], next - firstPart, isBody0, firstPart, partOf);
                            Build.UnitVessels[neighbourUnit[n]] = vessel;
                        }
                    }
                }

                Build.Vessels.Add(new VesselRecord
                {
                    FirstPart = firstPart,
                    PartCount = Build.PartStoreBodies.Length - firstPart,
                });
            }

            // Count the closures for each vessel, then place them in consecutive ranges grouped by vessel.
            int vesselCount = Build.Vessels.Length;
            var closureCount = new NativeArray<int>(vesselCount + 1, Allocator.Temp);
            for (int u = 0; u < unitCount; u++)
            {
                if (IsVesselUnit(u, JointForestRole.Closure, out int a, out _) && partOf[a] >= 0)
                {
                    closureCount[vesselOfRoot[Find(root, a)] + 1]++;
                }
            }

            for (int v = 0; v < vesselCount; v++)
            {
                closureCount[v + 1] += closureCount[v];
                VesselRecord record = Build.Vessels[v];
                record.FirstClosure = closureCount[v];
                record.ClosureCount = closureCount[v + 1] - closureCount[v];
                Build.Vessels[v] = record;
            }

            Build.BodyVessels.ResizeUninitialized(bodyCount);
            for (int body = 0; body < bodyCount; body++)
            {
                Build.BodyVessels[body] = -1;
            }

            for (int v = 0; v < vesselCount; v++)
            {
                VesselRecord record = Build.Vessels[v];
                for (int part = record.FirstPart; part < record.FirstPart + record.PartCount; part++)
                {
                    Build.BodyVessels[Build.PartStoreBodies[part]] = v;
                }
            }

            int closures = closureCount[vesselCount];
            Build.ClosureUnits.ResizeUninitialized(closures);
            Build.ClosurePartA.ResizeUninitialized(closures);
            Build.ClosurePartB.ResizeUninitialized(closures);
            Build.ClosureJointRecords.ResizeUninitialized(closures);
            for (int u = 0; u < unitCount; u++)
            {
                if (!IsVesselUnit(u, JointForestRole.Closure, out int a, out int b) || partOf[a] < 0)
                {
                    continue;
                }

                int vessel = vesselOfRoot[Find(root, a)];
                int slot = closureCount[vessel]++;
                int firstPart = Build.Vessels[vessel].FirstPart;
                Build.ClosureUnits[slot] = u;
                Build.ClosurePartA[slot] = partOf[a] - firstPart;
                Build.ClosurePartB[slot] = partOf[b] - firstPart;
                Build.ClosureJointRecords[slot] = (IntPtr)World->joints[Units[u].Joint];
                Build.UnitVessels[u] = vessel;
            }

            // List every edge and closure unit by its two bodies, so that DropIgnoredPairsJob can find them.
            Build.PairUnits.Capacity = math.max(Build.PairUnits.Capacity, Build.PartStoreBodies.Length + closures);
            for (int u = 0; u < unitCount; u++)
            {
                if (Build.UnitVessels[u] >= 0)
                {
                    NativeStore.JointData* joint = World->joints[Units[u].Joint];
                    Build.PairUnits.Add(VesselBuild.PairKey((int)joint->body0->index, (int)joint->body1->index), u);
                }
            }

            KeepOrRenewIds(oldVessels, oldBodies, oldJoints, oldClosures);
        }

        // The role assigned to each unit during this compilation: a vessel edge, a vessel closure, or neither.
        [NativeDisableUnsafePtrRestriction]
        private JointForestRole* kinds;

        /// <summary>
        /// Returns whether this compilation assigned the requested kind to the unit. Also returns the store indices of
        /// its bodies: a is the leading joint's body0, and b is its body1.
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
            if (body == null)
            {
                return false;
            }

            NativeStore.BodyFlags flags = StoreOps.Hot(body)->flags;
            return (flags & NativeStore.BodyFlags.Simulated) != 0
                && (flags & (NativeStore.BodyFlags.Static | NativeStore.BodyFlags.Kinematic)) == 0;
        }

        private void AddPart(int body, int unit, int parent, byte isBody0, int firstPart, NativeArray<int> partOf)
        {
            partOf[body] = Build.PartStoreBodies.Length;
            Build.PartStoreBodies.Add(body);
            Build.PartUnits.Add(unit);
            Build.PartParents.Add(parent);
            Build.PartIsBody0.Add(isBody0);
            Build.PartVesselFirst.Add(firstPart);
            Build.PartBodyRecords.Add((IntPtr)World->bodies[body].owner);
            Build.PartJointRecords.Add(unit >= 0 ? (IntPtr)World->joints[Units[unit].Joint] : IntPtr.Zero);
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
        /// Reuses the Id of a previously compiled vessel when it has the same root body, the same parts in the same
        /// order, the same connecting joints, and the same closures. Assigns a new Id to every other vessel.
        /// </summary>
        private void KeepOrRenewIds(NativeArray<VesselRecord> oldVessels, NativeArray<IntPtr> oldBodies, NativeArray<IntPtr> oldJoints,
            NativeArray<IntPtr> oldClosures)
        {
            var oldByRoot = new NativeHashMap<IntPtr, int>(math.max(1, oldVessels.Length), Allocator.Temp);
            for (int v = 0; v < oldVessels.Length; v++)
            {
                oldByRoot.TryAdd(oldBodies[oldVessels[v].FirstPart], v);
            }

            for (int v = 0; v < Build.Vessels.Length; v++)
            {
                VesselRecord vessel = Build.Vessels[v];
                if (oldByRoot.TryGetValue(Build.PartBodyRecords[vessel.FirstPart], out int old)
                    && Same(vessel, oldVessels[old], oldBodies, oldJoints, oldClosures))
                {
                    vessel.Id = oldVessels[old].Id;
                    vessel.StructureVersion = oldVessels[old].StructureVersion;
                }
                else
                {
                    vessel.Id = VesselPipeline.NewId();
                    vessel.StructureVersion = (uint)vessel.Id;
                }

                Build.Vessels[v] = vessel;
            }
        }

        private bool Same(in VesselRecord vessel, in VesselRecord old, NativeArray<IntPtr> oldBodies, NativeArray<IntPtr> oldJoints,
            NativeArray<IntPtr> oldClosures)
        {
            if (vessel.PartCount != old.PartCount || vessel.ClosureCount != old.ClosureCount)
            {
                return false;
            }

            for (int i = 0; i < vessel.PartCount; i++)
            {
                if (Build.PartBodyRecords[vessel.FirstPart + i] != oldBodies[old.FirstPart + i]
                    || Build.PartJointRecords[vessel.FirstPart + i] != oldJoints[old.FirstPart + i])
                {
                    return false;
                }
            }

            for (int i = 0; i < vessel.ClosureCount; i++)
            {
                if (Build.ClosureJointRecords[vessel.FirstClosure + i] != oldClosures[old.FirstClosure + i])
                {
                    return false;
                }
            }

            return true;
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
        // writes only those of the closures, and BuildJointsJob writes only the groups outside active vessels.
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

        public void Execute(int part)
        {
            int storeBody = PartStoreBodies[part];
            int unit = PartUnits[part];
            if (unit < 0)
            {
                Edges[part] = new VesselEdge { Parent = -1, State = VesselEdgeState.Yielding, MayLock = true };
                return;
            }

            if (Active[UnitVessels[unit]] != 0)
            {
                BuildJointsJob.PrepareVesselGroup(World, Units[unit], Groups, Members);
            }

            UnitParameters parameters = UnitParameters.Of(World, Units[unit], Groups, Members);
            NativeStore.BodyData* child = World->bodies + storeBody;
            NativeStore.BodyData* parent = World->bodies + PartStoreBodies[PartVesselFirst[part] + PartParents[part]];
            bool childIsBody0 = PartIsBody0[part] != 0;
            float3 childAnchor = childIsBody0 ? parameters.Anchor0 : parameters.Anchor1;
            float3 parentAnchor = childIsBody0 ? parameters.Anchor1 : parameters.Anchor0;
            quaternion childFrame = childIsBody0 ? parameters.Rotation0 : parameters.Rotation1;
            quaternion parentFrame = childIsBody0 ? parameters.Rotation1 : parameters.Rotation0;

            // Use the softer swing axis because the edge uses one spring for both swing axes. Use the damping from that
            // same axis.
            int swing = parameters.Stiffness.y <= parameters.Stiffness.z ? 1 : 2;
            Edges[part] = new VesselEdge
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

        public void Execute(int closure)
        {
            JointUnit unit = Units[ClosureUnits[closure]];
            if (Active[UnitVessels[ClosureUnits[closure]]] != 0)
            {
                BuildJointsJob.PrepareVesselGroup(World, unit, Groups, Members);
            }

            UnitParameters parameters = UnitParameters.Of(World, unit, Groups, Members);
            NativeStore.JointData* leader = World->joints[unit.Joint];
            NativeStore.BodyData* bodyA = StoreOps.Hot(leader->body0);
            NativeStore.BodyData* bodyB = StoreOps.Hot(leader->body1);
            int softest = 0;
            for (int axis = 1; axis < 3; axis++)
            {
                softest = parameters.Stiffness[axis] < parameters.Stiffness[softest] ? axis : softest;
            }

            Closures[closure] = new VesselClosure
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
