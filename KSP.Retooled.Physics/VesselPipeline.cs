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
    /// The switch for persistent vessels. When it is on, the backend compiles the part joints between moving bodies into
    /// vessels and hands them to the step in <see cref="SimulationStepInput.VesselSet"/>. The solver then solves those
    /// joints with the vessel kernel's own rows, and the backend builds no Unity Physics joints for them. The switch also
    /// turns on the forest path. The step reads the switch from Burst, so it lives in a SharedStatic. The environment
    /// variable RETOOLED_VESSEL=1 turns it on when the first world attaches.
    /// </summary>
    public static class VesselPipeline
    {
        private readonly struct Key { }

        private readonly struct IdKey { }

        private static readonly SharedStatic<bool> Shared = SharedStatic<bool>.GetOrCreate<Key>();

        // The last vessel Id that was handed out. Ids are counted for the whole process, so no two vessels in any world
        // ever have the same one.
        private static readonly SharedStatic<long> LastId = SharedStatic<long>.GetOrCreate<IdKey>();

        private static bool initialized;

        public static bool Enabled
        {
            get => Shared.Data;
            set => Shared.Data = value;
        }

        /// <summary>A vessel Id that no vessel has had before.</summary>
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
            if (!string.IsNullOrEmpty(value))
            {
                Enabled = value == "1" || value.Equals("true", StringComparison.OrdinalIgnoreCase);
                UnityEngine.Debug.Log($"[VesselPipeline] RETOOLED_VESSEL={value}: vessels {(Enabled ? "on" : "off")}");
            }
        }
    }

    /// <summary>
    /// The vessels that the backend compiled from the store's joints, and the set that it hands to each step.
    /// <para>
    /// A vessel is a group of moving bodies that part joints connect. The compile runs only when the joint topology
    /// changes (<see cref="JointTopology"/>). It lists each vessel's parts in parent-first order, with the edge that joins
    /// each part to its parent, and it lists the joints that close loops as closures. Every step, the backend works out the
    /// edges' parameters again from the store's joints and bodies, so that a changed drive, anchor or mass reaches the
    /// solver in the same step.
    /// </para>
    /// </summary>
    internal unsafe struct VesselBuild : IDisposable
    {
        /// <summary>Every compiled vessel, in the order of their root bodies' store indices.</summary>
        public NativeList<VesselRecord> Vessels;

        /// <summary>For each part, the store index of its body.</summary>
        public NativeList<int> PartStoreBodies;

        /// <summary>For each part, the joint unit of the edge to its parent, or -1 for a vessel's root.</summary>
        public NativeList<int> PartUnits;

        /// <summary>For each part, its parent counted from the vessel's first part, or -1 for the root.</summary>
        public NativeList<int> PartParents;

        /// <summary>For each part, 1 when the part's body is the edge joint's body0, and 0 when it is body1.</summary>
        public NativeList<byte> PartIsBody0;

        /// <summary>For each part, the first part of its vessel.</summary>
        public NativeList<int> PartVesselFirst;

        /// <summary>For each closure, its joint unit and its two parts, counted from the vessel's first part.</summary>
        public NativeList<int> ClosureUnits;

        public NativeList<int> ClosurePartA;
        public NativeList<int> ClosurePartB;

        /// <summary>For each joint unit, the vessel that it belongs to, or -1.</summary>
        public NativeList<int> UnitVessels;

        // What the last compile found, as record addresses that don't move, so that the next compile can tell which
        // vessels are still the same and keep their Ids.
        public NativeList<IntPtr> PartBodyRecords;
        public NativeList<IntPtr> PartJointRecords;
        public NativeList<IntPtr> ClosureJointRecords;

        /// <summary>For each compiled vessel, 1 when the vessel takes part in this step.</summary>
        public NativeList<byte> Active;

        /// <summary>The vessels that take part in this step, which are the ones that the step's set lists.</summary>
        public NativeList<VesselRecord> StepVessels;

        // The step's set points at these. They have one entry for each compiled part or closure.
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

        /// <summary>Forgets every vessel, as when the switch is off.</summary>
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
            UnitVessels.ResizeUninitialized(unitCount);
            for (int i = 0; i < unitCount; i++)
            {
                UnitVessels[i] = -1;
            }
        }

        /// <summary>The set for this step, which points at this build's lists.</summary>
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
        /// Chooses the vessels that take part in this step and sizes the step's lists. A vessel takes part when all of its
        /// bodies are built as moving bodies this step. The joints spread waking along themselves before the build, so a
        /// vessel's bodies are either all awake or all asleep. A vessel that is asleep stays out of the set, and its joints
        /// are built as Unity Physics joints with nothing to solve, as they would be without vessels. Main thread.
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
            Edges.ResizeUninitialized(parts);
            EdgeLoads.ResizeUninitialized(parts);
            EdgeLocked.ResizeUninitialized(parts);
            Closures.ResizeUninitialized(ClosureUnits.Length);
            ClosureLoads.ResizeUninitialized(ClosureUnits.Length);

            // The solver writes the loads of the vessels that it solves. The rest read as nothing.
            UnsafeUtility.MemClear(EdgeLoads.GetUnsafePtr(), (long)parts * sizeof(VesselEdgeLoad));
            UnsafeUtility.MemClear(ClosureLoads.GetUnsafePtr(), (long)ClosureLoads.Length * sizeof(VesselEdgeLoad));
            UnsafeUtility.MemClear(EdgeLocked.GetUnsafePtr(), parts);
        }
    }

    /// <summary>
    /// Compiles the vessels from the joint units that <see cref="GroupJointsJob"/> found. A vessel is a connected group of
    /// forest units between two moving bodies. The forest's tree units become the vessel's edges, and its closure units
    /// become the vessel's closures. Units that touch the world, a static body or a kinematic body stay Unity Physics
    /// joints, and so do joints that aren't mergeable, such as hinges, servos with targets and wheels.
    /// <para>
    /// A vessel's root is the body with the lowest store index, and its parts follow in breadth-first order from there, so
    /// every parent comes before its children. A vessel keeps its Id when the compile finds the same parts, in the same
    /// order and joined by the same joints, as the compile before it. Any other vessel gets a new Id.
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

            // What the last compile found, kept until the new vessels are compared with it.
            var oldVessels = new NativeArray<VesselRecord>(Build.Vessels.AsArray(), Allocator.Temp);
            var oldBodies = new NativeArray<IntPtr>(Build.PartBodyRecords.AsArray(), Allocator.Temp);
            var oldJoints = new NativeArray<IntPtr>(Build.PartJointRecords.AsArray(), Allocator.Temp);
            var oldClosures = new NativeArray<IntPtr>(Build.ClosureJointRecords.AsArray(), Allocator.Temp);
            Build.Clear(unitCount);

            // Kruskal over the forest units between two moving bodies, in the store's order, as the forest does it: a
            // unit that joins two trees is an edge, and a unit whose bodies are already in one tree closes a loop. The
            // forest's own roles count a body that isn't simulated as moving, so the units are sorted again here.
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

            // Each body's neighbours along tree units, as one list with a start for each body.
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

            // Walk each component from its lowest body, breadth first. A body with no tree units is no vessel.
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

            // The closures, grouped by vessel: count them, then place them.
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

            KeepOrRenewIds(oldVessels, oldBodies, oldJoints, oldClosures);
        }

        // What this compile found each unit to be: a vessel's edge, a vessel's closure, or neither.
        [NativeDisableUnsafePtrRestriction]
        private JointForestRole* kinds;

        /// <summary>
        /// Whether this compile found a unit to be of the given kind, and its bodies' store indices: a is the leading
        /// joint's body0, and b is its body1.
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

        /// <summary>Whether a body is built as a moving body: simulated, and neither static nor kinematic.</summary>
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
        /// Gives each new vessel the Id of the vessel from the last compile that had the same root body, if the two have the
        /// same parts in the same order, joined by the same joints, and the same closures. Every other vessel gets a new Id.
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
    /// What a joint unit asks of a vessel edge or closure, worked out from its joints: the joint point in each body's own
    /// frame, the joint frame in each body's own frame, each rotation axis's spring and damper, and the break limits. A
    /// group of joints between the same two bodies counts as one, at the mean of its members' anchors.
    /// </summary>
    internal unsafe struct UnitParameters
    {
        public float3 Anchor0;
        public float3 Anchor1;
        public quaternion Rotation0;
        public quaternion Rotation1;

        /// <summary>Per rotation axis of the joint frame (twist, then the two swings): the spring and the damper.</summary>
        public double3 Stiffness;

        public double3 Damping;
        public double BreakForce;
        public double BreakTorque;

        /// <summary>
        /// Below this, in square metres, the members of a group are counted as sitting at one point. Above it, the members'
        /// linear rows hold a turn about any axis that their offsets don't lie along, and those rows are always hard.
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

            // A locked axis is hard, and a driven axis gets the sum of its members' drives.
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

            // The members of a group sit at different points around the joint, and each member's linear rows are hard. So
            // the group can't turn about an axis that their offsets from the mean point don't all lie along.
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

        /// <summary>A point in a body's own frame, as a point in its motion frame: from the centre of mass, along the principal axes.</summary>
        public static double3 InMotionFrame(NativeStore.BodyData* body, float3 point)
        {
            RigidTransform bodyFromMotion = body->mass->Transform;
            return math.rotate(math.conjugate(bodyFromMotion.rot), point - bodyFromMotion.pos);
        }

        /// <summary>A rotation in a body's own frame, as a rotation in its motion frame.</summary>
        public static quaternion InMotionFrame(NativeStore.BodyData* body, quaternion rotation)
        {
            return math.mul(math.conjugate(body->mass->Transform.rot), rotation);
        }

        /// <summary>
        /// The rotation from part A's motion frame to part B's, when the joint between them is at rest. At rest the two
        /// joint frames line up in the world: A's orientation times A's joint frame equals B's orientation times B's joint
        /// frame.
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
    /// Fills each part's body index and edge for this step. The edge's anchors, frame and rest orientation come from the
    /// store's joints and are turned into the parts' motion frames. Its springs and break limits come from the joints too.
    /// </summary>
    [BurstCompile]
    internal unsafe struct FillVesselEdgesJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<int> BuiltIndices;

        [ReadOnly]
        public NativeArray<JointUnit> Units;

        [ReadOnly]
        public NativeArray<JointGroup> Groups;

        [ReadOnly]
        public NativeArray<JointGroupMember> Members;

        [ReadOnly]
        public NativeArray<int> PartStoreBodies;

        [ReadOnly]
        public NativeArray<int> PartUnits;

        [ReadOnly]
        public NativeArray<int> PartParents;

        [ReadOnly]
        public NativeArray<byte> PartIsBody0;

        public NativeArray<int> PartBodies;
        public NativeArray<VesselEdge> Edges;

        /// <summary>The first part of the vessel of each part, so that a parent's index can be found from its relative one.</summary>
        [ReadOnly]
        public NativeArray<int> PartVesselFirst;

        public void Execute(int part)
        {
            int storeBody = PartStoreBodies[part];
            PartBodies[part] = BuiltIndices[storeBody];
            int unit = PartUnits[part];
            if (unit < 0)
            {
                Edges[part] = new VesselEdge { Parent = -1, State = VesselEdgeState.Yielding, MayLock = true };
                return;
            }

            UnitParameters parameters = UnitParameters.Of(World, Units[unit], Groups, Members);
            NativeStore.BodyData* child = World->bodies + storeBody;
            NativeStore.BodyData* parent = World->bodies + PartStoreBodies[PartVesselFirst[part] + PartParents[part]];
            bool childIsBody0 = PartIsBody0[part] != 0;
            float3 childAnchor = childIsBody0 ? parameters.Anchor0 : parameters.Anchor1;
            float3 parentAnchor = childIsBody0 ? parameters.Anchor1 : parameters.Anchor0;
            quaternion childFrame = childIsBody0 ? parameters.Rotation0 : parameters.Rotation1;
            quaternion parentFrame = childIsBody0 ? parameters.Rotation1 : parameters.Rotation0;

            // The edge has one spring for both swing axes, so it takes the softer of the two.
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
    /// Fills each closure for this step. Part A is the leading joint's body0 and part B its body1. A closure's linear rows
    /// are hard, and its angular rows take the softest of the joint's rotation axes.
    /// </summary>
    [BurstCompile]
    internal unsafe struct FillVesselClosuresJob : IJobParallelFor
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.WorldData* World;

        [ReadOnly]
        public NativeArray<JointUnit> Units;

        [ReadOnly]
        public NativeArray<JointGroup> Groups;

        [ReadOnly]
        public NativeArray<JointGroupMember> Members;

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
    /// Turns the loads that the solve wrote for the vessels into the loads of their joint units, the same way that the
    /// impulse events of a Unity Physics joint become a load: the force and torque that act on the leading joint's body0.
    /// An edge's load is what the parent applied to the child, and a closure's load is what part A applied to part B.
    /// A group's load goes to the group, and <see cref="ShareGroupLoadsJob"/> shares it among the members afterwards.
    /// </summary>
    [BurstCompile]
    internal unsafe struct VesselLoadsJob : IJobParallelFor
    {
        [ReadOnly]
        public NativeArray<JointUnit> Units;

        /// <summary>For each part, the unit of its edge, and for each closure after the parts, the closure's unit.</summary>
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

        [NativeDisableParallelForRestriction]
        public NativeArray<JointLoad> JointLoads;

        [NativeDisableParallelForRestriction]
        public NativeArray<JointLoad> GroupLoads;

        /// <summary>Indexes run over the parts first and then over the closures.</summary>
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

                // The load acts on the child. When the child is body1, what acts on body0 is the opposite.
                load = EdgeLoads[index];
                sign = PartIsBody0[index] != 0 ? 1.0f : -1.0f;
            }
            else
            {
                unit = ClosureUnits[index - parts];

                // The load acts on part B, which is body1, so what acts on body0 is the opposite.
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
                JointLoads[jointUnit.Joint] = jointLoad;
            }
            else
            {
                GroupLoads[jointUnit.Group] = jointLoad;
            }
        }
    }
}
