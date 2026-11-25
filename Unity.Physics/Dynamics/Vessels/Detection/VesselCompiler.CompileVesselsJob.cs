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
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Configuration;
using Unity.Physics.Collision.Filter.Groups;

namespace Unity.Physics.Dynamics.Vessels.Detection
{
    internal static unsafe partial class VesselCompiler
    {
        /// <summary>
        /// Compile connected forest units into vessels, with parents before children and loop joints as closures.
        /// Keep eligible existing vessels and their IDs. Identify bodies by address and serial to detect reused
        /// allocations.
        /// Retain removed joints as unconstrained tombstones for TombstoneSteps, allowing autostruts to return without
        /// resetting locks.
        /// Recompile immediately for existing joints that become ineligible. Change structure versions when edges or
        /// closures change.
        /// New vessels must meet MinimumParts; matching structures retain their prior IDs.
        /// </summary>
        [BurstCompile]
        internal unsafe struct CompileVesselsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

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

            /// <summary>See <see cref="VesselSwitches.TombstoneSteps"/>.</summary>
            public int TombstoneSteps;

            /// <summary>See <see cref="VesselSwitches.MinimumParts"/>.</summary>
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
                public WorldData* World;
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
                        JointData* joint = World->joints[Units[u].Joint];
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
                            JointData* joint = World->joints[Units[u].Joint];
                            Build.PairUnits.Add(VesselBuild.PairKey((int)joint->body0->index, (int)joint->body1->index), u);
                        }
                    }
                }

                /// <summary>
                /// Keep eligible previous vessels in their existing order.
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

                        JointData* joint = World->joints[Units[u].Joint];
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
                        closureUnit[c] = Claim(unitsByBodies, a, b, oldClosures[c], oldClosureSerials[c], closure: true);
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
                        closureUnit[c] = Claim(unitsByBodies, a, b, IntPtr.Zero, 0, closure: true);
                        if (closureUnit[c] < 0 && (Expired(oldClosureTombstones[c]) || StillJoins(oldClosures[c], oldClosureSerials[c], a, b)))
                        {
                            return false;
                        }
                    }

                    return true;
                }

                /// <summary>
                /// Return whether the previous joint still exists between the same bodies and is unbroken.
                /// An existing but ineligible joint requires recompilation, not a tombstone.
                /// </summary>
                private bool StillJoins(IntPtr record, uint serial, int bodyA, int bodyB)
                {
                    if (record == IntPtr.Zero || !liveJoints.Contains(record))
                    {
                        return false;
                    }

                    var joint = (JointData*)record;
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

                private static bool Is(JointData* joint, IntPtr record, uint serial)
                {
                    return (IntPtr)joint == record && joint->serial == serial;
                }

                /// <summary>Returns whether a tombstone made at the given step (or one made now, for -1) has expired.</summary>
                private bool Expired(int stamp)
                {
                    return Step - (stamp >= 0 ? stamp : Step) >= TombstoneSteps;
                }

                /// <summary>
                /// Claim an eligible unit between two bodies, or return -1.
                /// If a joint record is supplied, require the unit to contain that record and serial.
                /// </summary>
                private int Claim(NativeParallelMultiHashMap<ulong, int> unitsByBodies, int bodyA, int bodyB, IntPtr record, uint serial,
                    bool closure = false)
                {
                    if (unitsByBodies.TryGetFirstValue(VesselBuild.PairKey(bodyA, bodyB), out int unit, out NativeParallelMultiHashMapIterator<ulong> iterator))
                    {
                        do
                        {
                            if (!claimed[unit] && (!closure || Roles[unit] != JointForestRole.ServoTree)
                                && (record == IntPtr.Zero || Holds(unit, record, serial)))
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
                            JointData* joint = World->joints[Units[unit].Joint];
                            isBody0 = (int)joint->body0->index == partBody[old] ? (byte)1 : (byte)0;
                            Build.UnitVessels[unit] = index;

                            // A different joint took this edge; invalidate its old lock state. Tombstones are
                            // handled below.
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
                            JointData* joint = World->joints[Units[unit].Joint];
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
                /// Compile unclaimed units into vessels using Kruskal's algorithm in store order.
                /// Units joining trees become edges; units within a tree become closures.
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

                        JointData* joint = World->joints[Units[u].Joint];
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
                            if (Roles[u] != JointForestRole.ServoTree)
                            {
                                kinds[u] = JointForestRole.Closure;
                            }
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

                    // List each connected component breadth-first from its lowest store index, excluding components
                    // below the minimum size.
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
                                JointData* joint = World->joints[Units[orderUnit[k]].Joint];
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
                /// Return whether a new unit has the requested role, with its store body indices (a = body0, b =
                /// body1).
                /// </summary>
                private bool IsVesselUnit(int unit, JointForestRole kind, out int a, out int b)
                {
                    a = b = -1;
                    if (kinds[unit] != kind)
                    {
                        return false;
                    }

                    JointData* joint = World->joints[Units[unit].Joint];
                    a = (int)joint->body0->index;
                    b = (int)joint->body1->index;
                    return true;
                }

                /// <summary>
                /// Returns whether a body is simulated and is neither static nor kinematic, so it can be built as a moving
                /// body.
                /// </summary>
                private static bool Moving(Body* body)
                {
                    return body != null && Moving(StoreMemory.Row(body));
                }

                private static bool Moving(BodyData* body)
                {
                    BodyFlags flags = body->flags;
                    return (flags & BodyFlags.Simulated) != 0
                        && (flags & (BodyFlags.Static | BodyFlags.Kinematic)) == 0;
                }

                /// <summary>
                /// Add a part. Roots and tombstones use unit -1; tombstone records the removal step, or -1.
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
                    Build.ClosureJointSerials.Add(record != IntPtr.Zero ? ((JointData*)record)->serial : 0u);
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
                /// Reuse prior vessel IDs when root, ordered parts, joints and closures match; assign new IDs
                /// otherwise.
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
                            vessel.Id = NewId();
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
    }
}
