using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the joint units, groups, and loads.
    /// Reuses groups while the topology version and forest setting match.
    /// </summary>
    internal unsafe struct JointBuildState
    {
        internal NativeList<JointGroup> jointGroups;
        internal NativeList<JointGroupMember> jointGroupMembers;
        internal NativeList<JointUnit> jointUnits;
        internal NativeList<JointForestRole> jointRoles;
        internal NativeList<int> jointsWritten;

        // The topology version and forest setting used to find the groups. The groups can be reused while these values
        // are unchanged. See JointTopology.
        internal uint groupedTopology;
        internal bool groupedForest;
        internal bool grouped;
        internal NativeList<JointLoad> jointLoads;
        internal NativeList<JointLoad> groupLoads;

        // For each joint, 1 if its smoothed load exceeded its break limits in this step, for JointBreaking.BreakJointsJob. The jobs
        // that settle the loads write the flags. A byte for each joint is much less memory to write and to scan than
        // a load.
        internal NativeList<byte> jointsOverloaded;

        // The load state of each joint, in the order of the store's list of joints. The joint commands keep it in that
        // order between steps. See JointLoadStates.
        internal UnsafeList<JointLoadState>* jointStates;

        // The joints and groups outside the active vessels of this step. See VesselLoads.ListOtherJointsJob.
        internal NativeList<int> otherJoints;
        internal NativeList<int> otherGroups;

        internal void Initialize(WorldData* store)
        {
            jointGroups = new NativeList<JointGroup>(Allocator.Persistent);
            jointGroupMembers = new NativeList<JointGroupMember>(Allocator.Persistent);
            jointUnits = new NativeList<JointUnit>(Allocator.Persistent);
            jointRoles = new NativeList<JointForestRole>(Allocator.Persistent);
            jointsWritten = new NativeList<int>(Allocator.Persistent);
            jointLoads = new NativeList<JointLoad>(Allocator.Persistent);
            groupLoads = new NativeList<JointLoad>(Allocator.Persistent);
            jointsOverloaded = new NativeList<byte>(Allocator.Persistent);
            jointStates = UnsafeList<JointLoadState>.Create((int)store->jointCount, Allocator.Persistent);
            JointLoadStates.Register(store, jointStates);
            otherJoints = new NativeList<int>(Allocator.Persistent);
            otherGroups = new NativeList<int>(Allocator.Persistent);
        }

        internal void Dispose(WorldData* store)
        {
            jointGroups.Dispose();
            jointGroupMembers.Dispose();
            jointUnits.Dispose();
            jointRoles.Dispose();
            jointsWritten.Dispose();
            jointLoads.Dispose();
            groupLoads.Dispose();
            jointsOverloaded.Dispose();
            JointLoadStates.Unregister(store);
            UnsafeList<JointLoadState>.Destroy(jointStates);
            otherJoints.Dispose();
            otherGroups.Dispose();
        }

        internal bool FindGroups(WorldData* store, bool forest, JobHandle dependency, out JobHandle groupsFound)
        {
            uint topology = JointTopology.Version;
            groupsFound = dependency;
            bool regrouped = !grouped || topology != groupedTopology || forest != groupedForest;
            if (regrouped)
            {
                groupsFound = new JointGrouping.GroupJointsJob
                {
                    World = store,
                    Groups = jointGroups,
                    Members = jointGroupMembers,
                    Units = jointUnits,
                    Roles = jointRoles,
                    Written = jointsWritten,
                    Forest = forest,
                }.Schedule(dependency);
                grouped = true;
                groupedTopology = topology;
                groupedForest = forest;
            }
            return regrouped;
        }
    }
}
