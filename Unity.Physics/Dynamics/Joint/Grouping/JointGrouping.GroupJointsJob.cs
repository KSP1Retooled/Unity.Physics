using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Physics.Math;
using SolverJoint = Unity.Physics.Joint;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Memory;
using Unity.Physics.Dynamics.Joint.Translation;

namespace Unity.Physics.Dynamics.Joint.Grouping
{
    internal static partial class JointGrouping
    {
        /// <summary>
        /// List ungrouped joints and joint groups as build units, preserving store order.
        /// </summary>
        [BurstCompile]
        internal unsafe struct GroupJointsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            public NativeList<JointGroup> Groups;
            public NativeList<JointGroupMember> Members;
            public NativeList<JointUnit> Units;

            /// <summary>
            /// The role of each unit in the vessel forest, or None for all units when the forest is disabled.
            /// </summary>
            public NativeList<JointForestRole> Roles;

            /// <summary>Resized here for <see cref="JointTranslation.BuildJointsJob.Written"/> ; values are written by that job.</summary>
            public NativeList<int> Written;

            public bool Forest;

            public void Execute()
            {
                Units.Clear();
                NativeArray<int> groupOf = FindGroups();
                for (int i = 0; i < World->jointCount; i++)
                {
                    int group = groupOf[i];
                    if (group < 0 || Groups[group].Leader == i)
                    {
                        Units.Add(new JointUnit { Joint = i, Group = group });
                    }
                }

                FindForest();
                Written.ResizeUninitialized(Units.Length);
            }

            /// <summary>
            /// Use Kruskal's algorithm in store order to build the forests for vessels and rows.
            /// Mergeable units that join trees become edges. Mergeable units within a tree become closures.
            /// </summary>
            private void FindForest()
            {
                Roles.ResizeUninitialized(Units.Length);
                if (!Forest)
                {
                    for (int u = 0; u < Units.Length; u++)
                    {
                        Roles[u] = JointForestRole.None;
                    }

                    return;
                }

                int bodyCount = (int)World->bodyCount;
                var root = new NativeArray<int>(bodyCount, Allocator.Temp);
                // A servo can join two vessel trees. The row forest tracks connections through mergeable joints.
                var vesselRoot = new NativeArray<int>(bodyCount, Allocator.Temp);
                for (int i = 0; i < bodyCount; i++)
                {
                    root[i] = i;
                    vesselRoot[i] = i;
                }

                for (int u = 0; u < Units.Length; u++)
                {
                    JointData* joint = World->joints[Units[u].Joint];
                    ServoAxis axis = JointTranslation.BuildJointsJob.ServoJoint(joint);
                    bool servo = axis != ServoAxis.None;
                    if (!servo && !JointTranslation.BuildJointsJob.Mergeable(joint))
                    {
                        Roles[u] = JointForestRole.None;
                        continue;
                    }

                    if (Immobile(joint->body0) || Immobile(joint->body1))
                    {
                        Roles[u] = servo ? JointForestRole.None : JointForestRole.Tree;
                        continue;
                    }

                    int body0 = (int)joint->body0->index, body1 = (int)joint->body1->index;
                    bool joinsVesselTrees = false;
                    if ((StoreMemory.Row(joint->body0)->flags & StoreMemory.Row(joint->body1)->flags & BodyFlags.Simulated) != 0)
                    {
                        int first = Find(vesselRoot, body0), second = Find(vesselRoot, body1);
                        joinsVesselTrees = first != second;
                        vesselRoot[first] = second;
                    }

                    if (servo)
                    {
                        Roles[u] = !joinsVesselTrees ? JointForestRole.None
                            : axis == ServoAxis.LinearX ? JointForestRole.PistonTree : JointForestRole.ServoTree;
                        continue;
                    }

                    int a = Find(root, body0), b = Find(root, body1);
                    if (a == b)
                    {
                        Roles[u] = JointForestRole.Closure;
                        continue;
                    }

                    root[a] = b;
                    Roles[u] = JointForestRole.Tree;
                }
            }

            internal static bool Immobile(Body* body)
            {
                const BodyFlags immobile = BodyFlags.Static | BodyFlags.Kinematic;
                return body == null || (StoreMemory.Row(body)->flags & immobile) != 0;
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
            /// Merge compatible joints between the same bodies into groups of at least two, avoiding redundant solver
            /// rows.
            /// </summary>
            private NativeArray<int> FindGroups()
            {
                Groups.Clear();
                Members.Clear();
                int count = (int)World->jointCount;
                var groupOf = new NativeArray<int>(count, Allocator.Temp);
                var leaderOf = new NativeArray<int>(count, Allocator.Temp);
                var memberCount = new NativeArray<int>(count, Allocator.Temp);
                var leaders = new NativeHashMap<int2, int>(count, Allocator.Temp);
                for (int i = 0; i < count; i++)
                {
                    groupOf[i] = -1;
                    leaderOf[i] = -1;
                    JointData* joint = World->joints[i];
                    if (JointTranslation.BuildJointsJob.ServoJoint(joint) != ServoAxis.None
                        || !JointTranslation.BuildJointsJob.Mergeable(joint))
                    {
                        continue;
                    }

                    var bodies = new int2(JointTranslation.BuildJointsJob.BodyKey(joint->body0), JointTranslation.BuildJointsJob.BodyKey(joint->body1));
                    if (!leaders.TryGetValue(bodies, out int leader))
                    {
                        leader = i;
                        leaders.Add(bodies, leader);
                    }
                    else if (!JointTranslation.BuildJointsJob.SameFrames(World->joints[leader], joint)
                        || !JointTranslation.BuildJointsJob.SameAngularLimits(World->joints[leader], joint))
                    {
                        continue;
                    }

                    leaderOf[i] = leader;
                    memberCount[leader]++;
                }

                // The leader appears before the other members in store order, so its group has been created before those
                // members are processed.
                for (int i = 0; i < count; i++)
                {
                    int leader = leaderOf[i];
                    if (leader < 0 || memberCount[leader] < 2)
                    {
                        continue;
                    }

                    if (leader == i)
                    {
                        groupOf[i] = Groups.Length;
                        Groups.Add(new JointGroup { Leader = i, FirstMember = Members.Length });
                        Members.ResizeUninitialized(Members.Length + memberCount[i]);
                    }

                    int group = groupOf[leader];
                    groupOf[i] = group;
                    JointGroup joined = Groups[group];
                    Members[joined.FirstMember + joined.MemberCount] = new JointGroupMember { Joint = i };
                    joined.MemberCount++;
                    Groups[group] = joined;
                }

                return groupOf;
            }
        }
    }
}
