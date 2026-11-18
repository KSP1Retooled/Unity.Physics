using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Base.Threading;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Dynamics.Vessels.Model;
using static Unity.Physics.Solver;

namespace Unity.Physics.Dynamics.Solver.Islands.Scheduling
{
    internal static partial class IslandSolveScheduler
    {
        /// <summary>
        /// Mark eligible vessel islands for fused solving, assign other cache slots and list unfused bodies for
        /// integration.
        /// </summary>
        [BurstCompile]
        unsafe struct MarkFusedVesselIslandsJob : IJob
        {
            public NativeList<DirectIsland> Islands;

            [ReadOnly]
            public NativeList<int> IslandVessels;

            /// <summary>The islands' bodies, in island order, and where each island's bodies start.</summary>
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            public NativeList<int> UnfusedBodies;

            public VesselSet Vessels;

            public IslandSolverCache Cache;

            /// <summary>
            /// Whether the fused job is on, and the value of <see cref="SolverSwitches.FusedVesselParts"/>, when the job was scheduled.
            /// </summary>
            public bool Fuse;

            public int MinimumParts;

            /// <summary>The value of <see cref="SolverSwitches.FusedContacts"/> when the job was scheduled.</summary>
            public bool Contacts;

            /// <summary>The buffers that hold a team for each fused island and the configured team that they copy.</summary>
            public DirectIslandBuffers Buffers;

            public void Execute()
            {
                for (int i = 0; Fuse && i < Islands.Length; i++)
                {
                    DirectIsland island = Islands[i];
                    if (!Fusable(island, MinimumParts))
                    {
                        continue;
                    }

                    island.Fused = true;
                    Islands[i] = island;
                }

                AssignSlots();
                ListUnfusedBodies();
                ListFusedTeams();
                AssignVesselForests(Cache, Islands.AsArray(), IslandVessels.AsArray(), Vessels);
            }

            /// <summary>
            /// Assign substep cache slots by first-body identity. Fused islands use their own slots.
            /// </summary>
            void AssignSlots()
            {
                if (!Cache.IsCreated)
                {
                    return;
                }

                UnsafeList<ulong>* identities = Buffers.FirstIdentities;
                for (int i = 0; i < Islands.Length; i++)
                {
                    if (Islands[i].Fused)
                    {
                        identities->Ptr[i] = IslandSolverCache.NoSlot;
                    }
                }

                Cache.Assign(View(Buffers.FirstBodies->Ptr, Islands.Length), View(identities->Ptr, Islands.Length));
            }

            /// <summary>
            /// List unfused bodies and record their workload for the next step's scheduling decision.
            /// </summary>
            void ListUnfusedBodies()
            {
                UnfusedBodies.Clear();
                int islandCount = Islands.Length;
                var work = new IslandWorkCounters();
                for (int i = 0; i < islandCount; i++)
                {
                    DirectIsland island = Islands[i];
                    if (island.Fused)
                    {
                        work.FusedParts += island.VesselParts;
                        continue;
                    }

                    int first = Info.FirstIslandBodyIndices[i];
                    int end = i + 1 < islandCount ? Info.FirstIslandBodyIndices[i + 1] : Info.IslandBodies.Length;
                    UnfusedBodies.AddRange((int*)Info.IslandBodies.GetUnsafeReadOnlyPtr() + first, end - first);
                    work.Islands++;
                    work.Rows += island.Rows;
                    work.TeamIslands += island.SolvesInTeam ? 1 : 0;
                }

                work.Bodies = UnfusedBodies.Length;
                Cache.NoteUnfusedWork(work);
            }

            /// <summary>
            /// Create a team per fused island, ordered by descending part count then island index.
            /// </summary>
            void ListFusedTeams()
            {
                UnsafeList<FusedTeam>* teams = Buffers.FusedTeams;
                WorkTeam configured = Buffers.Teams->Ptr[Buffers.Teams->Length - 1];
                teams->Clear();
                for (int i = 0; i < Islands.Length; i++)
                {
                    if (!Islands[i].Fused)
                    {
                        continue;
                    }

                    var team = new FusedTeam { Team = configured, Island = i, Parts = Islands[i].VesselParts };
                    int at = teams->Length;
                    teams->Add(team);

                    // The islands come in index order, so an island moves ahead only of islands with fewer parts.
                    while (at > 0 && teams->Ptr[at - 1].Parts < team.Parts)
                    {
                        teams->Ptr[at] = teams->Ptr[at - 1];
                        at--;
                    }

                    teams->Ptr[at] = team;
                }
            }

            /// <summary>
            /// Whether the fused job can take an island of at least <paramref name="minimumParts"/> vessel parts: all
            /// its bodies must be vessel parts.
            /// </summary>
            bool Fusable(in DirectIsland island, int minimumParts)
            {
                return island.VesselCount > 0 && island.VesselParts >= minimumParts && island.OtherBodies == 0
                    && (Contacts || (island.JointRows == 0 && island.Rows == 0 && island.JointEnd == island.HeaderEnd));
            }
        }
    }
}
