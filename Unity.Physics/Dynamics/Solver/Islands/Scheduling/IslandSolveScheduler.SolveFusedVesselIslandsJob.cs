using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Profiling;
using static Unity.Physics.Solver;
using static Unity.Physics.Dynamics.Solver.Vessels.VesselIslandSolver;

namespace Unity.Physics.Dynamics.Solver.Islands.Scheduling
{
    internal static partial class IslandSolveScheduler
    {
        /// <summary>
        /// Solve marked vessel islands across all substeps with a team per island.
        /// Idle lanes help active teams. Combine task results in fixed order for deterministic results.
        /// </summary>
        [BurstCompile]
        unsafe struct SolveFusedVesselIslandsJob : IJobParallelFor
        {
            // The substeps' jobs change the row counts of the islands while this job runs. This job reads only the
            // fields that say which islands are fused and which vessels they hold, and those fields do not change.
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<DirectIsland> Islands;

            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<int> IslandVessels;

            /// <summary>
            /// The Jacobians of every island. This job updates and reads only the Jacobians of the fused islands, which
            /// the substeps' jobs skip.
            /// </summary>
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<IslandHeader> Headers;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            public StepInput StepInput;

            /// <summary>A team for each fused island, the island with the most parts first.</summary>
            [NativeDisableUnsafePtrRestriction]
            public UnsafeList<FusedTeam>* Teams;

            /// <summary>
            /// The bodies of the islands that are not fused, which the lane that does the substeps' work of those
            /// islands integrates.
            /// </summary>
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<int> UnfusedBodies;

            // Unfused substep inputs and optional atomic lane claim. Without the claim word, chained jobs own
            // the work.
            public UnfusedSubsteps Unfused;

            [NativeDisableUnsafePtrRestriction]
            public int* UnfusedClaim;

            /// <summary>
            /// Scheduled lane count. With one lane, skip team stage publication and solve islands directly.
            /// </summary>
            public int Lanes;

            static readonly ProfilerMarker HelpMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Vessel.HelpFused");

            public void Execute(int lane)
            {
                FusedTeam* teams = Teams->Ptr;
                int count = Teams->Length;

                // Claim unled islands. Give unfused substeps to the first lane not leading the largest island.
                for (int t = 0; t < count; t++)
                {
                    if (teams[t].Team.TryLead())
                    {
                        DirectIsland island = Islands[teams[t].Island];
                        IslandWorkCounters.CountFusedSolve();
                        SolveFusedVesselIsland(island, (IslandHeader*)Headers.GetUnsafeReadOnlyPtr() + island.FirstHeader,
                            (int*)IslandVessels.GetUnsafeReadOnlyPtr() + island.FirstVessel, MotionVelocities, MotionDatas, Bodies,
                            StepInput, Lanes > 1 ? &teams[t].Team : null);
                        teams[t].Team.Finish();
                    }

                    if (t == 0)
                    {
                        RunUnfusedOnce();
                    }
                }

                if (count == 0)
                {
                    RunUnfusedOnce();
                }

                // Help open stages, largest island first. After the spin timeout, wait on the team with the
                // most parts per lane.
                using (HelpMarker.Auto())
                {
                    while (true)
                    {
                        if (count > 1 && !SweepOpenStages(teams, count))
                        {
                            return;
                        }

                        int best = -1;
                        long bestParts = 0, bestLanes = 1;
                        for (int t = 0; t < count; t++)
                        {
                            if (teams[t].Team.Finished)
                            {
                                continue;
                            }

                            // This island has more parts for each lane than the best so far when its parts times the best
                            // island's lanes are more than the best island's parts times its lanes.
                            long lanes = System.Threading.Volatile.Read(ref teams[t].Helpers) + 1;
                            if (best < 0 || teams[t].Parts * bestLanes > bestParts * lanes)
                            {
                                best = t;
                                bestParts = teams[t].Parts;
                                bestLanes = lanes;
                            }
                        }

                        if (best < 0)
                        {
                            return;
                        }

                        System.Threading.Interlocked.Increment(ref teams[best].Helpers);
                        teams[best].Team.Help<IslandTaskDispatch>();
                        System.Threading.Interlocked.Decrement(ref teams[best].Helpers);

                        // A helper that stayed idle past its patience gives its thread back to the job system rather than
                        // wait for another island.
                        if (!teams[best].Team.Finished)
                        {
                            return;
                        }
                    }
                }
            }

            /// <summary>
            /// Does the substeps' work of the islands that are not fused, if the step gave that work to this job and no
            /// other lane has taken it.
            /// </summary>
            void RunUnfusedOnce()
            {
                if (UnfusedClaim != null && System.Threading.Interlocked.CompareExchange(ref *UnfusedClaim, 1, 0) == 0)
                {
                    RunUnfusedSubsteps(Unfused, Headers, Islands, IslandVessels, UnfusedBodies, MotionVelocities, MotionDatas, Bodies);
                }
            }

            /// <summary>
            /// Help open island stages, largest first, until the spin timeout. Return false when all islands are
            /// solved.
            /// </summary>
            static bool SweepOpenStages(FusedTeam* teams, int count)
            {
                long spinTicks = teams[0].Team.SpinTicks;
                long idleSince = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                while (true)
                {
                    bool unsolved = false, worked = false;
                    for (int t = 0; t < count; t++)
                    {
                        if (!teams[t].Team.Finished)
                        {
                            unsolved = true;
                            worked |= teams[t].Team.TryWork<IslandTaskDispatch>();
                        }
                    }

                    if (!unsolved)
                    {
                        return false;
                    }

                    long now = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    if (worked)
                    {
                        idleSince = now;
                    }
                    else if (now - idleSince > spinTicks)
                    {
                        return true;
                    }
                    else
                    {
                        Unity.Burst.Intrinsics.Common.Pause();
                    }
                }
            }
        }
    }
}
