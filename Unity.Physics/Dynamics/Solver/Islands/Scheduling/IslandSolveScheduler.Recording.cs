using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Stepping.Recording;

namespace Unity.Physics.Dynamics.Solver.Islands.Scheduling
{
    internal static partial class IslandSolveScheduler
    {
        [BurstCompile]
        private unsafe struct RecordIslandsJob : IJob
        {
            [ReadOnly] public NativeList<DirectIsland> Islands;
            [ReadOnly] public NativeList<int> Starts;
            [ReadOnly] public NativeList<int> Bodies;
            [ReadOnly] public NativeArray<RigidBody> RigidBodies;
            [NativeDisableUnsafePtrRestriction] public RecordedIsland* Output;
            [NativeDisableUnsafePtrRestriction] public RecordedSolverPath* Paths;

            public void Execute()
            {
                for (int i = 0; i < Islands.Length; i++)
                {
                    DirectIsland island = Islands[i];
                    int start = Starts[i];
                    int end = i + 1 < Starts.Length ? Starts[i + 1] : Bodies.Length;
                    var record = new RecordedIsland
                    {
                        Index = i,
                        Size = end - start,
                        Path = island.Fused ? RecordedSolverPath.FusedVessel
                            : island.VesselCount > 0 ? RecordedSolverPath.UnfusedVessel : RecordedSolverPath.PlainIsland,
                    };
                    record.Path |= Paths[i];
                    for (int j = start; j < end; j++)
                    {
                        Output[RigidBodies[Bodies[j]].Entity.Index] = record;
                    }
                }
            }
        }
    }
}
