using System;
using Unity.Collections;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Sparse;
using static Unity.Physics.Solver;
using static Unity.Physics.Dynamics.Solver.Forest.ForestSolver;

namespace Unity.Physics.Dynamics.Solver.Vessels
{
    /// <summary>Calls the vessel solver and push-out pass when the island scheduler runs the sparse solve.</summary>
    internal unsafe struct SparseIslandPath : ISparseIslandPath
    {
        public void SolveVessel(in SparseConstraintSolver.Problem problem, JacobianHeader** headers,
            in NativeList<IntPtr> headerList, in NativeList<int> worldBodies, in NativeArray<int> compactBodies, bool bounded,
            bool holdsPositions, int islandIndex, int* islandVessels, int islandVesselCount, in NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, in StepInput stepInput, in JacobianRows jacobianRows, float* recovery,
            bool sparseContacts, WorkTeam* team, bool applyImpulses, float* impulses)
        {
            VesselIslandSolver.SolveVesselIsland(problem, headers, headerList, worldBodies, compactBodies, bounded,
                holdsPositions, islandIndex, islandVessels, islandVesselCount, motionVelocities, motionDatas, stepInput,
                jacobianRows, recovery, sparseContacts, team, applyImpulses, impulses);
        }

        public void PushOut(in SparseConstraintSolver.Problem problem, JacobianHeader** headers,
            in NativeList<int> worldBodies, float* recovery, in NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, float timestep, float fastest, ForestRun* forest)
        {
            VesselIslandSolver.SolvePushOut(problem, headers, worldBodies, recovery, motionVelocities, motionDatas,
                timestep, fastest, forest);
        }
    }
}
