using static Unity.Physics.Solver;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Profiling;
using static Unity.Physics.Math;

namespace Unity.Physics.Dynamics.Solver.Vessels
{
    internal static partial class VesselIslandSolver
    {
        /// <summary>
        /// Predict equality constraint motion over a substep using the integrator and supplied velocities.
        /// Write one change per row. Return false for ranged or position-free constraints.
        /// </summary>
        static unsafe bool PredictEqualityChange(ref JacobianHeader header, in NativeArray<MotionVelocity> motionVelocities,
            float timestep, float* change)
        {
            int ixBodyA = header.BodyPair.BodyIndexA;
            int ixBodyB = header.BodyPair.BodyIndexB;
            switch (header.Type)
            {
                case JacobianType.AngularLimit1D:
                {
                    ref AngularLimit1DJacobian angJacobian = ref header.AccessBaseJacobian<AngularLimit1DJacobian>();
                    if (math.abs(angJacobian.MinAngle - angJacobian.MaxAngle) >= math.EPSILON)
                    {
                        return false;
                    }

                    quaternion predictedBFromA = JacobianUtilities.IntegrateOrientationBFromA(angJacobian.MotionBFromA,
                        MotionAngularVelocity(ixBodyA, motionVelocities), MotionAngularVelocity(ixBodyB, motionVelocities), timestep);
                    float angleChange = angJacobian.CalculateAngle(predictedBFromA) - angJacobian.CalculateAngle(angJacobian.MotionBFromA);
                    change[0] = math.atan2(math.sin(angleChange), math.cos(angleChange));
                    return true;
                }
                case JacobianType.LinearLimit:
                {
                    ref LinearLimitJacobian limitJacobian = ref header.AccessBaseJacobian<LinearLimitJacobian>();
                    if (math.abs(limitJacobian.MinDistance - limitJacobian.MaxDistance) >= math.EPSILON)
                    {
                        return false;
                    }

                    RigidTransform predictedA = PredictedMotion(limitJacobian.WorldFromA, ixBodyA, motionVelocities, timestep);
                    RigidTransform predictedB = PredictedMotion(limitJacobian.WorldFromB, ixBodyB, motionVelocities, timestep);
                    float3 offset = math.transform(limitJacobian.WorldFromA, limitJacobian.BodyFromConstraintA.Translation)
                        - math.transform(limitJacobian.WorldFromB, limitJacobian.BodyFromConstraintB.Translation);
                    float3 predictedOffset = math.transform(predictedA, limitJacobian.BodyFromConstraintA.Translation)
                        - math.transform(predictedB, limitJacobian.BodyFromConstraintB.Translation);

                    // The same cases, in the same order, as the rows BuildJacobianRows builds for a linear limit.
                    bool lockAll = math.all(limitJacobian.ConstrainedAxes);
                    bool atZero = limitJacobian.MinDistance < math.EPSILON;
                    if (lockAll && atZero)
                    {
                        float3 offsetChange = predictedOffset - offset;
                        change[0] = offsetChange.x;
                        change[1] = offsetChange.y;
                        change[2] = offsetChange.z;
                        return true;
                    }

                    if (!lockAll && !limitJacobian.Is1D && atZero)
                    {
                        int axis0 = limitJacobian.ConstrainedAxes[0] ? 0 : 1;
                        int axis1 = axis0 == 1 ? 2 : (limitJacobian.ConstrainedAxes[1] ? 1 : 2);
                        float3x3 axes = limitJacobian.BodyFromConstraintB.Rotation;
                        change[0] = math.dot(predictedOffset, math.rotate(predictedB, axes[axis0]))
                            - math.dot(offset, math.rotate(limitJacobian.WorldFromB, axes[axis0]));
                        change[1] = math.dot(predictedOffset, math.rotate(predictedB, axes[axis1]))
                            - math.dot(offset, math.rotate(limitJacobian.WorldFromB, axes[axis1]));
                        return true;
                    }

                    if (limitJacobian.Is1D)
                    {
                        change[0] = math.dot(predictedOffset, math.rotate(predictedB, limitJacobian.AxisInB))
                            - math.dot(offset, math.rotate(limitJacobian.WorldFromB, limitJacobian.AxisInB));
                        return true;
                    }

                    // A fixed distance between the anchors. A line with a distance from it has no one direction to hold.
                    if (lockAll && math.lengthsq(offset) > math.EPSILON)
                    {
                        change[0] = math.length(predictedOffset) - math.length(offset);
                        return true;
                    }

                    return false;
                }
                default:
                    return false;
            }
        }

        /// <summary>
        /// Adjust positional row targets for nonlinear drift predicted from the solved velocities.
        /// Return whether any correction is nonzero. Use post-solve motion so that rejected initial motion does not
        /// distort the prediction.
        /// </summary>
        internal static unsafe bool AimAtPredictedDrift(in SparseConstraintSolver.Problem problem, SparseConstraintSolver* solver,
            float3* bodyLinear, float3* bodyAngular, int* worldBodies, in NativeList<IntPtr> headers, float* impulses,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, float timestep,
            in JacobianRows jacobianRows, float* rhsChange, WorkTeam* team, bool skipSlow = false, byte* skipBlocks = null)
        {
            using ProfilerMarker.AutoScope span = PredictDriftMarker.Auto();

            // Temporarily apply solved velocities in the island-owned motion array, then restore them after
            // prediction.
            var motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(motionVelocities);
            var saved = new NativeArray<MotionVelocity>(math.max(1, problem.BodyCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int compact = 0; compact < problem.BodyCount; compact++)
            {
                int body = worldBodies[compact];
                float3 linear, angular;
                if (solver != null)
                {
                    solver->BodyImpulse(problem, impulses, compact, out linear, out angular);
                }
                else
                {
                    linear = bodyLinear[compact];
                    angular = bodyAngular[compact];
                }

                MotionVelocity velocity = motions[body];
                saved[compact] = velocity;
                velocity.ApplyLinearImpulse(linear);
                velocity.ApplyAngularImpulse(math.rotate(math.conjugate(motionDatas[body].WorldFromMotion.rot), angular));
                motions[body] = velocity;
            }

            // How far each moving body turns this substep, and how far out its joints can reach, for the Jacobians whose
            // drift is too small to aim at.
            var turns = new NativeArray<float2>(skipSlow ? problem.BodyCount : 0, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int compact = 0; skipSlow && compact < problem.BodyCount; compact++)
            {
                MotionVelocity velocity = motions[worldBodies[compact]];
                turns[compact] = new float2(math.length(velocity.AngularVelocity) * timestep,
                    math.max(velocity.AngularExpansionFactor, 1.0f));
            }

            var prediction = new DriftPrediction
            {
                Headers = (JacobianHeader**)headers.GetUnsafeReadOnlyPtr(),
                Blocks = problem.Blocks,
                BlockCount = problem.BlockCount,
                JacobianRows = jacobianRows,
                Solved = motions,
                BodyCount = motionVelocities.Length,
                MotionDatas = (MotionData*)motionDatas.GetUnsafeReadOnlyPtr(),
                MotionDataCount = motionDatas.Length,
                Timestep = timestep,
                Change = rhsChange,
                Turns = skipSlow ? (float2*)turns.GetUnsafeReadOnlyPtr() : null,
                SkipBlocks = skipBlocks,
            };

            int tasks = (problem.BlockCount + DriftBlocksPerTask - 1) / DriftBlocksPerTask;
            WorkTeam.RunAll<DirectTaskDispatch>(team, &prediction, IslandTaskKind.PredictDriftKind, tasks, null);
            for (int compact = 0; compact < problem.BodyCount; compact++)
            {
                motions[worldBodies[compact]] = saved[compact];
            }

            saved.Dispose();
            turns.Dispose();
            for (int row = 0; row < problem.RowCount; row++)
            {
                if (rhsChange[row] != 0.0f)
                {
                    return true;
                }
            }

            return false;
        }

        static readonly ProfilerMarker PredictDriftMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Direct.PredictDrift");

        /// <summary>How many Jacobians one task of a drift prediction takes, which is a few microseconds of work.</summary>
        const int DriftBlocksPerTask = 16;

        /// <summary>What predicting the drift reads and writes, shared by the lanes that split its Jacobians.</summary>
        unsafe struct DriftPrediction
        {
            public JacobianHeader** Headers;
            public SparseConstraintSolver.ConstraintBlock* Blocks;
            public int BlockCount;
            public JacobianRows JacobianRows;
            public MotionVelocity* Solved;
            public int BodyCount;
            public MotionData* MotionDatas;
            public int MotionDataCount;
            public float Timestep;
            public float* Change;

            /// <summary>
            /// Each body's turn this substep in radians and the reach of its joints in metres, or null to aim every
            /// Jacobian.
            /// </summary>
            public float2* Turns;

            /// <summary>Blocks to leave unaimed, such as those inside a rigid section, or null.</summary>
            public byte* SkipBlocks;
        }

        /// <summary>
        /// Minimum predicted nonlinear drift requiring a target correction, near float position resolution at one
        /// metre.
        /// </summary>
        const float DriftTolerance = 1e-7f;

        /// <summary>
        /// A side's turn and reach: a moving body's from the table, a static body's none, and a kinematic body's from its
        /// velocity.
        /// </summary>
        static unsafe float2 Turn(DriftPrediction* prediction, int compact, JacobianHeader* header, bool sideA,
            in NativeArray<MotionVelocity> solved)
        {
            if (compact >= 0)
            {
                return prediction->Turns[compact];
            }

            int bodyIndex = sideA ? header->BodyPair.BodyIndexA : header->BodyPair.BodyIndexB;
            if (bodyIndex >= solved.Length)
            {
                return float2.zero;
            }

            MotionVelocity velocity = solved[bodyIndex];
            return new float2(math.length(velocity.AngularVelocity) * prediction->Timestep, math.max(velocity.AngularExpansionFactor, 1.0f));
        }

        /// <summary>One task of <see cref="AimAtPredictedDrift"/>, whose Jacobians write only their own rows.</summary>
        internal static unsafe void PredictDriftTask(void* context, int task)
        {
            var prediction = (DriftPrediction*)context;
            NativeArray<MotionVelocity> solved = View(prediction->Solved, prediction->BodyCount);
            NativeArray<MotionData> motionDatas = View(prediction->MotionDatas, prediction->MotionDataCount);
            float timestep = prediction->Timestep;
            float* change = stackalloc float[3];
            int end = math.min((task + 1) * DriftBlocksPerTask, prediction->BlockCount);
            for (int block = task * DriftBlocksPerTask; block < end; block++)
            {
                int startRow = prediction->Blocks[block].FirstRow;
                int jacRows = prediction->Blocks[block].RowCount;
                if (prediction->SkipBlocks != null && prediction->SkipBlocks[block] != 0)
                {
                    continue;
                }

                // The block's own body numbers spare reading the Jacobian, unless a side isn't one of the solve's bodies.
                JacobianHeader* headerPointer = prediction->Headers[block];
                if (prediction->Turns != null)
                {
                    float2 turnA = Turn(prediction, prediction->Blocks[block].BodyA, headerPointer, true, solved);
                    float2 turnB = Turn(prediction, prediction->Blocks[block].BodyB, headerPointer, false, solved);
                    float turn = turnA.x + turnB.x;
                    if (turn * turn * math.max(turnA.y, turnB.y) < DriftTolerance)
                    {
                        continue;
                    }
                }

                ref JacobianHeader header = ref *headerPointer;
                if (jacRows > 3 || !PredictEqualityChange(ref header, solved, timestep, change))
                {
                    continue;
                }

                int ixBodyA = header.BodyPair.BodyIndexA;
                int ixBodyB = header.BodyPair.BodyIndexB;
                for (int i = 0; i < jacRows; i++)
                {
                    // The row asks for -gamma * phi / dt - Ju. Swapping Ju's first-order drift for the predicted one at the
                    // solved velocities takes away the part of the drift the row misses.
                    float rate = RowRate(prediction->JacobianRows, startRow + i, ixBodyA, ixBodyB, ixBodyA < solved.Length,
                        ixBodyB < solved.Length, solved, motionDatas);
                    prediction->Change[startRow + i] = rate - change[i] / timestep;
                }
            }
        }

        /// <summary> Where a body's motion will be after a substep at the given velocity, as the integrator moves it. </summary>
        static RigidTransform PredictedMotion(in RigidTransform worldFromMotion, int bodyIndex, in NativeArray<MotionVelocity> motionVelocities,
            float timestep)
        {
            RigidTransform predicted = worldFromMotion;
            if (bodyIndex < motionVelocities.Length)
            {
                Integrator.Integrate(ref predicted, motionVelocities[bodyIndex], timestep);
            }

            return predicted;
        }

        static float3 MotionAngularVelocity(int bodyIndex, in NativeArray<MotionVelocity> motionVelocities)
        {
            return bodyIndex < motionVelocities.Length ? motionVelocities[bodyIndex].AngularVelocity : float3.zero;
        }
    }
}
