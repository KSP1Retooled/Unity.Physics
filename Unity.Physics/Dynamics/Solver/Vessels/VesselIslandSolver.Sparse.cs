using static Unity.Physics.Solver;
using static Unity.Physics.Dynamics.Solver.Islands.Data.IslandTaskKind;
using static Unity.Physics.Dynamics.Solver.Forest.ForestSolver;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Linear;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Numerics.Linear.Solvers.Lowlevel;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Profiling;

namespace Unity.Physics.Dynamics.Solver.Vessels
{
    internal static partial class VesselIslandSolver
    {
        static readonly ProfilerMarker SolveSparselyMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Sparse.Solve");
        static readonly ProfilerMarker SparseSetupMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Sparse.Setup");

        /// <summary>
        /// Solve the island's sparse rows and leave the impulses in the right-hand side.
        /// </summary>
        internal static unsafe void SolveSparsely(int islandIndex, in NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, in StepInput stepInput, int rows, in JacobianRows jacobianRows,
            in Vector eps, in Vector l, in Vector u, in Vector rhs, in NativeArray<LCP.MLCPIndexFlag> indexSetArray, bool bounded,
            int contactPairs, WorkTeam* team, IslandHeader* islandHeaders, int* headerRows, int headerCount,
            float* recovery, int* islandVessels, int islandVesselCount, RigidBody* rigidBodies, int rigidBodyCount)
        {
            using ProfilerMarker.AutoScope span = SolveSparselyMarker.Auto();

            SparseSetupMarker.Begin();

            // The island numbers its bodies in the index that all the islands of the step share, if the cache has one.
            // Every entry of that index is -1 while no island solves its body.
            int* sharedIndex = stepInput.IslandSolverCache.CompactIndex(motionVelocities.Length);
            var setup = new SparseSetup
            {
                Blocks = new NativeList<SparseConstraintSolver.ConstraintBlock>(Allocator.Temp),
                Couplings = new NativeList<BoundCoupling>(contactPairs, Allocator.Temp),
                Headers = new NativeList<IntPtr>(Allocator.Temp),
                WorldBodies = new NativeList<int>(Allocator.Temp),
                CompactBodies = sharedIndex != null ? View(sharedIndex, motionVelocities.Length)
                    : new NativeArray<int>(motionVelocities.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory),
            };

            // All ones is -1, for a body not in the island.
            if (sharedIndex == null)
            {
                UnsafeUtility.MemSet(setup.CompactBodies.GetUnsafePtr(), 0xff, setup.CompactBodies.Length * (long)sizeof(int));
            }

            if (islandHeaders != null)
            {
                GatherSparseBlocks(ref setup, islandHeaders, headerRows, headerCount, rows, motionVelocities, team);
            }

            // The parts of the island's vessels are its bodies too, whether or not any Jacobian reaches them.
            for (int v = 0; v < islandVesselCount; v++)
            {
                VesselRecord vessel = stepInput.VesselSet.Vessels[islandVessels[v]];
                for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    setup.Number(Moving(stepInput.VesselSet.PartBodies[part], (MotionVelocity*)motionVelocities.GetUnsafeReadOnlyPtr(),
                        motionVelocities.Length));
                }
            }

            // The island's moving bodies by their index in the world, in the order of the solve's bodies.
            var islandBodies = (int*)setup.WorldBodies.GetUnsafeReadOnlyPtr();

            // Each body's mass matrix is its own, so the lanes split them.
            var bodies = new NativeArray<SparseConstraintSolver.Body>(setup.WorldBodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var bodySetup = new BodySetup
            {
                WorldBodies = setup.WorldBodies.GetUnsafeReadOnlyPtr(),
                Count = bodies.Length,
                MotionVelocities = (MotionVelocity*)motionVelocities.GetUnsafeReadOnlyPtr(),
                MotionDatas = (MotionData*)motionDatas.GetUnsafeReadOnlyPtr(),
                Timestep = stepInput.Timestep,
                Gyroscopic = stepInput.EnableGyroscopicTorque,
                Bodies = (SparseConstraintSolver.Body*)bodies.GetUnsafePtr(),
            };

            WorkTeam.RunAll<DirectTaskDispatch>(team, &bodySetup, IslandTaskKind.SetupBodiesKind, (bodies.Length + BodiesPerSetupTask - 1) / BodiesPerSetupTask, null);

            NativeList<SparseConstraintSolver.ConstraintBlock> blocks = setup.Blocks;
            NativeList<BoundCoupling> couplings = setup.Couplings;
            NativeList<IntPtr> headers = setup.Headers;
            NativeArray<int> compactBodies = setup.CompactBodies;
            bool holdsPositions = setup.HoldsPositions;

            var problem = new SparseConstraintSolver.Problem
            {
                BodyCount = bodies.Length,
                Bodies = (SparseConstraintSolver.Body*)bodies.GetUnsafePtr(),
                BlockCount = blocks.Length,
                Blocks = blocks.GetUnsafePtr(),
                RowCount = rows,
                Jacobian = jacobianRows.Start,
                Epsilon = rows > 0 ? (float*)UnsafeUtility.AddressOf(ref eps[0]) : null,
                RightHandSide = rows > 0 ? (float*)UnsafeUtility.AddressOf(ref rhs[0]) : null,
                Lower = rows > 0 ? (float*)UnsafeUtility.AddressOf(ref l[0]) : null,
                Upper = rows > 0 ? (float*)UnsafeUtility.AddressOf(ref u[0]) : null,
                InitialStates = (byte*)indexSetArray.GetUnsafeReadOnlyPtr(),
                CouplingCount = couplings.Length,
                Couplings = couplings.GetUnsafePtr(),
            };
            SparseSetupMarker.End();

            var impulses = new NativeArray<float>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            float* impulsesPtr = (float*)impulses.GetUnsafePtr();
            long solveStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;

            // A forest island solves its forest joints' rows structurally and the rest as a boundary block against them.
            ForestSettings forestSettings = stepInput.IslandSolverCache.Forest;

            // An island with a vessel has no Jacobians for the vessel's edges, so it can only solve with the vessel's own rows.
            if (islandVesselCount > 0)
            {
                SolveVesselIsland(problem, (JacobianHeader**)headers.GetUnsafeReadOnlyPtr(), headers, setup.WorldBodies, compactBodies,
                    bounded, holdsPositions, islandIndex, islandVessels, islandVesselCount, motionVelocities, motionDatas, stepInput,
                    jacobianRows, recovery, team, impulsesPtr);
                SparseProblemObserver.Report(&problem, (float*)impulses.GetUnsafePtr(), bounded);
                for (int row = 0; row < rows; row++)
                {
                    rhs[row] = impulses[row];
                }

                bodies.Dispose();
                blocks.Dispose();
                couplings.Dispose();
                headers.Dispose();
                setup.ReleaseCompactBodies(sharedIndex != null);
                setup.WorldBodies.Dispose();
                impulses.Dispose();
                return;
            }

            if (forestSettings.Enabled && headers.Length == blocks.Length && blocks.Length > 0)
            {
                ForestIsland* forest = stepInput.IslandSolverCache.ForestOf(islandIndex);
                int builds = forest != null ? forest->Builds : 0;
                ForestRun run = default;
                int fallbackBoundaryRows = 0;
                bool started = forest != null && TryStartForest(problem, (JacobianHeader**)headers.GetUnsafeReadOnlyPtr(), bounded,
                    forest, forestSettings, team, (int*)setup.WorldBodies.GetUnsafeReadOnlyPtr(), rigidBodies, rigidBodyCount,
                    motionVelocities, motionDatas,
                    stepInput.Timestep, stepInput.CurrentSubstep, out run, out fallbackBoundaryRows);
                long lap = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                if (started)
                {
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestStartTime, lap - solveStarted);
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestFactorTime, run.FactorTicks);
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestBoundaryTime, run.BoundaryTicks);
                    stepInput.IslandSolverCache.ForestCounters.CountSections(forest->SectionCount, forest->LockedPairs, run.Rebuilt,
                        run.LockChanges, run.StructureTicks);
                    stepInput.IslandSolverCache.ForestCounters.AddCount(ForestCounters.ForestContactLockChanges, run.ContactLockChanges);
                }

                stepInput.IslandSolverCache.ForestCounters.CountForest(started ? ForestCounters.ForestSolves : ForestCounters.ForestFallbacks);
                if (!started && forest != null)
                {
                    stepInput.IslandSolverCache.ForestCounters.NoteMax(ForestCounters.ForestFallbackBoundaryMax, fallbackBoundaryRows);
                }
                if (forest != null && forest->Builds != builds)
                {
                    stepInput.IslandSolverCache.ForestCounters.CountForest(ForestCounters.ForestBuilds);
                }

                if (started)
                {
                    SolveForest(ref run, problem.RightHandSide, impulsesPtr);
                    long solvedAt = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestSolveTime, solvedAt - lap);
                    var bodyLinear = new NativeArray<float3>(bodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    var bodyAngular = new NativeArray<float3>(bodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    var linearPtr = (float3*)bodyLinear.GetUnsafePtr();
                    var angularPtr = (float3*)bodyAngular.GetUnsafePtr();
                    SettleForest(ref run, problem, impulsesPtr, linearPtr, angularPtr, false);
                    long settledAt = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestSettleTime, settledAt - solvedAt);
                    if (holdsPositions)
                    {
                        var forestDrift = new NativeArray<float>(rows, Allocator.Temp);
                        float* forestDriftPtr = (float*)forestDrift.GetUnsafePtr();
                        bool drifts = AimAtPredictedDrift(problem, null, linearPtr, angularPtr, islandBodies, headers,
                            impulsesPtr, motionVelocities, motionDatas, stepInput.Timestep, jacobianRows, forestDriftPtr, team, true,
                            run.BlockInSection.IsCreated ? (byte*)run.BlockInSection.GetUnsafeReadOnlyPtr() : null);
                        stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestDriftTime,
                            Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - settledAt);
                        if (drifts)
                        {
                            for (int row = 0; row < rows; row++)
                            {
                                rhs[row] += forestDrift[row];
                            }

                            SolveForest(ref run, problem.RightHandSide, impulsesPtr);
                            SettleForest(ref run, problem, impulsesPtr, linearPtr, angularPtr, false);
                        }

                        forestDrift.Dispose();
                    }

                    ApplySparseImpulses(problem, null, linearPtr, angularPtr, islandBodies, headers, impulsesPtr,
                        motionVelocities, motionDatas, team);
                    if (recovery != null)
                    {
                        SolvePushOut(problem, (JacobianHeader**)headers.GetUnsafeReadOnlyPtr(), setup.WorldBodies, recovery,
                            motionVelocities, motionDatas, stepInput.Timestep,
                            math.max(stepInput.MaxStaticDepenetrationVelocity, stepInput.MaxDynamicDepenetrationVelocity), &run);
                    }

                    float deflection = RecordDeflections(ref run, (JacobianHeader**)headers.GetUnsafeReadOnlyPtr(), impulsesPtr,
                        stepInput.CurrentSubstep, forestSettings.DeflectionSmoothingSteps);
                    stepInput.IslandSolverCache.ForestCounters.NoteDeflection(deflection);

                    stepInput.IslandSolverCache.ForestCounters.NoteResidual(run.Residual);
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestClassifyTime, run.ClassifyTicks);
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestDecideTime, run.DecideTicks);
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestSumsTime, run.SumsTicks);
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestBodyImpulseTime, run.BodyImpulseTicks);
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestSubtreeTime, run.SubtreeTicks);
                    run.Dispose();
                    bodyLinear.Dispose();
                    bodyAngular.Dispose();
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestFinishTime,
                        Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solvedAt);
                    SparseProblemObserver.Report(&problem, impulsesPtr, bounded);
                    for (int row = 0; row < rows; row++)
                    {
                        rhs[row] = impulses[row];
                    }

                    bodies.Dispose();
                    blocks.Dispose();
                    couplings.Dispose();
                    headers.Dispose();
                    setup.ReleaseCompactBodies(sharedIndex != null);
                    setup.WorldBodies.Dispose();
                    impulses.Dispose();
                    return;
                }
            }

            SparseConstraintSolver* kept = stepInput.IslandSolverCache.Solver(islandIndex);
            var temporary = default(SparseConstraintSolver);
            if (kept == null)
            {
                temporary = new SparseConstraintSolver(Allocator.Temp);
                kept = &temporary;
            }

            // The team lives for one substep, and a kept solver for many.
            kept->Team = team;
            kept->Solve(problem, impulsesPtr, bounded);

            var drift = new NativeArray<float>(rows, Allocator.Temp);
            float* driftPtr = (float*)drift.GetUnsafePtr();
            if (holdsPositions && AimAtPredictedDrift(problem, kept, null, null, islandBodies, headers, impulsesPtr,
                motionVelocities, motionDatas, stepInput.Timestep, jacobianRows, driftPtr, team)
                && !kept->SolveChange(problem, driftPtr, impulsesPtr))
            {
                for (int row = 0; row < rows; row++)
                {
                    rhs[row] += drift[row];
                }

                kept->Solve(problem, impulsesPtr, bounded);
            }

            drift.Dispose();
            ApplySparseImpulses(problem, kept, null, null, islandBodies, headers, impulsesPtr, motionVelocities,
                motionDatas, team);
            if (recovery != null)
            {
                SolvePushOut(problem, (JacobianHeader**)headers.GetUnsafeReadOnlyPtr(), setup.WorldBodies, recovery,
                    motionVelocities, motionDatas, stepInput.Timestep,
                    math.max(stepInput.MaxStaticDepenetrationVelocity, stepInput.MaxDynamicDepenetrationVelocity));
            }

            kept->Team = null;
            stepInput.IslandSolverCache.SparseCounters.AddTime(
                Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solveStarted);

            if (kept == &temporary)
            {
                temporary.Dispose();
            }

            SparseProblemObserver.Report(&problem, (float*)impulses.GetUnsafePtr(), bounded);

            for (int row = 0; row < rows; row++)
            {
                rhs[row] = impulses[row];
            }

            bodies.Dispose();
            blocks.Dispose();
            couplings.Dispose();
            headers.Dispose();
            setup.ReleaseCompactBodies(sharedIndex != null);
            setup.WorldBodies.Dispose();
            impulses.Dispose();
        }

        /// <summary>Runs the tasks that build, correct and apply the direct solver rows.</summary>
        unsafe struct DirectTaskDispatch : ITeamDispatch
        {

            public void Run(in WorkTeam.Stage stage, int task)
            {
                switch (stage.Kind)
                {
                    case PredictDriftKind:
                        PredictDriftTask(stage.Context, task);
                        break;
                    case ApplyImpulsesKind:
                        ApplyImpulsesTask(stage.Context, task);
                        break;
                    case SetupBodiesKind:
                        SetupBodiesTask(stage.Context, task);
                        break;
                    case SummarizeJacobiansKind:
                        SummarizeJacobiansTask(stage.Context, task);
                        break;
                    case FillBlocksKind:
                        FillBlocksTask(stage.Context, task);
                        break;
                }
            }
        }
    }
}
