using static Unity.Physics.Solver;
using static Unity.Physics.Dynamics.Solver.Forest.ForestSolver;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Configuration;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Numerics.Linear.Solvers.Lowlevel;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Dynamics.Vessels.Structure;
using Unity.Profiling;

namespace Unity.Physics.Dynamics.Solver.Vessels
{
    /// <summary>Build the rows for compiled vessels and solve them with the forest and vessel kernels.</summary>
    internal static partial class VesselIslandSolver
    {
        static readonly ProfilerMarker VesselSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Vessel.Solve");

        /// <summary>
        /// Minimum steps between lock state changes. Yield immediately at break strength or when locking is
        /// disallowed.
        /// </summary>
        const int VesselHoldSteps = 10;

        /// <summary>
        /// Default yield and relock fractions when configured values are nonpositive: yield at half break strength,
        /// relock below a quarter after the hold time.
        /// </summary>
        const double DefaultYieldFraction = 0.5;

        const double DefaultRelockFraction = 0.5;

        /// <summary>
        /// Solve vessel rows and couple contact/non-vessel joint rows through the boundary block.
        /// Treat other moving bodies as separate parts. Compute section velocities, body impulses and vessel edge
        /// loads.
        /// </summary>
        internal static unsafe void SolveVesselIsland(in SparseConstraintSolver.Problem problem, JacobianHeader** headers,
            in NativeList<System.IntPtr> headerList, in NativeList<int> worldBodies, in NativeArray<int> compactBodies, bool bounded,
            bool holdsPositions, int islandIndex, int* islandVessels, int islandVesselCount, in NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, in StepInput stepInput, in JacobianRows jacobianRows, float* recovery,
            bool sparseContacts, WorkTeam* team, bool applyImpulses, float* impulses)
        {
            using ProfilerMarker.AutoScope span = VesselSolveMarker.Auto();

            // If no forest is cached for this island because the cache is full or missing, create one for this solve
            // only.
            ForestIsland* forest = stepInput.IslandSolverCache.ForestOf(islandIndex);
            bool temporary = forest == null;
            if (temporary)
            {
                forest = ForestIsland.Create();
            }

            int* world = (int*)worldBodies.GetUnsafeReadOnlyPtr();
            var run = new ForestRun
            {
                Forest = forest,
                Problem = problem,
                Bounded = bounded,
                Headers = headers,
                VesselMode = true,
                Vessels = stepInput.VesselSet,
                IslandVessels = islandVessels,
                IslandVesselCount = islandVesselCount,
                WorldBodies = world,
                CompactOfWorld = (int*)compactBodies.GetUnsafeReadOnlyPtr(),
                CompactOfWorldLength = compactBodies.Length,
                MotionVelocities = (MotionVelocity*)motionVelocities.GetUnsafeReadOnlyPtr(),
                MotionDatas = (MotionData*)motionDatas.GetUnsafeReadOnlyPtr(),
                Timestep = stepInput.Timestep,
                Team = team,
                Caps = stepInput.IslandSolverCache.Forest.Caps,
                Substep = stepInput.CurrentSubstep,
                RigidBodies = stepInput.Bodies,
                RigidBodyCount = stepInput.BodyCount,
                Substeps = stepInput.NumSubsteps,

                // A forest made for this solve alone ends with the substep, and its sums with it.
                DeferStrutLoads = !temporary && stepInput.NumSubsteps > 1 && stepInput.CurrentSubstep >= 0,
            };

            int builds = forest->StructureBuilds;
            long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            PrepareVesselParts(ref run, stepInput.CurrentSubstep);
            StartVesselRun(ref run);
            long lap = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestStartTime, lap - started);
            stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestFactorTime, run.FactorTicks);
            stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestBoundaryTime, run.BoundaryTicks);
            stepInput.IslandSolverCache.ForestCounters.CountForest(ForestCounters.ForestSolves);
            stepInput.IslandSolverCache.ForestCounters.CountSections(forest->SectionCount, forest->LockedPairs, forest->StructureBuilds != builds, 0, 0);

            var bodyLinear = new NativeArray<float3>(math.max(1, problem.BodyCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var bodyAngular = new NativeArray<float3>(math.max(1, problem.BodyCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var linear = (float3*)bodyLinear.GetUnsafePtr();
            var angular = (float3*)bodyAngular.GetUnsafePtr();
            long solved = SolveVesselBoundary(ref run, problem, headers, headerList, worldBodies, compactBodies, holdsPositions,
                motionVelocities, motionDatas, stepInput, jacobianRows, recovery, sparseContacts, team, applyImpulses, impulses, linear,
                angular);
            WriteVesselLoads(ref run, stepInput.CurrentSubstep, stepInput.NumSubsteps);
            FinishStrutLoads(ref run);
            stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestFinishTime,
                Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solved);
            bodyLinear.Dispose();
            bodyAngular.Dispose();
            run.Dispose();
            if (temporary)
            {
                ForestIsland.Destroy(forest);
            }
        }

        /// <summary>
        /// Solve the vessel boundary and recover body impulses. Correct predicted positional drift with a second
        /// solve when needed.
        /// Optionally apply impulses and deep-contact push-out. Return first-solve completion time for profiling.
        /// </summary>
        static unsafe long SolveVesselBoundary(ref ForestRun run, in SparseConstraintSolver.Problem problem, JacobianHeader** headers,
            in NativeList<System.IntPtr> headerList, in NativeList<int> worldBodies, in NativeArray<int> compactBodies, bool holdsPositions,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, in StepInput stepInput,
            in JacobianRows jacobianRows, float* recovery, bool sparseContacts, WorkTeam* team, bool applyImpulses, float* impulses,
            float3* linear, float3* angular)
        {
            long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;

            // If the boundary ran out of pivoting rounds, SolveForest returns false, but its impulses are still inside
            // their bounds and near the answer, so the vessel applies them. BoundaryCounters counts such solves.
            SolveForest(ref run, problem.RightHandSide, impulses);
            long solved = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestSolveTime, solved - started);
            SettleCompiledVessel(ref run, problem, impulses, linear, angular, false);
            stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestSettleTime,
                Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solved);

            // Correct boundary positional drift only; vessel rows already include pose errors each substep.
            if (holdsPositions && problem.RowCount > 0)
            {
                var drift = new NativeArray<float>(problem.RowCount, Allocator.Temp);
                float* driftPtr = (float*)drift.GetUnsafePtr();
                if (AimAtPredictedDrift(problem, null, linear, angular, (int*)worldBodies.GetUnsafeReadOnlyPtr(), headerList,
                    impulses, motionVelocities, motionDatas, stepInput.Timestep, jacobianRows, driftPtr, team, true))
                {
                    for (int row = 0; row < problem.RowCount; row++)
                    {
                        problem.RightHandSide[row] += driftPtr[row];
                    }

                    SolveForest(ref run, problem.RightHandSide, impulses);
                    SettleCompiledVessel(ref run, problem, impulses, linear, angular, false);
                }

                drift.Dispose();
            }

            if (applyImpulses)
            {
                long applying = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                ApplySparseImpulses(problem, null, linear, angular, (int*)worldBodies.GetUnsafeReadOnlyPtr(), headerList, impulses,
                    motionVelocities, motionDatas, sparseContacts, team);
                long pushing = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                if (recovery != null)
                {
                    SolvePushOut(problem, headers, worldBodies, recovery, motionVelocities, motionDatas, stepInput.Timestep,
                        math.max(stepInput.MaxStaticDepenetrationVelocity, stepInput.MaxDynamicDepenetrationVelocity),
                        (ForestRun*)UnsafeUtility.AddressOf(ref run));
                }

                BoundaryCounters.AddFinish(pushing - applying, Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - pushing);
            }

            return solved;
        }

        static readonly ProfilerMarker VesselFusedMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Vessel.Fused");

        /// <summary>Whether a body of the world is kinematic: it has a motion, and no mass or inertia.</summary>
        static unsafe bool IsKinematicBody(int body, MotionVelocity* motions, int motionCount)
        {
            return (uint)body < (uint)motionCount && motions[body].IsKinematic;
        }

        /// <summary>
        /// Solve all substeps for an island of vessel parts in one job, retaining state by first vessel ID.
        /// Update boundary rows, solve vessel constraints, compute loads and integrate bodies with shared kernels.
        /// </summary>
        internal static unsafe void SolveFusedVesselIsland(in DirectIsland island, IslandHeader* islandHeaders, int* islandVessels,
            NativeArray<MotionVelocity> motionVelocities, NativeArray<MotionData> motionDatas, NativeArray<RigidBody> rigidBodies,
            StepInput stepInput, WorkTeam* team)
        {
            using ProfilerMarker.AutoScope span = VesselFusedMarker.Auto();
            VesselSet set = stepInput.VesselSet;
            var motions = (MotionVelocity*)motionVelocities.GetUnsafePtr();
            int motionCount = motionVelocities.Length;

            // The island's state is kept by the ID of its first vessel, not by the island's index or its first body. The
            // job that marked the fused islands has already given the island its forest.
            ForestIsland* forest = stepInput.IslandSolverCache.FindFusedForest(set.Vessels[islandVessels[0]].Id);
            int islandVesselCount = island.VesselCount;
            bool temporary = forest == null;
            if (temporary)
            {
                forest = ForestIsland.Create();
            }

            // Reuse layout only when vessels, versions, counts and body numbering match.
            // Imported lock states require a fresh layout.
            bool known = forest->VesselMode && !forest->TakesStates && forest->VesselSignature.Length == 1 + 3 * islandVesselCount
                && forest->VesselSignature[0] == (ulong)islandVesselCount;
            int closureStart = 0;
            for (int v = 0; known && v < islandVesselCount; v++)
            {
                VesselRecord vessel = set.Vessels[islandVessels[v]];
                known = forest->VesselSignature[1 + 3 * v] == vessel.Id && forest->VesselSignature[2 + 3 * v] == vessel.StructureVersion
                    && forest->VesselSignature[3 + 3 * v] == (((ulong)(uint)vessel.PartCount << 32) | (uint)vessel.ClosureCount);

                // The signature leaves out where the vessel's parts and closures start in the set. Those move when an
                // earlier vessel in the set changes, so check them against the kept tables.
                int partStart = known && v < forest->VesselPartStart.Length ? forest->VesselPartStart[v] : -1;
                known = known && partStart >= 0 && (vessel.PartCount == 0
                    || (partStart < forest->PartSetPart.Length && forest->PartSetPart[partStart] == vessel.FirstPart));
                known = known && (vessel.ClosureCount == 0
                    || (closureStart < forest->ClosureSetIndex.Length && forest->ClosureSetIndex[closureStart] == vessel.FirstClosure));
                closureStart += vessel.ClosureCount;
            }

            // The kinematic parts' bodies. The island's moving bodies leave them out, so the stages below don't move
            // them, and the substeps' jobs skip the fused islands. So this job integrates them at the end of each
            // substep.
            var kinematic = new NativeList<int>(Allocator.Temp);
            int numbered = 0;
            for (int v = 0; known && v < islandVesselCount; v++)
            {
                VesselRecord vessel = set.Vessels[islandVessels[v]];
                for (int part = vessel.FirstPart; known && part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    int body = Moving(set.PartBodies[part], motions, motionCount);
                    if (body >= 0)
                    {
                        known = numbered < forest->FusedWorldBodies.Length && forest->FusedWorldBodies[numbered] == body;
                        numbered++;
                    }
                    else if (IsKinematicBody(set.PartBodies[part], motions, motionCount))
                    {
                        kinematic.Add(set.PartBodies[part]);
                    }
                }
            }

            known &= numbered == forest->FusedWorldBodies.Length;

            // The island writes its numbering into the index of bodies that all the islands of the step share, and sets
            // its entries to -1 again when it is done. Without a shared index, the island fills one of its own.
            int* compactOfWorld = stepInput.IslandSolverCache.CompactIndex(motionCount);
            NativeArray<int> ownIndex = default;
            if (compactOfWorld == null)
            {
                ownIndex = new NativeArray<int>(math.max(1, motionCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                UnsafeUtility.MemSet(ownIndex.GetUnsafePtr(), 0xFF, ownIndex.Length * (long)sizeof(int));
                compactOfWorld = (int*)ownIndex.GetUnsafePtr();
            }

            if (known)
            {
                for (int compact = 0; compact < forest->FusedWorldBodies.Length; compact++)
                {
                    compactOfWorld[forest->FusedWorldBodies[compact]] = compact;
                }
            }
            else
            {
                // Number the moving parts again, one vessel after another.
                forest->FusedWorldBodies.Clear();
                kinematic.Clear();
                for (int v = 0; v < islandVesselCount; v++)
                {
                    VesselRecord vessel = set.Vessels[islandVessels[v]];
                    for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                    {
                        int body = Moving(set.PartBodies[part], motions, motionCount);
                        if (body >= 0 && compactOfWorld[body] < 0)
                        {
                            compactOfWorld[body] = forest->FusedWorldBodies.Length;
                            forest->FusedWorldBodies.Add(body);
                        }
                        else if (body < 0 && IsKinematicBody(set.PartBodies[part], motions, motionCount))
                        {
                            kinematic.Add(set.PartBodies[part]);
                        }
                    }
                }
            }

            // Two parts can share a body. Integrate each kinematic body once.
            if (kinematic.Length > 1)
            {
                kinematic.Sort();
                int unique = 1;
                for (int k = 1; k < kinematic.Length; k++)
                {
                    if (kinematic[k] != kinematic[unique - 1])
                    {
                        kinematic[unique++] = kinematic[k];
                    }
                }

                kinematic.Resize(unique, NativeArrayOptions.UninitializedMemory);
            }

            // All the island's moving bodies are vessel parts, so the numbering above numbers every body that the island's
            // contacts and joints reach.
            NativeArray<int> compactBodies = View(compactOfWorld, motionCount);
            int headerCount = island.HeaderEnd - island.FirstHeader;
            int solveCount = island.SolveEnd - island.FirstHeader;
            var headerRows = new NativeArray<int>(math.max(1, solveCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var bodyLinear = new NativeArray<float3>(math.max(1, forest->FusedWorldBodies.Length), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var bodyAngular = new NativeArray<float3>(math.max(1, forest->FusedWorldBodies.Length), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var linear = (float3*)bodyLinear.GetUnsafePtr();
            var angular = (float3*)bodyAngular.GetUnsafePtr();

            // Whether the last substep's stage that moved the bodies also filled the parts' poses for this substep.
            bool posesFilled = false;
            for (int substep = 0; substep < stepInput.NumSubsteps; substep++)
            {
                StepInput substepInput = stepInput;
                substepInput.CurrentSubstep = substep;

                // The Jacobians were built from the poses at the start of the step. After the first substep, update
                // them from the current poses, as the substeps' jobs do for the other islands.
                var tasks = new VesselTasks
                {
                    Headers = islandHeaders,
                    HeaderRows = (int*)headerRows.GetUnsafePtr(),
                    MotionVelocities = motionVelocities,
                    MotionDatas = motionDatas,
                    RigidBodies = rigidBodies,
                    StepInput = substepInput,
                    SparseContacts = island.ContactPairs > 0,
                };
                if (substep > 0 && headerCount > 0)
                {
                    tasks.Phase = VesselTasks.UpdateJacobians;
                    tasks.Count = headerCount;
                    WorkTeam.RunAll<VesselTaskDispatch>(team, &tasks, IslandTaskKind.ForestVesselKind, (headerCount + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
                }

                int rows = 0;
                for (int h = 0; h < solveCount; h++)
                {
                    headerRows[h] = rows;
                    rows += GetConstraintRows(ref *islandHeaders[h].Header);
                }

                // The boundary rows: one block of floats for the Jacobian, epsilon, the bounds, the right-hand side and
                // the push-out of every row.
                var rowValues = new NativeArray<float>(math.max(1, rows * (SparseConstraintSolver.ColumnsPerRow + 6)), Allocator.Temp,
                    NativeArrayOptions.UninitializedMemory);
                var flags = new NativeArray<LCP.MLCPIndexFlag>(math.max(1, rows), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                float* values = (float*)rowValues.GetUnsafePtr();
                var buffers = new IslandRowBuffers
                {
                    Jacobian = values,
                    Epsilon = values + rows * SparseConstraintSolver.ColumnsPerRow,
                    Lower = values + rows * (SparseConstraintSolver.ColumnsPerRow + 1),
                    Upper = values + rows * (SparseConstraintSolver.ColumnsPerRow + 2),
                    RightHandSide = values + rows * (SparseConstraintSolver.ColumnsPerRow + 3),
                    Recovery = values + rows * (SparseConstraintSolver.ColumnsPerRow + 4),
                    Flags = (LCP.MLCPIndexFlag*)flags.GetUnsafePtr(),
                    Rows = rows,
                };

                var setup = new SparseSetup
                {
                    Blocks = new NativeList<SparseConstraintSolver.ConstraintBlock>(Allocator.Temp),
                    Couplings = new NativeList<BoundCoupling>(island.ContactPairs, Allocator.Temp),
                    Headers = new NativeList<System.IntPtr>(Allocator.Temp),
                    WorldBodies = new NativeList<int>(Allocator.Temp),
                    CompactBodies = compactBodies,
                };

                var problem = new SparseConstraintSolver.Problem { BodyCount = forest->FusedWorldBodies.Length };
                if (rows > 0)
                {
                    tasks.Phase = VesselTasks.BuildRows;
                    tasks.Count = solveCount;
                    tasks.Rows = buffers;
                    tasks.Regularization = RowRegularization.Of(substepInput);
                    WorkTeam.RunAll<VesselTaskDispatch>(team, &tasks, IslandTaskKind.ForestVesselKind, (solveCount + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
                    GatherSparseBlocks(ref setup, islandHeaders, (int*)headerRows.GetUnsafeReadOnlyPtr(), solveCount, rows, motionVelocities, team);

                    // The solve that pushes bodies out of deep contacts needs the island's bodies in a list.
                    setup.WorldBodies.AddRange(forest->FusedWorldBodies.Ptr, forest->FusedWorldBodies.Length);
                    problem.BlockCount = setup.Blocks.Length;
                    problem.Blocks = setup.Blocks.GetUnsafePtr();
                    problem.RowCount = rows;
                    problem.Jacobian = buffers.Jacobian;
                    problem.Epsilon = buffers.Epsilon;
                    problem.RightHandSide = buffers.RightHandSide;
                    problem.Lower = buffers.Lower;
                    problem.Upper = buffers.Upper;
                    problem.InitialStates = (byte*)buffers.Flags;
                    problem.CouplingCount = setup.Couplings.Length;
                    problem.Couplings = setup.Couplings.GetUnsafePtr();
                }

                var run = new ForestRun
                {
                    Forest = forest,
                    Problem = problem,
                    Bounded = tasks.Bounded != 0,
                    Headers = (JacobianHeader**)setup.Headers.GetUnsafeReadOnlyPtr(),
                    // After the first substep, the layout is the one that the first substep built or kept.
                    KnownLayout = known || substep > 0,
                    VesselMode = true,
                    Vessels = set,
                    IslandVessels = islandVessels,
                    IslandVesselCount = islandVesselCount,
                    WorldBodies = forest->FusedWorldBodies.Ptr,
                    CompactOfWorld = compactOfWorld,
                    CompactOfWorldLength = motionCount,
                    MotionVelocities = motions,
                    MotionDatas = (MotionData*)motionDatas.GetUnsafeReadOnlyPtr(),
                    Timestep = stepInput.Timestep,
                    Team = team,
                    Caps = stepInput.IslandSolverCache.Forest.Caps,
                    Substep = substep,
                    RigidBodies = stepInput.Bodies,
                    RigidBodyCount = stepInput.BodyCount,
                    Substeps = stepInput.NumSubsteps,
                    DeferStrutLoads = stepInput.NumSubsteps > 1,
                };

                int builds = forest->StructureBuilds;
                long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                PrepareVesselParts(ref run, substep, posesFilled);
                StartVesselRun(ref run);
                long lap = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestStartTime, lap - started);
                stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestFactorTime, run.FactorTicks);
                stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestBoundaryTime, run.BoundaryTicks);
                stepInput.IslandSolverCache.ForestCounters.CountForest(ForestCounters.ForestSolves);
                stepInput.IslandSolverCache.ForestCounters.CountSections(forest->SectionCount, forest->LockedPairs, forest->StructureBuilds != builds, 0, 0);

                // The lanes move or integrate the bodies and write the loads in one stage: its items are the parts and
                // then the closures, as in WriteVesselLoads.
                var bodyTasks = new VesselTasks
                {
                    Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                    Count = forest->Parts.Length + forest->Closures.Length,
                    VesselParts = forest->Parts.Length,
                    Substep = substep,
                    Substeps = stepInput.NumSubsteps,
                    Linear = linear,
                    Angular = angular,
                    MotionVelocities = motionVelocities,
                    MotionDatas = motionDatas,
                };

                long solved;
                if (rows > 0)
                {
                    // The same boundary solve as the ordinary vessel island's, and then the lanes integrate the bodies.
                    var impulses = new NativeArray<float>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    solved = SolveVesselBoundary(ref run, problem, (JacobianHeader**)setup.Headers.GetUnsafeReadOnlyPtr(), setup.Headers,
                        setup.WorldBodies, compactBodies, setup.HoldsPositions, motionVelocities, motionDatas, substepInput,
                        new JacobianRows(buffers.Jacobian, rows), buffers.Recovery, island.ContactPairs > 0, team, true,
                        (float*)impulses.GetUnsafePtr(), linear, angular);
                    bodyTasks.Phase = VesselTasks.IntegrateBodiesAndWriteLoads;
                    WorkTeam.RunAll<VesselTaskDispatch>(team, &bodyTasks, IslandTaskKind.ForestVesselKind,
                        (bodyTasks.Count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
                    impulses.Dispose();
                    posesFilled = false;
                }
                else
                {
                    // With no boundary rows, the sections' velocities are the ones from the vessels' own rows.
                    UnsafeUtility.MemCpy(run.Velocity.GetUnsafePtr(), run.BaseVelocity.GetUnsafeReadOnlyPtr(), run.Velocity.Length * (long)sizeof(double));
                    solved = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestSolveTime, solved - lap);

                    ref VesselContext context = ref forest->Context;
                    UnsafeUtility.MemCpy(context.Solution, run.Velocity.GetUnsafeReadOnlyPtr(), 6L * forest->SectionCount * sizeof(double));
                    UnsafeUtility.MemClear(forest->KernelClosureLoads.Ptr, forest->KernelClosureLoads.Length * (long)sizeof(VesselEdgeLoad));
                    RunVesselStage(ref run, VesselTasks.LinkImpulses, forest->LinkCount);
                    FinishVesselSections(ref run);
                    // After integrating a body, fill its part motion in the same task so that the next substep
                    // can skip that stage.
                    bodyTasks.Phase = VesselTasks.MoveBodiesAndWriteLoads;
                    bodyTasks.FillNext = substep < stepInput.NumSubsteps - 1;
                    WorkTeam.RunAll<VesselTaskDispatch>(team, &bodyTasks, IslandTaskKind.ForestVesselKind,
                        (bodyTasks.Count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
                    posesFilled = bodyTasks.FillNext;
                    stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestSettleTime,
                        Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solved);
                }

                FinishStrutLoads(ref run);
                for (int k = 0; k < kinematic.Length; k++)
                {
                    Integrator.IntegrateMotion(kinematic[k], motionDatas, motionVelocities, stepInput.Timestep);
                }

                stepInput.IslandSolverCache.ForestCounters.AddTime(ForestCounters.ForestFinishTime,
                    Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solved);
                run.Dispose();
                setup.Blocks.Dispose();
                setup.Couplings.Dispose();
                setup.Headers.Dispose();
                setup.WorldBodies.Dispose();
                rowValues.Dispose();
                flags.Dispose();
            }

            headerRows.Dispose();
            bodyLinear.Dispose();
            bodyAngular.Dispose();
            kinematic.Dispose();
            for (int compact = 0; compact < forest->FusedWorldBodies.Length; compact++)
            {
                compactOfWorld[forest->FusedWorldBodies[compact]] = -1;
            }

            if (ownIndex.IsCreated)
            {
                ownIndex.Dispose();
            }

            if (temporary)
            {
                ForestIsland.Destroy(forest);
            }
        }

        /// <summary>
        /// Integrate a fused part body after boundary impulses. Each body is owned by exactly one part.
        /// </summary>
        static unsafe void IntegrateVesselBody(ref ForestRun run, VesselTasks* tasks, int part)
        {
            int body = run.Forest->PartBody[part];
            if (body >= 0 && run.Forest->BodyPart[body] == part)
            {
                Integrator.IntegrateMotion(run.WorldBodies[body], tasks->MotionDatas, tasks->MotionVelocities, run.Timestep);
            }
        }

        /// <summary>
        /// For the fused vessel solve: updates or builds the rows of one of the island's Jacobians, as the substeps'
        /// jobs do for the other islands.
        /// </summary>
        static unsafe void FusedJacobianTask(VesselTasks* tasks, int index)
        {
            ref JacobianHeader header = ref *tasks->Headers[index].Header;
            if (tasks->Phase == VesselTasks.UpdateJacobians)
            {
                UpdateJacobian(ref header, tasks->MotionDatas, tasks->MotionVelocities, tasks->RigidBodies, tasks->StepInput);
                return;
            }

            int startRow = tasks->HeaderRows[index];
            int jacRows = GetConstraintRows(ref header);
            if (jacRows == 0)
            {
                return;
            }

            // Every lane that sees a bounded row stores the same value, so the lanes need no lock.
            if (BuildHeaderRows(ref header, startRow, jacRows, tasks->Rows, tasks->MotionVelocities, tasks->MotionDatas, tasks->StepInput,
                tasks->SparseContacts, &tasks->Regularization) && tasks->Bounded == 0)
            {
                tasks->Bounded = 1;
            }
        }

        /// <summary>
        /// Apply section motion as a body impulse, then integrate over the fused substep.
        /// </summary>
        static unsafe void MoveVesselBody(ref ForestRun run, VesselTasks* tasks, int part)
        {
            int body = run.Forest->PartBody[part];
            if (body < 0)
            {
                return;
            }

            PartImpulse(ref run, part, tasks->Linear, tasks->Angular, false);
            int worldBody = run.WorldBodies[body];
            ref MotionVelocity velocity = ref UnsafeUtility.ArrayElementAsRef<MotionVelocity>(tasks->MotionVelocities.GetUnsafePtr(), worldBody);
            velocity.ApplyLinearImpulse(tasks->Linear[body]);
            quaternion motionFromWorld = math.conjugate(run.MotionDatas[worldBody].WorldFromMotion.rot);
            velocity.ApplyAngularImpulse(math.rotate(motionFromWorld, tasks->Angular[body]));
            Integrator.IntegrateMotion(worldBody, tasks->MotionDatas, tasks->MotionVelocities, run.Timestep);
        }

        /// <summary>
        /// Arrange vessel parts first, then other moving bodies. Rebuild when layout changes while preserving
        /// matching vessel states.
        /// Test locks on the first substep. posesFilled permits reuse of poses already filled by the preceding
        /// substep.
        /// </summary>
        static unsafe void PrepareVesselParts(ref ForestRun run, int substep, bool posesFilled = false)
        {
            ForestIsland* forest = run.Forest;
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            VesselSet set = run.Vessels;
            int bodies = problem.BodyCount;
            if (run.KnownLayout)
            {
                PrepareKnownVesselParts(ref run, substep, posesFilled);
                return;
            }

            // Record which bodies in the compact array are vessel parts, and build a signature for the island's layout.
            var bodyIsPart = new NativeArray<byte>(math.max(1, bodies), Allocator.Temp);
            var signature = new NativeList<ulong>(4 + 4 * run.IslandVesselCount, Allocator.Temp);
            signature.Add((ulong)run.IslandVesselCount);
            int partCount = 0, closureCount = 0;
            for (int i = 0; i < run.IslandVesselCount; i++)
            {
                VesselRecord vessel = set.Vessels[run.IslandVessels[i]];
                signature.Add(vessel.Id);
                signature.Add(vessel.StructureVersion);
                signature.Add(((ulong)(uint)vessel.PartCount << 32) | (uint)vessel.ClosureCount);
                partCount += vessel.PartCount;
                closureCount += vessel.ClosureCount;
            }

            int vesselPartCount = partCount;

            for (int i = 0; i < run.IslandVesselCount; i++)
            {
                VesselRecord vessel = set.Vessels[run.IslandVessels[i]];
                for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    int body = CompactOf(run, set.PartBodies[part]);
                    if (body >= 0)
                    {
                        bodyIsPart[body] = 1;
                    }
                }
            }

            for (int body = 0; body < bodies; body++)
            {
                if (bodyIsPart[body] == 0)
                {
                    signature.Add(0x8000000000000000UL | (uint)run.WorldBodies[body]);
                    partCount++;
                }
            }

            bool same = forest->VesselMode && !forest->TakesStates && forest->VesselSignature.Length == signature.Length;
            for (int i = 0; same && i < signature.Length; i++)
            {
                same = forest->VesselSignature[i] == signature[i];
            }

            bool rebuild = !same;
            if (!same)
            {
                CarryVesselStates(forest, signature, run, partCount, closureCount);
            }

            // The edges, the closures and the mapping between parts and bodies stay the same for every substep of a step,
            // so fill them only on the first substep, or when the island's layout has changed.
            bool firstSubstep = substep <= 0;
            bool refill = !same || firstSubstep;
            if (refill)
            {
                forest->Parts.Resize(partCount);
                forest->Edges.Resize(partCount);
                forest->Closures.Resize(closureCount);
                forest->PartBody.Resize(partCount);
                forest->BodyPart.Resize(math.max(1, bodies));
                UnsafeUtility.MemSet(forest->BodyPart.Ptr, 0xFF, forest->BodyPart.Length * (long)sizeof(int));

                // Refresh vessel-set part and closure indices on every refill; their offsets can move without
                // changing the signature.
                {
                    forest->PartSetPart.Resize(partCount);
                    forest->PartVesselStart.Resize(partCount);
                    forest->ClosureSetIndex.Resize(math.max(1, closureCount));
                    forest->ClosureVesselStart.Resize(math.max(1, closureCount));
                    int next = 0, nextClosure = 0;
                    for (int i = 0; i < run.IslandVesselCount; i++)
                    {
                        VesselRecord vessel = set.Vessels[run.IslandVessels[i]];
                        int start = next;
                        for (int k = 0; k < vessel.PartCount; k++)
                        {
                            forest->PartSetPart[next] = vessel.FirstPart + k;
                            forest->PartVesselStart[next++] = start;
                        }

                        for (int c = vessel.FirstClosure; c < vessel.FirstClosure + vessel.ClosureCount; c++)
                        {
                            forest->ClosureSetIndex[nextClosure] = c;
                            forest->ClosureVesselStart[nextClosure++] = start;
                        }
                    }
                }

                // The team's lanes copy the edges and closures, test the locks, and map the parts to their bodies.
                var tasks = new VesselTasks
                {
                    Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                    Phase = VesselTasks.FillEdges,
                    Count = vesselPartCount + closureCount,
                    VesselParts = vesselPartCount,
                    Substep = substep,
                };
                WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind,
                    (tasks.Count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
                rebuild |= tasks.Rebuild != 0;

                // Each moving body that is not in a vessel is a separate part after the vessels' parts.
                int extra = vesselPartCount;
                for (int body = 0; body < bodies; body++)
                {
                    if (bodyIsPart[body] == 0)
                    {
                        int part = extra++;
                        forest->PartBody[part] = body;
                        forest->PartSetPart[part] = -1;
                        forest->PartVesselStart[part] = -1;
                        forest->BodyPart[body] = part;
                        forest->Edges[part] = new VesselEdge { Parent = -1, State = VesselEdgeState.Yielding };
                    }
                }
            }

            // The parts' poses and velocities change every substep. The team's lanes fill them from the bodies by
            // ranges of parts.
            RunVesselStage(ref run, VesselTasks.FillParts, partCount);

            if (rebuild)
            {
                BuildVesselStructure(forest);
            }

            // The compact body indices can change between steps, so rebuild the mapping from bodies to sections
            // whenever the parts were filled again.
            if (refill)
            {
                forest->BodySection.Resize(math.max(1, bodies));
                for (int body = 0; body < bodies; body++)
                {
                    int part = forest->BodyPart[body];
                    forest->BodySection[body] = part >= 0 ? forest->Structure.PartSection[part] : -1;
                }
            }

            bodyIsPart.Dispose();
            signature.Dispose();
        }

        /// <summary>
        /// Prepare an unchanged vessel-only layout without renumbering.
        /// Copy edges and test locks on the first substep; fill motion each substep unless posesFilled is true.
        /// </summary>
        static unsafe void PrepareKnownVesselParts(ref ForestRun run, int substep, bool posesFilled)
        {
            ForestIsland* forest = run.Forest;
            int partCount = forest->Parts.Length;
            int closureCount = forest->Closures.Length;
            bool rebuild = false;
            if (substep <= 0)
            {
                var tasks = new VesselTasks
                {
                    Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                    Phase = VesselTasks.FillEdges,
                    Count = partCount + closureCount,
                    VesselParts = partCount,
                    Substep = substep,
                };
                WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind,
                    (tasks.Count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
                rebuild = tasks.Rebuild != 0;
            }

            // The parts' masses and inertias do not change within a step, so after the first substep the lanes fill
            // only the poses and velocities.
            if (substep <= 0 || !posesFilled)
            {
                RunVesselStage(ref run, substep <= 0 ? VesselTasks.FillParts : VesselTasks.FillPoses, partCount);
            }

            if (rebuild)
            {
                BuildVesselStructure(forest);

                // A change of lock state can change the sections, so map the bodies to their sections again.
                for (int body = 0; body < forest->BodySection.Length && body < forest->BodyPart.Length; body++)
                {
                    int part = forest->BodyPart[body];
                    forest->BodySection[body] = part >= 0 ? forest->Structure.PartSection[part] : -1;
                }
            }
        }

        /// <summary>
        /// Copy a part edge and body mapping from the vessel set. On the first substep, test locking.
        /// Return whether the structure must be rebuilt.
        /// </summary>
        static unsafe bool FillVesselEdge(ref ForestRun run, int part, bool firstSubstep)
        {
            ForestIsland* forest = run.Forest;
            VesselSet set = run.Vessels;
            int setPart = forest->PartSetPart[part];

            // With a known layout, the numbering of the bodies is the same, so the mapping between parts and bodies
            // is already right.
            if (!run.KnownLayout)
            {
                int body = CompactOf(run, set.PartBodies[setPart]);
                forest->PartBody[part] = body;
                if (body >= 0)
                {
                    forest->BodyPart[body] = part;
                }
            }

            bool changed = false;
            VesselEdge edge = set.Edges[setPart];
            edge.Parent = edge.Parent >= 0 ? forest->PartVesselStart[part] + edge.Parent : -1;
            if (edge.Parent >= 0 && firstSubstep && TestLock(forest, part, edge))
            {
                changed = true;
            }

            if (edge.Parent >= 0 && forest->PartLocked[part] != 0)
            {
                edge.State = VesselEdgeState.Locked;
            }
            else if (edge.State == VesselEdgeState.Locked)
            {
                edge.State = VesselEdgeState.Yielding;
            }

            forest->Edges[part] = edge;
            return changed;
        }

        /// <summary>Copies one closure from the vessel set, with its parts numbered as in the island.</summary>
        static unsafe void FillVesselClosure(ref ForestRun run, int closure)
        {
            ForestIsland* forest = run.Forest;
            VesselClosure copy = run.Vessels.Closures[forest->ClosureSetIndex[closure]];
            int start = forest->ClosureVesselStart[closure];
            copy.PartA += start;
            copy.PartB += start;
            forest->Closures[closure] = copy;
        }

        /// <summary>
        /// Returns the compact index of a body in the dynamics world, or -1 if the body is not in this island.
        /// </summary>
        static unsafe int CompactOf(in ForestRun run, int worldBody) =>
            worldBody >= 0 && worldBody < run.CompactOfWorldLength ? run.CompactOfWorld[worldBody] : -1;

        /// <summary>
        /// Supply world body pointers to the step without safety checks; writer jobs may still be running during
        /// scheduling.
        /// </summary>
        internal static unsafe void SetBodies(ref StepInput stepInput, in PhysicsWorld world)
        {
            stepInput.BodyCount = world.NumBodies;
            stepInput.Bodies = world.NumBodies > 0
                ? (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies)
                : null;
        }

        /// <summary>
        /// Converts a body to a kernel part with mass, principal inertia, and a pose and velocity in world space.
        /// </summary>
        static unsafe VesselPart PartOf(in ForestRun run, int compact)
        {
            if (compact < 0)
            {
                return new VesselPart { Mass = 1.0, Inertia = 1.0, Orientation = new double4(0, 0, 0, 1) };
            }

            int body = run.WorldBodies[compact];
            MotionVelocity motion = run.MotionVelocities[body];
            RigidTransform pose = run.MotionDatas[body].WorldFromMotion;
            return new VesselPart
            {
                Mass = 1.0 / math.max((double)motion.InverseMass, 1e-12),
                Inertia = 1.0 / math.max((double3)motion.InverseInertia, 1e-12),
                Position = pose.pos,
                Orientation = pose.rot.value,
                LinearVelocity = motion.LinearVelocity,
                AngularVelocity = math.rotate(pose.rot, motion.AngularVelocity),
            };
        }

        /// <summary>
        /// Fill part pose and velocity, clear external impulses and preserve first-substep mass/inertia.
        /// </summary>
        static unsafe void FillPartPose(in ForestRun run, int part)
        {
            int compact = run.Forest->PartBody[part];
            if (compact < 0)
            {
                run.Forest->Parts[part] = PartOf(run, compact);
                return;
            }

            VesselPart* p = run.Forest->Parts.Ptr + part;
            int body = run.WorldBodies[compact];
            MotionVelocity motion = run.MotionVelocities[body];
            RigidTransform pose = run.MotionDatas[body].WorldFromMotion;
            p->Position = pose.pos;
            p->Orientation = pose.rot.value;
            p->LinearVelocity = motion.LinearVelocity;
            p->AngularVelocity = math.rotate(pose.rot, motion.AngularVelocity);
            p->LinearImpulse = 0.0;
            p->AngularImpulse = 0.0;
        }

        /// <summary>
        /// Finds where a vessel's parts start in a forest's lock states, if the forest's layout holds the vessel with
        /// the same structure version.
        /// </summary>
        internal static unsafe bool FindVesselStates(ForestIsland* forest, in VesselRecord vessel, out int start)
        {
            start = -1;
            if (!forest->VesselMode || forest->VesselSignature.Length == 0)
            {
                return false;
            }

            int count = (int)forest->VesselSignature[0];
            for (int i = 0; i < count && 3 + 3 * i < forest->VesselSignature.Length && i < forest->VesselPartStart.Length; i++)
            {
                if (forest->VesselSignature[1 + 3 * i] == vessel.Id && forest->VesselSignature[2 + 3 * i] == vessel.StructureVersion)
                {
                    start = forest->VesselPartStart[i];
                    return start >= 0 && start + vessel.PartCount <= forest->PartLocked.Length;
                }
            }

            return false;
        }

        /// <summary>
        /// Rebuild part layout while preserving states for matching vessel IDs and structure versions.
        /// Prefer states imported from other forests; new eligible edges start locked.
        /// </summary>
        static unsafe void CarryVesselStates(ForestIsland* forest, NativeList<ulong> signature, in ForestRun run, int partCount,
            int closureCount)
        {
            VesselSet set = run.Vessels;

            // Record each cached vessel's ID, structure version and index of its first part in the previous layout.
            var old = new NativeHashMap<ulong, int2>(math.max(1, forest->VesselPartStart.Length), Allocator.Temp);
            if (forest->VesselMode)
            {
                int oldCount = forest->VesselSignature.Length > 0 ? (int)forest->VesselSignature[0] : 0;
                for (int i = 0; i < oldCount && 3 * i + 2 < forest->VesselSignature.Length; i++)
                {
                    old.TryAdd(forest->VesselSignature[1 + 3 * i], new int2((int)forest->VesselSignature[2 + 3 * i], forest->VesselPartStart[i]));
                }
            }

            var locked = new NativeArray<byte>(math.max(1, partCount), Allocator.Temp);
            var held = new NativeArray<int>(math.max(1, partCount), Allocator.Temp);
            var ratio = new NativeArray<float>(math.max(1, partCount), Allocator.Temp);
            forest->VesselPartStart.Resize(run.IslandVesselCount);
            int next = 0;
            for (int i = 0; i < run.IslandVesselCount; i++)
            {
                VesselRecord vessel = set.Vessels[run.IslandVessels[i]];
                forest->VesselPartStart[i] = next;
                int taken = -1;
                for (int t = 0; forest->TakesStates && t < forest->TakenIds.Length; t++)
                {
                    taken = forest->TakenIds[t] == vessel.Id && forest->TakenVersions[t] == vessel.StructureVersion ? forest->TakenStarts[t] : taken;
                }

                bool kept = old.TryGetValue(vessel.Id, out int2 before) && (uint)before.x == vessel.StructureVersion;
                for (int k = 0; k < vessel.PartCount; k++)
                {
                    int part = next + k;
                    if (taken >= 0)
                    {
                        locked[part] = forest->TakenLocked[taken + k];
                        held[part] = forest->TakenHeld[taken + k];
                        ratio[part] = forest->TakenRatio[taken + k];
                    }
                    else if (kept)
                    {
                        locked[part] = forest->PartLocked[before.y + k];
                        held[part] = forest->PartHeld[before.y + k];
                        ratio[part] = forest->PartRatio[before.y + k];
                    }
                    else
                    {
                        VesselEdge edge = set.Edges[vessel.FirstPart + k];
                        locked[part] = (byte)(edge.Parent >= 0 && edge.MayLock ? 1 : 0);
                        held[part] = VesselHoldSteps;
                        ratio[part] = 0.0f;
                    }
                }

                next += vessel.PartCount;
            }

            forest->PartLocked.Resize(math.max(1, partCount));
            forest->PartHeld.Resize(math.max(1, partCount));
            forest->PartRatio.Resize(math.max(1, partCount));
            for (int part = 0; part < partCount; part++)
            {
                forest->PartLocked[part] = part < next ? locked[part] : (byte)0;
                forest->PartHeld[part] = part < next ? held[part] : 0;
                forest->PartRatio[part] = part < next ? ratio[part] : 0.0f;
            }

            forest->VesselSignature.Clear();
            for (int i = 0; i < signature.Length; i++)
            {
                forest->VesselSignature.Add(signature[i]);
            }

            forest->VesselMode = true;
            forest->TakesStates = false;
            forest->TakenIds.Clear();
            forest->TakenVersions.Clear();
            forest->TakenStarts.Clear();
            forest->TakenLocked.Clear();
            forest->TakenHeld.Clear();
            forest->TakenRatio.Clear();

            // Invalidate the signature for the forest built from supplied Jacobian rows, so it is rebuilt if the island
            // returns to that solver path.
            forest->Signature.Clear();
            forest->Signature.Add(ulong.MaxValue);
            old.Dispose();
            locked.Dispose();
            held.Dispose();
            ratio.Dispose();
        }

        /// <summary>
        /// Test locking once per step from the previous load. Return whether state changed.
        /// After VesselHoldSteps, yield/relock at configured thresholds; yield immediately at break strength or
        /// revoked permission.
        /// </summary>
        static unsafe bool TestLock(ForestIsland* forest, int part, in VesselEdge edge)
        {
            int held = forest->PartHeld[part];
            if (held < int.MaxValue / 2)
            {
                forest->PartHeld[part] = ++held;
            }

            bool locked = forest->PartLocked[part] != 0;
            double ratio = forest->PartRatio[part];
            double yield = edge.YieldFraction > 0.0 ? edge.YieldFraction : DefaultYieldFraction;
            double relock = (edge.RelockFraction > 0.0 ? edge.RelockFraction : DefaultRelockFraction) * yield;
            bool settled = held >= VesselHoldSteps;
            bool next = locked
                ? ratio < 1.0 && (ratio < yield || !settled)
                : edge.MayLock && settled && ratio < relock;
            if (!edge.MayLock)
            {
                next = false;
            }

            if (next == locked)
            {
                return false;
            }

            forest->PartLocked[part] = (byte)(next ? 1 : 0);
            forest->PartHeld[part] = 0;
            return true;
        }

        /// <summary>
        /// Build vessel constraint structure and a fresh context pointing into its arrays.
        /// </summary>
        static unsafe void BuildVesselStructure(ForestIsland* forest)
        {
            forest->StructureBuilds++;
            new BuildVesselStructureJob
            {
                Parts = View(forest->Parts.Ptr, forest->Parts.Length),
                Edges = View(forest->Edges.Ptr, forest->Edges.Length),
                Closures = View(forest->Closures.Ptr, forest->Closures.Length),
                HubClosures = 16,
                ChunkTarget = 96,
                MinimumDegree = true,
                Structure = &forest->Structure,
            }.Execute();

            ref VesselStructure s = ref forest->Structure;
            int parts = forest->Parts.Length;
            int sections = forest->SectionCount;
            int blocks = sections + s.Header[VesselStructure.OffDiagonalCount] + s.Header[VesselStructure.BufferBlockCount];
            int rhs = sections + s.Header[VesselStructure.BufferRhsCount];
            forest->Sections.Resize(sections);
            forest->Blocks.Resize(Block6x6.Size * blocks);
            forest->Inverse.Resize(Block6x6.Size * sections);
            forest->Rhs.Resize(6 * rhs);
            forest->Solution.Resize(6 * sections);
            forest->LinkRows.Resize(math.max(1, forest->LinkCount));
            forest->PartForce.Resize(math.max(1, parts));
            forest->PartTorque.Resize(math.max(1, parts));
            forest->ChildPoint.Resize(math.max(1, parts));
            forest->SectionResidual.Resize(math.max(1, sections));
            forest->KernelEdgeLoads.Resize(math.max(1, parts));
            forest->KernelClosureLoads.Resize(math.max(1, forest->Closures.Length));
            forest->NextState.Resize(math.max(1, parts));

            var settings = VesselSolverSettings.Default;
            forest->Context = new VesselContext
            {
                Parts = forest->Parts.Ptr,
                Edges = forest->Edges.Ptr,
                Closures = forest->Closures.Ptr,
                Sections = forest->Sections.Ptr,
                Blocks = forest->Blocks.Ptr,
                Inverse = forest->Inverse.Ptr,
                Rhs = forest->Rhs.Ptr,
                Solution = forest->Solution.Ptr,
                LinkRows = forest->LinkRows.Ptr,
                PartForce = forest->PartForce.Ptr,
                PartTorque = forest->PartTorque.Ptr,
                ChildPoint = forest->ChildPoint.Ptr,
                SectionResidual = forest->SectionResidual.Ptr,
                EdgeLoads = forest->KernelEdgeLoads.Ptr,
                ClosureLoads = forest->KernelClosureLoads.Ptr,
                NextState = forest->NextState.Ptr,
                Counters = forest->Counters.Ptr,
                PartSection = s.PartSection.Ptr,
                SectionPartStart = s.SectionPartStart.Ptr,
                SectionParts = s.SectionParts.Ptr,
                ColumnStart = s.ColumnStart.Ptr,
                Rows = s.Rows.Ptr,
                UpdateStart = s.UpdateStart.Ptr,
                UpdateTarget = s.UpdateTarget.Ptr,
                RhsTarget = s.RhsTarget.Ptr,
                LinkStart = s.LinkStart.Ptr,
                Links = s.Links.Ptr,
                IncidentStart = s.IncidentStart.Ptr,
                Incident = s.Incident.Ptr,
                PartLink = s.PartLink.Ptr,
                ChunkStart = s.ChunkStart.Ptr,
                ChunkColumns = s.ChunkColumns.Ptr,
                TopColumns = s.TopColumns.Ptr,
                ChunkBufferStart = s.ChunkBufferStart.Ptr,
                BufferTarget = s.BufferTarget.Ptr,
                ChunkRhsStart = s.ChunkRhsStart.Ptr,
                RhsBufferTarget = s.RhsBufferTarget.Ptr,
                EliminationParent = s.EliminationParent.Ptr,
                SectionCount = sections,
                OffDiagonalCount = s.Header[VesselStructure.OffDiagonalCount],
                ChunkCount = s.Header[VesselStructure.ChunkCount],
                TopCount = s.Header[VesselStructure.TopCount],
                MaxStructure = s.Header[VesselStructure.MaxStructure],
                HardFrequency = 2.0 * math.PI_DBL / settings.HardPeriod,
                HardDampingRatio = settings.HardDampingRatio,
                Centripetal = settings.CentripetalLoads,
                LoadsOnly = true,
                Equilibrium = &forest->Equilibrium,
            };

            int locked = 0;
            for (int part = 0; part < parts; part++)
            {
                locked += forest->Edges[part].Parent >= 0 && forest->Edges[part].State == VesselEdgeState.Locked ? 1 : 0;
            }

            forest->LockedPairs = locked;
            OrderSharedSections(forest);
        }

        /// <summary>
        /// A section with more parts than this is shared among the team's lanes by ranges of parts. A smaller section is
        /// one task, as before.
        /// </summary>
        const int SharedSectionParts = 256;

        /// <summary>
        /// Separate small and shared sections. Order shared parts depth-first so that subtree loads are differences
        /// of prefix sums.
        /// </summary>
        static unsafe void OrderSharedSections(ForestIsland* forest)
        {
            ref VesselStructure structure = ref forest->Structure;
            int parts = forest->Parts.Length;
            int sections = forest->SectionCount;
            forest->SmallSections.Clear();
            forest->SharedSections.Clear();
            forest->DepthFirstPart.Resize(math.max(1, parts));
            forest->DepthFirstPosition.Resize(math.max(1, parts));
            forest->SubtreeLast.Resize(math.max(1, parts));
            forest->PrefixForce.Resize(math.max(1, parts));
            forest->PrefixTorque.Resize(math.max(1, parts));
            int* sectionParts = structure.SectionParts.Ptr;
            int* depthFirstPart = forest->DepthFirstPart.Ptr;
            int* position = forest->DepthFirstPosition.Ptr;
            int* last = forest->SubtreeLast.Ptr;
            var size = new NativeArray<int>(math.max(1, parts), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var cursor = new NativeArray<int>(math.max(1, parts), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int s = 0; s < sections; s++)
            {
                int start = structure.SectionPartStart[s], end = structure.SectionPartStart[s + 1];
                bool shared = end - start > SharedSectionParts;

                // Each part after the first must hang from a parent in the same section, which comes before it.
                for (int k = start + 1; shared && k < end; k++)
                {
                    int parent = forest->Edges[sectionParts[k]].Parent;
                    shared = parent >= 0 && structure.PartSection[parent] == s;
                }

                if (!shared)
                {
                    forest->SmallSections.Add(s);
                    continue;
                }

                forest->SharedSections.Add(s);
                for (int k = start; k < end; k++)
                {
                    size[sectionParts[k]] = 1;
                }

                for (int k = end - 1; k > start; k--)
                {
                    int i = sectionParts[k];
                    size[forest->Edges[i].Parent] += size[i];
                }

                // Parents come before their children, so each parent has its position before its children need it.
                int root = sectionParts[start];
                position[root] = start;
                cursor[root] = start + 1;
                for (int k = start + 1; k < end; k++)
                {
                    int i = sectionParts[k];
                    int parent = forest->Edges[i].Parent;
                    position[i] = cursor[parent];
                    cursor[parent] += size[i];
                    cursor[i] = position[i] + 1;
                }

                for (int k = start; k < end; k++)
                {
                    int i = sectionParts[k];
                    depthFirstPart[position[i]] = i;
                    last[i] = position[i] + size[i] - 1;
                }
            }

            size.Dispose();
            cursor.Dispose();
        }

        /// <summary>
        /// Compute section properties, using one task per small section and two shared passes for centre and
        /// inertia.
        /// </summary>
        static unsafe void ComputeVesselSections(ref ForestRun run)
        {
            ForestIsland* forest = run.Forest;
            if (forest->SharedSections.Length == 0)
            {
                RunVesselStage(ref run, VesselTasks.ComputeSections, forest->SectionCount);
                return;
            }

            RunVesselStage(ref run, VesselTasks.ComputeSmallSections, forest->SmallSections.Length);
            ref VesselContext kernel = ref forest->Context;
            for (int i = 0; i < forest->SharedSections.Length; i++)
            {
                int s = forest->SharedSections[i];
                int start = kernel.SectionPartStart[s], end = kernel.SectionPartStart[s + 1];
                int chunks = (end - start + VesselItemsPerTask - 1) / VesselItemsPerTask;
                var mass = new NativeArray<VesselContext.MassSums>(chunks, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var inertia = new NativeArray<VesselContext.InertiaSums>(chunks, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var tasks = new VesselTasks
                {
                    Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                    Phase = VesselTasks.SharedMass,
                    Start = start,
                    End = end,
                    MassSums = (VesselContext.MassSums*)mass.GetUnsafePtr(),
                    InertiaSums = (VesselContext.InertiaSums*)inertia.GetUnsafePtr(),
                };
                WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind, chunks, null);

                // Add the ranges' sums in order, so that the result does not depend on which lane summed which range.
                VesselContext.MassSums totalMass = mass[0];
                for (int c = 1; c < chunks; c++)
                {
                    totalMass.Add(mass[c]);
                }

                tasks.Phase = VesselTasks.SharedInertia;
                tasks.Center = totalMass.Moment / totalMass.Mass;
                WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind, chunks, null);
                VesselContext.InertiaSums totalInertia = inertia[0];
                for (int c = 1; c < chunks; c++)
                {
                    totalInertia.Add(inertia[c]);
                }

                kernel.StoreSection(s, totalMass, totalInertia);
                mass.Dispose();
                inertia.Dispose();
            }
        }

        /// <summary>
        /// Compute edge loads, using one task per small section and parallel depth-first prefix sums for shared
        /// sections.
        /// </summary>
        static unsafe void FinishVesselSections(ref ForestRun run)
        {
            ForestIsland* forest = run.Forest;
            PrepareEquilibrium(ref run);
            forest->Context.DeferStrutLoads = run.DeferStrutLoads;
            if (run.DeferStrutLoads && run.Substep <= 0)
            {
                forest->Equilibrium.BeginStep();
            }

            if (forest->SharedSections.Length == 0)
            {
                RunVesselStage(ref run, VesselTasks.FinishSections, forest->SectionCount);
                return;
            }

            RunVesselStage(ref run, VesselTasks.FinishSmallSections, forest->SmallSections.Length);
            ref VesselContext kernel = ref forest->Context;
            double3* prefixForce = forest->PrefixForce.Ptr;
            double3* prefixTorque = forest->PrefixTorque.Ptr;
            for (int i = 0; i < forest->SharedSections.Length; i++)
            {
                int s = forest->SharedSections[i];
                int start = kernel.SectionPartStart[s], end = kernel.SectionPartStart[s + 1];
                int chunks = (end - start + VesselItemsPerTask - 1) / VesselItemsPerTask;
                var sums = new NativeArray<double3>(2 * chunks, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                double3* chunkForce = (double3*)sums.GetUnsafePtr();
                double3* chunkTorque = chunkForce + chunks;
                var tasks = new VesselTasks
                {
                    Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                    Phase = VesselTasks.SharedPartLoads,
                    Section = s,
                    Start = start,
                    End = end,
                    ChunkForce = chunkForce,
                    ChunkTorque = chunkTorque,
                };
                WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind, chunks, null);

                // The live rows' forces on this section's parts. There are few of these, so the leader adds them.
                for (int e = kernel.IncidentStart[s]; e < kernel.IncidentStart[s + 1]; e++)
                {
                    int part = kernel.LiveRowLoad(s, e, out double3 force, out double3 torque);
                    int at = forest->DepthFirstPosition[part];
                    int chunk = (at - start) / VesselItemsPerTask;
                    kernel.PartForce[part] -= force;
                    kernel.PartTorque[part] -= torque;
                    prefixForce[at] -= force;
                    prefixTorque[at] -= torque;
                    chunkForce[chunk] -= force;
                    chunkTorque[chunk] -= torque;
                }

                // Turn each range's sum into the sum of all the ranges before it.
                double3 runningForce = 0.0, runningTorque = 0.0;
                for (int c = 0; c < chunks; c++)
                {
                    double3 force = chunkForce[c], torque = chunkTorque[c];
                    chunkForce[c] = runningForce;
                    chunkTorque[c] = runningTorque;
                    runningForce += force;
                    runningTorque += torque;
                }

                tasks.Phase = VesselTasks.SharedPrefix;
                WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind, chunks, null);
                tasks.Phase = VesselTasks.SharedEdgeLoads;
                WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind, chunks, null);
                kernel.StoreRootLoad(s, prefixForce[end - 1]);
                kernel.ApplyStrutLoads(s);
                sums.Dispose();
            }
        }

        /// <summary>
        /// At a step's last substep, when the substeps only summed the loads of the sections that have closures inside them:
        /// solves each such section once with the mean of its substeps' loads, and writes the loads of the edges on the
        /// closures' loops and of the closures to the vessel set in place of the means that the substeps wrote there.
        /// </summary>
        static unsafe void FinishStrutLoads(ref ForestRun run)
        {
            if (!run.DeferStrutLoads || run.Substep != run.Substeps - 1)
            {
                return;
            }

            ForestIsland* forest = run.Forest;
            ref VesselContext kernel = ref forest->Context;
            ref VesselEquilibrium equilibrium = ref forest->Equilibrium;
            VesselSet set = run.Vessels;
            int sections = math.min(forest->SectionCount, equilibrium.SectionEntry.Length);
            for (int s = 0; s < sections; s++)
            {
                int entryIndex = equilibrium.SectionEntry[s];
                if (entryIndex < 0)
                {
                    continue;
                }

                equilibrium.Apply(ref kernel, s, VesselEquilibrium.ApplyMode.Finish);
                SectionEquilibrium* entry = equilibrium.Entries.Ptr + entryIndex;
                int start = kernel.SectionPartStart[s];
                for (int l = 0; l < entry->Links.Length; l++)
                {
                    int target = entry->Links[l].Target;
                    if (target < 0)
                    {
                        continue;
                    }

                    int part = kernel.SectionParts[start + target];
                    int setPart = forest->PartSetPart[part];
                    if (setPart < 0)
                    {
                        continue;
                    }

                    VesselEdge* edge = forest->Edges.Ptr + part;
                    VesselEdgeLoad load = forest->KernelEdgeLoads[part];
                    load.Ratio = Ratio(load, edge->BreakForce, edge->BreakTorque);
                    set.EdgeLoads[setPart] = load;
                    forest->PartRatio[part] = (float)load.Ratio;
                }

                for (int c = equilibrium.SectionClosureStart[s]; c < equilibrium.SectionClosureStart[s + 1]; c++)
                {
                    int closure = equilibrium.SectionClosures[c];
                    VesselEdgeLoad load = forest->KernelClosureLoads[closure];
                    set.ClosureLoads[forest->ClosureSetIndex[closure]] = new VesselEdgeLoad { Force = load.Force, Torque = load.Torque };
                }
            }
        }

        /// <summary>
        /// Matches the sections that have closures inside them with their equilibrium systems after each structure build, and
        /// factors the systems of new sections. A system is kept while its vessel's Id and structure version, the time
        /// step and the section's parts stay the same.
        /// </summary>
        static unsafe void PrepareEquilibrium(ref ForestRun run)
        {
            ForestIsland* forest = run.Forest;
            bool enabled = VesselSwitches.StrutLoadsEnabled;
            if (forest->Equilibrium.BuiltFor == forest->StructureBuilds && forest->EquilibriumTimeStep == run.Timestep
                && forest->EquilibriumEnabled == enabled)
            {
                return;
            }

            long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            int parts = forest->Parts.Length;
            var id = new NativeArray<ulong>(math.max(1, parts), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var version = new NativeArray<uint>(math.max(1, parts), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var index = new NativeArray<int>(math.max(1, parts), Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            // A body outside the vessels is a section of its own, which never holds a closure.
            for (int part = 0; part < parts; part++)
            {
                id[part] = ulong.MaxValue;
                version[part] = 0;
                index[part] = part;
            }

            int vessels = forest->VesselSignature.Length > 0 ? (int)forest->VesselSignature[0] : 0;
            for (int v = 0; v < vessels && v < forest->VesselPartStart.Length; v++)
            {
                int start = forest->VesselPartStart[v];
                int count = (int)(forest->VesselSignature[3 + 3 * v] >> 32);
                for (int k = 0; k < count && start + k < parts; k++)
                {
                    id[start + k] = forest->VesselSignature[1 + 3 * v];
                    version[start + k] = (uint)forest->VesselSignature[2 + 3 * v];
                    index[start + k] = k;
                }
            }

            forest->Equilibrium.Prepare(forest->Context, parts, forest->Closures.Length, (ulong*)id.GetUnsafePtr(),
                (uint*)version.GetUnsafePtr(), (int*)index.GetUnsafePtr(), run.Timestep);
            forest->Equilibrium.BuiltFor = forest->StructureBuilds;
            forest->Equilibrium.PrepareTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - started;
            forest->EquilibriumTimeStep = run.Timestep;
            forest->EquilibriumEnabled = enabled;
        }

        /// <summary>
        /// Processes one range of a shared section's parts in one of the stages of <see cref="ComputeVesselSections"/>
        /// or <see cref="FinishVesselSections"/>.
        /// </summary>
        static unsafe void SharedSectionTask(VesselTasks* tasks, int chunk)
        {
            ForestIsland* forest = tasks->Run->Forest;
            ref VesselContext kernel = ref forest->Context;
            int from = tasks->Start + chunk * VesselItemsPerTask;
            int to = math.min(from + VesselItemsPerTask, tasks->End);
            double3* prefixForce = forest->PrefixForce.Ptr;
            double3* prefixTorque = forest->PrefixTorque.Ptr;
            switch (tasks->Phase)
            {
                case VesselTasks.SharedMass:
                    tasks->MassSums[chunk] = kernel.SumMass(from, to);
                    break;
                case VesselTasks.SharedInertia:
                    tasks->InertiaSums[chunk] = kernel.SumInertia(from, to, tasks->Center);
                    break;
                case VesselTasks.SharedPartLoads:
                {
                    VesselSection section = kernel.Sections[tasks->Section];
                    double* solution = kernel.Solution + 6 * tasks->Section;
                    double3 linear = DoubleMath.Load3(solution), angular = DoubleMath.Load3(solution + 3);
                    double3 sumForce = 0.0, sumTorque = 0.0;
                    for (int at = from; at < to; at++)
                    {
                        int i = forest->DepthFirstPart[at];
                        kernel.PartLoad(i, section.Center, linear, angular, out double3 force, out double3 torque);
                        kernel.PartForce[i] = force;
                        kernel.PartTorque[i] = torque;
                        prefixForce[at] = force;
                        prefixTorque[at] = torque;
                        sumForce += force;
                        sumTorque += torque;
                    }

                    tasks->ChunkForce[chunk] = sumForce;
                    tasks->ChunkTorque[chunk] = sumTorque;
                    break;
                }
                case VesselTasks.SharedPrefix:
                {
                    double3 force = tasks->ChunkForce[chunk], torque = tasks->ChunkTorque[chunk];
                    for (int at = from; at < to; at++)
                    {
                        force += prefixForce[at];
                        torque += prefixTorque[at];
                        prefixForce[at] = force;
                        prefixTorque[at] = torque;
                    }

                    break;
                }
                default:
                {
                    // The first position is the section's root, whose load the leader stores afterwards.
                    double3 center = kernel.Sections[tasks->Section].Center;
                    for (int at = math.max(from, tasks->Start + 1); at < to; at++)
                    {
                        int i = forest->DepthFirstPart[at];
                        int last = forest->SubtreeLast[i];
                        double3 force, torque;
                        if (last == at)
                        {
                            // A leaf carries only its own terms, so take them directly rather than as a difference.
                            force = kernel.PartForce[i];
                            torque = kernel.PartTorque[i];
                        }
                        else
                        {
                            force = prefixForce[last] - prefixForce[at - 1];
                            torque = prefixTorque[last] - prefixTorque[at - 1];
                        }

                        kernel.StoreEdgeLoad(i, center, force, torque);
                    }

                    break;
                }
            }
        }

        /// <summary>
        /// Compute section properties and factor the vessel system with fixed-row impulses.
        /// Convert boundary rows to section coordinates and build their dense block.
        /// </summary>
        static unsafe void StartVesselRun(ref ForestRun run)
        {
            ForestIsland* forest = run.Forest;
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            ref VesselContext context = ref forest->Context;
            int rows = problem.RowCount;
            int sections = forest->SectionCount;
            context.TimeStep = run.Timestep;
            context.FactorOnly = false;
            context.LinkRowStart = null;
            context.ExtraDiagonal = null;
            context.ExtraRhs = null;

            // Every row is a boundary row except the fixed ones, whose impulses are known.
            run.RowClass = new NativeArray<byte>(math.max(1, rows), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            run.RowBlock = new NativeArray<int>(math.max(1, rows), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            run.BlockSections = new NativeArray<int2>(math.max(1, problem.BlockCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var coupled = new NativeArray<bool>(math.max(1, rows), Allocator.Temp);
            for (int c = 0; run.Bounded && c < problem.CouplingCount; c++)
            {
                BoundCoupling coupling = problem.Couplings[c];
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    coupled[row] = true;
                }
            }

            int boundaryRows = 0;
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                run.BlockSections[b] = new int2(block.BodyA >= 0 ? forest->BodySection[block.BodyA] : -1,
                    block.BodyB >= 0 ? forest->BodySection[block.BodyB] : -1);
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    run.RowBlock[row] = b;
                    bool fixedRow = run.Bounded && problem.Lower[row] == problem.Upper[row] && !coupled[row];
                    run.RowClass[row] = fixedRow ? FixedRowClass : BoundaryRowClass;
                    boundaryRows += fixedRow ? 0 : 1;
                }
            }

            coupled.Dispose();

            // The fixed rows' known impulses act on the parts like any other outside impulse.
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (run.RowClass[row] == FixedRowClass && problem.Lower[row] != 0.0f)
                    {
                        float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                        AddPartImpulse(forest, block.BodyA, columns, problem.Lower[row]);
                        AddPartImpulse(forest, block.BodyB, columns + 6, problem.Lower[row]);
                    }
                }
            }

            long factorStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            ComputeVesselSections(ref run);

            VesselContext* contextPtr = (VesselContext*)UnsafeUtility.AddressOf(ref context);
            WorkTeam.RunAll<ForestTaskDispatch>(run.Team, contextPtr, IslandTaskKind.ForestFactorKind, context.ChunkCount, null);
            var scratch = new NativeArray<double>(Block6x6.Size * math.max(1, context.MaxStructure), Allocator.Temp,
                NativeArrayOptions.UninitializedMemory);
            context.SolveTop((double*)scratch.GetUnsafePtr());
            scratch.Dispose();
            WorkTeam.RunAll<ForestTaskDispatch>(run.Team, contextPtr, IslandTaskKind.ForestSubstituteKind, context.ChunkCount, null);
            run.FactorTicks = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - factorStarted;

            run.Velocity = new NativeArray<double>(6 * sections, Allocator.Temp);
            run.Work = new NativeArray<double>(6 * sections, Allocator.Temp);
            run.BaseVelocity = new NativeArray<double>(6 * sections, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            UnsafeUtility.MemCpy(run.BaseVelocity.GetUnsafePtr(), context.Solution, 6L * sections * sizeof(double));

            // Transform rows to section coordinates and add current body velocity contributions back to
            // boundary targets.
            long floats = SparseConstraintSolver.ColumnsPerRow * (long)rows;
            run.SectionJacobian = new NativeArray<float>((int)math.max(1, floats), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            float* jacobian = (float*)run.SectionJacobian.GetUnsafePtr();
            if (rows > 0)
            {
                UnsafeUtility.MemCpy(jacobian, problem.Jacobian, floats * sizeof(float));
            }

            run.Jacobian = jacobian;
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    float* columns = jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    double rate = TurnToSection(forest, block.BodyA, columns) + TurnToSection(forest, block.BodyB, columns + 6);
                    if (run.RowClass[row] == BoundaryRowClass)
                    {
                        problem.RightHandSide[row] += (float)rate;
                    }
                }
            }

            BuildBoundary(ref run, boundaryRows);
        }

        /// <summary>
        /// Adds a row's known impulse to the part on one side of the row, as linear and angular impulses about the
        /// part's centre.
        /// </summary>
        static unsafe void AddPartImpulse(ForestIsland* forest, int body, float* columns, float impulse)
        {
            if (body < 0 || forest->BodyPart[body] < 0)
            {
                return;
            }

            VesselPart* part = forest->Parts.Ptr + forest->BodyPart[body];
            part->LinearImpulse += impulse * new double3(columns[0], columns[1], columns[2]);
            part->AngularImpulse += impulse * new double3(columns[3], columns[4], columns[5]);
        }

        /// <summary>
        /// Convert row columns to section coordinates: angular += r cross linear.
        /// Return the row-rate contribution of the body's current velocity.
        /// </summary>
        static unsafe double TurnToSection(ForestIsland* forest, int body, float* columns)
        {
            if (body < 0 || forest->BodyPart[body] < 0)
            {
                return 0.0;
            }

            VesselPart* part = forest->Parts.Ptr + forest->BodyPart[body];
            VesselSection* section = forest->Sections.Ptr + forest->BodySection[body];
            double3 linear = new double3(columns[0], columns[1], columns[2]);
            double3 angular = new double3(columns[3], columns[4], columns[5]);
            double rate = math.dot(linear, part->LinearVelocity) + math.dot(angular, part->AngularVelocity);
            double3 turned = angular + math.cross(part->Position - section->Center, linear);
            columns[3] = (float)turned.x;
            columns[4] = (float)turned.y;
            columns[5] = (float)turned.z;
            return rate;
        }

        /// <summary>Sum the rows with the forest solver, then recover the vessel bodies' impulses.</summary>
        internal static unsafe void SettleCompiledVessel(ref ForestRun run, in SparseConstraintSolver.Problem problem,
            float* impulses, float3* linear, float3* angular, bool pushOut)
        {
            SumForestBodyImpulses(ref run, problem, impulses, linear, angular);
            SettleVessel(ref run, problem, impulses, linear, angular, pushOut);
        }

        /// <summary>
        /// Recover vessel-row impulses and loads from solved section velocities, treating boundary impulses as
        /// external forces.
        /// Use inverse dynamics for locked edges. Push-out starts sections at rest and writes body impulses only.
        /// </summary>
        static unsafe void SettleVessel(ref ForestRun run, in SparseConstraintSolver.Problem problem, float* impulses,
            float3* linear, float3* angular, bool pushOut)
        {
            ForestIsland* forest = run.Forest;
            ref VesselContext context = ref forest->Context;
            int sections = forest->SectionCount;
            double* v = (double*)run.Velocity.GetUnsafePtr();
            if (!pushOut)
            {
                // At this point, linear and angular contain each body's impulses from the boundary rows and fixed rows.
                // The kernel uses these as external impulses on the parts when computing the loads on locked edges.
                for (int part = 0; part < forest->Parts.Length; part++)
                {
                    int body = forest->PartBody[part];
                    VesselPart* p = forest->Parts.Ptr + part;
                    p->LinearImpulse = body >= 0 ? (double3)linear[body] : 0.0;
                    p->AngularImpulse = body >= 0 ? (double3)angular[body] : 0.0;
                }

                UnsafeUtility.MemCpy(context.Solution, v, 6L * sections * sizeof(double));
                UnsafeUtility.MemClear(forest->KernelClosureLoads.Ptr, forest->KernelClosureLoads.Length * (long)sizeof(VesselEdgeLoad));
                RunVesselStage(ref run, VesselTasks.LinkImpulses, forest->LinkCount);

                // The context has LoadsOnly set, so FinishSections leaves the parts where they are. Later calculations
                // in this substep use the poses from the start of the substep.
                FinishVesselSections(ref run);
            }

            var tasks = new VesselTasks
            {
                Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                Phase = VesselTasks.BodyImpulses,
                Count = forest->Parts.Length,
                Linear = linear,
                Angular = angular,
                PushOut = pushOut,
            };
            WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind,
                (tasks.Count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
        }

        /// <summary>
        /// The number of sections, links or parts processed by one task in a vessel stage.
        /// </summary>
        const int VesselItemsPerTask = 64;

        /// <summary>
        /// The data shared by the team's lanes while processing a stage for a vessel island.
        /// </summary>
        unsafe struct VesselTasks
        {
            public const int ComputeSections = 0;
            public const int LinkImpulses = 1;
            public const int FinishSections = 2;
            public const int BodyImpulses = 3;
            public const int FillParts = 4;
            public const int WriteLoads = 5;
            public const int ComputeSmallSections = 6;
            public const int FinishSmallSections = 7;

            // The stages for a shared section, whose tasks are ranges of its parts.
            public const int SharedMass = 8;
            public const int SharedInertia = 9;
            public const int SharedPartLoads = 10;
            public const int SharedPrefix = 11;
            public const int SharedEdgeLoads = 12;
            public const int FillEdges = 13;
            public const int MoveBodies = 14;
            public const int FillPoses = 15;

            // The stages of the fused vessel solve that update the island's Jacobians, build their rows, and integrate
            // the bodies after a boundary solve.
            public const int UpdateJacobians = 16;
            public const int BuildRows = 17;
            public const int IntegrateBodies = 18;

            // Fused stages that move/integrate bodies and write loads, with part tasks followed by closures.
            // FillNext also prepares part motion for the next substep.
            public const int MoveBodiesAndWriteLoads = 19;
            public const int IntegrateBodiesAndWriteLoads = 20;
            public bool FillNext;

            public ForestRun* Run;
            public int Phase;
            public int Count;
            public float3* Linear;
            public float3* Angular;
            public bool PushOut;

            /// <summary>For <see cref="WriteLoads"/>: the substep and the number of substeps in the step.</summary>
            public int Substep;
            public int Substeps;

            // Part count before closure tasks. FillEdges also uses the state-change flag.
            public int VesselParts;
            public int Rebuild;

            /// <summary>For <see cref="MoveBodies"/>: the bodies that the fused vessel solve moves and integrates.</summary>
            public NativeArray<MotionVelocity> MotionVelocities;
            public NativeArray<MotionData> MotionDatas;

            // Jacobians, row ranges and shared build inputs for UpdateJacobians and BuildRows.
            public IslandHeader* Headers;

            public int* HeaderRows;
            public NativeArray<RigidBody> RigidBodies;
            public StepInput StepInput;
            public IslandRowBuffers Rows;
            public RowRegularization Regularization;
            public bool SparseContacts;
            public int Bounded;

            /// <summary>
            /// For the stages of a shared section: the section, its range of positions, the centre for the inertia sums,
            /// and each range's sums.
            /// </summary>
            public int Section;
            public int Start;
            public int End;
            public double3 Center;
            public VesselContext.MassSums* MassSums;
            public VesselContext.InertiaSums* InertiaSums;
            public double3* ChunkForce;
            public double3* ChunkTorque;
        }

        /// <summary>Runs one of the kernel's stages over sections or links on the run's team.</summary>
        static unsafe void RunVesselStage(ref ForestRun run, int phase, int count)
        {
            var tasks = new VesselTasks
            {
                Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                Phase = phase,
                Count = count,
            };
            WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind, (count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
        }

        /// <summary>
        /// Processes one task in a vessel stage on a team lane.
        /// </summary>
        internal static unsafe void ForestVesselTask(void* context, int task)
        {
            var tasks = (VesselTasks*)context;
            if (tasks->Phase >= VesselTasks.SharedMass && tasks->Phase <= VesselTasks.SharedEdgeLoads)
            {
                SharedSectionTask(tasks, task);
                return;
            }

            if (tasks->Phase == VesselTasks.UpdateJacobians || tasks->Phase == VesselTasks.BuildRows)
            {
                int last = math.min((task + 1) * VesselItemsPerTask, tasks->Count);
                for (int index = task * VesselItemsPerTask; index < last; index++)
                {
                    FusedJacobianTask(tasks, index);
                }

                return;
            }

            ref ForestRun run = ref *tasks->Run;
            ref VesselContext kernel = ref run.Forest->Context;
            int end = math.min((task + 1) * VesselItemsPerTask, tasks->Count);
            for (int item = task * VesselItemsPerTask; item < end; item++)
            {
                switch (tasks->Phase)
                {
                    case VesselTasks.ComputeSections:
                        kernel.ComputeSection(item);
                        break;
                    case VesselTasks.LinkImpulses:
                        kernel.LinkImpulses(item);
                        break;
                    case VesselTasks.FinishSections:
                        kernel.FinishSection(item);
                        break;
                    case VesselTasks.FillParts:
                        run.Forest->Parts[item] = PartOf(run, run.Forest->PartBody[item]);
                        break;
                    case VesselTasks.FillPoses:
                        FillPartPose(run, item);
                        break;
                    case VesselTasks.WriteLoads:
                        WriteVesselLoad(ref run, item, tasks->Substep, tasks->Substeps);
                        break;
                    case VesselTasks.MoveBodies:
                        MoveVesselBody(ref run, tasks, item);
                        break;
                    case VesselTasks.IntegrateBodies:
                        IntegrateVesselBody(ref run, tasks, item);
                        break;
                    case VesselTasks.MoveBodiesAndWriteLoads:
                        if (item < tasks->VesselParts)
                        {
                            MoveVesselBody(ref run, tasks, item);
                            if (tasks->FillNext)
                            {
                                FillPartPose(run, item);
                            }
                        }

                        WriteVesselLoad(ref run, item, tasks->Substep, tasks->Substeps);
                        break;
                    case VesselTasks.IntegrateBodiesAndWriteLoads:
                        if (item < tasks->VesselParts)
                        {
                            IntegrateVesselBody(ref run, tasks, item);
                        }

                        WriteVesselLoad(ref run, item, tasks->Substep, tasks->Substeps);
                        break;
                    case VesselTasks.FillEdges:
                        if (item >= tasks->VesselParts)
                        {
                            FillVesselClosure(ref run, item - tasks->VesselParts);
                        }
                        else if (FillVesselEdge(ref run, item, tasks->Substep <= 0))
                        {
                            // Every task that sees a change writes the same value, so the tasks need no lock.
                            tasks->Rebuild = 1;
                        }

                        break;
                    case VesselTasks.ComputeSmallSections:
                        kernel.ComputeSection(run.Forest->SmallSections[item]);
                        break;
                    case VesselTasks.FinishSmallSections:
                        kernel.FinishSection(run.Forest->SmallSections[item]);
                        break;
                    default:
                        PartImpulse(ref run, item, tasks->Linear, tasks->Angular, tasks->PushOut);
                        break;
                }
            }
        }

        /// <summary>
        /// Compute the body's impulse to follow its section. Include rotation over the timestep to preserve
        /// rigidity.
        /// For push-out, treat the section as initially at rest.
        /// </summary>
        static unsafe void PartImpulse(ref ForestRun run, int part, float3* linear, float3* angular, bool pushOut)
        {
            ForestIsland* forest = run.Forest;
            double* v = (double*)run.Velocity.GetUnsafePtr();
            double h = run.Timestep;
            int body = forest->PartBody[part];
            if (body < 0)
            {
                return;
            }

            int s = forest->BodySection[body];
            VesselPart p = forest->Parts[part];
            VesselSection section = forest->Sections[s];
            double3 sectionLinear = new double3(v[6 * s], v[6 * s + 1], v[6 * s + 2]);
            double3 sectionAngular = new double3(v[6 * s + 3], v[6 * s + 4], v[6 * s + 5]);
            if (!pushOut && math.lengthsq(sectionAngular) < MinimumSpin * MinimumSpin)
            {
                sectionAngular = 0.0;
            }

            double3 r = p.Position - section.Center;
            double3 targetLinear;
            if (pushOut)
            {
                targetLinear = sectionLinear + math.cross(sectionAngular, r);
            }
            else
            {
                // Choose a velocity that places the part at its offset after the section rotates, so the locked section
                // remains rigid.
                double angle = math.length(sectionAngular) * h;
                double4 turn = angle > 1e-12 ? new double4(math.normalize(sectionAngular) * math.sin(0.5 * angle), math.cos(0.5 * angle))
                    : new double4(0.0, 0.0, 0.0, 1.0);
                targetLinear = sectionLinear + (math.mul(DoubleMath.Rotation(turn), r) - r) / h;
            }

            double3 changeLinear = pushOut ? targetLinear : targetLinear - p.LinearVelocity;
            double3 changeAngular = pushOut ? sectionAngular : sectionAngular - p.AngularVelocity;
            int worldBody = run.WorldBodies[body];
            MotionVelocity motion = run.MotionVelocities[worldBody];
            quaternion rotation = run.MotionDatas[worldBody].WorldFromMotion.rot;
            double3 inverseInertia = math.max((double3)motion.InverseInertia, 1e-12);
            double3 local = math.rotate(math.conjugate(rotation), (float3)changeAngular);
            linear[body] = DoubleMath.Flushed(changeLinear / math.max((double)motion.InverseMass, 1e-12));
            angular[body] = DoubleMath.Flushed((double3)math.rotate(rotation, (float3)(local / inverseInertia)));
        }

        /// <summary>
        /// Write mean substep loads, break ratios and lock flags to the vessel set.
        /// Save final ratios for next-step lock tests. Split work by part and closure ranges.
        /// </summary>
        static unsafe void WriteVesselLoads(ref ForestRun run, int substep, int substeps)
        {
            ForestIsland* forest = run.Forest;
            int count = forest->Parts.Length + forest->Closures.Length;
            var tasks = new VesselTasks
            {
                Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                Phase = VesselTasks.WriteLoads,
                Count = count,
                Substep = substep,
                Substeps = substeps,
            };
            WorkTeam.RunAll<VesselTaskDispatch>(run.Team, &tasks, IslandTaskKind.ForestVesselKind, (count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
        }

        /// <summary>
        /// Writes the load on one item to the vessel set. Items below the part count are parts, and the items after
        /// them are closures.
        /// </summary>
        static unsafe void WriteVesselLoad(ref ForestRun run, int item, int substep, int substeps)
        {
            ForestIsland* forest = run.Forest;
            VesselSet set = run.Vessels;
            int count = substep < 0 ? 1 : math.max(1, substeps);
            bool first = substep <= 0;
            bool last = substep < 0 || substep >= substeps - 1;
            int parts = forest->Parts.Length;
            if (item < parts)
            {
                int setPart = forest->PartSetPart[item];
                if (setPart < 0)
                {
                    return;
                }

                // The island's copy of the edge has the same parent sign and break strengths as the set's edge.
                VesselEdge* edge = forest->Edges.Ptr + item;
                VesselEdgeLoad load = edge->Parent >= 0 ? forest->KernelEdgeLoads[item] : default;
                VesselEdgeLoad sum = first ? default : set.EdgeLoads[setPart];
                sum.Force += load.Force / count;
                sum.Torque += load.Torque / count;
                sum.Ratio = Ratio(sum, edge->BreakForce, edge->BreakTorque);
                set.EdgeLoads[setPart] = sum;
                set.EdgeLocked[setPart] = edge->State == VesselEdgeState.Locked;
                if (last)
                {
                    forest->PartRatio[item] = (float)sum.Ratio;
                }

                return;
            }

            int closure = item - parts;
            int setClosure = forest->ClosureSetIndex[closure];
            VesselEdgeLoad closureLoad = forest->KernelClosureLoads[closure];
            VesselEdgeLoad closureSum = first ? default : set.ClosureLoads[setClosure];
            closureSum.Force += closureLoad.Force / count;
            closureSum.Torque += closureLoad.Torque / count;
            set.ClosureLoads[setClosure] = closureSum;
        }

        static double Ratio(in VesselEdgeLoad load, double breakForce, double breakTorque)
        {
            double force = breakForce > 0.0 && !double.IsInfinity(breakForce) ? math.length(load.Force) / breakForce : 0.0;
            double torque = breakTorque > 0.0 && !double.IsInfinity(breakTorque) ? math.length(load.Torque) / breakTorque : 0.0;
            return math.max(force, torque);
        }
    }
}
