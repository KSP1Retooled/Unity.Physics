using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Numerics.Linear.Solvers.Lowlevel;
using Unity.Profiling;

namespace Unity.Physics
{
    public static partial class Solver
    {
        static readonly ProfilerMarker VesselSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Vessel.Solve");

        /// <summary>
        /// The number of steps that an edge must remain locked before it can yield under load, or remain yielding
        /// before it can lock again. A locked edge yields immediately if its load reaches its break strength or if
        /// locking is no longer allowed.
        /// </summary>
        const int VesselHoldSteps = 10;

        /// <summary>
        /// The default fractions used when an edge's configured fractions are not positive. A locked edge yields at
        /// half its break strength after the required hold time. A yielding edge can lock again below a quarter of its
        /// break strength after that hold time.
        /// </summary>
        const double DefaultYieldFraction = 0.5;

        const double DefaultRelockFraction = 0.5;

        /// <summary>
        /// Solves a direct island that contains one or more vessels. The vessel kernel solves the vessels' edges and
        /// closures using its own rows. The island's Jacobian rows represent contacts and joints that are not vessel
        /// edges. These rows form a boundary block that is coupled to the vessels through the kernel's factorization.
        ///
        /// Every moving body in the island is included in the factorization. A body that is not in a vessel is treated
        /// as a separate part. The solver computes section velocities rather than changes in velocity, so it adds the
        /// contribution from the bodies' current velocities back to each boundary row's right-hand side. It then
        /// computes the impulse needed for each body to follow its section's motion and writes the edge loads to the
        /// vessel set.
        /// </summary>
        static unsafe void SolveVesselIsland(in SparseConstraintSolver.Problem problem, JacobianHeader** headers,
            in NativeList<System.IntPtr> headerList, in NativeList<int> worldBodies, in NativeArray<int> compactBodies, bool bounded,
            bool holdsPositions, int islandIndex, int* islandVessels, int islandVesselCount, in NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, in StepInput stepInput, in JacobianRows jacobianRows, float* recovery,
            bool sparseContacts, SparseSolveTeam* team, bool applyImpulses, float* impulses)
        {
            using ProfilerMarker.AutoScope span = VesselSolveMarker.Auto();

            // If no forest is cached for this island because the cache is full or missing, create one for this solve
            // only.
            ForestIsland* forest = stepInput.DirectSolverCache.ForestOf(islandIndex);
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
                Caps = stepInput.DirectSolverCache.Forest.Caps,
                Substep = stepInput.CurrentSubstep,
                RigidBodies = stepInput.Bodies,
                RigidBodyCount = stepInput.BodyCount,
            };

            int builds = forest->StructureBuilds;
            long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            PrepareVesselParts(ref run, stepInput.CurrentSubstep);
            StartVesselRun(ref run);
            long lap = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestStartTime, lap - started);
            stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestFactorTime, run.FactorTicks);
            stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestBoundaryTime, run.BoundaryTicks);
            stepInput.DirectSolverCache.CountForest(DirectSolverCache.ForestSolves);
            stepInput.DirectSolverCache.CountSections(forest->SectionCount, forest->LockedPairs, forest->StructureBuilds != builds, 0, 0);

            var bodyLinear = new NativeArray<float3>(math.max(1, problem.BodyCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var bodyAngular = new NativeArray<float3>(math.max(1, problem.BodyCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var linear = (float3*)bodyLinear.GetUnsafePtr();
            var angular = (float3*)bodyAngular.GetUnsafePtr();
            long solved = SolveVesselBoundary(ref run, problem, headers, headerList, worldBodies, compactBodies, holdsPositions,
                motionVelocities, motionDatas, stepInput, jacobianRows, recovery, sparseContacts, team, applyImpulses, impulses, linear,
                angular);
            WriteVesselLoads(ref run, stepInput.CurrentSubstep, stepInput.NumSubsteps);
            stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestFinishTime,
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
        /// Solves a vessel island's boundary rows against the factor that <see cref="StartVesselRun"/> made, and finds
        /// each body's impulse from its section's new motion. For boundary rows that hold positions, aims the rows at
        /// the drift that the integration will cause, and solves again. Then, if <paramref name="applyImpulses"/> is
        /// set, gives the bodies their impulses and pushes the bodies of deep contacts out. Returns the time at which
        /// the first solve ended, for the island's timers.
        /// </summary>
        static unsafe long SolveVesselBoundary(ref ForestRun run, in SparseConstraintSolver.Problem problem, JacobianHeader** headers,
            in NativeList<System.IntPtr> headerList, in NativeList<int> worldBodies, in NativeArray<int> compactBodies, bool holdsPositions,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, in StepInput stepInput,
            in JacobianRows jacobianRows, float* recovery, bool sparseContacts, SparseSolveTeam* team, bool applyImpulses, float* impulses,
            float3* linear, float3* angular)
        {
            long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;

            // If the boundary ran out of pivoting rounds, SolveForest returns false, but its impulses are still inside
            // their bounds and near the answer, so the vessel applies them. BoundaryCounters counts such solves.
            SolveForest(ref run, problem.RightHandSide, impulses);
            long solved = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestSolveTime, solved - started);
            SettleForest(ref run, problem, impulses, linear, angular, false);
            stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestSettleTime,
                Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solved);

            // For boundary rows that constrain positions, correct for the drift predicted from integration. The vessel
            // kernel builds its own rows from the parts' poses every substep and includes their position errors, so
            // those rows do not need this correction.
            if (holdsPositions && problem.RowCount > 0)
            {
                var drift = new NativeArray<float>(problem.RowCount, Allocator.Temp);
                float* driftPtr = (float*)drift.GetUnsafePtr();
                if (AimAtPredictedDrift(problem, null, linear, angular, compactBodies, headerList, impulses, motionVelocities,
                    motionDatas, stepInput.Timestep, jacobianRows, driftPtr, team, true))
                {
                    for (int row = 0; row < problem.RowCount; row++)
                    {
                        problem.RightHandSide[row] += driftPtr[row];
                    }

                    SolveForest(ref run, problem.RightHandSide, impulses);
                    SettleForest(ref run, problem, impulses, linear, angular, false);
                }

                drift.Dispose();
            }

            if (applyImpulses)
            {
                long applying = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                ApplySparseImpulses(problem, null, linear, angular, compactBodies, headerList, impulses, motionVelocities, motionDatas,
                    sparseContacts, team);
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

        static readonly ProfilerMarker VesselFusedMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Vessel.Fused");

        /// <summary>
        /// Solves an island whose bodies are all vessel parts for every substep of the step. The island's state is kept
        /// by the ID of its first vessel, and the island keeps the numbering of its moving parts from step to step
        /// while its vessels and bodies stay the same. In the first substep, the team's lanes copy the edges and test
        /// the locks. In each substep, the lanes update the island's contacts and joints and build their rows, and fill
        /// the parts' poses and velocities from the bodies. The kernel factors and solves the vessels' own rows, and the
        /// contacts and joints are solved against that factor as the boundary block, with the same code as the
        /// ordinary vessel island's. Then the lanes compute the loads, give each body its impulse and integrate the
        /// bodies. An island without boundary rows gives each body the motion of its section and integrates it in the
        /// same pass. The island needs no per-body sparse setup, no jobs and waits of its own in each substep, and no
        /// separate integration job.
        /// </summary>
        static unsafe void SolveFusedVesselIsland(in DirectIsland island, IslandHeader* islandHeaders, int* islandVessels,
            NativeArray<MotionVelocity> motionVelocities, NativeArray<MotionData> motionDatas, NativeArray<RigidBody> rigidBodies,
            StepInput stepInput, SparseSolveTeam* team)
        {
            using ProfilerMarker.AutoScope span = VesselFusedMarker.Auto();
            VesselSet set = stepInput.VesselSet;
            var motions = (MotionVelocity*)motionVelocities.GetUnsafePtr();
            int motionCount = motionVelocities.Length;

            // The island's state is kept by the ID of its first vessel, not by the island's index or its first body. The
            // job that marked the fused islands has already given the island its forest.
            ForestIsland* forest = stepInput.DirectSolverCache.FindFusedForest(set.Vessels[islandVessels[0]].Id);
            int islandVesselCount = island.VesselCount;
            bool temporary = forest == null;
            if (temporary)
            {
                forest = ForestIsland.Create();
            }

            // The layout is known if the island holds the same vessels, with the same structure versions and counts, as
            // when the forest's layout was last built, and if each moving part's body still has the same number. An
            // island that takes lock states over from another forest lays its parts out again to take them.
            bool known = forest->VesselMode && !forest->TakesStates && forest->VesselSignature.Length == 1 + 3 * islandVesselCount
                && forest->VesselSignature[0] == (ulong)islandVesselCount && forest->FusedCompactOfWorld.Length == math.max(1, motionCount);
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

            int numbered = 0;
            for (int v = 0; known && v < islandVesselCount; v++)
            {
                VesselRecord vessel = set.Vessels[islandVessels[v]];
                for (int part = vessel.FirstPart; known && part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    int body = Moving(set.PartBodies[part], motions, motionCount);
                    if (body >= 0)
                    {
                        known = numbered < forest->FusedWorldBodies.Length && forest->FusedWorldBodies[numbered] == body
                            && forest->FusedCompactOfWorld[body] == numbered;
                        numbered++;
                    }
                }
            }

            known &= numbered == forest->FusedWorldBodies.Length;
            if (!known)
            {
                // Number the moving parts again, one vessel after another.
                forest->FusedCompactOfWorld.Resize(math.max(1, motionCount));
                UnsafeUtility.MemSet(forest->FusedCompactOfWorld.Ptr, 0xFF, forest->FusedCompactOfWorld.Length * (long)sizeof(int));
                forest->FusedWorldBodies.Clear();
                for (int v = 0; v < islandVesselCount; v++)
                {
                    VesselRecord vessel = set.Vessels[islandVessels[v]];
                    for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                    {
                        int body = Moving(set.PartBodies[part], motions, motionCount);
                        if (body >= 0 && forest->FusedCompactOfWorld[body] < 0)
                        {
                            forest->FusedCompactOfWorld[body] = forest->FusedWorldBodies.Length;
                            forest->FusedWorldBodies.Add(body);
                        }
                    }
                }
            }

            // All the island's moving bodies are vessel parts, so the numbering above numbers every body that the island's
            // contacts and joints reach.
            NativeArray<int> compactBodies = View(forest->FusedCompactOfWorld.Ptr, forest->FusedCompactOfWorld.Length);
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
                    SparseSolveTeam.RunAll(team, &tasks, SparseSolveTeam.ForestVesselKind, (headerCount + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
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
                    SparseSolveTeam.RunAll(team, &tasks, SparseSolveTeam.ForestVesselKind, (solveCount + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
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
                    CompactOfWorld = forest->FusedCompactOfWorld.Ptr,
                    CompactOfWorldLength = forest->FusedCompactOfWorld.Length,
                    MotionVelocities = motions,
                    MotionDatas = (MotionData*)motionDatas.GetUnsafeReadOnlyPtr(),
                    Timestep = stepInput.Timestep,
                    Team = team,
                    Caps = stepInput.DirectSolverCache.Forest.Caps,
                    Substep = substep,
                    RigidBodies = stepInput.Bodies,
                    RigidBodyCount = stepInput.BodyCount,
                };

                int builds = forest->StructureBuilds;
                long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                PrepareVesselParts(ref run, substep, posesFilled);
                StartVesselRun(ref run);
                long lap = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestStartTime, lap - started);
                stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestFactorTime, run.FactorTicks);
                stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestBoundaryTime, run.BoundaryTicks);
                stepInput.DirectSolverCache.CountForest(DirectSolverCache.ForestSolves);
                stepInput.DirectSolverCache.CountSections(forest->SectionCount, forest->LockedPairs, forest->StructureBuilds != builds, 0, 0);

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
                    SparseSolveTeam.RunAll(team, &bodyTasks, SparseSolveTeam.ForestVesselKind,
                        (bodyTasks.Count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
                    impulses.Dispose();
                    posesFilled = false;
                }
                else
                {
                    // With no boundary rows, the sections' velocities are the ones from the vessels' own rows.
                    UnsafeUtility.MemCpy(run.Velocity.GetUnsafePtr(), run.BaseVelocity.GetUnsafeReadOnlyPtr(), run.Velocity.Length * (long)sizeof(double));
                    solved = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestSolveTime, solved - lap);

                    ref VesselContext context = ref forest->Context;
                    UnsafeUtility.MemCpy(context.Solution, run.Velocity.GetUnsafeReadOnlyPtr(), 6L * forest->SectionCount * sizeof(double));
                    UnsafeUtility.MemClear(forest->KernelClosureLoads.Ptr, forest->KernelClosureLoads.Length * (long)sizeof(VesselEdgeLoad));
                    RunVesselStage(ref run, VesselTasks.LinkImpulses, forest->LinkCount);
                    FinishVesselSections(ref run);
                    // Each part's body moves only in its own task, so the same task can fill the part's pose and
                    // velocity for the next substep right after it integrates the body. The next substep then skips
                    // the stage that fills the poses.
                    bodyTasks.Phase = VesselTasks.MoveBodiesAndWriteLoads;
                    bodyTasks.FillNext = substep < stepInput.NumSubsteps - 1;
                    SparseSolveTeam.RunAll(team, &bodyTasks, SparseSolveTeam.ForestVesselKind,
                        (bodyTasks.Count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
                    posesFilled = bodyTasks.FillNext;
                    stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestSettleTime,
                        Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solved);
                }

                stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestFinishTime,
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
            if (temporary)
            {
                ForestIsland.Destroy(forest);
            }
        }

        /// <summary>
        /// For the fused vessel solve with boundary rows: integrates one part's body over the substep, after the
        /// boundary solve has given the body its impulse. A body is integrated once, by the part that the island maps
        /// it to.
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
        /// For the fused vessel solve: gives one part's body the motion of its section, as an impulse in the same way
        /// as the sparse solve applies it, and then integrates the body over the substep.
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
        /// Arranges the island's parts in order: first the parts of each vessel, then a separate part for each moving
        /// body that is not in a vessel. If the vessels or extra bodies have changed, preserves the lock states for
        /// vessels with the same ID and structure version, then rebuilds the structure. On the first substep of each
        /// step, tests each edge's lock state against its load from the previous step. With a known layout after the
        /// first substep, <paramref name="posesFilled"/> says that the last substep already filled the parts' poses.
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

                // Number the parts and closures, one vessel after another, and record where each one comes from in the
                // vessel set. Do this on every refill, even when the signature is the same: the signature leaves out
                // each vessel's first part and first closure in the set, and those move when an earlier vessel in the
                // set changes.
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
                SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind,
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
        /// Prepares the parts of an island whose layout the caller knows is the same as when the forest's layout was
        /// last built: the same vessels with the same structure versions, the same numbering of the bodies, and no body
        /// outside the vessels. The part numbering, the closures and the mapping from bodies to sections stay as they
        /// are. On the first substep of each step, the team's lanes copy the edges again and test the locks, because
        /// the edges' drive and break values can change between steps. In every substep, the lanes fill the parts'
        /// poses and velocities from the bodies, unless <paramref name="posesFilled"/> says that the last substep's
        /// stage that moved the bodies already filled them.
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
                SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind,
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
        /// Copies one vessel part's edge from the vessel set and maps the part to its body. On the first substep, also
        /// tests the edge's lock state. Returns true if the lock state changed, which means that the structure must be
        /// built again.
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
        /// Gives the step's solver input the world's bodies, see <see cref="StepInput.Bodies"/>. The pointer is taken
        /// without safety checks, because jobs that write the bodies can still be running when the step is scheduled.
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
        /// Fills one part's pose and velocity from its body and clears its outside impulses, like
        /// <see cref="PartOf"/>, but keeps the part's mass and inertia from the first substep of the step.
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
        /// Gives each vessel island of the step the forest that will solve it, and hands each vessel's lock states to
        /// that forest if another forest holds the newest ones. A fused island's forest is kept by the ID of its first
        /// vessel, and any other island's forest is kept by the island's first body, so a vessel changes forests when
        /// it moves in or out of the fused job or when its island's first body changes. The states are copied in two
        /// passes, all of them before any forest is handed out, because handing out a fused forest can destroy the
        /// forest of another vessel. Runs on one thread, after the cache has assigned the islands' slots and before
        /// any island is solved.
        /// </summary>
        static unsafe void AssignVesselForests(DirectSolverCache cache, NativeArray<DirectIsland> islands, NativeArray<int> islandVessels,
            VesselSet set)
        {
            if (!cache.IsCreated || !set.IsCreated)
            {
                return;
            }

            // For each of the island vessels, where its copied states start in the lists below, or -1 if the vessel
            // keeps its states in the forest that will solve it.
            var copied = new NativeArray<int>(math.max(1, islandVessels.Length), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var locked = new NativeList<byte>(Allocator.Temp);
            var held = new NativeList<int>(Allocator.Temp);
            var ratio = new NativeList<float>(Allocator.Temp);
            for (int i = 0; i < islands.Length; i++)
            {
                DirectIsland island = islands[i];
                if (island.VesselCount == 0)
                {
                    continue;
                }

                ForestIsland* planned = island.Fused ? cache.FindFusedForest(set.Vessels[islandVessels[island.FirstVessel]].Id) : cache.ForestOf(i);
                for (int k = island.FirstVessel; k < island.FirstVessel + island.VesselCount; k++)
                {
                    copied[k] = -1;
                    VesselRecord vessel = set.Vessels[islandVessels[k]];
                    ForestIsland* owner = cache.OwnerOf(vessel.Id);
                    if (owner == null || owner == planned || !FindVesselStates(owner, vessel, out int start))
                    {
                        continue;
                    }

                    copied[k] = locked.Length;
                    locked.AddRange(owner->PartLocked.Ptr + start, vessel.PartCount);
                    held.AddRange(owner->PartHeld.Ptr + start, vessel.PartCount);
                    ratio.AddRange(owner->PartRatio.Ptr + start, vessel.PartCount);
                }
            }

            for (int i = 0; i < islands.Length; i++)
            {
                DirectIsland island = islands[i];
                if (island.VesselCount == 0)
                {
                    continue;
                }

                ForestIsland* forest = island.Fused ? cache.FusedForestOf(set.Vessels[islandVessels[island.FirstVessel]].Id) : cache.ForestOf(i);
                for (int k = island.FirstVessel; k < island.FirstVessel + island.VesselCount; k++)
                {
                    VesselRecord vessel = set.Vessels[islandVessels[k]];
                    cache.SetOwner(vessel.Id, forest);
                    if (forest == null || copied[k] < 0)
                    {
                        continue;
                    }

                    if (!forest->TakesStates)
                    {
                        forest->TakenIds.Clear();
                        forest->TakenVersions.Clear();
                        forest->TakenStarts.Clear();
                        forest->TakenLocked.Clear();
                        forest->TakenHeld.Clear();
                        forest->TakenRatio.Clear();
                        forest->TakesStates = true;
                    }

                    forest->TakenIds.Add(vessel.Id);
                    forest->TakenVersions.Add(vessel.StructureVersion);
                    forest->TakenStarts.Add(forest->TakenLocked.Length);
                    forest->TakenLocked.AddRange(locked.GetUnsafeReadOnlyPtr() + copied[k], vessel.PartCount);
                    forest->TakenHeld.AddRange(held.GetUnsafeReadOnlyPtr() + copied[k], vessel.PartCount);
                    forest->TakenRatio.AddRange(ratio.GetUnsafeReadOnlyPtr() + copied[k], vessel.PartCount);
                }
            }

            cache.PruneOwners();
            copied.Dispose();
            locked.Dispose();
            held.Dispose();
            ratio.Dispose();
        }

        /// <summary>
        /// Finds where a vessel's parts start in a forest's lock states, if the forest's layout holds the vessel with
        /// the same structure version.
        /// </summary>
        static unsafe bool FindVesselStates(ForestIsland* forest, in VesselRecord vessel, out int start)
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
        /// Rebuilds the arrangement of the island's parts and preserves each part's lock state if its vessel has the
        /// same ID and structure version. A vessel that the island takes over from another forest in this step takes
        /// the states that it had there. For a new vessel, edges that are allowed to lock start locked, so the craft
        /// starts with those connections rigid.
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
        /// Tests an edge's lock state once per step using its load from the previous step. Returns whether the state
        /// changed. After <see cref="VesselHoldSteps"/> steps in the same state, a locked edge yields when its load
        /// reaches the yield fraction of its break strength. A yielding edge can lock again when its load falls below
        /// the relock fraction multiplied by the yield threshold. A locked edge yields without waiting if its load
        /// reaches the break strength or if locking is no longer allowed.
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
        /// Builds the kernel's structure for the island's parts, edges and closures, and the context that points into
        /// the
        /// island's arrays. The kernel builds the rows itself from the parts' poses.
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
            forest->Blocks.Resize(VesselBlocks.Size * blocks);
            forest->Inverse.Resize(VesselBlocks.Size * sections);
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
        /// Sorts the sections into small ones and shared ones. For each shared section, numbers its parts depth first,
        /// so that each part's subtree is one range of positions. Then a load that sums over a subtree is the difference
        /// of two prefix sums, and the lanes can compute the prefix sums by ranges of positions.
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
        /// Computes every section's properties. The small sections are one task each. Each shared section is summed by
        /// ranges of parts in two stages: the first gives its centre, and the second sums the inertia about that centre.
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
                SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind, chunks, null);

                // Add the ranges' sums in order, so that the result does not depend on which lane summed which range.
                VesselContext.MassSums totalMass = mass[0];
                for (int c = 1; c < chunks; c++)
                {
                    totalMass.Add(mass[c]);
                }

                tasks.Phase = VesselTasks.SharedInertia;
                tasks.Center = totalMass.Moment / totalMass.Mass;
                SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind, chunks, null);
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
        /// Computes the loads on every section's edges. The small sections are one task each. For each shared section,
        /// the lanes compute each part's own force and torque by ranges of depth-first positions, then the prefix sums
        /// of those terms, and then each edge's load as the sum over its child's subtree, which is the difference of
        /// two prefix sums.
        /// </summary>
        static unsafe void FinishVesselSections(ref ForestRun run)
        {
            ForestIsland* forest = run.Forest;
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
                SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind, chunks, null);

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
                SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind, chunks, null);
                tasks.Phase = VesselTasks.SharedEdgeLoads;
                SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind, chunks, null);
                kernel.StoreRootLoad(s, prefixForce[end - 1]);
                sums.Dispose();
            }
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
                    double3 linear = VectorMath.Load3(solution), angular = VectorMath.Load3(solution + 3);
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
        /// Prepares the solve before solving the boundary block. Computes the section properties, factorizes the
        /// kernel's system, and solves it without boundary impulses but with the known impulses from fixed rows. Then
        /// converts the Jacobian rows to section coordinates and builds the boundary block.
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
            SparseSolveTeam.RunAll(run.Team, contextPtr, SparseSolveTeam.ForestFactorKind, context.ChunkCount, null);
            var scratch = new NativeArray<double>(VesselBlocks.Size * math.max(1, context.MaxStructure), Allocator.Temp,
                NativeArrayOptions.UninitializedMemory);
            context.SolveTop((double*)scratch.GetUnsafePtr());
            scratch.Dispose();
            SparseSolveTeam.RunAll(run.Team, contextPtr, SparseSolveTeam.ForestSubstituteKind, context.ChunkCount, null);
            run.FactorTicks = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - factorStarted;

            run.Velocity = new NativeArray<double>(6 * sections, Allocator.Temp);
            run.Work = new NativeArray<double>(6 * sections, Allocator.Temp);
            run.BaseVelocity = new NativeArray<double>(6 * sections, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            UnsafeUtility.MemCpy(run.BaseVelocity.GetUnsafePtr(), context.Solution, 6L * sections * sizeof(double));

            // Express each row in section coordinates, accounting for the part's offset from its section's centre. The
            // solver computes velocities rather than changes in velocity, so add the contribution from the bodies'
            // current velocities back to each boundary row's right-hand side.
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
        /// Converts the columns on one side of a row from body coordinates to section coordinates. The angular columns
        /// become angular + r x linear, where r points from the section's centre to the part's centre. Returns the
        /// contribution to the row's rate from the body's current velocity.
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

        /// <summary>
        /// After solving a vessel island, computes the impulses for the kernel's rows from the new section velocities.
        /// Uses the boundary impulses as external impulses on the parts, then computes every edge's load, using inverse
        /// dynamics for locked edges. Finally, computes the impulse needed for each body to follow its section's
        /// motion. For a push-out, treats the sections as initially at rest and writes only the body impulses.
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
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind,
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

            // The stages of the fused vessel solve that move or integrate the bodies and also write the loads. Their
            // items are the parts and then the closures. With FillNext, the stage that moves the bodies also fills
            // each part's pose and velocity for the next substep.
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

            /// <summary>
            /// For <see cref="FillEdges"/>, <see cref="MoveBodiesAndWriteLoads"/> and
            /// <see cref="IntegrateBodiesAndWriteLoads"/>: the number of parts, which come before the closures in the
            /// stage's items. For <see cref="FillEdges"/>, also a flag that a task sets when a lock state changes.
            /// </summary>
            public int VesselParts;
            public int Rebuild;

            /// <summary>For <see cref="MoveBodies"/>: the bodies that the fused vessel solve moves and integrates.</summary>
            public NativeArray<MotionVelocity> MotionVelocities;
            public NativeArray<MotionData> MotionDatas;

            /// <summary>
            /// For <see cref="UpdateJacobians"/> and <see cref="BuildRows"/>: the island's Jacobians and each one's first
            /// row, the world's bodies, the substep, where the rows go, the regularization that the rows share, whether
            /// the island's contacts are in the solve, and a flag that a task sets when a row needs a bounded solve.
            /// </summary>
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
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind, (count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
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
        /// Computes the impulse needed for a part's body to follow the linear and angular motion of its section. After
        /// a solve, the target position accounts for the section's rotation over the timestep, so a locked section
        /// remains rigid. For a push-out, treats the section as initially at rest.
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
                targetLinear = sectionLinear + (math.mul(VectorMath.Rotation(turn), r) - r) / h;
            }

            double3 changeLinear = pushOut ? targetLinear : targetLinear - p.LinearVelocity;
            double3 changeAngular = pushOut ? sectionAngular : sectionAngular - p.AngularVelocity;
            int worldBody = run.WorldBodies[body];
            MotionVelocity motion = run.MotionVelocities[worldBody];
            quaternion rotation = run.MotionDatas[worldBody].WorldFromMotion.rot;
            double3 inverseInertia = math.max((double3)motion.InverseInertia, 1e-12);
            double3 local = math.rotate(math.conjugate(rotation), (float3)changeAngular);
            linear[body] = VectorMath.Flushed(changeLinear / math.max((double)motion.InverseMass, 1e-12));
            angular[body] = VectorMath.Flushed((double3)math.rotate(rotation, (float3)(local / inverseInertia)));
        }

        /// <summary>
        /// Writes the loads on the island's edges and closures to the vessel set, averaging the force and torque over
        /// the step's substeps. Also records each edge's load ratio relative to its break strengths and whether it was
        /// locked. On the last substep, saves each edge's ratio for the next step's lock test. The team's lanes share
        /// this work by ranges of parts and closures.
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
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestVesselKind, (count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
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
