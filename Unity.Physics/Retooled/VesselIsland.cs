using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Numerics.Retooled;
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
            SolveForest(ref run, problem.RightHandSide, impulses);
            long solved = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            stepInput.DirectSolverCache.AddTime(DirectSolverCache.ForestSolveTime, solved - lap);
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
                ApplySparseImpulses(problem, null, linear, angular, compactBodies, headerList, impulses, motionVelocities, motionDatas,
                    sparseContacts, team);
                if (recovery != null)
                {
                    SolvePushOut(problem, headers, worldBodies, recovery, motionVelocities, motionDatas, stepInput.Timestep,
                        math.max(stepInput.MaxStaticDepenetrationVelocity, stepInput.MaxDynamicDepenetrationVelocity), &run);
                }
            }

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
        /// Arranges the island's parts in order: first the parts of each vessel, then a separate part for each moving
        /// body that is not in a vessel. If the vessels or extra bodies have changed, preserves the lock states for
        /// vessels with the same ID and structure version, then rebuilds the structure. On the first substep of each
        /// step, tests each edge's lock state against its load from the previous step.
        /// </summary>
        static unsafe void PrepareVesselParts(ref ForestRun run, int substep)
        {
            ForestIsland* forest = run.Forest;
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            VesselSet set = run.Vessels;
            int bodies = problem.BodyCount;

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

            bool same = forest->VesselMode && forest->VesselSignature.Length == signature.Length;
            for (int i = 0; same && i < signature.Length; i++)
            {
                same = forest->VesselSignature[i] == signature[i];
            }

            bool rebuild = !same;
            if (!same)
            {
                CarryVesselStates(forest, signature, run, partCount, closureCount);
            }

            // Fill this substep's parts, edges and closures.
            forest->Parts.Resize(partCount);
            forest->Edges.Resize(partCount);
            forest->Closures.Resize(closureCount);
            forest->PartBody.Resize(partCount);
            forest->BodyPart.Resize(math.max(1, bodies));
            for (int body = 0; body < bodies; body++)
            {
                forest->BodyPart[body] = -1;
            }

            int next = 0, nextClosure = 0;
            bool firstSubstep = substep <= 0;
            for (int i = 0; i < run.IslandVesselCount; i++)
            {
                VesselRecord vessel = set.Vessels[run.IslandVessels[i]];
                int start = next;
                for (int k = 0; k < vessel.PartCount; k++)
                {
                    int setPart = vessel.FirstPart + k;
                    int part = next++;
                    int body = CompactOf(run, set.PartBodies[setPart]);
                    forest->PartBody[part] = body;
                    if (body >= 0)
                    {
                        forest->BodyPart[body] = part;
                    }

                    VesselEdge edge = set.Edges[setPart];
                    edge.Parent = edge.Parent >= 0 ? start + edge.Parent : -1;
                    if (edge.Parent >= 0 && firstSubstep && TestLock(forest, part, edge))
                    {
                        rebuild = true;
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
                }

                for (int c = vessel.FirstClosure; c < vessel.FirstClosure + vessel.ClosureCount; c++)
                {
                    VesselClosure closure = set.Closures[c];
                    closure.PartA += start;
                    closure.PartB += start;
                    forest->Closures[nextClosure++] = closure;
                }
            }

            for (int body = 0; body < bodies; body++)
            {
                if (bodyIsPart[body] == 0)
                {
                    int part = next++;
                    forest->PartBody[part] = body;
                    forest->BodyPart[body] = part;
                    forest->Edges[part] = new VesselEdge { Parent = -1, State = VesselEdgeState.Yielding };
                }
            }

            for (int part = 0; part < partCount; part++)
            {
                forest->Parts[part] = PartOf(run, forest->PartBody[part]);
            }

            if (rebuild)
            {
                BuildVesselStructure(forest);
            }

            // The compact body indices can change between solves, so rebuild the mapping from bodies to sections.
            forest->BodySection.Resize(math.max(1, bodies));
            for (int body = 0; body < bodies; body++)
            {
                int part = forest->BodyPart[body];
                forest->BodySection[body] = part >= 0 ? forest->Structure.PartSection[part] : -1;
            }

            bodyIsPart.Dispose();
            signature.Dispose();
        }

        /// <summary>
        /// Returns the compact index of a body in the dynamics world, or -1 if the body is not in this island.
        /// </summary>
        static unsafe int CompactOf(in ForestRun run, int worldBody) =>
            worldBody >= 0 && worldBody < run.CompactOfWorldLength ? run.CompactOfWorld[worldBody] : -1;

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
        /// Rebuilds the arrangement of the island's parts and preserves each part's lock state if its vessel has the
        /// same ID and structure version. For a new vessel, edges that are allowed to lock start locked, so the craft
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
                bool kept = old.TryGetValue(vessel.Id, out int2 before) && (uint)before.x == vessel.StructureVersion;
                for (int k = 0; k < vessel.PartCount; k++)
                {
                    int part = next + k;
                    if (kept)
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
            };

            int locked = 0;
            for (int part = 0; part < parts; part++)
            {
                locked += forest->Edges[part].Parent >= 0 && forest->Edges[part].State == VesselEdgeState.Locked ? 1 : 0;
            }

            forest->LockedPairs = locked;
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
            RunVesselStage(ref run, VesselTasks.ComputeSections, sections);

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

                // FinishSections changes the kernel's copies of the parts. Save and restore those copies here so later
                // calculations in this substep use the original poses. The next substep fills them from the bodies
                // again.
                var poses = new NativeArray<VesselPart>(forest->Parts.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                UnsafeUtility.MemCpy(poses.GetUnsafePtr(), forest->Parts.Ptr, forest->Parts.Length * (long)sizeof(VesselPart));
                RunVesselStage(ref run, VesselTasks.FinishSections, sections);

                UnsafeUtility.MemCpy(forest->Parts.Ptr, poses.GetUnsafeReadOnlyPtr(), forest->Parts.Length * (long)sizeof(VesselPart));
                poses.Dispose();
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

            public ForestRun* Run;
            public int Phase;
            public int Count;
            public float3* Linear;
            public float3* Angular;
            public bool PushOut;
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
        /// locked. On the last substep, saves each edge's ratio for the next step's lock test.
        /// </summary>
        static unsafe void WriteVesselLoads(ref ForestRun run, int substep, int substeps)
        {
            ForestIsland* forest = run.Forest;
            VesselSet set = run.Vessels;
            int count = substep < 0 ? 1 : math.max(1, substeps);
            bool first = substep <= 0;
            bool last = substep < 0 || substep >= substeps - 1;
            for (int i = 0; i < run.IslandVesselCount; i++)
            {
                VesselRecord vessel = set.Vessels[run.IslandVessels[i]];
                int start = forest->VesselPartStart[i];
                for (int k = 0; k < vessel.PartCount; k++)
                {
                    int part = start + k;
                    int setPart = vessel.FirstPart + k;
                    VesselEdge edge = set.Edges[setPart];
                    VesselEdgeLoad load = edge.Parent >= 0 ? forest->KernelEdgeLoads[part] : default;
                    VesselEdgeLoad sum = first ? default : set.EdgeLoads[setPart];
                    sum.Force += load.Force / count;
                    sum.Torque += load.Torque / count;
                    sum.Ratio = Ratio(sum, edge.BreakForce, edge.BreakTorque);
                    set.EdgeLoads[setPart] = sum;
                    set.EdgeLocked[setPart] = forest->Edges[part].State == VesselEdgeState.Locked;
                    if (last)
                    {
                        forest->PartRatio[part] = (float)sum.Ratio;
                    }
                }
            }

            int nextClosure = 0;
            for (int i = 0; i < run.IslandVesselCount; i++)
            {
                VesselRecord vessel = set.Vessels[run.IslandVessels[i]];
                for (int c = 0; c < vessel.ClosureCount; c++)
                {
                    int setClosure = vessel.FirstClosure + c;
                    VesselEdgeLoad load = forest->KernelClosureLoads[nextClosure++];
                    VesselEdgeLoad sum = first ? default : set.ClosureLoads[setClosure];
                    sum.Force += load.Force / count;
                    sum.Torque += load.Torque / count;
                    set.ClosureLoads[setClosure] = sum;
                }
            }
        }

        static double Ratio(in VesselEdgeLoad load, double breakForce, double breakTorque)
        {
            double force = breakForce > 0.0 && !double.IsInfinity(breakForce) ? math.length(load.Force) / breakForce : 0.0;
            double torque = breakTorque > 0.0 && !double.IsInfinity(breakTorque) ? math.length(load.Torque) / breakTorque : 0.0;
            return math.max(force, torque);
        }
    }
}
