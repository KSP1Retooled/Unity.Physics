using static Unity.Physics.Solver;
using System;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Linear.Solvers.Lowlevel;
using Unity.Numerics.Kernels;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Dynamics.Vessels.Structure;
using Unity.Profiling;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    /// <summary>Factor and solve forest sections through the vessel kernels, then recover the row impulses.</summary>
    internal static class ForestSolver
    {
        static readonly ProfilerMarker ForestFactorMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.Factor");
        static readonly ProfilerMarker ForestBoundaryMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.Boundary");
        static readonly ProfilerMarker ForestSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.Solve");

        // The serial parts of SolveForest, which the leader runs alone.
        static readonly ProfilerMarker ForestHeldMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.HeldRows");
        static readonly ProfilerMarker ForestBoundaryInputMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.BoundaryInput");
        static readonly ProfilerMarker ForestDenseMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.Dense");
        static readonly ProfilerMarker ForestSpreadMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.Spread");
        static readonly ProfilerMarker ForestRowImpulsesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.RowImpulses");
        static readonly ProfilerMarker ForestStructureMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Forest.Structure");

        const byte ForestRowClass = 0;
        internal const byte BoundaryRowClass = 1;
        internal const byte FixedRowClass = 2;

        /// <summary>A forest row between two bodies of one section: a locked pair's, whose load comes from inverse dynamics, or an inert closure's.</summary>
        internal const byte SectionRowClass = 3;

        /// <summary>
        /// Boundary row excluded by a cap. Keep its warm impulse unless SolveHeldRows adjusts a normal row
        /// for separation, closing motion or deep penetration.
        /// </summary>
        const byte HeldRowClass = 4;

        /// <summary>
        /// One-sided joint row predicted to remain beyond DormantMargin from its stop.
        /// Exclude it from this boundary solve and reevaluate next solve.
        /// </summary>
        const byte DormantRowClass = 5;

        /// <summary>
        /// Forest and boundary state for one island solve, including reusable factor, row classes and W =
        /// J_b*A^-1*J_b^T + E_b.
        /// </summary>
        internal unsafe struct ForestRun
        {
            public ForestIsland* Forest;
            public SparseConstraintSolver.Problem Problem;
            public bool Bounded;

            /// <summary>
            /// Caller guarantee that vessels, structure versions and body numbering match the cached layout,
            /// and that no bodies lie outside vessels. Allows skipping signature comparison.
            /// </summary>
            public bool KnownLayout;

            /// <summary>Each block's Jacobian, for its regularization.</summary>
            public JacobianHeader** Headers;

            // Rows transformed to section coordinates for merged bodies; otherwise use the problem's original
            // rows.
            public float* Jacobian;

            public NativeArray<float> SectionJacobian;

            /// <summary>Each row's class, and each block's two sections, or -1 for a side that doesn't move.</summary>
            public NativeArray<byte> RowClass;
            public NativeArray<int2> BlockSections;
            public NativeArray<int> RowBlock;

            /// <summary>The problem's forest blocks, in order, as the island's pairs number them.</summary>
            public NativeList<int> ForestBlocks;

            /// <summary>The boundary rows, the dense block over them, and the bounds and couplings they take.</summary>
            public NativeArray<int> Boundary;
            public NativeArray<int> BoundaryOf;
            public NativeArray<double> W;
            public int BoundaryCount;

            // Vessel boundary caps and stable row keys for state reuse between solves.
            public BoundaryCaps Caps;
            public NativeArray<ulong> BoundaryKey;

            /// <summary>For a vessel island: the substep that the run solves, for the boundary's counters.</summary>
            public int Substep;

            // Optional world bodies used to identify boundary bodies outside vessels by entity version.
            public RigidBody* RigidBodies;
            public int RigidBodyCount;

            /// <summary>The change of velocity of every section, and a vector of the same size to work in.</summary>
            public NativeArray<double> Velocity;
            public NativeArray<double> Work;

            /// <summary>
            /// Section centres and momentum-conserving velocities, body offsets and original velocities.
            /// Empty when each section has only one body.
            /// </summary>
            public NativeArray<double3> SectionCenter;
            public NativeArray<double3> SectionLinear;
            public NativeArray<double3> SectionAngular;
            public NativeArray<double3> Arm;
            public NativeArray<double3> OwnLinear;
            public NativeArray<double3> OwnAngular;
            public bool MultiPart;

            // Solved section motion and each merged body's remaining impulse required from locked edges.
            public NativeArray<double3> SettleLinear;
            public NativeArray<double3> SettleAngular;
            public NativeArray<double3x3> SettleTurn;
            public NativeArray<double6> Need;

            /// <summary>Each block whose bodies share a section, whose rows have no drift to aim at; empty without such.</summary>
            public NativeArray<byte> BlockInSection;

            public int* WorldBodies;

            /// <summary>Each body of the dynamics world's compact number in the island, or -1.</summary>
            public int* CompactOfWorld;
            public int CompactOfWorldLength;
            public MotionVelocity* MotionVelocities;
            public MotionData* MotionDatas;
            public float Timestep;

            /// <summary>The lanes that share the island's solve, or null.</summary>
            public WorkTeam* Team;

            // Section velocities from vessel and fixed rows before boundary impulses are applied.
            public NativeArray<double> BaseVelocity;

            /// <summary>Whether the run solves a vessel island with the vessel kernel's own rows.</summary>
            public bool VesselMode;

            /// <summary>
            /// For a vessel island: the number of substeps in the step, and whether the strut loads are solved once at the
            /// step's last substep from the mean of the substeps' loads, which needs a forest that lasts the whole step.
            /// </summary>
            public int Substeps;

            public bool DeferStrutLoads;

            /// <summary>For a vessel island: the step's vessels and which of them the island holds.</summary>
            public VesselSet Vessels;
            public int* IslandVessels;
            public int IslandVesselCount;

            /// <summary>How long the factor, the boundary block and any structure rebuild took, in profiler timestamp units.</summary>
            public long FactorTicks;
            public long BoundaryTicks;
            public long StructureTicks;

            /// <summary>Whether the structure was rebuilt for this solve, and how many pairs changed lock state.</summary>
            public bool Rebuilt;
            public int LockChanges;
            public int ContactLockChanges;

            // Profiler ticks for serial classification, lock decisions and section/body/subtree sums.
            public long ClassifyTicks;
            public long DecideTicks;
            public long SumsTicks;
            public long BodyImpulseTicks;
            public long SubtreeTicks;

            /// <summary>The largest momentum left over at a section's first body by its inverse dynamics, in N s.</summary>
            public double Residual;

            public void Dispose()
            {
                RowClass.Dispose();
                BlockSections.Dispose();
                RowBlock.Dispose();
                if (ForestBlocks.IsCreated)
                {
                    ForestBlocks.Dispose();
                }

                Boundary.Dispose();
                BoundaryOf.Dispose();
                W.Dispose();
                if (BoundaryKey.IsCreated)
                {
                    BoundaryKey.Dispose();
                }
                Velocity.Dispose();
                Work.Dispose();
                if (SectionJacobian.IsCreated)
                {
                    SectionJacobian.Dispose();
                }

                if (SectionCenter.IsCreated)
                {
                    SectionCenter.Dispose();
                    SectionLinear.Dispose();
                    SectionAngular.Dispose();
                    Arm.Dispose();
                    OwnLinear.Dispose();
                    OwnAngular.Dispose();
                }

                if (BaseVelocity.IsCreated)
                {
                    BaseVelocity.Dispose();
                }

                if (BlockInSection.IsCreated)
                {
                    BlockInSection.Dispose();
                    SettleLinear.Dispose();
                    SettleAngular.Dispose();
                    SettleTurn.Dispose();
                    Need.Dispose();
                }
            }
        }

        /// <summary>
        /// Solve forest and boundary rows, writing impulses. Return false without changes if no forest rows exist
        /// or the boundary exceeds its cap.
        /// </summary>
        internal static unsafe bool TryStartForest(in SparseConstraintSolver.Problem problem, JacobianHeader** headers, bool bounded,
            ForestIsland* forest, in ForestSettings settings, WorkTeam* team, int* worldBodies, RigidBody* rigidBodies,
            int rigidBodyCount, in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, float timestep, int substep,
            out ForestRun run, out int fallbackBoundaryRows)
        {
            run = default;
            fallbackBoundaryRows = 0;
            long classifyStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            int rows = problem.RowCount;
            var coupled = new NativeArray<bool>(rows, Allocator.Temp);
            for (int c = 0; bounded && c < problem.CouplingCount; c++)
            {
                BoundCoupling coupling = problem.Couplings[c];
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    coupled[row] = true;
                }
            }

            var rowClass = new NativeArray<byte>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var rowBlock = new NativeArray<int>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var forestBlocks = new NativeList<int>(problem.BlockCount, Allocator.Temp);
            var treeBlocks = new NativeList<bool>(problem.BlockCount, Allocator.Temp);

            // What each forest block allows its pair: bit 0 lockable, bit 1 holdable, bit 2 every row a forest row or idle.
            var blockAllows = new NativeList<byte>(problem.BlockCount, Allocator.Temp);
            var bodyContact = new NativeArray<byte>(problem.BodyCount, Allocator.Temp);

            // Each block's rows by tasks on the team, then the forest blocks in order on this lane.
            int classifyTasks = (problem.BlockCount + BlocksPerSectionTask - 1) / BlocksPerSectionTask;
            var blockKind = new NativeArray<byte>(problem.BlockCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var taskRows = new NativeArray<int2>(math.max(1, classifyTasks), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var classify = new ClassifyTasks
            {
                Problem = problem,
                Headers = headers,
                Bounded = bounded,
                Coupled = (bool*)coupled.GetUnsafeReadOnlyPtr(),
                RowClass = (byte*)rowClass.GetUnsafePtr(),
                RowBlock = (int*)rowBlock.GetUnsafePtr(),
                BlockKind = (byte*)blockKind.GetUnsafePtr(),
                BodyContact = (byte*)bodyContact.GetUnsafePtr(),
                TaskRows = (int2*)taskRows.GetUnsafePtr(),
            };
            WorkTeam.RunAll(team, &classify, WorkTeam.ForestClassifyKind, classifyTasks, null);
            int forestRows = 0, boundaryRows = 0;
            for (int t = 0; t < classifyTasks; t++)
            {
                forestRows += taskRows[t].x;
                boundaryRows += taskRows[t].y;
            }

            for (int b = 0; b < problem.BlockCount; b++)
            {
                byte kind = blockKind[b];
                if ((kind & ClassifyTasks.Forest) != 0)
                {
                    forestBlocks.Add(b);
                    treeBlocks.Add((kind & ClassifyTasks.Tree) != 0);
                    blockAllows.Add((byte)(kind & 7));
                }
            }

            blockKind.Dispose();
            taskRows.Dispose();

            // A coupling's rows and its sources have to be boundary rows, which contacts' always are.
            bool couplingsInBoundary = true;
            for (int c = 0; bounded && c < problem.CouplingCount; c++)
            {
                BoundCoupling coupling = problem.Couplings[c];
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    couplingsInBoundary &= rowClass[row] == BoundaryRowClass;
                }

                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    couplingsInBoundary &= rowClass[row] == BoundaryRowClass;
                }
            }

            fallbackBoundaryRows = boundaryRows;
            if (forestRows == 0 || boundaryRows > settings.BoundaryRowCap || !couplingsInBoundary)
            {
                bodyContact.Dispose();
                coupled.Dispose();
                rowClass.Dispose();
                rowBlock.Dispose();
                forestBlocks.Dispose();
                treeBlocks.Dispose();
                blockAllows.Dispose();
                return false;
            }

            long classified = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;

            // Cache pairs by body identity so that reused world indices cannot inherit destroyed bodies' lock
            // states.
            int bodyCount = problem.BodyCount;
            var identities = new NativeArray<ulong>(math.max(1, bodyCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int b = 0; b < bodyCount; b++)
            {
                identities[b] = ForestIsland.BodyIdentity(rigidBodies, rigidBodyCount, worldBodies[b]);
            }

            ref UnsafeList<ulong> signature = ref forest->Signature;
            bool same = signature.Length == 1 + bodyCount + forestBlocks.Length && signature[0] == (ulong)bodyCount;
            for (int b = 0; same && b < bodyCount; b++)
            {
                same = signature[1 + b] == identities[b];
            }

            for (int f = 0; same && f < forestBlocks.Length; f++)
            {
                same = signature[1 + bodyCount + f] == BlockKey(problem.Blocks[forestBlocks[f]], treeBlocks[f]);
            }

            long structureStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            bool rebuild = !same || forest->StructureBuilds == 0;
            if (!same)
            {
                signature.Clear();
                signature.Add((ulong)bodyCount);
                for (int b = 0; b < bodyCount; b++)
                {
                    signature.Add(identities[b]);
                }

                for (int f = 0; f < forestBlocks.Length; f++)
                {
                    signature.Add(BlockKey(problem.Blocks[forestBlocks[f]], treeBlocks[f]));
                }

                forest->BuildPairs(problem, (int*)forestBlocks.GetUnsafeReadOnlyPtr(), (bool*)treeBlocks.GetUnsafeReadOnlyPtr(),
                    forestBlocks.Length, (ulong*)identities.GetUnsafeReadOnlyPtr());
            }

            identities.Dispose();

            long decideStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;

            // Lock eligible forest pairs below the lock angle and keep them locked until their predicted
            // deflection exceeds yield.
            int lockChanges = 0, contactChanges = 0;
            int pairCount = forest->PairCount;
            for (int p = 0; p < pairCount; p++)
            {
                int2 bodies = forest->PairBodies[p];
                byte touches = (byte)(bodyContact[bodies.x] | bodyContact[bodies.y]);
                forest->PairContact[p] = touches;
                bool locked = forest->PairLocked[p] != 0;
                int held = forest->PairHeld[p];
                if (substep <= 0 && held < int.MaxValue / 2)
                {
                    forest->PairHeld[p] = ++held;
                }

                bool settled = held >= settings.MinimumHoldSteps;
                bool next = false;
                if (settings.Locking && forest->PairEdge[p] != 0 && forest->PairTree[p] != 0)
                {
                    int allows = 7;
                    for (int k = forest->PairBlockStart[p]; k < forest->PairBlockStart[p + 1]; k++)
                    {
                        allows &= blockAllows[forest->PairBlocks[k]];
                    }

                    float deflection = forest->PairDeflection[p];
                    float last = forest->PairDeflectionStep[p];
                    next = locked
                        ? (allows & 6) == 6 && last <= 2.0f * settings.YieldAngle && (deflection <= settings.YieldAngle || !settled)
                        : (allows & 5) == 5 && deflection < settings.LockAngle && settled;
                }

                if (next != locked)
                {
                    forest->PairLocked[p] = (byte)(next ? 1 : 0);
                    forest->PairHeld[p] = 0;
                    lockChanges++;
                    contactChanges += touches;
                }
            }

            bodyContact.Dispose();

            long decided = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            rebuild |= lockChanges > 0;
            if (rebuild)
            {
                using (ForestStructureMarker.Auto())
                {
                    forest->BuildStructure(problem, (int*)forestBlocks.GetUnsafeReadOnlyPtr(), forestBlocks.Length);
                }
            }

            long structureTicks = rebuild ? Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - structureStarted : 0;
            treeBlocks.Dispose();
            blockAllows.Dispose();

            int sections = forest->SectionCount;
            ref VesselContext context = ref forest->Context;
            int* bodySection = forest->BodySection.Ptr;
            var blockSections = new NativeArray<int2>(problem.BlockCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                blockSections[b] = new int2(block.BodyA >= 0 ? bodySection[block.BodyA] : -1, block.BodyB >= 0 ? bodySection[block.BodyB] : -1);
            }

            // A forest row between two bodies of one section leaves the solve.
            for (int f = 0; f < forestBlocks.Length; f++)
            {
                if (forest->BlockLink[f] != ForestIsland.InSection)
                {
                    continue;
                }

                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[forestBlocks[f]];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (rowClass[row] == ForestRowClass)
                    {
                        rowClass[row] = SectionRowClass;
                    }
                }
            }

            run = new ForestRun
            {
                Forest = forest,
                Problem = problem,
                Bounded = bounded,
                Headers = headers,
                Jacobian = problem.Jacobian,
                RowClass = rowClass,
                BlockSections = blockSections,
                RowBlock = rowBlock,
                ForestBlocks = forestBlocks,
                Velocity = new NativeArray<double>(6 * sections, Allocator.Temp),
                Work = new NativeArray<double>(6 * sections, Allocator.Temp),
                MultiPart = forest->MultiPartSections > 0,
                WorldBodies = worldBodies,
                MotionVelocities = (MotionVelocity*)motionVelocities.GetUnsafeReadOnlyPtr(),
                MotionDatas = (MotionData*)motionDatas.GetUnsafeReadOnlyPtr(),
                Timestep = timestep,
                Team = team,
                StructureTicks = structureTicks,
                Rebuilt = rebuild,
                LockChanges = lockChanges,
                ContactLockChanges = contactChanges,
                ClassifyTicks = classified - classifyStarted,
                DecideTicks = decided - decideStarted,
            };

            for (int i = 0; i < problem.BodyCount; i++)
            {
                forest->Sections[bodySection[i]] = new VesselSection
                {
                    Mass = problem.Bodies[i].Mass,
                    Inertia = problem.Bodies[i].Inertia,
                };
            }

            if (run.MultiPart)
            {
                PrepareSections(ref run);
            }

            float* jacobian = run.Jacobian;
            long factorStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            using (ForestFactorMarker.Auto())
            {
                // Each link's rows, and the rows on one moving body, which go to its section's diagonal.
                int links = forest->LinkCount;
                int* linkRowStart = forest->LinkRowStart.Ptr;
                UnsafeUtility.MemClear(linkRowStart, (links + 1) * (long)sizeof(int));
                UnsafeUtility.MemClear(forest->ExtraDiagonal.Ptr, forest->ExtraDiagonal.Length * (long)sizeof(double));
                var rowEntry = new NativeArray<int>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int f = 0; f < forestBlocks.Length; f++)
                {
                    int b = forestBlocks[f];
                    SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                    int linked = forest->BlockLink[f];
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        rowEntry[row] = -1;
                        if (rowClass[row] != ForestRowClass)
                        {
                            continue;
                        }

                        if (linked >= 0)
                        {
                            rowEntry[row] = row * 2 + (linked & 1);
                            linkRowStart[(linked >> 1) + 1]++;
                        }
                        else
                        {
                            // One moving side: w j j^T on its section's diagonal.
                            float* columns = jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                            int2 sides = blockSections[b];
                            int s = sides.x >= 0 ? sides.x : sides.y;
                            double6 j = JacobianRowBlocks.Load(sides.x >= 0 ? columns : columns + 6);
                            Block6x6.AddOuter(forest->ExtraDiagonal.Ptr + Block6x6.Size * s, 1.0 / problem.Epsilon[row], j, j);
                        }
                    }
                }

                for (int k = 0; k < links; k++)
                {
                    linkRowStart[k + 1] += linkRowStart[k];
                }

                forest->LinkRowList.Resize(math.max(1, linkRowStart[links]));
                int* linkRowList = forest->LinkRowList.Ptr;
                var linkFill = new NativeArray<int>(math.max(1, links), Allocator.Temp);
                for (int f = 0; f < forestBlocks.Length; f++)
                {
                    int linked = forest->BlockLink[f];
                    if (linked < 0)
                    {
                        continue;
                    }

                    SparseConstraintSolver.ConstraintBlock block = problem.Blocks[forestBlocks[f]];
                    int link = linked >> 1;
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        if (rowClass[row] == ForestRowClass)
                        {
                            linkRowList[linkRowStart[link] + linkFill[link]++] = rowEntry[row];
                        }
                    }
                }

                linkFill.Dispose();
                rowEntry.Dispose();

                context.RowJacobian = jacobian;
                context.RowEpsilon = problem.Epsilon;
                context.RowRhs = problem.RightHandSide;
                context.LinkRowStart = linkRowStart;
                context.LinkRowList = linkRowList;
                context.ExtraDiagonal = forest->ExtraDiagonal.Ptr;
                context.ExtraRhs = null;

                VesselContext* contextPtr = (VesselContext*)UnsafeUtility.AddressOf(ref context);
                WorkTeam.RunAll(team, contextPtr, WorkTeam.ForestFactorKind, context.ChunkCount, null);
                var scratch = new NativeArray<double>(Block6x6.Size * math.max(1, context.MaxStructure), Allocator.Temp,
                    NativeArrayOptions.UninitializedMemory);
                context.SolveTop((double*)scratch.GetUnsafePtr());
                scratch.Dispose();
            }

            coupled.Dispose();
            run.FactorTicks = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - factorStarted;
            BuildBoundary(ref run, boundaryRows);
            return true;
        }

        /// <summary>
        /// Build W = J_b*A^-1*J_b^T + E_b from boundary-row elimination-tree paths.
        /// Each entry sums contributions from sections shared by the two paths.
        /// </summary>
        internal static unsafe void BuildBoundary(ref ForestRun run, int boundaryRows)
        {
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            ForestIsland* forest = run.Forest;
            ref VesselContext context = ref forest->Context;
            int rows = problem.RowCount;
            int sections = forest->SectionCount;
            float* jacobian = run.Jacobian;
            NativeArray<byte> rowClass = run.RowClass;
            NativeArray<int> rowBlock = run.RowBlock;
            NativeArray<int2> blockSections = run.BlockSections;
            long chooseStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            if (run.VesselMode && run.Bounded)
            {
                CapBoundary(ref run);
                if (!run.Caps.KeepDormantRows)
                {
                    MarkDormantRows(ref run);
                }
            }

            var boundary = new NativeArray<int>(math.max(1, boundaryRows), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var boundaryOf = new NativeArray<int>(rows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            int nb = 0;
            for (int row = 0; row < rows; row++)
            {
                boundaryOf[row] = -1;
                if (rowClass[row] == BoundaryRowClass)
                {
                    boundaryOf[row] = nb;
                    boundary[nb++] = row;
                }
            }

            long boundaryStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            long pathsWalked = boundaryStarted;
            var w = new NativeArray<double>(math.max(1, nb * nb), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            using (ForestBoundaryMarker.Auto())
            {
                var y = new NativeArray<double>(6 * sections, Allocator.Temp);
                var mark = new NativeArray<int>(sections, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var path = new NativeArray<int>(sections, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int s = 0; s < sections; s++)
                {
                    mark[s] = -1;
                }

                var pathStart = new NativeArray<int>(nb + 1, Allocator.Temp);
                var pathSection = new NativeList<int>(nb * 8, Allocator.Temp);
                var pathY = new NativeList<double6>(nb * 8, Allocator.Temp);
                var pathZ = new NativeList<double6>(nb * 8, Allocator.Temp);
                int* seeds = stackalloc int[2];
                double* yPtr = (double*)y.GetUnsafePtr();
                for (int i = 0; i < nb; i++)
                {
                    int row = boundary[i];
                    int2 sides = blockSections[rowBlock[row]];
                    float* columns = jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    int seedCount = 0;
                    if (sides.x >= 0)
                    {
                        seeds[seedCount++] = sides.x;
                        (double6.Load(yPtr + 6 * sides.x) + JacobianRowBlocks.Load(columns)).Store(yPtr + 6 * sides.x);
                    }

                    if (sides.y >= 0)
                    {
                        if (sides.y != sides.x)
                        {
                            seeds[seedCount++] = sides.y;
                        }

                        (double6.Load(yPtr + 6 * sides.y) + JacobianRowBlocks.Load(columns + 6)).Store(yPtr + 6 * sides.y);
                    }

                    int count = context.CollectPath(seeds, seedCount, (int*)path.GetUnsafePtr(), (int*)mark.GetUnsafePtr(), i);
                    context.ForwardPath(yPtr, (int*)path.GetUnsafePtr(), count);
                    pathStart[i] = pathSection.Length;
                    for (int k = 0; k < count; k++)
                    {
                        int s = path[k];
                        double6 ys = double6.Load(yPtr + 6 * s);
                        pathSection.Add(s);
                        pathY.Add(ys);
                        pathZ.Add(context.InverseDiagonal(s, ys));
                        default(double6).Store(yPtr + 6 * s);
                    }
                }

                pathStart[nb] = pathSection.Length;
                pathsWalked = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;

                // The team's lanes fill W by ranges of its rows.
                var tasks = new BoundaryTasks
                {
                    Count = nb,
                    PathStart = (int*)pathStart.GetUnsafePtr(),
                    PathSection = pathSection.GetUnsafePtr(),
                    PathY = pathY.GetUnsafePtr(),
                    PathZ = pathZ.GetUnsafePtr(),
                    W = (double*)w.GetUnsafePtr(),
                    Epsilon = problem.Epsilon,
                    Boundary = (int*)boundary.GetUnsafePtr(),
                };
                WorkTeam.RunAll(run.Team, &tasks, WorkTeam.ForestBoundaryKind,
                    (nb + BoundaryRowsPerTask - 1) / BoundaryRowsPerTask, null);

                y.Dispose();
                mark.Dispose();
                path.Dispose();
                pathStart.Dispose();
                pathSection.Dispose();
                pathY.Dispose();
                pathZ.Dispose();
            }

            long filled = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            run.Boundary = boundary;
            run.BoundaryOf = boundaryOf;
            run.W = w;
            run.BoundaryCount = nb;
            if (run.VesselMode && run.Bounded)
            {
                run.BoundaryKey = BoundaryKeys(ref run, (int*)boundary.GetUnsafeReadOnlyPtr(), nb);
            }
            long ended = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            run.BoundaryTicks = ended - boundaryStarted;
            if (run.VesselMode)
            {
                BoundaryCounters.AddBuild(boundaryStarted - chooseStarted + ended - filled, pathsWalked - boundaryStarted,
                    filled - pathsWalked);
            }
        }

        /// <summary>The number of rows of the boundary block W that one task fills.</summary>
        const int BoundaryRowsPerTask = 8;

        /// <summary>The data that the team's lanes share while they fill the boundary block W.</summary>
        unsafe struct BoundaryTasks
        {
            public int Count;
            public int* PathStart;
            public int* PathSection;
            public double6* PathY;
            public double6* PathZ;
            public double* W;
            public float* Epsilon;
            public int* Boundary;
        }

        /// <summary>
        /// Fill assigned upper-triangular rows of W and mirror entries below the diagonal.
        /// Each entry belongs to the task owning its smaller index, preventing overlapping writes.
        /// </summary>
        internal static unsafe void ForestBoundaryTask(void* context, int task)
        {
            var tasks = (BoundaryTasks*)context;
            int nb = tasks->Count;
            int* pathStart = tasks->PathStart;
            int* pathSection = tasks->PathSection;
            double6* pathY = tasks->PathY;
            double6* pathZ = tasks->PathZ;
            double* w = tasks->W;
            int end = math.min((task + 1) * BoundaryRowsPerTask, nb);
            for (int i = task * BoundaryRowsPerTask; i < end; i++)
            {
                for (int j = i; j < nb; j++)
                {
                    double sum = 0.0;
                    int p = pathStart[i], pEnd = pathStart[i + 1], q = pathStart[j], qEnd = pathStart[j + 1];
                    while (p < pEnd && q < qEnd)
                    {
                        int sp = pathSection[p], sq = pathSection[q];
                        if (sp == sq)
                        {
                            sum += double6.Dot(pathY[p], pathZ[q]);
                            p++;
                            q++;
                        }
                        else if (sp < sq)
                        {
                            p++;
                        }
                        else
                        {
                            q++;
                        }
                    }

                    w[i * nb + j] = sum;
                    w[j * nb + i] = sum;
                }

                w[i * nb + i] += tasks->Epsilon[tasks->Boundary[i]];
            }
        }

        /// <summary>
        /// Rank contact points or manifolds, then sort deepest first. Vary Tie between solves to avoid persistent
        /// tie bias.
        /// </summary>
        struct DepthItem : IComparable<DepthItem>
        {
            public int Rank;
            public float Depth;
            public uint Tie;
            public int Index;

            /// <summary>For a point that the cap on outside points sorts, the index of its manifold in the cap's list.</summary>
            public int Group;

            public int CompareTo(DepthItem other)
            {
                int order = Rank.CompareTo(other.Rank);
                if (order == 0)
                {
                    order = other.Depth.CompareTo(Depth);
                }

                if (order == 0)
                {
                    order = Tie.CompareTo(other.Tie);
                }

                return order != 0 ? order : Index.CompareTo(other.Index);
            }
        }

        /// <summary>
        /// Cap vessel boundary contact points and self-contact manifolds, marking excluded rows held.
        /// Keep deepest points, giving each outside manifold one point before selecting second points.
        /// Retain friction only for manifolds with selected points, subject to the combined row cap.
        /// </summary>
        static unsafe void CapBoundary(ref ForestRun run)
        {
            if (run.Headers == null)
            {
                return;
            }

            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            ForestIsland* forest = run.Forest;
            int perManifold = run.Caps.PointLimit;
            int selfLimit = run.Caps.SelfManifoldLimit;
            int outsideLimit = run.Caps.OutsidePointLimit;
            int vesselParts = 0;
            for (int v = 0; v < run.IslandVesselCount; v++)
            {
                vesselParts += run.Vessels.Vessels[run.IslandVessels[v]].PartCount;
            }

            ulong turn = (ulong)forest->BoundaryMemory.CapCount++ << 32;
            var points = new NativeList<DepthItem>(8, Allocator.Temp);
            var outsidePoints = new NativeList<DepthItem>(64, Allocator.Temp);
            var outsideBlocks = new NativeList<int2>(16, Allocator.Temp);
            var selfManifolds = new NativeList<DepthItem>(16, Allocator.Temp);
            int heldPoints = 0;
            long outsideRows = 0;
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                JacobianHeader* header = run.Headers[b];
                if (block.RowCount == 0 || header == null || header->Type != JacobianType.Contact ||
                    run.RowClass[block.FirstRow] != BoundaryRowClass)
                {
                    continue;
                }

                // The block's first rows are the normal rows of the points that are within reach, in the manifold's order.
                // The friction rows follow, if there are any.
                ref JacobianHeader contact = ref *header;
                int count = contact.AccessBaseJacobian<ContactJacobian>().BaseJacobian.NumContacts;
                points.Clear();
                for (int i = 0; i < count; i++)
                {
                    float distance = contact.AccessAngularJacobian(i).ContactDistance;
                    if (distance <= ContactRowReach)
                    {
                        int row = block.FirstRow + points.Length;
                        points.Add(new DepthItem { Depth = -distance, Tie = (uint)Mix((ulong)row | turn), Index = row });
                    }
                }

                int frictionRows = block.RowCount - points.Length;
                if (points.Length == 0 || (frictionRows != 0 && frictionRows != 3))
                {
                    continue;
                }

                points.Sort();
                for (int k = perManifold; k < points.Length; k++)
                {
                    run.RowClass[points[k].Index] = HeldRowClass;
                    heldPoints++;
                }

                if (IsSelfContact(run, forest, block, vesselParts))
                {
                    selfManifolds.Add(new DepthItem { Depth = points[0].Depth, Tie = (uint)Mix((ulong)b | turn), Index = b });
                }
                else
                {
                    int kept = math.min(perManifold, points.Length);
                    outsideRows += kept + frictionRows;
                    for (int k = 0; k < kept; k++)
                    {
                        DepthItem point = points[k];
                        point.Rank = k;
                        point.Group = outsideBlocks.Length;
                        outsidePoints.Add(point);
                    }

                    outsideBlocks.Add(new int2(b, points.Length));
                }
            }

            int heldManifolds = 0;
            if (selfManifolds.Length > selfLimit)
            {
                selfManifolds.Sort();
                for (int k = selfLimit; k < selfManifolds.Length; k++)
                {
                    SparseConstraintSolver.ConstraintBlock block = problem.Blocks[selfManifolds[k].Index];
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        run.RowClass[row] = HeldRowClass;
                    }

                    heldManifolds++;
                }
            }

            // Include retained manifolds' friction rows in the outside row cap.
            long rowLimit = outsideLimit >= int.MaxValue
                ? long.MaxValue
                : outsideLimit + 3L * (outsideLimit / math.min(perManifold, BoundaryCaps.DefaultPointsPerManifold));
            int keptOutside = outsidePoints.Length;
            int emptied = 0;
            if (outsidePoints.Length > outsideLimit || outsideRows > rowLimit)
            {
                outsidePoints.Sort();
                var touched = new NativeArray<bool>(outsideBlocks.Length, Allocator.Temp);
                long used = 0;
                keptOutside = 0;
                for (int k = 0; k < outsidePoints.Length; k++)
                {
                    DepthItem point = outsidePoints[k];
                    int2 owner = outsideBlocks[point.Group];
                    int cost = 1 + (touched[point.Group] ? 0 : problem.Blocks[owner.x].RowCount - owner.y);
                    if (keptOutside < outsideLimit && used + cost <= rowLimit)
                    {
                        touched[point.Group] = true;
                        used += cost;
                        keptOutside++;
                    }
                    else
                    {
                        run.RowClass[point.Index] = HeldRowClass;
                        heldPoints++;
                    }
                }

                touched.Dispose();
                for (int k = 0; k < outsideBlocks.Length; k++)
                {
                    SparseConstraintSolver.ConstraintBlock block = problem.Blocks[outsideBlocks[k].x];
                    int normalRows = outsideBlocks[k].y;
                    bool anyKept = false;
                    for (int row = block.FirstRow; row < block.FirstRow + normalRows; row++)
                    {
                        anyKept |= run.RowClass[row] == BoundaryRowClass;
                    }

                    emptied += anyKept ? 0 : 1;
                    for (int row = block.FirstRow + normalRows; !anyKept && row < block.FirstRow + block.RowCount; row++)
                    {
                        run.RowClass[row] = HeldRowClass;
                    }
                }
            }

            BoundaryCounters.AddCaps(heldPoints, heldManifolds, keptOutside, math.min(selfManifolds.Length, selfLimit), emptied);
            points.Dispose();
            outsidePoints.Dispose();
            outsideBlocks.Dispose();
            selfManifolds.Dispose();
        }

        /// <summary>
        /// Mark one-sided joint rows dormant when predicted motion keeps them beyond the stop margin.
        /// Use forest and fixed-row velocities before the boundary acts. Never exclude contact or coupled rows.
        /// </summary>
        static unsafe void MarkDormantRows(ref ForestRun run)
        {
            if (run.Headers == null || !run.BaseVelocity.IsCreated || run.Timestep <= 0.0f)
            {
                return;
            }

            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            int rows = problem.RowCount;
            var coupled = new NativeArray<bool>(math.max(1, rows), Allocator.Temp);
            for (int c = 0; c < problem.CouplingCount; c++)
            {
                BoundCoupling coupling = problem.Couplings[c];
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    coupled[row] = true;
                }

                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    coupled[row] = true;
                }
            }

            double* velocity = (double*)run.BaseVelocity.GetUnsafeReadOnlyPtr();
            double margin = BoundaryCaps.DormantMargin / (double)run.Timestep;
            int dormant = 0;
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                JacobianHeader* header = run.Headers[b];
                if (block.RowCount == 0 || header == null || header->Type == JacobianType.Contact)
                {
                    continue;
                }

                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (run.RowClass[row] != BoundaryRowClass || coupled[row])
                    {
                        continue;
                    }

                    float lower = problem.Lower[row], upper = problem.Upper[row];
                    bool pushes = lower == 0.0f && upper >= float.MaxValue;
                    bool pulls = upper == 0.0f && lower <= -float.MaxValue;
                    if (!pushes && !pulls)
                    {
                        continue;
                    }

                    // The row's free impulse has the sign of rhs - J v. A row that may only push stays slack while J v is
                    // above rhs, and a row that may only pull stays slack while J v is below rhs.
                    double speed = JacobianRowBlocks.Dot(velocity, run.BlockSections[run.RowBlock[row]],
                        run.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
                    double slack = speed - problem.RightHandSide[row];
                    if ((pushes && slack > margin) || (pulls && -slack > margin))
                    {
                        run.RowClass[row] = DormantRowClass;
                        dormant++;
                    }
                }
            }

            coupled.Dispose();
            BoundaryCounters.AddDormant(dormant);
        }

        /// <summary>Whether a block joins two parts of the same vessel.</summary>
        static unsafe bool IsSelfContact(in ForestRun run, ForestIsland* forest, in SparseConstraintSolver.ConstraintBlock block,
            int vesselParts)
        {
            if (block.BodyA < 0 || block.BodyB < 0)
            {
                return false;
            }

            int partA = forest->BodyPart[block.BodyA];
            int partB = forest->BodyPart[block.BodyB];
            return partA >= 0 && partB >= 0 && partA < vesselParts && partB < vesselParts &&
                forest->PartVesselStart[partA] == forest->PartVesselStart[partB];
        }

        /// <summary>
        /// Build stable boundary keys from body identities, block type/collider keys, duplicate-block occurrence
        /// and row position.
        /// Leave the low eight bits free for BoundaryMemory state and age. Count blocks even when capped or
        /// dormant.
        /// </summary>
        static unsafe NativeArray<ulong> BoundaryKeys(ref ForestRun run, int* boundary, int nb)
        {
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            var keys = new NativeArray<ulong>(math.max(1, nb), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var blockKeys = new NativeArray<ulong>(math.max(1, problem.BlockCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var seen = new NativeHashMap<ulong, int>(16, Allocator.Temp);
            var vesselIds = new NativeHashMap<int, ulong>(math.max(1, run.IslandVesselCount), Allocator.Temp);
            int vesselParts = 0;
            for (int v = 0; v < run.IslandVesselCount; v++)
            {
                VesselRecord vessel = run.Vessels.Vessels[run.IslandVessels[v]];
                vesselIds[vesselParts] = vessel.Id;
                vesselParts += vessel.PartCount;
            }

            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                JacobianHeader* header = run.Headers != null ? run.Headers[b] : null;
                ulong identity;
                if (header != null)
                {
                    identity = Mix(BodyKey(run, header->BodyPair.BodyIndexA, vesselParts, vesselIds) * 0x9E3779B97F4A7C15UL
                        + BodyKey(run, header->BodyPair.BodyIndexB, vesselParts, vesselIds)) ^ (ulong)header->Type;
                    if (header->Type == JacobianType.Contact && header->HasContactManifold)
                    {
                        ColliderKeyPair colliders = header->AccessColliderKeys();
                        identity = Mix(identity + (((ulong)colliders.ColliderKeyA.Value << 32) | colliders.ColliderKeyB.Value));
                    }
                }
                else
                {
                    int worldA = block.BodyA >= 0 ? run.WorldBodies[block.BodyA] : -1;
                    int worldB = block.BodyB >= 0 ? run.WorldBodies[block.BodyB] : -1;
                    identity = Mix(BodyKey(run, worldA, vesselParts, vesselIds) * 0x9E3779B97F4A7C15UL
                        + BodyKey(run, worldB, vesselParts, vesselIds)) ^ 0xFFUL;
                }

                int occurrence = 0;
                if (seen.TryGetValue(identity, out int before))
                {
                    occurrence = before + 1;
                }

                seen[identity] = occurrence;
                blockKeys[b] = Mix(identity + (ulong)occurrence * 0x9E3779B97F4A7C15UL);
            }

            for (int i = 0; i < nb; i++)
            {
                int row = boundary[i];
                int b = run.RowBlock[row];
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                JacobianHeader* header = run.Headers != null ? run.Headers[b] : null;
                ulong place = (ulong)(row - block.FirstRow);
                if (header != null && header->Type == JacobianType.Contact)
                {
                    place = ContactRowPlace(ref *header, row - block.FirstRow);
                }

                keys[i] = Mix(blockKeys[b] + place) & ~(ulong)BoundaryMemory.ValueMask;
            }

            blockKeys.Dispose();
            seen.Dispose();
            vesselIds.Dispose();
            return keys;
        }

        /// <summary>
        /// Return a stable body key from vessel ID and part index, or entity version for other bodies.
        /// Return zero for no body. Fall back to world index without body data or when KeyByWorldIndex is enabled.
        /// </summary>
        static unsafe ulong BodyKey(in ForestRun run, int world, int vesselParts, in NativeHashMap<int, ulong> vesselIds)
        {
            if (world < 0)
            {
                return 0;
            }

            int compact = world < run.CompactOfWorldLength ? run.CompactOfWorld[world] : -1;
            int part = compact >= 0 && compact < run.Forest->BodyPart.Length ? run.Forest->BodyPart[compact] : -1;
            if (part >= 0 && part < vesselParts)
            {
                int start = run.Forest->PartVesselStart[part];
                if (vesselIds.TryGetValue(start, out ulong id))
                {
                    return Mix(Mix(id) + (((ulong)(uint)(part - start) << 2) | 3));
                }
            }

            if (run.RigidBodies != null && world < run.RigidBodyCount && !run.Caps.KeyByWorldIndex)
            {
                return Mix(((ulong)(uint)run.RigidBodies[world].Entity.Version << 2) | 2);
            }

            // Without the bodies, a body that does not move is known by its index among the bodies that do not move,
            // and a moving body by its index in the world.
            return world >= run.CompactOfWorldLength
                ? Mix(((ulong)(uint)(world - run.CompactOfWorldLength) << 2) | 2)
                : Mix(((ulong)(uint)world << 2) | 1);
        }

        /// <summary>
        /// Contact row identity: manifold point index for normals, or 0x10000 plus friction-row index.
        /// </summary>
        static unsafe ulong ContactRowPlace(ref JacobianHeader header, int offset)
        {
            int count = header.AccessBaseJacobian<ContactJacobian>().BaseJacobian.NumContacts;
            int normals = 0;
            int point = -1;
            for (int k = 0; k < count; k++)
            {
                if (header.AccessAngularJacobian(k).ContactDistance <= ContactRowReach)
                {
                    if (normals == offset)
                    {
                        point = k;
                    }

                    normals++;
                }
            }

            return offset < normals && point >= 0 ? (ulong)point : 0x10000UL + (ulong)(offset - normals);
        }

        /// <summary>Mixes the bits of a 64-bit value, as the last step of MurmurHash3 does.</summary>
        static ulong Mix(ulong x)
        {
            x ^= x >> 33;
            x *= 0xFF51AFD7ED558CCDUL;
            x ^= x >> 33;
            x *= 0xC4CEB9FE1A85EC53UL;
            x ^= x >> 33;
            return x;
        }

        static ulong BlockKey(in SparseConstraintSolver.ConstraintBlock block, bool tree)
        {
            return (ulong)(uint)(block.BodyA + 1) | ((ulong)(uint)(block.BodyB + 1) << 31) | (tree ? 1UL << 62 : 0UL);
        }

        /// <summary>
        /// Compute section mass, centre, inertia and momentum-conserving velocities, then transform rows and
        /// targets to section motion.
        /// </summary>
        static unsafe void PrepareSections(ref ForestRun run)
        {
            long sumsStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            ForestIsland* forest = run.Forest;
            int sections = forest->SectionCount;
            int bodies = problem.BodyCount;
            int* sectionPartStart = forest->Structure.SectionPartStart.Ptr;
            int* sectionParts = forest->Structure.SectionParts.Ptr;
            int* partBody = forest->PartBody.Ptr;
            int* bodySection = forest->BodySection.Ptr;
            run.SectionCenter = new NativeArray<double3>(sections, Allocator.Temp);
            run.SectionLinear = new NativeArray<double3>(sections, Allocator.Temp);
            run.SectionAngular = new NativeArray<double3>(sections, Allocator.Temp);
            run.Arm = new NativeArray<double3>(bodies, Allocator.Temp);
            run.OwnLinear = new NativeArray<double3>(bodies, Allocator.Temp);
            run.OwnAngular = new NativeArray<double3>(bodies, Allocator.Temp);
            run.BlockInSection = new NativeArray<byte>(problem.BlockCount, Allocator.Temp);
            run.SettleLinear = new NativeArray<double3>(sections, Allocator.Temp);
            run.SettleAngular = new NativeArray<double3>(sections, Allocator.Temp);
            run.SettleTurn = new NativeArray<double3x3>(sections, Allocator.Temp);
            run.Need = new NativeArray<double6>(bodies, Allocator.Temp);

            for (int s = 0; s < sections; s++)
            {
                int start = sectionPartStart[s], end = sectionPartStart[s + 1];
                if (end - start < 2)
                {
                    continue;
                }

                double mass = 0.0;
                double3 moment = 0.0, momentum = 0.0;
                for (int k = start; k < end; k++)
                {
                    int i = partBody[sectionParts[k]];
                    int world = run.WorldBodies[i];
                    double m = problem.Bodies[i].Mass;
                    mass += m;
                    moment += m * (double3)run.MotionDatas[world].WorldFromMotion.pos;
                    momentum += m * (double3)run.MotionVelocities[world].LinearVelocity;
                }

                double3 center = moment / mass;
                double3x3 inertia = double3x3.zero;
                double3 angularMomentum = 0.0;
                for (int k = start; k < end; k++)
                {
                    int i = partBody[sectionParts[k]];
                    int world = run.WorldBodies[i];
                    double m = problem.Bodies[i].Mass;
                    double3 r = (double3)run.MotionDatas[world].WorldFromMotion.pos - center;
                    run.Arm[i] = r;
                    run.OwnLinear[i] = run.MotionVelocities[world].LinearVelocity;
                    run.OwnAngular[i] = BodyAngular(run, world);
                    double3x3 own = problem.Bodies[i].Inertia;
                    inertia += own + m * (math.dot(r, r) * double3x3.identity - new double3x3(r * r.x, r * r.y, r * r.z));
                    angularMomentum += math.mul(own, run.OwnAngular[i]) + m * math.cross(r, run.OwnLinear[i]);
                }

                run.SectionCenter[s] = center;
                run.SectionLinear[s] = momentum / mass;
                run.SectionAngular[s] = math.mul(math.inverse(inertia), angularMomentum);
                run.Forest->Sections[s] = new VesselSection { Mass = mass, Center = center, Inertia = inertia };
            }

            run.SumsTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - sumsStarted;

            // The rows on a body of a section of several act on the section at the body's offset: j_angular + r x j_linear.
            // Each block's rows are copied and turned by one task, on the team's lanes.
            int rows = problem.RowCount;
            long floats = SparseConstraintSolver.ColumnsPerRow * (long)rows;
            run.SectionJacobian = new NativeArray<float>((int)floats, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            run.Jacobian = (float*)run.SectionJacobian.GetUnsafePtr();
            var tasks = new SectionTasks
            {
                Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                Problem = problem,
                Phase = SectionTasks.PrepareRows,
            };
            WorkTeam.RunAll(run.Team, &tasks, WorkTeam.ForestSectionsKind,
                (problem.BlockCount + BlocksPerSectionTask - 1) / BlocksPerSectionTask, null);
        }

        const int BlocksPerSectionTask = 128;

        /// <summary>What classifying an island's rows shares among a team's lanes; each task takes a run of blocks.</summary>
        unsafe struct ClassifyTasks
        {
            /// <summary>A block's kind: what it allows its pair (bits 0 to 2), and whether it is a forest block, a tree one.</summary>
            public const byte Forest = 1 << 3;
            public const byte Tree = 1 << 4;

            public SparseConstraintSolver.Problem Problem;
            public JacobianHeader** Headers;
            public bool Bounded;
            public bool* Coupled;
            public byte* RowClass;
            public int* RowBlock;
            public byte* BlockKind;
            public byte* BodyContact;

            /// <summary>Each task's forest rows and boundary rows.</summary>
            public int2* TaskRows;
        }

        /// <summary>One task of classifying rows: its blocks' rows by class, and each block's kind.</summary>
        internal static unsafe void ForestClassifyTask(void* context, int task)
        {
            var tasks = (ClassifyTasks*)context;
            ref SparseConstraintSolver.Problem problem = ref tasks->Problem;
            bool bounded = tasks->Bounded;
            int forestRows = 0, boundaryRows = 0;
            int end = math.min((task + 1) * BlocksPerSectionTask, problem.BlockCount);
            for (int b = task * BlocksPerSectionTask; b < end; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                JacobianHeader* header = tasks->Headers[b];
                JacobianFlags flags = header->Flags;
                if (header->Type == JacobianType.Contact)
                {
                    // Several tasks may mark one body; they all write the same value.
                    if (block.BodyA >= 0)
                    {
                        tasks->BodyContact[block.BodyA] = 1;
                    }

                    if (block.BodyB >= 0)
                    {
                        tasks->BodyContact[block.BodyB] = 1;
                    }
                }

                bool forestBlock = header->Type != JacobianType.Contact && (flags & JacobianFlags.ForestRow) != 0
                    && (block.BodyA >= 0 || block.BodyB >= 0);
                bool allForest = true;
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    tasks->RowBlock[row] = b;
                    float lower = problem.Lower[row], upper = problem.Upper[row];
                    bool free = !bounded || (lower <= -float.MaxValue && upper >= float.MaxValue);
                    if (bounded && lower == upper && !tasks->Coupled[row])
                    {
                        // A limit out of reach is a fixed row with no impulse, which doesn't keep its pair from locking.
                        tasks->RowClass[row] = FixedRowClass;
                        allForest &= lower == 0.0f;
                    }
                    else if (forestBlock && free && problem.Epsilon[row] > 0.0f)
                    {
                        tasks->RowClass[row] = ForestRowClass;
                        forestRows++;
                    }
                    else
                    {
                        tasks->RowClass[row] = BoundaryRowClass;
                        boundaryRows++;
                        allForest = false;
                    }
                }

                tasks->BlockKind[b] = !forestBlock ? (byte)0 : (byte)(ClassifyTasks.Forest
                    | ((flags & JacobianFlags.ForestTree) != 0 ? ClassifyTasks.Tree : 0)
                    | ((flags & JacobianFlags.ForestLockable) != 0 ? 1 : 0)
                    | ((flags & JacobianFlags.ForestHoldable) != 0 ? 2 : 0) | (allForest ? 4 : 0));
            }

            tasks->TaskRows[task] = new int2(forestRows, boundaryRows);
        }
        const int PartsPerSectionTask = 64;

        /// <summary>What the team's section tasks share: the run, its problem, and the arrays a settle writes.</summary>
        unsafe struct SectionTasks
        {
            public const int PrepareRows = 0;
            public const int SettleBodies = 1;
            public const int SplitLoads = 2;

            public ForestRun* Run;
            public SparseConstraintSolver.Problem Problem;
            public float* Impulses;
            public float3* Linear;
            public float3* Angular;
            public bool PushOut;
            public int Phase;
        }

        /// <summary>One task of a section stage, for a team.</summary>
        internal static unsafe void ForestSectionTask(void* context, int task)
        {
            var tasks = (SectionTasks*)context;
            if (tasks->Phase == SectionTasks.PrepareRows)
            {
                PrepareRowsTask(ref *tasks->Run, tasks->Problem, task);
            }
            else if (tasks->Phase == SectionTasks.SettleBodies)
            {
                SettleBodiesTask(ref *tasks->Run, tasks, task);
            }
            else
            {
                SplitLoadsTask(ref *tasks->Run, tasks, task);
            }
        }

        /// <summary>One task's blocks of <see cref="PrepareSections"/>: their rows copied, turned to the sections, and moved.</summary>
        static unsafe void PrepareRowsTask(ref ForestRun run, in SparseConstraintSolver.Problem problem, int task)
        {
            ForestIsland* forest = run.Forest;
            int* bodySection = forest->BodySection.Ptr;
            float* jacobian = run.Jacobian;
            int end = math.min((task + 1) * BlocksPerSectionTask, problem.BlockCount);
            for (int b = task * BlocksPerSectionTask; b < end; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                long first = SparseConstraintSolver.ColumnsPerRow * (long)block.FirstRow;
                UnsafeUtility.MemCpy(jacobian + first, problem.Jacobian + first, SparseConstraintSolver.ColumnsPerRow * (long)block.RowCount * sizeof(float));
                bool multiA = block.BodyA >= 0 && Multi(forest, bodySection[block.BodyA]);
                bool multiB = block.BodyB >= 0 && Multi(forest, bodySection[block.BodyB]);
                if (!multiA && !multiB)
                {
                    continue;
                }

                run.BlockInSection[b] = (byte)(block.BodyA >= 0 && block.BodyB >= 0 && bodySection[block.BodyA] == bodySection[block.BodyB] ? 1 : 0);
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    // A row inside a section isn't solved, and an idle fixed row adds nothing: neither needs turning.
                    byte rowClass = run.RowClass[row];
                    if (rowClass == SectionRowClass || (rowClass == FixedRowClass && problem.Lower[row] == 0.0f))
                    {
                        continue;
                    }

                    float* columns = jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    double change = 0.0;
                    if (multiA)
                    {
                        change += ToSection(ref run, columns, block.BodyA);
                    }

                    if (multiB)
                    {
                        change += ToSection(ref run, columns + 6, block.BodyB);
                    }

                    if (rowClass != FixedRowClass)
                    {
                        problem.RightHandSide[row] += (float)change;
                    }
                }
            }
        }

        /// <summary>Radians per second below which a section's spin is taken as none.</summary>
        internal const double MinimumSpin = 1e-9;

        /// <summary>Whether section s has more than one body.</summary>
        internal static unsafe bool Multi(ForestIsland* forest, int s) =>
            forest->Structure.SectionPartStart[s + 1] - forest->Structure.SectionPartStart[s] > 1;

        /// <summary>A body's angular velocity in world space.</summary>
        static unsafe double3 BodyAngular(in ForestRun run, int world) =>
            math.rotate(run.MotionDatas[world].WorldFromMotion.rot, run.MotionVelocities[world].AngularVelocity);

        /// <summary>
        /// Convert row columns to section coordinates and return the target correction:
        /// j dot u_body - j_section dot u_section.
        /// </summary>
        static unsafe double ToSection(ref ForestRun run, float* columns, int body)
        {
            int s = run.Forest->BodySection[body];
            double3 linear = new double3(columns[0], columns[1], columns[2]);
            double3 angular = new double3(columns[3], columns[4], columns[5]);
            double own = math.dot(linear, run.OwnLinear[body]) + math.dot(angular, run.OwnAngular[body]);
            double3 moved = angular + math.cross(run.Arm[body], linear);
            columns[3] = (float)moved.x;
            columns[4] = (float)moved.y;
            columns[5] = (float)moved.z;
            return own - (math.dot(linear, run.SectionLinear[s]) + math.dot(moved, run.SectionAngular[s]));
        }

        /// <summary>
        /// Solve using the retained forest factor and boundary block, writing row impulses.
        /// Return false if pivoting exhausts its rounds without a feasible answer; impulses then come from bounded
        /// finishing sweeps.
        /// Velocity solves may apply them and count an unsettled solve; push-out skips them.
        /// </summary>
        internal static unsafe bool SolveForest(ref ForestRun run, float* rhs, float* impulses)
        {
            using ProfilerMarker.AutoScope span = ForestSolveMarker.Auto();

            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            float* jacobian = run.Jacobian;
            int rows = problem.RowCount;
            double* v = (double*)run.Velocity.GetUnsafePtr();
            double* work = (double*)run.Work.GetUnsafePtr();
            UnsafeUtility.MemClear(v, run.Velocity.Length * (long)sizeof(double));

            // A vessel run already has the sections' velocities without the boundary, from its own rows and the fixed rows.
            bool vesselStart = run.BaseVelocity.IsCreated;
            if (vesselStart)
            {
                UnsafeUtility.MemCpy(v, run.BaseVelocity.GetUnsafeReadOnlyPtr(), run.Velocity.Length * (long)sizeof(double));
            }

            // b = J_f^T E_f^-1 r_f, plus the known impulses of fixed rows; then the forest's own answer.
            for (int row = 0; row < rows && !vesselStart; row++)
            {
                byte rowClass = run.RowClass[row];
                double scale;
                if (rowClass == ForestRowClass)
                {
                    scale = rhs[row] / (double)problem.Epsilon[row];
                }
                else if (rowClass == FixedRowClass)
                {
                    scale = problem.Lower[row];
                }
                else
                {
                    continue;
                }

                JacobianRowBlocks.Add(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row, scale);
            }

            if (!vesselStart)
            {
                SolveAgainstFactor(ref run, v);
            }

            // In a vessel island's velocity solve, the held rows whose contacts separate give back their warm start before
            // the boundary is solved, so that the boundary sees the velocities without it.
            bool heldSolved = run.VesselMode && run.Bounded;
            if (heldSolved && vesselStart)
            {
                using (ForestHeldMarker.Auto())
                {
                    SolveHeldRows(ref run, rhs, impulses, v, false);
                }
            }

            // The boundary: (J_b A^-1 J_b^T + E_b) z = r_b - J_b v0, bounded.
            int nb = run.BoundaryCount;
            bool solved = true;
            if (nb > 0)
            {
                ProfilerMarker.AutoScope inputSpan = ForestBoundaryInputMarker.Auto();
                var q = new NativeArray<double>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var lower = new NativeArray<double>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var upper = new NativeArray<double>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var states = new NativeArray<byte>(nb, Allocator.Temp, NativeArrayOptions.ClearMemory);
                var z = new NativeArray<double>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

                // Warm start vessel boundary states by stable row keys and cap pivot rounds. Push-out uses the
                // plain solve.
                bool vesselSolve = vesselStart && run.Bounded && run.BoundaryKey.IsCreated;
                bool warm = vesselSolve && !run.Caps.ColdStart;
                ref UnsafeList<ulong> memory = ref run.Forest->BoundaryMemory.States;
                int warmRows = 0;
                var remembered = new NativeArray<byte>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var ages = new NativeArray<byte>(nb, Allocator.Temp, NativeArrayOptions.ClearMemory);
                for (int i = 0; i < nb; i++)
                {
                    remembered[i] = 0xFF;
                    int row = run.Boundary[i];
                    q[i] = rhs[row] - JacobianRowBlocks.Dot(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
                    lower[i] = run.Bounded ? problem.Lower[row] : double.NegativeInfinity;
                    upper[i] = run.Bounded ? problem.Upper[row] : double.PositiveInfinity;
                    if (run.Bounded && problem.InitialStates != null)
                    {
                        states[i] = problem.InitialStates[row];
                    }

                    if (warm && memory.IsCreated)
                    {
                        int found = BoundaryMemory.Find(memory, run.BoundaryKey[i]);
                        if (found >= 0)
                        {
                            states[i] = (byte)(memory[found] & BoundaryMemory.StateMask);
                            remembered[i] = states[i];
                            ages[i] = (byte)(BoundaryMemory.Age(memory[found]) + 1);
                            warmRows++;
                        }
                    }
                }

                // Move capped-out source impulses into the held contribution. Omit couplings whose own rows are
                // held.
                var couplings = new NativeList<BoundCoupling>(problem.CouplingCount, Allocator.Temp);
                for (int c = 0; run.Bounded && c < problem.CouplingCount; c++)
                {
                    BoundCoupling coupling = problem.Couplings[c];
                    if (run.BoundaryOf[coupling.FirstRow] < 0)
                    {
                        continue;
                    }

                    int firstSource = -1;
                    int sourceCount = 0;
                    for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                    {
                        if (run.BoundaryOf[row] >= 0)
                        {
                            firstSource = firstSource < 0 ? run.BoundaryOf[row] : firstSource;
                            sourceCount++;
                        }
                        else if (heldSolved && run.RowClass[row] == HeldRowClass)
                        {
                            coupling.HeldSourceImpulse += impulses[row];
                        }
                    }

                    coupling.FirstRow = run.BoundaryOf[coupling.FirstRow];
                    coupling.FirstSourceRow = math.max(firstSource, 0);
                    coupling.SourceRowCount = sourceCount;
                    couplings.Add(coupling);
                }

                inputSpan.Dispose();
                ProfilerMarker.AutoScope denseSpan = ForestDenseMarker.Auto();
                if (vesselSolve)
                {
                    int maxRounds = run.Caps.PivotRoundLimit;
                    var stats = new BlockPrincipalPivotingStats();
                    long solveStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    solved = BlockPrincipalPivoting.Solve(nb, (double*)run.W.GetUnsafeReadOnlyPtr(), (double*)q.GetUnsafePtr(),
                        (double*)lower.GetUnsafePtr(), (double*)upper.GetUnsafePtr(), (byte*)states.GetUnsafePtr(),
                        couplings.GetUnsafePtr(), couplings.Length, run.Bounded, (double*)z.GetUnsafePtr(), maxRounds, warm,
                        run.Caps.FactorUpdate, run.Caps.UnorderedFactor ? null : (byte*)ages.GetUnsafePtr(), &stats, out int rounds);
                    long solveTicks = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solveStarted;
                    int changes = 0;
                    for (int i = 0; i < nb; i++)
                    {
                        bool changed = remembered[i] != 0xFF && remembered[i] != states[i];
                        changes += changed ? 1 : 0;
                        ages[i] = changed || remembered[i] == 0xFF ? (byte)0 : ages[i];
                    }

                    BoundaryMemory.Remember(ref memory, (ulong*)run.BoundaryKey.GetUnsafeReadOnlyPtr(),
                        (byte*)states.GetUnsafeReadOnlyPtr(), (byte*)ages.GetUnsafeReadOnlyPtr(), nb);
                    BoundaryCounters.AddSolve(nb, rounds, rounds >= maxRounds, !solved, warmRows);
                    BoundaryCounters.AddFactor(stats, solveTicks, changes, run.Substep <= 0);
                }
                else
                {
                    long solveStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    solved = BlockPrincipalPivoting.Solve(nb, (double*)run.W.GetUnsafeReadOnlyPtr(), (double*)q.GetUnsafePtr(),
                        (double*)lower.GetUnsafePtr(), (double*)upper.GetUnsafePtr(), (byte*)states.GetUnsafePtr(),
                        couplings.GetUnsafePtr(), couplings.Length, run.Bounded, (double*)z.GetUnsafePtr(),
                        BlockPrincipalPivoting.MaxPivotingIterations, false, run.Caps.FactorUpdate, null, null, out int rounds);
                    if (run.VesselMode)
                    {
                        BoundaryCounters.AddPush(rounds, Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solveStarted);
                    }
                }

                denseSpan.Dispose();

                // v += A^-1 J_b^T z.
                using ProfilerMarker.AutoScope spreadSpan = ForestSpreadMarker.Auto();
                UnsafeUtility.MemClear(work, run.Work.Length * (long)sizeof(double));
                for (int i = 0; i < nb; i++)
                {
                    int row = run.Boundary[i];
                    JacobianRowBlocks.Add(work, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row, z[i]);
                    impulses[row] = (float)z[i];
                }

                SolveAgainstFactor(ref run, work);
                for (int i = 0; i < run.Velocity.Length; i++)
                {
                    v[i] += work[i];
                }

                q.Dispose();
                lower.Dispose();
                upper.Dispose();
                states.Dispose();
                remembered.Dispose();
                ages.Dispose();
                z.Dispose();
                couplings.Dispose();
            }

            // In a push-out, the held rows whose contacts are still too deep after the boundary's push get a push of
            // their own.
            if (heldSolved && !vesselStart)
            {
                using (ForestHeldMarker.Auto())
                {
                    SolveHeldRows(ref run, rhs, impulses, v, true);
                }
            }

            // Every forest row's impulse from the velocities: z = E^-1 (r - J v).
            using ProfilerMarker.AutoScope rowSpan = ForestRowImpulsesMarker.Auto();
            for (int row = 0; row < rows; row++)
            {
                byte rowClass = run.RowClass[row];
                if (rowClass == ForestRowClass)
                {
                    double speed = JacobianRowBlocks.Dot(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
                    impulses[row] = (float)((rhs[row] - speed) / problem.Epsilon[row]);
                }
                else if (rowClass == FixedRowClass)
                {
                    impulses[row] = problem.Lower[row];
                }
                else if (rowClass == SectionRowClass || rowClass == DormantRowClass || (rowClass == HeldRowClass && !heldSolved))
                {
                    impulses[row] = 0.0f;
                }
            }

            return solved;
        }

        /// <summary>
        /// Adjust held normal rows for separating or closing contacts, or residual penetration in push-out.
        /// Compute independent impulses, then scale them together to best fit targets without overshoot.
        /// Update section velocity changes in v and return the adjusted row count. IdleHeldRows disables this
        /// correction.
        /// </summary>
        static unsafe int SolveHeldRows(ref ForestRun run, float* rhs, float* impulses, double* v, bool pushOut)
        {
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            float* jacobian = run.Jacobian;
            var held = new NativeList<int>(16, Allocator.Temp);
            var target = new NativeList<double>(16, Allocator.Temp);
            for (int row = 0; row < problem.RowCount; row++)
            {
                if (run.RowClass[row] != HeldRowClass)
                {
                    continue;
                }

                impulses[row] = 0.0f;

                // A normal row may push without limit, and a friction row has a finite upper bound.
                bool candidate = !run.Caps.IdleHeldRows && problem.Upper[row] >= float.MaxValue && (!pushOut || rhs[row] > 0.0f);
                if (!candidate)
                {
                    continue;
                }

                // In a velocity solve, a row whose contact separates may give back no more than its warm start, so a
                // row without a warm start has nothing to give back.
                double change = rhs[row] - JacobianRowBlocks.Dot(v, run.BlockSections[run.RowBlock[row]],
                    jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
                if (change > 0.0 || (!pushOut && change < 0.0 && problem.Lower[row] < 0.0f))
                {
                    held.Add(row);
                    target.Add(change);
                }
            }

            int given = held.Length;
            if (given > 0)
            {
                ForestIsland* forest = run.Forest;
                ref VesselContext context = ref forest->Context;
                int sections = forest->SectionCount;
                var y = new NativeArray<double>(6 * sections, Allocator.Temp);
                var mark = new NativeArray<int>(sections, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var path = new NativeArray<int>(sections, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                var z = new NativeArray<double>(given, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int s = 0; s < sections; s++)
                {
                    mark[s] = -1;
                }

                int* seeds = stackalloc int[2];
                double* yPtr = (double*)y.GetUnsafePtr();
                double* work = (double*)run.Work.GetUnsafePtr();
                UnsafeUtility.MemClear(work, run.Work.Length * (long)sizeof(double));
                for (int k = 0; k < given; k++)
                {
                    int row = held[k];
                    int2 sides = run.BlockSections[run.RowBlock[row]];
                    float* columns = jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    int seedCount = 0;
                    if (sides.x >= 0)
                    {
                        seeds[seedCount++] = sides.x;
                        (double6.Load(yPtr + 6 * sides.x) + JacobianRowBlocks.Load(columns)).Store(yPtr + 6 * sides.x);
                    }

                    if (sides.y >= 0)
                    {
                        if (sides.y != sides.x)
                        {
                            seeds[seedCount++] = sides.y;
                        }

                        (double6.Load(yPtr + 6 * sides.y) + JacobianRowBlocks.Load(columns + 6)).Store(yPtr + 6 * sides.y);
                    }

                    int count = context.CollectPath(seeds, seedCount, (int*)path.GetUnsafePtr(), (int*)mark.GetUnsafePtr(), k);
                    context.ForwardPath(yPtr, (int*)path.GetUnsafePtr(), count);
                    double diagonal = problem.Epsilon[row];
                    for (int p = 0; p < count; p++)
                    {
                        int s = path[p];
                        double6 ys = double6.Load(yPtr + 6 * s);
                        diagonal += double6.Dot(ys, context.InverseDiagonal(s, ys));
                        default(double6).Store(yPtr + 6 * s);
                    }

                    double impulse = target[k] / math.max(diagonal, 1e-300);
                    if (!pushOut)
                    {
                        impulse = math.max(impulse, problem.Lower[row]);
                    }

                    z[k] = impulse;
                    JacobianRowBlocks.Add(work, sides, columns, impulse);
                }

                SolveAgainstFactor(ref run, work);
                // The scale is the one that best meets the targets in the least-squares sense, kept between 0 and 1.
                double along = 0.0, squared = 0.0;
                for (int k = 0; k < given; k++)
                {
                    int row = held[k];
                    double moved = JacobianRowBlocks.Dot(work, run.BlockSections[run.RowBlock[row]],
                        jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
                    along += moved * target[k];
                    squared += moved * moved;
                }

                double scale = squared > 0.0 ? math.clamp(along / squared, 0.0, 1.0) : 0.0;

                for (int i = 0; i < run.Velocity.Length; i++)
                {
                    v[i] += scale * work[i];
                }

                for (int k = 0; k < given; k++)
                {
                    impulses[held[k]] = (float)(scale * z[k]);
                }

                y.Dispose();
                mark.Dispose();
                path.Dispose();
                z.Dispose();
                int closing = 0;
                for (int k = 0; k < given; k++)
                {
                    closing += target[k] > 0.0 ? 1 : 0;
                }

                BoundaryCounters.AddHeld(pushOut ? 0 : given - closing, pushOut ? given : 0, pushOut ? 0 : closing);
            }

            held.Dispose();
            target.Dispose();
            return given;
        }

        /// <summary>
        /// Convert section velocity changes to body impulses in linear and angular.
        /// For velocity solves, recover locked-row impulses by inverse dynamics; push-out starts sections at rest.
        /// </summary>
        internal static unsafe void SettleForest(ref ForestRun run, in SparseConstraintSolver.Problem problem, float* impulses,
            float3* linear, float3* angular, bool pushOut)
        {
            SumForestBodyImpulses(ref run, problem, impulses, linear, angular);

            if (!run.MultiPart)
            {
                return;
            }

            ForestIsland* forest = run.Forest;
            int sections = forest->SectionCount;
            int* sectionPartStart = forest->Structure.SectionPartStart.Ptr;
            int* sectionParts = forest->Structure.SectionParts.Ptr;
            int* partBody = forest->PartBody.Ptr;
            double h = run.Timestep;
            double* v = (double*)run.Velocity.GetUnsafePtr();
            for (int s = 0; s < sections; s++)
            {
                if (!Multi(forest, s))
                {
                    continue;
                }

                double3 dLinear = new double3(v[6 * s], v[6 * s + 1], v[6 * s + 2]);
                double3 dAngular = new double3(v[6 * s + 3], v[6 * s + 4], v[6 * s + 5]);
                double3 sectionLinear = pushOut ? dLinear : run.SectionLinear[s] + dLinear;
                double3 sectionAngular = pushOut ? dAngular : run.SectionAngular[s] + dAngular;

                // Remove negligible spin to avoid denormal arithmetic in later Jacobian construction and
                // integration.
                if (math.lengthsq(sectionAngular) < MinimumSpin * MinimumSpin)
                {
                    sectionAngular = 0.0;
                }

                // Each body lands where the section's rotation takes it, not along the tangent, so that a turning section
                // stays rigid however long it stays locked.
                double angle = math.length(sectionAngular) * h;
                double4 turn = angle > 1e-12 ? new double4(math.normalize(sectionAngular) * math.sin(0.5 * angle), math.cos(0.5 * angle))
                    : new double4(0.0, 0.0, 0.0, 1.0);
                run.SettleLinear[s] = sectionLinear;
                run.SettleAngular[s] = sectionAngular;
                run.SettleTurn[s] = DoubleMath.Rotation(turn);
            }

            int parts = forest->Structure.SectionParts.Length;
            int partTasks = (parts + PartsPerSectionTask - 1) / PartsPerSectionTask;
            var tasks = new SectionTasks
            {
                Run = (ForestRun*)UnsafeUtility.AddressOf(ref run),
                Problem = problem,
                Impulses = impulses,
                Linear = linear,
                Angular = angular,
                PushOut = pushOut,
                Phase = SectionTasks.SettleBodies,
            };
            WorkTeam.RunAll(run.Team, &tasks, WorkTeam.ForestSectionsKind, partTasks, null);
            if (pushOut)
            {
                return;
            }

            long subtreeStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;

            // Leaves inward: each locked edge carries what its child's subtree still needs, moved to the parent's centre.
            // Short work on one lane; the split of each edge's load over its rows goes back to the team.
            run.Residual = 0.0;
            double6* need = (double6*)run.Need.GetUnsafePtr();
            for (int s = 0; s < sections; s++)
            {
                int start = sectionPartStart[s], end = sectionPartStart[s + 1];
                if (end - start < 2)
                {
                    continue;
                }

                for (int k = end - 1; k > start; k--)
                {
                    int part = sectionParts[k];
                    int child = partBody[part];
                    int parent = partBody[forest->Edges[part].Parent];
                    double6 carried = need[child];
                    double3 force = new double3(carried.Low.x, carried.Low.y, carried.Low.z);
                    double3 torque = new double3(carried.Low.w, carried.High.x, carried.High.y) + math.cross(run.Arm[child] - run.Arm[parent], force);
                    need[parent] = need[parent] + new double6(new double4(force, torque.x), new double2(torque.y, torque.z));
                }

                double6 left = need[partBody[sectionParts[start]]];
                run.Residual = math.max(run.Residual, math.length(new double3(left.Low.x, left.Low.y, left.Low.z)));
            }

            run.SubtreeTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - subtreeStarted;
            tasks.Phase = SectionTasks.SplitLoads;
            WorkTeam.RunAll(run.Team, &tasks, WorkTeam.ForestSectionsKind, partTasks, null);
            ForestStrutLoads.Recover(ref run, problem, impulses);
        }

        /// <summary>
        /// Compute body impulses from section motion and record the remaining impulse needed from locked edges.
        /// </summary>
        static unsafe void SettleBodiesTask(ref ForestRun run, SectionTasks* tasks, int task)
        {
            ForestIsland* forest = run.Forest;
            int* sectionParts = forest->Structure.SectionParts.Ptr;
            int* partSection = forest->Structure.PartSection.Ptr;
            int* partBody = forest->PartBody.Ptr;
            double h = run.Timestep;
            bool pushOut = tasks->PushOut;
            int end = math.min((task + 1) * PartsPerSectionTask, forest->Structure.SectionParts.Length);
            for (int k = task * PartsPerSectionTask; k < end; k++)
            {
                int part = sectionParts[k];
                int s = partSection[part];
                if (!Multi(forest, s))
                {
                    continue;
                }

                int i = partBody[part];
                int world = run.WorldBodies[i];
                MotionVelocity motion = run.MotionVelocities[world];
                quaternion rotation = run.MotionDatas[world].WorldFromMotion.rot;
                double3 r = run.Arm[i];
                double3 sectionLinear = run.SettleLinear[s], sectionAngular = run.SettleAngular[s];
                double3 targetLinear = pushOut ? sectionLinear + math.cross(sectionAngular, r)
                    : sectionLinear + (math.mul(run.SettleTurn[s], r) - r) / h;
                double3 changeLinear = pushOut ? targetLinear : targetLinear - run.OwnLinear[i];
                double3 changeAngular = pushOut ? sectionAngular : sectionAngular - run.OwnAngular[i];

                // The impulses that give exactly these changes as the bodies take impulses: by their own mass, and by their
                // principal inertia in their own frame.
                double3 inverseInertia = math.max((double3)motion.InverseInertia, 1e-12);
                double3 local = math.rotate(math.conjugate(rotation), (float3)changeAngular);
                double3 linearImpulse = changeLinear / math.max((double)motion.InverseMass, 1e-12);
                double3 angularImpulse = math.rotate(rotation, (float3)(local / inverseInertia));
                if (!pushOut)
                {
                    float3 known = tasks->Linear[i], knownAngular = tasks->Angular[i];
                    run.Need[i] = new double6(new double4(linearImpulse - (double3)known, angularImpulse.x - knownAngular.x),
                        new double2(angularImpulse.y - knownAngular.y, angularImpulse.z - knownAngular.z));
                }

                tasks->Linear[i] = DoubleMath.Flushed(linearImpulse);
                tasks->Angular[i] = DoubleMath.Flushed(angularImpulse);
            }
        }

        /// <summary>One task's parts of a settle: the load each locked edge to a parent carries, split over its rows.</summary>
        static unsafe void SplitLoadsTask(ref ForestRun run, SectionTasks* tasks, int task)
        {
            ForestIsland* forest = run.Forest;
            int* sectionParts = forest->Structure.SectionParts.Ptr;
            int* sectionPartStart = forest->Structure.SectionPartStart.Ptr;
            int* partSection = forest->Structure.PartSection.Ptr;
            int end = math.min((task + 1) * PartsPerSectionTask, forest->Structure.SectionParts.Length);
            for (int k = task * PartsPerSectionTask; k < end; k++)
            {
                int part = sectionParts[k];
                int s = partSection[part];
                if (k == sectionPartStart[s] || !Multi(forest, s))
                {
                    continue;
                }

                int child = forest->PartBody[part];
                LockedPairImpulses(ref run, tasks->Problem, forest->PartPair[part], child, run.Need[child], tasks->Impulses);
            }
        }

        /// <summary>
        /// Distribute a locked pair's required impulse by stiffness-weighted least squares:
        /// z = W*J*(J^T*W*J)^-1*need. Cache the inverse in the child's frame while locking and compliances remain
        /// unchanged.
        /// </summary>
        static unsafe void LockedPairImpulses(ref ForestRun run, in SparseConstraintSolver.Problem problem, int pair, int child,
            double6 need, float* impulses)
        {
            if (pair < 0)
            {
                return;
            }

            ForestIsland* forest = run.Forest;
            double3x3 rotation = new double3x3(new float3x3(run.MotionDatas[run.WorldBodies[child]].WorldFromMotion.rot));
            double3x3 back = math.transpose(rotation);
            float key = 0.0f;
            int count = 0;
            for (int k = forest->PairBlockStart[pair]; k < forest->PairBlockStart[pair + 1]; k++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (run.RowClass[row] == SectionRowClass)
                    {
                        key += problem.Epsilon[row] / PairBlockGamma(run, k);
                        count++;
                    }
                }
            }

            double* inverse = forest->PairInverse.Ptr + 36 * pair;
            float kept = forest->PairInverseKey[pair];
            if (!(math.abs(kept - key) <= 1e-6f * key) || forest->PairInverseRows[pair] != count)
            {
                PairLeastSquares(ref run, problem, pair, child, back, inverse);
                forest->PairInverseKey[pair] = key;
                forest->PairInverseRows[pair] = count;
            }

            double6 local = new double6(new double4(math.mul(back, new double3(need.Low.x, need.Low.y, need.Low.z)), 0.0), default);
            double3 torque = math.mul(back, new double3(need.Low.w, need.High.x, need.High.y));
            local = new double6(new double4(local.Low.xyz, torque.x), new double2(torque.y, torque.z));
            double6 y = ((double6x6*)inverse)->Mul(local);
            double3 xLinear = math.mul(rotation, new double3(y.Low.x, y.Low.y, y.Low.z));
            double3 xAngular = math.mul(rotation, new double3(y.Low.w, y.High.x, y.High.y));
            double6 x = new double6(new double4(xLinear, xAngular.x), new double2(xAngular.y, xAngular.z));
            for (int k = forest->PairBlockStart[pair]; k < forest->PairBlockStart[pair + 1]; k++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                int side = block.BodyA == child ? 0 : 6;
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (run.RowClass[row] == SectionRowClass)
                    {
                        double6 j = JacobianRowBlocks.Load(problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row + side);
                        impulses[row] = DoubleMath.Flushed(double6.Dot(j, x) * RowStiffness(ref run, problem, pair, k, row));
                    }
                }
            }
        }

        /// <summary>
        /// Weight a locked row by spring stiffness gamma/epsilon, excluding damper effects so that static load
        /// sharing matches yielding rows.
        /// </summary>
        internal static unsafe double RowStiffness(ref ForestRun run, in SparseConstraintSolver.Problem problem, int pair, int k, int row) =>
            PairBlockGamma(run, k) / (double)problem.Epsilon[row];

        /// <summary>A row's rate at the bodies' velocities after the solve, which the motions hold by then.</summary>
        static unsafe double SolvedRowSpeed(in ForestRun run, in SparseConstraintSolver.ConstraintBlock block, float* columns)
        {
            double speed = 0.0;
            if (block.BodyA >= 0)
            {
                speed += BodyRowSpeed(run, block.BodyA, columns);
            }

            if (block.BodyB >= 0)
            {
                speed += BodyRowSpeed(run, block.BodyB, columns + 6);
            }

            return speed;
        }

        static unsafe double BodyRowSpeed(in ForestRun run, int body, float* columns)
        {
            int world = run.WorldBodies[body];
            MotionVelocity motion = run.MotionVelocities[world];
            float3 angular = math.rotate(run.MotionDatas[world].WorldFromMotion.rot, motion.AngularVelocity);
            return math.dot(new float3(columns[0], columns[1], columns[2]), motion.LinearVelocity)
                + math.dot(new float3(columns[3], columns[4], columns[5]), angular);
        }

        /// <summary>The gamma of the k-th of the pairs' forest blocks, see <see cref="SpringGamma"/>.</summary>
        static unsafe float PairBlockGamma(in ForestRun run, int k) => SpringGamma(run.Headers[run.ForestBlocks[run.Forest->PairBlocks[k]]]);

        /// <summary>
        /// The gamma a Jacobian's rows hold their position with, h k / (h k + c), from its regularization, or 1 for none.
        /// </summary>
        static unsafe float SpringGamma(JacobianHeader* header)
        {
            float gamma = header->AccessDirectSolverRegularizationData().Gamma;
            return gamma > 1e-6f && gamma <= 1.0f ? gamma : 1.0f;
        }

        /// <summary>
        /// (J^T W J)^-1 over a locked pair's rows on the child's side, W the rows' stiffness 1 / epsilon, with the rows
        /// taken into the child's frame by <paramref name="back"/>.
        /// </summary>
        static unsafe void PairLeastSquares(ref ForestRun run, in SparseConstraintSolver.Problem problem, int pair, int child,
            double3x3 back, double* inverse)
        {
            ForestIsland* forest = run.Forest;
            double* m = stackalloc double[36];
            UnsafeUtility.MemClear(m, 36 * sizeof(double));
            for (int k = forest->PairBlockStart[pair]; k < forest->PairBlockStart[pair + 1]; k++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[run.ForestBlocks[forest->PairBlocks[k]]];
                int side = block.BodyA == child ? 0 : 6;
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (run.RowClass[row] == SectionRowClass)
                    {
                        float* c = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row + side;
                        double3 l = math.mul(back, new double3(c[0], c[1], c[2]));
                        double3 a = math.mul(back, new double3(c[3], c[4], c[5]));
                        double6 j = new double6(new double4(l, a.x), new double2(a.y, a.z));
                        Block6x6.AddOuter(m, RowStiffness(ref run, problem, pair, k, row), j, j);
                    }
                }
            }

            // Rows that don't reach every direction leave the rest of the load to rounding, not to a singular matrix.
            double trace = m[0] + m[7] + m[14] + m[21] + m[28] + m[35];
            for (int d = 0; d < 6; d++)
            {
                m[7 * d] += 1e-12 * trace + 1e-300;
            }

            Block6x6.InvertSymmetric(inverse, m);
        }

        /// <summary>
        /// Record each tree pair's maximum spring-only angular deflection from its solved load.
        /// Accumulate across substeps and use the mean for the next step's lock tests. Return the largest current
        /// deflection.
        /// </summary>
        internal static unsafe float RecordDeflections(ref ForestRun run, JacobianHeader** headers, float* impulses, int substep,
            int smoothingSteps)
        {
            float largest = 0.0f;
            ForestIsland* forest = run.Forest;
            ref SparseConstraintSolver.Problem problem = ref run.Problem;
            float h = run.Timestep;
            for (int p = 0; p < forest->PairCount; p++)
            {
                if (forest->PairEdge[p] == 0 || forest->PairTree[p] == 0)
                {
                    continue;
                }

                float deflection = 0.0f;
                for (int k = forest->PairBlockStart[p]; k < forest->PairBlockStart[p + 1]; k++)
                {
                    int b = run.ForestBlocks[forest->PairBlocks[k]];
                    SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];

                    // The rows hold h^2 k + h c = 1 / epsilon and gamma = h k / (h k + c), so the spring alone is
                    // k = gamma / (epsilon h^2): a damped row's compliance alone would read the deflection low by 1 + c / (h k).
                    float gamma = SpringGamma(headers[b]);
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        byte rowClass = run.RowClass[row];
                        if (rowClass != ForestRowClass && rowClass != SectionRowClass)
                        {
                            continue;
                        }

                        float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                        float linear = columns[0] * columns[0] + columns[1] * columns[1] + columns[2] * columns[2];
                        float angular = columns[3] * columns[3] + columns[4] * columns[4] + columns[5] * columns[5];
                        if (linear <= 1e-8f * angular)
                        {
                            // Recover spring deflection without damping: phi = -(z*epsilon + J*v)*h/gamma.
                            // Locked rows have J*v = 0.
                            double speed = rowClass == SectionRowClass ? 0.0 : SolvedRowSpeed(run, block, columns);
                            double turn = (impulses[row] * (double)problem.Epsilon[row] + speed) * h / gamma;
                            deflection = math.max(deflection, (float)math.abs(turn));
                        }
                    }
                }

                // A step's first substep closes the step before it: its mean becomes the deflection the lock tests read.
                float sum = forest->PairDeflectionSum[p];
                if (substep <= 0)
                {
                    if (forest->DeflectionCount > 0)
                    {
                        float mean = sum / forest->DeflectionCount;
                        float before = forest->PairDeflection[p];
                        forest->PairDeflectionStep[p] = mean;
                        forest->PairDeflection[p] = smoothingSteps > 1 && math.isfinite(before)
                            ? before + (mean - before) / smoothingSteps : mean;
                    }

                    sum = 0.0f;
                }

                forest->PairDeflectionSum[p] = sum + deflection;
                largest = math.max(largest, forest->PairDeflection[p]);
            }

            forest->DeflectionCount = substep <= 0 ? 1 : forest->DeflectionCount + 1;
            return largest;
        }

        /// <summary>x = A^-1 x against the forest's factor, by chunks on the team's lanes.</summary>
        internal static unsafe void SolveAgainstFactor(ref ForestRun run, double* x)
        {
            ref VesselContext context = ref run.Forest->Context;
            VesselContext* contextPtr = (VesselContext*)UnsafeUtility.AddressOf(ref context);
            long bytes = 6L * context.SectionCount * sizeof(double);
            UnsafeUtility.MemCpy(context.Rhs, x, bytes);
            WorkTeam.RunAll(run.Team, contextPtr, WorkTeam.ForestForwardKind, context.ChunkCount, null);
            context.SubstituteTop();
            WorkTeam.RunAll(run.Team, contextPtr, WorkTeam.ForestSubstituteKind, context.ChunkCount, null);
            UnsafeUtility.MemCpy(x, context.Solution, bytes);
        }

        /// <summary>One chunk of a forest's factor, for a team.</summary>
        internal static unsafe void ForestFactorTask(void* context, int chunk)
        {
            var vessel = (VesselContext*)context;
            var scratch = new NativeArray<double>(Block6x6.Size * math.max(1, vessel->MaxStructure), Allocator.Temp,
                NativeArrayOptions.UninitializedMemory);
            vessel->FactorChunk(chunk, (double*)scratch.GetUnsafePtr());
            scratch.Dispose();
        }

        /// <summary>
        /// Each moving body's share of J^T z, in the frame of the Jacobian's columns, as
        /// <see cref="SparseConstraintSolver.BodyImpulse"/> gives it.
        /// </summary>
        /// <summary>Sum the row impulses and record the time that the forest spent summing them.</summary>
        internal static unsafe void SumForestBodyImpulses(ref ForestRun run, in SparseConstraintSolver.Problem problem,
            float* impulses, float3* linear, float3* angular)
        {
            long summing = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            SumBodyImpulses(problem, impulses, linear, angular);
            run.BodyImpulseTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - summing;
        }

        static unsafe void SumBodyImpulses(in SparseConstraintSolver.Problem problem, float* impulses, float3* linear, float3* angular)
        {
            UnsafeUtility.MemClear(linear, problem.BodyCount * (long)sizeof(float3));
            UnsafeUtility.MemClear(angular, problem.BodyCount * (long)sizeof(float3));
            for (int b = 0; b < problem.BlockCount; b++)
            {
                SparseConstraintSolver.ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    float* columns = problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row;
                    float impulse = impulses[row];
                    if (block.BodyA >= 0)
                    {
                        linear[block.BodyA] += impulse * new float3(columns[0], columns[1], columns[2]);
                        angular[block.BodyA] += impulse * new float3(columns[3], columns[4], columns[5]);
                    }

                    if (block.BodyB >= 0)
                    {
                        linear[block.BodyB] += impulse * new float3(columns[6], columns[7], columns[8]);
                        angular[block.BodyB] += impulse * new float3(columns[9], columns[10], columns[11]);
                    }
                }
            }
        }
    }
}
