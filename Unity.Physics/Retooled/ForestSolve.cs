using System;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Numerics.Retooled;
using Unity.Profiling;

namespace Unity.Physics
{
    public static partial class Solver
    {
        static readonly ProfilerMarker ForestFactorMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Factor");
        static readonly ProfilerMarker ForestBoundaryMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Boundary");
        static readonly ProfilerMarker ForestSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Solve");

        // The serial parts of SolveForest, which the leader runs alone.
        static readonly ProfilerMarker ForestHeldMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.HeldRows");
        static readonly ProfilerMarker ForestBoundaryInputMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.BoundaryInput");
        static readonly ProfilerMarker ForestDenseMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Dense");
        static readonly ProfilerMarker ForestSpreadMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Spread");
        static readonly ProfilerMarker ForestRowImpulsesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.RowImpulses");
        static readonly ProfilerMarker ForestStructureMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Forest.Structure");

        const byte ForestRowClass = 0;
        const byte BoundaryRowClass = 1;
        const byte FixedRowClass = 2;

        /// <summary>A forest row between two bodies of one section: a locked pair's, whose load comes from inverse dynamics, or an inert closure's.</summary>
        const byte SectionRowClass = 3;

        /// <summary>
        /// A row that a cap on the boundary keeps out of a vessel island's solve. The solve gives the row no impulse of its
        /// own, so the contact impulse that the row had before the solve, its warm start, stays applied: a sparse solve's
        /// contact rows solve for the change from their warm-start impulse. There are exceptions for normal rows, see
        /// <see cref="SolveHeldRows"/>. A held normal row whose contact separates gives back its warm start, a held normal
        /// row whose contact closes pushes, and in a push-out, a held normal row whose contact is too deep gets a push of
        /// its own.
        /// </summary>
        const byte HeldRowClass = 4;

        /// <summary>
        /// A one-sided joint row that stays out of a vessel island's boundary for this solve, because the bodies' motion
        /// over the step keeps the row more than <see cref="BoundaryCaps.DormantMargin"/> from its stop. Examples are a
        /// limit far from its stop and a wheel that is far from bottoming out. The row takes no impulse in this solve.
        /// The next solve checks the row again, and the row joins the boundary when it comes near its stop.
        /// </summary>
        const byte DormantRowClass = 5;

        /// <summary>
        /// One island's solve as a forest plus a boundary, see <see cref="ForestSettings"/>. Holds what a second solve with
        /// a changed right-hand side reuses: the factor (in the island's <see cref="ForestIsland"/>), the classes of the
        /// rows and the boundary block W = J_b A^-1 J_b^T + E_b; and, for sections of more than one body, what turns the
        /// sections' answer back into each body's.
        /// </summary>
        unsafe struct ForestRun
        {
            public ForestIsland* Forest;
            public SparseConstraintSolver.Problem Problem;
            public bool Bounded;

            /// <summary>
            /// Whether the caller knows that the island's vessels, their structure versions and the numbering of
            /// its bodies are the same as when the forest's layout was last built, and that the island has no body
            /// outside its vessels. Then the vessel preparation does not build and compare the layout's signature.
            /// </summary>
            public bool KnownLayout;

            /// <summary>Each block's Jacobian, for its regularization.</summary>
            public JacobianHeader** Headers;

            /// <summary>
            /// The rows as the sections see them: a row on a body of a section of several acts on the section's velocity at
            /// the body's offset from its centre. The problem's own rows when every section is one body.
            /// </summary>
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

            /// <summary>
            /// For a vessel island: the limits on its boundary, and a key for each boundary row that stays the same from one
            /// solve to the next. The keys let a solve start from the states that the rows had in the last solve.
            /// </summary>
            public BoundaryCaps Caps;
            public NativeArray<ulong> BoundaryKey;

            /// <summary>For a vessel island: the substep that the run solves, for the boundary's counters.</summary>
            public int Substep;

            /// <summary>
            /// For a vessel island: the world's bodies, so that a boundary row's key can know a body outside the island's
            /// vessels by its entity. Null if the caller has none.
            /// </summary>
            public RigidBody* RigidBodies;
            public int RigidBodyCount;

            /// <summary>The change of velocity of every section, and a vector of the same size to work in.</summary>
            public NativeArray<double> Velocity;
            public NativeArray<double> Work;

            /// <summary>
            /// For sections of several bodies: each section's centre and the velocity its bodies had together before the
            /// solve (momentum-conserving), each body's offset from its section's centre, and the bodies' own velocities.
            /// Empty when every section is one body.
            /// </summary>
            public NativeArray<double3> SectionCenter;
            public NativeArray<double3> SectionLinear;
            public NativeArray<double3> SectionAngular;
            public NativeArray<double3> Arm;
            public NativeArray<double3> OwnLinear;
            public NativeArray<double3> OwnAngular;
            public bool MultiPart;

            /// <summary>
            /// For a settle: each section's velocity after the solve and the turn it makes in a substep, and what each body of
            /// a section of several still needs from its locked edges (an impulse and its moment about the body's centre).
            /// </summary>
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
            public SparseSolveTeam* Team;

            /// <summary>
            /// For a vessel island: its sections' velocities from the vessel's own rows and the fixed rows, before the
            /// boundary acts. The boundary's solve starts from these rather than from rest.
            /// </summary>
            public NativeArray<double> BaseVelocity;

            /// <summary>Whether the run solves a vessel island with the vessel kernel's own rows.</summary>
            public bool VesselMode;

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

            /// <summary>
            /// Ticks in the parts that run on one lane: classifying rows, the lock decisions, the sections' sums, the body
            /// impulse sums and the subtree sums, for the bench.
            /// </summary>
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
        /// Solves an island's rows as a forest plus a boundary and writes every row's impulse, or returns false, changing
        /// nothing, when the island isn't one for it: no forest rows, or more boundary rows than the cap.
        /// </summary>
        static unsafe bool TryStartForest(in SparseConstraintSolver.Problem problem, JacobianHeader** headers, bool bounded,
            ForestIsland* forest, in ForestSettings settings, SparseSolveTeam* team, int* worldBodies, RigidBody* rigidBodies,
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
            SparseSolveTeam.RunAll(team, &classify, SparseSolveTeam.ForestClassifyKind, classifyTasks, null);
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

            // The pairs, kept while the island's bodies and forest blocks stay the same. The signature names the bodies by
            // their identities, not by their indices in the world: a body that the world made where a destroyed body was
            // has the destroyed body's index, and it must not take over the destroyed body's lock states.
            int bodyCount = problem.BodyCount;
            var identities = new NativeArray<ulong>(math.max(1, bodyCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int b = 0; b < bodyCount; b++)
            {
                identities[b] = DirectSolverCache.BodyIdentity(rigidBodies, rigidBodyCount, worldBodies[b]);
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

            // Which pairs are locked this solve: a tree edge whose joints all allow it, with nothing but forest rows, locks
            // when its springs would deflect less than the lock angle under its last load, and stays locked until its load
            // would deflect them past the yield angle.
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
                            double6 j = BlockRows.Load(sides.x >= 0 ? columns : columns + 6);
                            VesselBlocks.AddOuter(forest->ExtraDiagonal.Ptr + VesselBlocks.Size * s, 1.0 / problem.Epsilon[row], j, j);
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
                SparseSolveTeam.RunAll(team, contextPtr, SparseSolveTeam.ForestFactorKind, context.ChunkCount, null);
                var scratch = new NativeArray<double>(VesselBlocks.Size * math.max(1, context.MaxStructure), Allocator.Temp,
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
        /// The boundary block W = J_b A^-1 J_b^T + E_b over the run's boundary rows, against the factor the run's forest holds.
        /// Each boundary row's forward substitution only walks the sections on its elimination-tree path, and two rows'
        /// entry in W is the sum over the sections that both paths share.
        /// </summary>
        static unsafe void BuildBoundary(ref ForestRun run, int boundaryRows)
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
                        (double6.Load(yPtr + 6 * sides.x) + BlockRows.Load(columns)).Store(yPtr + 6 * sides.x);
                    }

                    if (sides.y >= 0)
                    {
                        if (sides.y != sides.x)
                        {
                            seeds[seedCount++] = sides.y;
                        }

                        (double6.Load(yPtr + 6 * sides.y) + BlockRows.Load(columns + 6)).Store(yPtr + 6 * sides.y);
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
                SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestBoundaryKind,
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
        /// Fills the rows of W that one task owns. For each row i, the task computes the entries from the diagonal to
        /// the end of the row, and writes each one to both of its places in the symmetric matrix. Two tasks never write
        /// the same entry, because an entry belongs to the task that owns the smaller of its row and column.
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
        /// A contact point or a contact manifold, with its depth, for sorting. Items sort by their rank first, lowest
        /// first, then by depth, deepest first. Items of the same rank and the same depth sort by <see cref="Tie"/>, which
        /// changes from one solve to the next, so that a tie does not always go against the same rows.
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
        /// Applies the run's caps to a vessel island's boundary. The rows that a cap drops become held rows, see
        /// <see cref="HeldRowClass"/>. The caps are these:
        /// <list type="bullet">
        /// <item>Each contact manifold keeps at most <see cref="BoundaryCaps.PointsPerManifold"/> of its points, the deepest.</item>
        /// <item>
        /// Contacts between two parts of the same vessel keep at most <see cref="BoundaryCaps.SelfContactManifolds"/>
        /// manifolds, the deepest by their deepest point. A dropped manifold's friction rows are held too.
        /// </item>
        /// <item>
        /// Other contacts, for example against the ground or against another body, keep at most
        /// <see cref="BoundaryCaps.OutsideContactPoints"/> points over the whole island. The cap takes every manifold's
        /// deepest point first, deepest first, then every manifold's second deepest point, and so on, so that the loss
        /// is spread over the manifolds. A manifold that keeps a point keeps its friction rows too, and those count
        /// against a limit on rows: the cap on points plus the friction rows of the manifolds that the cap would allow
        /// at <see cref="BoundaryCaps.DefaultPointsPerManifold"/> points each, so 448 rows for 256 points. If a manifold
        /// loses every point, its friction rows are held too.
        /// </item>
        /// </list>
        /// A point's depth is how far its two surfaces overlap, which is minus the contact's distance. When two points or
        /// two manifolds tie, the cap breaks the tie by a hash that changes with every solve of the island.
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

            // The cap on outside points also limits the rows that the kept points bring with them: each manifold that keeps
            // a point brings its friction rows too. The limit on rows is what the cap on points allows when every kept
            // manifold keeps all of its points, for example 256 points and the friction rows of 64 manifolds.
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
        /// Marks the dormant rows of a vessel island's boundary, see <see cref="DormantRowClass"/>. A dormant row is a
        /// joint row with one bound at zero and no other bound, which is how a limit's row or a wheel's bottom-out row
        /// looks once the row is armed. The test uses the velocities that the vessel's own rows and the fixed rows give
        /// the sections, before the boundary acts. A row whose impulse may only be positive is out of reach if those
        /// velocities move it away from its stop by more than the margin over one step, and the same holds the other way
        /// round for a row whose impulse may only be negative. Contact rows and coupled rows always stay in the boundary.
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
                    double speed = BlockRows.Dot(velocity, run.BlockSections[run.RowBlock[row]],
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
        /// Makes a key for each boundary row that stays the same from one solve to the next while the row's constraint
        /// lasts. The key comes from the row's block and from the row's own place in its constraint. The eight lowest
        /// bits of each key are zero, so that <see cref="BoundaryMemory"/> can keep a state and an age in them.
        /// <list type="bullet">
        /// <item>
        /// A block is known by its two bodies, see <see cref="BodyKey"/>, its type and, for a contact that has them, its
        /// two collider keys. If the island has several blocks with all of these the same, a block is also known by how
        /// many of them come before it among all of the island's blocks. The count includes blocks that have no
        /// boundary rows, so that a cap or a dormant row in an earlier block does not change it.
        /// </item>
        /// <item>
        /// A contact's normal row is known by its point's index in the manifold, and a friction row by its place among
        /// the three friction rows. So when a point drops out of the rows, the other rows keep their keys. A joint row
        /// is known by its place in its block, because a joint always has the same rows.
        /// </item>
        /// </list>
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
        /// A key for one side of a block that stays the same while the body stays in the island. A part of one of the
        /// island's vessels is known by its vessel's ID and its place in the vessel. Any other body, moving or not, is
        /// known by the version of its entity. The backend sets that version to the body's serial number, which no other
        /// body receives, and it keeps it while the body lives. The entity's index is not part of the key, because the
        /// index is the body's row in the backend's store, and the store moves a body to another row when an earlier
        /// body is destroyed. So the key does not change when other bodies join or leave the world, and it does not pass
        /// to a new body. A world whose entities share versions gives several bodies the same key. Then the blocks are
        /// told apart by how many blocks with the same identity come before them, and the worst case is a missed warm
        /// start. Without the world's bodies, or with <see cref="BoundaryCaps.KeyByWorldIndex"/>, a body is known by its
        /// place in the world, as before. Returns 0 for no body.
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
        /// The place of a contact block's row in its contact: for a normal row, the index of the row's point in the
        /// manifold, and for a friction row, 0x10000 plus its place among the three friction rows. The block's first
        /// rows are the normal rows of the points within reach, in the manifold's order, and the friction rows follow.
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
        /// For sections of several bodies: each section's mass, centre and inertia from its bodies, the momentum-conserving
        /// velocity its bodies have together, the rows as the sections see them, and each row's right-hand side moved from
        /// the bodies' own velocities to the section's. A section's bodies need not move rigidly when it starts, after a
        /// lock or a force on one of them; the solve takes them as one, and each body's answer is the section's.
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
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestSectionsKind,
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
        const double MinimumSpin = 1e-9;

        /// <summary>Whether section s has more than one body.</summary>
        static unsafe bool Multi(ForestIsland* forest, int s) =>
            forest->Structure.SectionPartStart[s + 1] - forest->Structure.SectionPartStart[s] > 1;

        /// <summary>A body's angular velocity in world space.</summary>
        static unsafe double3 BodyAngular(in ForestRun run, int world) =>
            math.rotate(run.MotionDatas[world].WorldFromMotion.rot, run.MotionVelocities[world].AngularVelocity);

        /// <summary>
        /// Turns one side of a row from a body's columns to its section's, and returns how much the row's right-hand side
        /// moves when the body's velocity is taken as the section's: j . u_body - j_section . u_section.
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
        /// Solves the run's island for the right-hand side <paramref name="rhs"/> against the factor and boundary block it
        /// holds, and writes every row's impulse; a row inside a section gets none here. Returns false if the boundary's
        /// pivoting ran out of iterations before it found an answer. The boundary's impulses are then the result of a
        /// few sweeps of projected Gauss-Seidel: they are inside their bounds and near the answer, so a velocity solve
        /// can apply them. A vessel island's velocity solve counts such a solve in
        /// <see cref="BoundaryCounters.UnsettledSolves"/>. A push-out does not apply them, see <see cref="SolvePushOut"/>.
        /// </summary>
        static unsafe bool SolveForest(ref ForestRun run, float* rhs, float* impulses)
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

                BlockRows.Add(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row, scale);
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

                // A vessel island's velocity solve has a key for each boundary row. It starts each row in the state that
                // the row had at the end of the last solve, if the island remembers one, and it takes a limited number of
                // pivoting rounds. A push-out solve has no base velocity, and it keeps the plain solve.
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
                    q[i] = rhs[row] - BlockRows.Dot(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
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

                // A cap can keep some of a coupling's source rows out of the boundary. The rows that stay are next to each
                // other in the boundary, because the boundary keeps the rows' order. A held row's impulse is known before
                // the boundary is solved, so the coupling counts it with the impulse that was applied before the solve,
                // and the coupling only needs its range of source rows moved. A coupling whose own rows are held is left
                // out.
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
                    var stats = new BoundarySolveStats();
                    long solveStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                    solved = BoundarySolver.Solve(nb, (double*)run.W.GetUnsafeReadOnlyPtr(), (double*)q.GetUnsafePtr(),
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
                    solved = BoundarySolver.Solve(nb, (double*)run.W.GetUnsafeReadOnlyPtr(), (double*)q.GetUnsafePtr(),
                        (double*)lower.GetUnsafePtr(), (double*)upper.GetUnsafePtr(), (byte*)states.GetUnsafePtr(),
                        couplings.GetUnsafePtr(), couplings.Length, run.Bounded, (double*)z.GetUnsafePtr(),
                        BoundarySolver.MaxPivotingIterations, false, run.Caps.FactorUpdate, null, null, out int rounds);
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
                    BlockRows.Add(work, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row, z[i]);
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
                    double speed = BlockRows.Dot(v, run.BlockSections[run.RowBlock[row]], jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row);
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
        /// Gives some of a vessel island's held normal rows an impulse, see <see cref="HeldRowClass"/>. Every other held
        /// row gets no impulse. There are two cases:
        /// <list type="bullet">
        /// <item>
        /// In a velocity solve (<paramref name="pushOut"/> false), a held row whose contact separates gives back its warm
        /// start, and a held row whose contact closes pushes. The row's target is the change of its velocity that its
        /// right-hand side asks for. When that change is negative, the row gives back at most its warm start, which is
        /// minus its lower bound. Without the push, a body whose contacts the cap holds would get only its warm start.
        /// That is 95% of the last step's impulse, and the next step's warm start comes from it, so the body would sink
        /// a little more in every step.
        /// </item>
        /// <item>
        /// In a push-out (<paramref name="pushOut"/> true), a held row whose contact is still too deep after the
        /// boundary's push gets a push of its own. The row's target is the rest of the push that its right-hand side
        /// asks for.
        /// </item>
        /// </list>
        /// Each row takes the impulse that would meet its target if the row were alone: the target divided by the row's
        /// diagonal entry of W. That entry costs one walk along the row's path to the root of the factor, as in
        /// <see cref="BuildBoundary"/>, but no other entry of W is needed. Several held rows can move the same bodies,
        /// for example the four points of one box, so the routine then scales all of their impulses by one factor. The
        /// factor is the one that meets the targets best in the least-squares sense, between 0 and 1. It costs one more
        /// solve against the factor. The routine adds the change of the sections' velocities to <paramref name="v"/>,
        /// writes each held row's impulse, and returns how many rows got one. With
        /// <see cref="BoundaryCaps.IdleHeldRows"/>, every held row gets no impulse.
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
                double change = rhs[row] - BlockRows.Dot(v, run.BlockSections[run.RowBlock[row]],
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
                        (double6.Load(yPtr + 6 * sides.x) + BlockRows.Load(columns)).Store(yPtr + 6 * sides.x);
                    }

                    if (sides.y >= 0)
                    {
                        if (sides.y != sides.x)
                        {
                            seeds[seedCount++] = sides.y;
                        }

                        (double6.Load(yPtr + 6 * sides.y) + BlockRows.Load(columns + 6)).Store(yPtr + 6 * sides.y);
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
                    BlockRows.Add(work, sides, columns, impulse);
                }

                SolveAgainstFactor(ref run, work);
                // The scale is the one that best meets the targets in the least-squares sense, kept between 0 and 1.
                double along = 0.0, squared = 0.0;
                for (int k = 0; k < given; k++)
                {
                    int row = held[k];
                    double moved = BlockRows.Dot(work, run.BlockSections[run.RowBlock[row]],
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
        /// Each body's impulse once the run's sections have their change of velocity, as <see cref="SumBodyImpulses"/>
        /// gives it, written to <paramref name="linear"/> and <paramref name="angular"/>. A body of a section of several
        /// takes the impulse that gives it its section's motion, rigid about the section's centre; for a solve, the
        /// locked pairs' rows then get the impulses that carry those bodies' share, by inverse dynamics. For a push-out,
        /// the sections start from rest and nothing else is written.
        /// </summary>
        static unsafe void SettleForest(ref ForestRun run, in SparseConstraintSolver.Problem problem, float* impulses,
            float3* linear, float3* angular, bool pushOut)
        {
            long summing = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            SumBodyImpulses(problem, impulses, linear, angular);
            run.BodyImpulseTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - summing;
            if (run.VesselMode)
            {
                SettleVessel(ref run, problem, impulses, linear, angular, pushOut);
                return;
            }

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

                // A locked section at rest keeps a spin of order 1e-19 rad/s from rounding, whose squares the next step's
                // row and Jacobian builds and the integration carry into denormal floats, several times slower to compute
                // with. Nothing that small is a real turn.
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
                run.SettleTurn[s] = VectorMath.Rotation(turn);
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
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestSectionsKind, partTasks, null);
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
            SparseSolveTeam.RunAll(run.Team, &tasks, SparseSolveTeam.ForestSectionsKind, partTasks, null);
        }

        /// <summary>
        /// One task's parts of a settle: each body of a section of several takes the impulse that gives it the section's
        /// motion, and for a solve, what it still needs from its locked edges is noted.
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

                tasks->Linear[i] = VectorMath.Flushed(linearImpulse);
                tasks->Angular[i] = VectorMath.Flushed(angularImpulse);
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
        /// The impulses of a locked pair's rows that carry <paramref name="need"/> (an impulse and its moment about the
        /// child's centre) into the child: the least-squares split weighted by each row's stiffness, z = W J (J^T W J)^-1
        /// need, which is how the rows would share it if the pair were a very stiff spring. (J^T W J)^-1 is kept in the
        /// child's own frame, where a locked pair's rows stay put, from one solve to the next while the pair stays locked
        /// and its rows' compliances don't change.
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
                        double6 j = BlockRows.Load(problem.Jacobian + SparseConstraintSolver.ColumnsPerRow * (long)row + side);
                        impulses[row] = VectorMath.Flushed(double6.Dot(j, x) * RowStiffness(ref run, problem, pair, k, row));
                    }
                }
            }
        }

        /// <summary>
        /// A locked row's weight in its pair's split: its spring alone, gamma / epsilon (h^2 k), so that the rows share a load
        /// as their springs would at rest; their compliance alone, which counts the dampers too, would give the damped rows
        /// more than their share, and the pair would read a deflection locked that it doesn't have unlocked.
        /// </summary>
        static unsafe double RowStiffness(ref ForestRun run, in SparseConstraintSolver.Problem problem, int pair, int k, int row) =>
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
                        VesselBlocks.AddOuter(m, RowStiffness(ref run, problem, pair, k, row), j, j);
                    }
                }
            }

            // Rows that don't reach every direction leave the rest of the load to rounding, not to a singular matrix.
            double trace = m[0] + m[7] + m[14] + m[21] + m[28] + m[35];
            for (int d = 0; d < 6; d++)
            {
                m[7 * d] += 1e-12 * trace + 1e-300;
            }

            VesselBlocks.InvertSymmetric(inverse, m);
        }

        /// <summary>
        /// Each tree pair's deflection under this solve's load: the largest over its rotation rows of how far the row's
        /// spring would turn to carry its impulse, z / (h k) with k = gamma / (epsilon h^2), the spring without its damper.
        /// Summed over a step's substeps; the
        /// next step's first substep turns the sum into their mean, the deflection the lock tests read from then on (see
        /// <see cref="ForestIsland.PairDeflectionSum"/>). Returns the largest deflection the lock tests now read.
        /// </summary>
        static unsafe float RecordDeflections(ref ForestRun run, JacobianHeader** headers, float* impulses, int substep,
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
                            // The row's impulse is (-gamma phi / h - J v) / epsilon with v the velocities after the solve,
                            // so its spring's turn is phi = -(z epsilon + J v) h / gamma: the damper's share, which a
                            // vibrating pair rectifies into a load it doesn't carry, comes out. Locked, J v is zero.
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
        static unsafe void SolveAgainstFactor(ref ForestRun run, double* x)
        {
            ref VesselContext context = ref run.Forest->Context;
            VesselContext* contextPtr = (VesselContext*)UnsafeUtility.AddressOf(ref context);
            long bytes = 6L * context.SectionCount * sizeof(double);
            UnsafeUtility.MemCpy(context.Rhs, x, bytes);
            SparseSolveTeam.RunAll(run.Team, contextPtr, SparseSolveTeam.ForestForwardKind, context.ChunkCount, null);
            context.SubstituteTop();
            SparseSolveTeam.RunAll(run.Team, contextPtr, SparseSolveTeam.ForestSubstituteKind, context.ChunkCount, null);
            UnsafeUtility.MemCpy(x, context.Solution, bytes);
        }

        /// <summary>One chunk of a forest's factor, for a team.</summary>
        internal static unsafe void ForestFactorTask(void* context, int chunk)
        {
            var vessel = (VesselContext*)context;
            var scratch = new NativeArray<double>(VesselBlocks.Size * math.max(1, vessel->MaxStructure), Allocator.Temp,
                NativeArrayOptions.UninitializedMemory);
            vessel->FactorChunk(chunk, (double*)scratch.GetUnsafePtr());
            scratch.Dispose();
        }

        /// <summary>
        /// Each moving body's share of J^T z, in the frame of the Jacobian's columns, as
        /// <see cref="SparseConstraintSolver.BodyImpulse"/> gives it.
        /// </summary>
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

namespace Unity.Physics
{
    /// <summary>
    /// Limits on the boundary block of a vessel island, so that the dense solve of the boundary stays small. The
    /// defaults come from the vessel solver study. A field that is zero takes its default, and a field that is negative
    /// turns its limit off, except for <see cref="SelfContactManifolds"/>, which never goes above
    /// <see cref="MaxSelfContactManifolds"/>. See <see cref="Solver"/>'s CapBoundary for how the caps choose rows.
    /// </summary>
    public struct BoundaryCaps
    {
        /// <summary>
        /// The most contact points against the world or against bodies outside the vessel, over the whole island. The
        /// solve keeps the deepest points.
        /// </summary>
        public int OutsideContactPoints;

        /// <summary>The most points that the solve keeps from one contact manifold. It keeps the deepest.</summary>
        public int PointsPerManifold;

        /// <summary>
        /// The most contact manifolds between two parts of the same vessel. The solve keeps the deepest manifolds, up to
        /// <see cref="MaxSelfContactManifolds"/>. A manifold that the cap drops keeps the impulse that it had before the
        /// solve.
        /// </summary>
        public int SelfContactManifolds;

        /// <summary>The most rounds of block pivoting in one solve of the boundary.</summary>
        public int PivotRounds;

        /// <summary>
        /// If true, every solve of the boundary starts from the guess that the rows' builder makes, instead of from the
        /// states that the rows had at the end of the last solve. For measuring what the warm start saves.
        /// </summary>
        public bool ColdStart;

        /// <summary>
        /// If true, every round of pivoting factors its free rows from the start. If false, a round keeps what it can of
        /// the last round's factor, see
        /// <see cref="Unity.Physics.Numerics.Retooled.BoundarySolver"/>. For measuring what the updates save.
        /// </summary>
        public bool RefactorEveryRound;

        /// <summary>
        /// If true, and <see cref="RefactorEveryRound"/> is false, a round keeps the last round's factor up to the first
        /// row that left the free set and inserts the rest after it, instead of deleting the rows that left wherever they
        /// are, see <see cref="Unity.Physics.Numerics.Retooled.FactorUpdate"/>. For measuring the two ways.
        /// </summary>
        public bool KeepFactorPrefix;

        /// <summary>
        /// If true, the first factor of a vessel island's boundary solve takes its free rows in the boundary's order. If
        /// false, it takes the rows that have kept their state for the most solves first, so that the rows that are
        /// likely to leave the free set sit at the end of the factor, where deleting them costs little. For measuring.
        /// </summary>
        public bool UnorderedFactor;

        /// <summary>The way that the boundary's rounds change their factor, from <see cref="RefactorEveryRound"/> and <see cref="KeepFactorPrefix"/>.</summary>
        public Unity.Physics.Numerics.Retooled.FactorUpdate FactorUpdate => RefactorEveryRound
            ? Unity.Physics.Numerics.Retooled.FactorUpdate.Refactor
            : KeepFactorPrefix ? Unity.Physics.Numerics.Retooled.FactorUpdate.KeepPrefix : Unity.Physics.Numerics.Retooled.BoundarySolver.DefaultUpdate;

        /// <summary>
        /// If true, the key of a boundary row knows a body outside the island's vessels by its place in the world, as the
        /// keys did before, instead of by its entity. For measuring what the entity keys keep.
        /// </summary>
        public bool KeyByWorldIndex;

        /// <summary>
        /// If true, every one-sided joint row stays in the boundary, even when the row is far from its stop. If false, the
        /// solve leaves out a dormant row, see <see cref="Solver"/>'s MarkDormantRows. RETOOLED_DORMANT_ROWS=0 sets this.
        /// </summary>
        public bool KeepDormantRows;

        /// <summary>
        /// If true, a held row never takes an impulse of its own: a held normal row keeps its warm start even while its
        /// contact separates, and it gets no push-out. If false, see <see cref="Solver"/>'s SolveHeldRows. For measuring
        /// what the held rows' own impulses do.
        /// </summary>
        public bool IdleHeldRows;

        /// <summary>
        /// How far, in metres or radians, a one-sided joint row must stay from its stop over the step for the solve to
        /// leave it out of the boundary.
        /// </summary>
        public const float DormantMargin = 0.005f;

        public const int DefaultOutsideContactPoints = 256;
        public const int DefaultPointsPerManifold = 4;
        public const int DefaultSelfContactManifolds = 8;
        public const int MaxSelfContactManifolds = 25;
        public const int DefaultPivotRounds = 8;

        /// <summary>The limit on outside contact points that applies, after defaults.</summary>
        public int OutsidePointLimit => Limit(OutsideContactPoints, DefaultOutsideContactPoints);

        /// <summary>The limit on points per manifold that applies, after defaults.</summary>
        public int PointLimit => Limit(PointsPerManifold, DefaultPointsPerManifold);

        /// <summary>
        /// The limit on self-contact manifolds that applies, after defaults. It is never more than 25, so a negative
        /// setting, which turns the other limits off, gives 25 here.
        /// </summary>
        public int SelfManifoldLimit =>
            Unity.Mathematics.math.min(Limit(SelfContactManifolds, DefaultSelfContactManifolds), MaxSelfContactManifolds);

        /// <summary>The limit on pivoting rounds that applies, after defaults.</summary>
        public int PivotRoundLimit => Limit(PivotRounds, DefaultPivotRounds);

        static int Limit(int value, int fallback) => value == 0 ? fallback : value < 0 ? int.MaxValue : value;
    }

    /// <summary>
    /// What one island remembers of its boundary rows between solves: for each row's key, the row's state at the end of
    /// the last solve, and the row's age, which is how many solves in a row before that one ended in the same state, up
    /// to <see cref="MostAge"/>. Each entry is the key with the state in its two lowest bits and the age in the six bits
    /// above them. The entries are sorted, so that a lookup is a binary search.
    /// </summary>
    internal unsafe struct BoundaryMemory
    {
        public const int StateMask = 3;
        public const int AgeShift = 2;
        public const int MostAge = 63;

        /// <summary>The bits of an entry that hold the state and the age. A key has these bits zero.</summary>
        public const int ValueMask = 0xFF;

        public UnsafeList<ulong> States;

        /// <summary>How many times the caps have chosen this island's boundary rows. It seeds the hash that breaks ties.</summary>
        public uint CapCount;

        public void Dispose()
        {
            if (States.IsCreated)
            {
                States.Dispose();
            }
        }

        /// <summary>Returns the index of the entry for <paramref name="key"/>, or -1 if there is none.</summary>
        public static int Find(in UnsafeList<ulong> states, ulong key)
        {
            int low = 0, high = states.Length;
            while (low < high)
            {
                int middle = (low + high) >> 1;
                if ((states[middle] & ~(ulong)ValueMask) < key)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }

            return low < states.Length && (states[low] & ~(ulong)ValueMask) == key ? low : -1;
        }

        /// <summary>The age that entry <paramref name="entry"/> holds.</summary>
        public static byte Age(ulong entry) => (byte)((entry & ValueMask) >> AgeShift);

        /// <summary>
        /// Replaces what <paramref name="states"/> holds with the keys, states and ages of this solve's rows. A row's age
        /// is its remembered age plus one if it ended in its remembered state, and zero otherwise.
        /// </summary>
        public static void Remember(ref UnsafeList<ulong> states, ulong* keys, byte* rowStates, byte* ages, int count)
        {
            if (!states.IsCreated)
            {
                states = new UnsafeList<ulong>(math.max(16, count), Allocator.Persistent);
            }

            states.Clear();
            for (int i = 0; i < count; i++)
            {
                states.Add(keys[i] | (ulong)(rowStates[i] & StateMask) | ((ulong)math.min((int)ages[i], MostAge) << AgeShift));
            }

            NativeSortExtension.Sort(states.Ptr, states.Length);
        }
    }

    /// <summary>
    /// Counts of what the vessel islands' boundary solves did, summed over every solve since the last
    /// <see cref="Reset"/>. Tests and benches read them between steps.
    /// </summary>
    public struct BoundaryCounters
    {
        /// <summary>Velocity solves of a vessel island's boundary, and the rows and pivoting rounds that they took.</summary>
        public long Solves;
        public long Rows;
        public long Rounds;

        /// <summary>The most rounds that one solve took.</summary>
        public long MostRounds;

        /// <summary>Solves that used every round that the cap allows.</summary>
        public long CappedSolves;

        /// <summary>
        /// Solves that ran out of rounds before they found any answer that met every condition. Sweeps of projected
        /// Gauss-Seidel finished these solves, see <see cref="Unity.Physics.Numerics.Retooled.BoundarySolver"/>.
        /// </summary>
        public long UnsettledSolves;

        /// <summary>Rows that started from the state that the island remembered for them.</summary>
        public long WarmRows;

        /// <summary>Contact points held out of the boundary, and self-contact manifolds held out of it.</summary>
        public long HeldPoints;
        public long HeldSelfManifolds;

        /// <summary>Contact manifolds against bodies outside the vessel that the cap on outside points left with no point.</summary>
        public long EmptiedOutsideManifolds;

        /// <summary>
        /// Held normal rows that gave back some of their warm start because their contact was separating, and held
        /// normal rows that got a push-out of their own because their contact was too deep.
        /// </summary>
        public long HeldReleases;
        public long HeldPushes;

        /// <summary>Held normal rows that pushed in a velocity solve because their contact was closing.</summary>
        public long HeldSupports;

        /// <summary>The most outside contact points and the most self-contact manifolds that one boundary kept.</summary>
        public long MostOutsidePoints;
        public long MostSelfManifolds;

        /// <summary>Joint rows that the solves left out of the boundary because they were dormant.</summary>
        public long DormantRows;

        /// <summary>
        /// What the solves did with the factor of their free rows, see <see cref="BoundarySolveStats"/>. Ticks are in
        /// profiler timestamp units.
        /// </summary>
        public BoundarySolveStats Factor;

        /// <summary>The time spent in the dense solve of the boundary, in profiler timestamp units.</summary>
        public long SolveTicks;

        /// <summary>
        /// Warm rows whose state at the end of the solve differs from the state that the island remembered for them, in
        /// the first substep of a step (against the last solve of the step before) and in later substeps.
        /// </summary>
        public long FirstSubstepChanges;
        public long LaterSubstepChanges;
        public long FirstSubstepSolves;

        /// <summary>
        /// The dense solves of a vessel island's push-out, which reuse the boundary block with other bounds, and the
        /// pivoting rounds and the time that they took.
        /// </summary>
        public long PushSolves;
        public long PushRounds;
        public long PushTicks;

        /// <summary>
        /// The time spent building vessel islands' boundary blocks: choosing the rows (the caps, the dormant rows and
        /// the rows' keys), walking each row's path through the factor on one lane, and filling W on the team.
        /// </summary>
        public long ChooseTicks;
        public long PathTicks;
        public long FillTicks;

        /// <summary>
        /// The time spent giving a vessel island's bodies the impulses of its rows, and pushing its bodies out of deep
        /// contacts, which includes checking whether any contact is deep.
        /// </summary>
        public long ApplyTicks;
        public long PushOutTicks;

        static readonly SharedStatic<BoundaryCounters> Shared = SharedStatic<BoundaryCounters>.GetOrCreate<BoundaryCounters>();

        /// <summary>The counts so far. Read them on the main thread between steps.</summary>
        public static BoundaryCounters Current => Shared.Data;

        public static void Reset()
        {
            Shared.Data = default;
        }

        internal static void AddSolve(int rows, int rounds, bool capped, bool unsettled, int warmRows)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Increment(ref data.Solves);
            Interlocked.Add(ref data.Rows, rows);
            Interlocked.Add(ref data.Rounds, rounds);
            Interlocked.Add(ref data.WarmRows, warmRows);
            if (capped)
            {
                Interlocked.Increment(ref data.CappedSolves);
            }

            if (unsettled)
            {
                Interlocked.Increment(ref data.UnsettledSolves);
            }

            Most(ref data.MostRounds, rounds);
        }

        internal static void AddCaps(int heldPoints, int heldManifolds, int outsidePoints, int selfManifolds, int emptiedManifolds)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Add(ref data.HeldPoints, heldPoints);
            Interlocked.Add(ref data.HeldSelfManifolds, heldManifolds);
            Interlocked.Add(ref data.EmptiedOutsideManifolds, emptiedManifolds);
            Most(ref data.MostOutsidePoints, outsidePoints);
            Most(ref data.MostSelfManifolds, selfManifolds);
        }

        internal static void AddFactor(in BoundarySolveStats stats, long solveTicks, int changes, bool firstSubstep)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Add(ref data.Factor.Rounds, stats.Rounds);
            Interlocked.Add(ref data.Factor.CouplingRounds, stats.CouplingRounds);
            Interlocked.Add(ref data.Factor.Factorizations, stats.Factorizations);
            Interlocked.Add(ref data.Factor.Updates, stats.Updates);
            Interlocked.Add(ref data.Factor.Reuses, stats.Reuses);
            Interlocked.Add(ref data.Factor.AddedRows, stats.AddedRows);
            Interlocked.Add(ref data.Factor.RemovedRows, stats.RemovedRows);
            Interlocked.Add(ref data.Factor.FactorTicks, stats.FactorTicks);
            Interlocked.Add(ref data.Factor.SolveTicks, stats.SolveTicks);
            Interlocked.Add(ref data.Factor.PivotTicks, stats.PivotTicks);
            Interlocked.Add(ref data.Factor.FullFactorTicks, stats.FullFactorTicks);
            Interlocked.Add(ref data.Factor.FreeRows, stats.FreeRows);
            Most(ref data.Factor.MostFreeRows, stats.MostFreeRows);
            Fewest(ref data.Factor.FewestFreeRows, stats.FewestFreeRows);
            Interlocked.Add(ref data.SolveTicks, solveTicks);
            if (firstSubstep)
            {
                Interlocked.Increment(ref data.FirstSubstepSolves);
                Interlocked.Add(ref data.FirstSubstepChanges, changes);
            }
            else
            {
                Interlocked.Add(ref data.LaterSubstepChanges, changes);
            }
        }

        internal static void AddPush(int rounds, long ticks)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Increment(ref data.PushSolves);
            Interlocked.Add(ref data.PushRounds, rounds);
            Interlocked.Add(ref data.PushTicks, ticks);
        }

        internal static void AddFinish(long applyTicks, long pushOutTicks)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Add(ref data.ApplyTicks, applyTicks);
            Interlocked.Add(ref data.PushOutTicks, pushOutTicks);
        }

        internal static void AddBuild(long chooseTicks, long pathTicks, long fillTicks)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Add(ref data.ChooseTicks, chooseTicks);
            Interlocked.Add(ref data.PathTicks, pathTicks);
            Interlocked.Add(ref data.FillTicks, fillTicks);
        }

        internal static void AddDormant(int dormantRows)
        {
            Interlocked.Add(ref Shared.Data.DormantRows, dormantRows);
        }

        internal static void AddHeld(int releases, int pushes, int supports)
        {
            Interlocked.Add(ref Shared.Data.HeldReleases, releases);
            Interlocked.Add(ref Shared.Data.HeldPushes, pushes);
            Interlocked.Add(ref Shared.Data.HeldSupports, supports);
        }

        /// <summary>Lowers <paramref name="fewest"/> to <paramref name="value"/>. Zero counts as no value in both.</summary>
        static void Fewest(ref long fewest, long value)
        {
            long seen = Interlocked.Read(ref fewest);
            while (value > 0 && (seen == 0 || value < seen))
            {
                long previous = Interlocked.CompareExchange(ref fewest, value, seen);
                if (previous == seen)
                {
                    break;
                }

                seen = previous;
            }
        }

        static void Most(ref long most, long value)
        {
            long seen = Interlocked.Read(ref most);
            while (value > seen)
            {
                long previous = Interlocked.CompareExchange(ref most, value, seen);
                if (previous == seen)
                {
                    break;
                }

                seen = previous;
            }
        }
    }
}
