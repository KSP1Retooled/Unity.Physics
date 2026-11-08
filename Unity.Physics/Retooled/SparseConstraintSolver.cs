using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Solves the same constraint problem as the direct solver's dense path, without forming J M^-1 J^t. That matrix is dense
    /// wherever constraints share a body, so a part with thirty attachments costs thirty squared blocks and a craft costs the
    /// cube of its rows. This keeps the bodies as unknowns instead:
    ///
    ///     | M   J^t | | du |   | 0 |
    ///     | J   -E  | | -z | = | b |
    ///
    /// Eliminating du gives (J M^-1 J^t + E) z = b, so the answer is the same. The graph of this matrix is the craft itself:
    /// bodies and the joints between them. A tree factors with no fill at all from its leaves inwards, and each strut adds a
    /// little. With M and E positive the matrix is quasi-definite, so every symmetric ordering factors stably without pivoting
    /// and the structure can be settled before any numbers are seen.
    ///
    /// Bounds on z are solved by block principal pivoting, as <see cref="LCP.SolveMLCP"/> does. A row at a bound only changes
    /// values in this matrix, so each pivoting step is one numeric refactorization of the same structure.
    /// </summary>
    internal unsafe struct SparseConstraintSolver : IDisposable
    {
        /// <summary>The columns of one row of the constraint Jacobian: linear then angular, for body A then body B.</summary>
        public const int ColumnsPerRow = 12;

        const int BodySize = 6;
        const int SideColumns = 6;
        const float Tolerance = 1e-4f;
        const int MaxPivotingIterations = 200;
        const int MaxBlockIterationsWithoutProgress = 3;
        const int MaxCouplingIterations = 25;

        /// <summary>How far a coupled bound may move, as a share of itself, before it asks for another answer.</summary>
        const float CouplingTolerance = 1e-3f;
        const double SmallestPivot = 1e-30;
        const int InitialCapacity = 16;

        /// <summary>How close a refined answer has to come, in m/s or rad/s at any body or row.</summary>
        const double RefinementTolerance = 1e-5;
        const int MaxRefinements = 3;

        /// <summary>A body's mass matrix, with its inertia in the same frame as the Jacobian's angular columns.</summary>
        public struct Body
        {
            public double Mass;
            public double3x3 Inertia;
        }

        /// <summary>Consecutive rows acting between two bodies. A body index of -1 is something that doesn't move.</summary>
        public struct ConstraintBlock
        {
            public int BodyA;
            public int BodyB;
            public int FirstRow;
            public int RowCount;
        }

        /// <summary>The problem to solve, which has to outlive the call and is only read.</summary>
        public struct Problem
        {
            public int BodyCount;
            public Body* Bodies;

            public int BlockCount;
            public ConstraintBlock* Blocks;

            public int RowCount;

            /// <summary><see cref="ColumnsPerRow"/> values for each row.</summary>
            public float* Jacobian;

            /// <summary>The regularization on each row's diagonal, which has to be positive.</summary>
            public float* Epsilon;

            public float* RightHandSide;
            public float* Lower;
            public float* Upper;

            /// <summary>
            /// A first guess at which rows sit at a bound, with the values of <see cref="LCP.MLCPIndexFlag"/>, or null for none.
            /// A good guess saves pivoting steps and doesn't change the answer.
            /// </summary>
            public byte* InitialStates;

            public int CouplingCount;

            /// <summary>Rows whose bounds follow other rows' impulses, or null for none.</summary>
            public Coupling* Couplings;
        }

        /// <summary>
        /// Rows whose bounds are ±<see cref="Factor"/> times the sum of other rows' impulses: the friction of a contact, bounded
        /// by its normal impulses. Their own bounds in the problem are ignored.
        /// </summary>
        public struct Coupling
        {
            public int FirstRow;
            public int RowCount;
            public int FirstSourceRow;
            public int SourceRowCount;
            public float Factor;
        }

        /// <summary>
        /// The size of a solve, which doesn't depend on the machine, and its time by stage. Times are in the units of
        /// <see cref="ProfilerUnsafeUtility.Timestamp"/>.
        /// </summary>
        public struct Statistics
        {
            public int Nodes;

            /// <summary>The independent subtrees the factorization splits into, and the nodes above them that run alone.</summary>
            public int Tasks;
            public int TopNodes;

            public int FactorBlocks;
            public long FactorScalars;
            public int PivotingIterations;
            public int Factorizations;
            public int Refinements;
            public int CouplingIterations;

            /// <summary>What a reused factorization's answer finally missed by, in m/s or rad/s.</summary>
            public double Residual;

            public long AnalyzeTime;
            public long AssembleTime;
            public long FactorizeTime;
            public long SubstituteTime;
            public long PivotTime;
        }

        static readonly ProfilerMarker AnalyzeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Sparse.Analyze");
        static readonly ProfilerMarker AssembleMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Sparse.Assemble");
        static readonly ProfilerMarker FactorizeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Sparse.Factorize");
        static readonly ProfilerMarker SubstituteMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Sparse.Substitute");
        static readonly ProfilerMarker PivotMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Sparse.Pivot");
        static readonly ProfilerMarker ResidualMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Sparse.Residual");
        static readonly ProfilerMarker RightHandSideMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Sparse.RightHandSide");

        const byte Free = 0;
        const byte LowerTight = 1;
        const byte UpperTight = 2;

        const int NoNode = -1;
        const int Top = -1;

        /// <summary>How many subtrees a factorization aims to split into, enough for every lane to keep busy.</summary>
        const int TasksPerFactorization = 64;

        // What a task of a team's stage does.
        const int FactorizeTask = 0;
        const int ForwardTask = 1;
        const int BackwardTask = 2;
        const int BoundaryUpdateTask = 3;
        const int TopUpdateTask = 4;
        const int ClearTask = 5;
        const int AssembleTask = 6;
        const int RowResidualTask = 7;
        const int BodyResidualTask = 8;
        const int BoundaryPushTask = 9;

        /// <summary>How much of the factor one task clears, and how many blocks or bodies one task assembles or checks.</summary>
        const int ScalarsPerClearTask = 16384;
        const int BlocksPerTask = 64;
        const int BodiesPerTask = 64;

        /// <summary>
        /// How many neighbours a top node needs before its updates are worth handing out one neighbour each. Fewer finish
        /// before the lanes would notice them.
        /// </summary>
        const int SharedUpdateNeighbours = 6;

        /// <summary>A node of the graph: a body, or all the rows between one pair of bodies.</summary>
        struct Node
        {
            public int Size;
            public int Position;
            public double Sign;
            public int BodyA;
            public int BodyB;

            // Where its unknowns start in the solution.
            public int Start;

            // Its diagonal block in the factor, and its later neighbours in elimination order.
            public int Diagonal;
            public int FirstNeighbour;
            public int NeighbourCount;

            // The subtree it is eliminated in, or Top, and how many of its neighbours come first and are in that subtree too.
            public int Task;
            public int OwnNeighbours;
        }

        struct Neighbour
        {
            public int Node;

            // Where its block starts in the factor, with this neighbour's rows and the column node's columns.
            public int Offset;
        }

        UnsafeList<Node> nodes;
        UnsafeList<int> order;
        UnsafeList<Neighbour> neighbours;

        // The subtrees of the elimination tree that factor and substitute independently: each one's positions, in order, start
        // at its entry in taskStarts. The top joins them and runs alone, and the boundary is every subtree node that has a
        // neighbour in the top, whose updates of the top wait until the subtrees are done.
        UnsafeList<int> taskStarts;
        UnsafeList<int> taskPositions;
        UnsafeList<int> topPositions;
        UnsafeList<int> boundaryPositions;

        // The boundary's updates of the top, by the top node whose blocks they land in: a boundary node's position and which
        // of its neighbours that is. Each top node's list starts at its entry in topUpdateStarts.
        UnsafeList<int> topUpdateStarts;
        UnsafeList<int2> topUpdates;

        // Each body's rows, as row * 2 + 1 for side B, starting at its entry in bodyRowStarts.
        UnsafeList<int> bodyRowStarts;
        UnsafeList<int> bodyRows;

        // The top node whose updates a stage splits by neighbour, the problem a stage reads, and each task's largest miss.
        int stageNode;
        Problem stageProblem;
        UnsafeList<double> stageWorst;

        UnsafeList<int> rowNode;
        UnsafeList<int> rowInNode;
        UnsafeList<double> factor;
        UnsafeList<double> right;
        UnsafeList<double> solution;
        UnsafeList<double> correction;

        // One over each diagonal of the factor, by unknown, so that a substitution multiplies where it would divide.
        UnsafeList<double> inverseDiagonal;

        // One over each body's mass and over its smallest inertia, which turn its residual into speeds.
        UnsafeList<double2> bodySpeedScale;

        UnsafeList<byte> state;

        // Each row's bounds: the problem's, or for a coupled row the last ones its coupling gave it. Coupled rows keep theirs
        // between solves of one shape, as the best first guess at the next.
        UnsafeList<float> lower;
        UnsafeList<float> upper;
        UnsafeList<bool> coupled;

        // The shape of the problem the structure above belongs to.
        UnsafeList<ConstraintBlock> analyzedBlocks;
        int analyzedBodies;
        bool analyzedBounded;

        // Whether factor holds a factorization of a recent problem of this shape, for the current state.
        bool factored;

        Statistics statistics;

        public SparseConstraintSolver(Allocator allocator)
        {
            nodes = new UnsafeList<Node>(InitialCapacity, allocator);
            order = new UnsafeList<int>(InitialCapacity, allocator);
            neighbours = new UnsafeList<Neighbour>(InitialCapacity, allocator);
            taskStarts = new UnsafeList<int>(InitialCapacity, allocator);
            taskPositions = new UnsafeList<int>(InitialCapacity, allocator);
            topPositions = new UnsafeList<int>(InitialCapacity, allocator);
            boundaryPositions = new UnsafeList<int>(InitialCapacity, allocator);
            topUpdateStarts = new UnsafeList<int>(InitialCapacity, allocator);
            topUpdates = new UnsafeList<int2>(InitialCapacity, allocator);
            bodyRowStarts = new UnsafeList<int>(InitialCapacity, allocator);
            bodyRows = new UnsafeList<int>(InitialCapacity, allocator);
            stageWorst = new UnsafeList<double>(InitialCapacity, allocator);
            stageNode = NoNode;
            Team = null;
            rowNode = new UnsafeList<int>(InitialCapacity, allocator);
            rowInNode = new UnsafeList<int>(InitialCapacity, allocator);
            factor = new UnsafeList<double>(InitialCapacity, allocator);
            right = new UnsafeList<double>(InitialCapacity, allocator);
            solution = new UnsafeList<double>(InitialCapacity, allocator);
            correction = new UnsafeList<double>(InitialCapacity, allocator);
            inverseDiagonal = new UnsafeList<double>(InitialCapacity, allocator);
            bodySpeedScale = new UnsafeList<double2>(InitialCapacity, allocator);
            state = new UnsafeList<byte>(InitialCapacity, allocator);
            lower = new UnsafeList<float>(InitialCapacity, allocator);
            upper = new UnsafeList<float>(InitialCapacity, allocator);
            coupled = new UnsafeList<bool>(InitialCapacity, allocator);
            analyzedBlocks = new UnsafeList<ConstraintBlock>(InitialCapacity, allocator);
            analyzedBodies = -1;
            analyzedBounded = false;
            factored = false;
            statistics = default;
        }

        public void Dispose()
        {
            nodes.Dispose();
            order.Dispose();
            neighbours.Dispose();
            taskStarts.Dispose();
            taskPositions.Dispose();
            topPositions.Dispose();
            boundaryPositions.Dispose();
            topUpdateStarts.Dispose();
            topUpdates.Dispose();
            bodyRowStarts.Dispose();
            bodyRows.Dispose();
            stageWorst.Dispose();
            rowNode.Dispose();
            rowInNode.Dispose();
            factor.Dispose();
            right.Dispose();
            solution.Dispose();
            correction.Dispose();
            inverseDiagonal.Dispose();
            bodySpeedScale.Dispose();
            state.Dispose();
            lower.Dispose();
            upper.Dispose();
            coupled.Dispose();
            analyzedBlocks.Dispose();
        }

        /// <summary>
        /// The lanes that share this solver's factorizations and substitutions, or null to run them on the caller's thread
        /// alone. The answer is the same either way.
        /// </summary>
        public SparseSolveTeam* Team;

        public Statistics Stats => statistics;

        /// <summary>
        /// What a body takes from the impulses of the rows on it, in the frame of the Jacobian's columns: its part of J^t z. A
        /// body reads only its own rows, so bodies can do this on any thread.
        /// </summary>
        public void BodyImpulse(in Problem problem, float* impulses, int body, out float3 linear, out float3 angular)
        {
            linear = float3.zero;
            angular = float3.zero;
            for (int i = bodyRowStarts[body]; i < bodyRowStarts[body + 1]; i++)
            {
                int row = bodyRows[i] >> 1;
                float* columns = problem.Jacobian + row * ColumnsPerRow + (bodyRows[i] & 1) * SideColumns;
                linear += impulses[row] * new float3(columns[0], columns[1], columns[2]);
                angular += impulses[row] * new float3(columns[3], columns[4], columns[5]);
            }
        }

        /// <summary>The bounds the last solve held a row to, which for a coupled row are what its coupling settled on.</summary>
        public float2 Bounds(int row) => new float2(lower[row], upper[row]);

        /// <summary>
        /// Writes each row's impulse to <paramref name="impulses"/>. With <paramref name="bounded"/> false the bounds are
        /// ignored. Returns false if pivoting ran out of iterations, leaving the last impulses clamped to their bounds.
        ///
        /// A solver kept between calls reuses its work. Bodies barely move from one solve to the next, so the last factorization
        /// is nearly right for the next problem of the same shape, and iterative refinement closes the gap for a fraction of the
        /// cost of a new one. The answer is checked against the problem it is for, so a factorization too old to help costs a
        /// little time and never accuracy.
        /// </summary>
        public bool Solve(in Problem problem, float* impulses, bool bounded)
        {
            statistics = new Statistics
            {
                Nodes = statistics.Nodes,
                Tasks = statistics.Tasks,
                TopNodes = statistics.TopNodes,
                FactorBlocks = statistics.FactorBlocks,
                FactorScalars = statistics.FactorScalars,
            };

            long started = ProfilerUnsafeUtility.Timestamp;
            MarkCoupledRows(problem, bounded);
            bool sameShape = HasAnalyzedShape(problem, bounded);
            if (!sameShape)
            {
                Analyze(problem, bounded);
                factored = false;
            }

            LoadBounds(problem, sameShape);
            statistics.AnalyzeTime = Lap(ref started);

            bodySpeedScale.Resize(problem.BodyCount);
            for (int body = 0; body < problem.BodyCount; body++)
            {
                double3x3 inertia = problem.Bodies[body].Inertia;
                double smallestInertia = math.min(inertia.c0.x, math.min(inertia.c1.y, inertia.c2.z));
                bodySpeedScale[body] = new double2(1.0 / problem.Bodies[body].Mass, 1.0 / smallestInertia);
            }

            if (!factored)
            {
                SetInitialState(problem, bounded);
            }

            int leastInfeasible = int.MaxValue;
            int stalled = 0;
            bool flipAll = true;
            int couplings = 0;
            for (int iteration = 1; iteration <= MaxPivotingIterations; iteration++)
            {
                statistics.PivotingIterations = iteration;
                started = ProfilerUnsafeUtility.Timestamp;
                AssembleRightHandSide(problem);
                statistics.AssembleTime += Lap(ref started);

                bool solved = factored && SolveWithOldFactor(problem);
                statistics.SubstituteTime += Lap(ref started);
                if (!solved)
                {
                    AssembleMatrix(problem);
                    statistics.AssembleTime += Lap(ref started);
                    Factorize();
                    factored = true;
                    statistics.Factorizations++;
                    statistics.FactorizeTime += Lap(ref started);
                    UnsafeUtility.MemCpy(solution.Ptr, right.Ptr, solution.Length * sizeof(double));
                    Substitute(solution.Ptr);
                }

                ReadImpulses(problem, impulses);
                statistics.SubstituteTime += Lap(ref started);
                if (!bounded)
                {
                    return true;
                }

                int infeasible = Pivot(problem, impulses, flipAll);
                statistics.PivotTime += Lap(ref started);
                if (infeasible == 0)
                {
                    // Each answer moves the normal impulses that bound the friction, and new bounds ask for a new answer.
                    // Most rounds only move the bounds of tight rows, which reuses the factor. A capped run keeps its last
                    // bounds, which the answer does respect.
                    if (couplings == MaxCouplingIterations || !Couple(problem, impulses))
                    {
                        return true;
                    }

                    couplings++;
                    statistics.CouplingIterations = couplings;
                    leastInfeasible = int.MaxValue;
                    stalled = 0;
                    flipAll = true;
                    continue;
                }

                // A row that changed sides changed the matrix.
                factored = false;

                // Flipping every infeasible row usually lands in a few steps but can cycle. A run that stops improving falls
                // back to Murty's rule, which flips only the last of them and is guaranteed to finish.
                if (infeasible < leastInfeasible)
                {
                    leastInfeasible = infeasible;
                    stalled = 0;
                    flipAll = true;
                }
                else if (++stalled >= MaxBlockIterationsWithoutProgress)
                {
                    flipAll = false;
                }
            }

            for (int row = 0; row < problem.RowCount; row++)
            {
                impulses[row] = math.clamp(impulses[row], lower[row], upper[row]);
            }

            return false;
        }

        /// <summary>
        /// Adds to <paramref name="impulses"/>, the answer of the last <see cref="Solve"/>, what a small change to the free
        /// rows' right-hand side asks of them. The matrix is the same, so one substitution with the factor in hand is the
        /// whole of it: the factor may be of an earlier problem, but its error is a fraction of a correction that is itself
        /// small, so it isn't refined. Tight rows stay where they are. Returns false, changing nothing, if there is no factor.
        /// </summary>
        public bool SolveChange(in Problem problem, float* rightHandSideChange, float* impulses)
        {
            if (!factored)
            {
                return false;
            }

            long started = ProfilerUnsafeUtility.Timestamp;
            double* step = correction.Ptr;
            UnsafeUtility.MemClear(step, correction.Length * sizeof(double));
            for (int row = 0; row < problem.RowCount; row++)
            {
                if (state[row] == Free)
                {
                    step[nodes[rowNode[row]].Start + rowInNode[row]] = rightHandSideChange[row];
                }
            }

            Substitute(step);
            for (int row = 0; row < problem.RowCount; row++)
            {
                if (state[row] == Free)
                {
                    float impulse = impulses[row] - (float)step[nodes[rowNode[row]].Start + rowInNode[row]];
                    impulses[row] = math.clamp(impulse, lower[row], upper[row]);
                }
            }

            statistics.SubstituteTime += Lap(ref started);
            return true;
        }

        static long Lap(ref long started)
        {
            long now = ProfilerUnsafeUtility.Timestamp;
            long elapsed = now - started;
            started = now;
            return elapsed;
        }

        void SetInitialState(in Problem problem, bool bounded)
        {
            state.Clear();
            state.Resize(problem.RowCount, NativeArrayOptions.ClearMemory);
            if (!bounded)
            {
                return;
            }

            if (problem.InitialStates != null)
            {
                UnsafeUtility.MemCpy(state.Ptr, problem.InitialStates, problem.RowCount);
            }

            for (int row = 0; row < problem.RowCount; row++)
            {
                if (lower[row] == upper[row])
                {
                    state[row] = LowerTight;
                }
                else if (coupled[row])
                {
                    // Kept bounds are a guess, and a row tight at one would start with a full friction impulse.
                    state[row] = Free;
                }
            }
        }

        void MarkCoupledRows(in Problem problem, bool bounded)
        {
            coupled.Clear();
            coupled.Resize(problem.RowCount, NativeArrayOptions.ClearMemory);
            for (int i = 0; bounded && i < problem.CouplingCount; i++)
            {
                Coupling coupling = problem.Couplings[i];
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    coupled[row] = true;
                }
            }
        }

        /// <summary>
        /// A row whose bounds meet in the problem has one possible impulse, so it is a known force and not an unknown. A
        /// coupled row's bounds can meet now and part later, so it is always an unknown.
        /// </summary>
        bool IsFixed(in Problem problem, int row)
        {
            return problem.Lower[row] == problem.Upper[row] && !coupled[row];
        }

        void LoadBounds(in Problem problem, bool keepCoupled)
        {
            bool kept = keepCoupled && lower.Length == problem.RowCount;
            lower.Resize(problem.RowCount);
            upper.Resize(problem.RowCount);
            for (int row = 0; row < problem.RowCount; row++)
            {
                if (!coupled[row])
                {
                    lower[row] = problem.Lower[row];
                    upper[row] = problem.Upper[row];
                }
                else if (!kept)
                {
                    lower[row] = 0.0f;
                    upper[row] = 0.0f;
                }
            }
        }

        /// <summary>
        /// Sets each coupled row's bounds from the impulses its sources have now, and returns whether any moved. A row
        /// tight at a bound stays tight at where the bound moved to, and pivoting frees it if that is wrong.
        /// </summary>
        bool Couple(in Problem problem, float* impulses)
        {
            bool moved = false;
            for (int i = 0; i < problem.CouplingCount; i++)
            {
                Coupling coupling = problem.Couplings[i];
                float sum = 0.0f;
                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    sum += impulses[row];
                }

                float bound = math.max(sum, 0.0f) * coupling.Factor;
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    if (math.abs(upper[row] - bound) <= math.max(Tolerance, CouplingTolerance * bound))
                    {
                        continue;
                    }

                    bool wasFixed = lower[row] == upper[row];
                    lower[row] = -bound;
                    upper[row] = bound;
                    moved = true;
                    if (bound == 0.0f && state[row] == Free)
                    {
                        state[row] = LowerTight;
                        factored = false;
                    }
                    else if (wasFixed && bound > 0.0f)
                    {
                        state[row] = Free;
                        factored = false;
                    }
                }
            }

            return moved;
        }

        // Structure

        /// <summary>Whether the nodes and their order from the last analysis fit this problem.</summary>
        bool HasAnalyzedShape(in Problem problem, bool bounded)
        {
            if (problem.BodyCount != analyzedBodies || bounded != analyzedBounded || problem.RowCount != rowNode.Length
                || problem.BlockCount != analyzedBlocks.Length)
            {
                return false;
            }

            if (UnsafeUtility.MemCmp(problem.Blocks, analyzedBlocks.Ptr, problem.BlockCount * sizeof(ConstraintBlock)) != 0)
            {
                return false;
            }

            if (bounded)
            {
                for (int row = 0; row < problem.RowCount; row++)
                {
                    if (IsFixed(problem, row) != (rowNode[row] == NoNode))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        void Analyze(in Problem problem, bool bounded)
        {
            using ProfilerMarker.AutoScope span = AnalyzeMarker.Auto();

            nodes.Clear();
            for (int body = 0; body < problem.BodyCount; body++)
            {
                nodes.Add(new Node { Size = BodySize, Sign = 1.0, BodyA = -1, BodyB = -1 });
            }

            MergeBlocksByBodyPair(problem, bounded);
            Order();
            LayOut();
            Partition();

            analyzedBodies = problem.BodyCount;
            analyzedBounded = bounded;
            analyzedBlocks.Resize(problem.BlockCount);
            UnsafeUtility.MemCpy(analyzedBlocks.Ptr, problem.Blocks, problem.BlockCount * sizeof(ConstraintBlock));

            statistics.Nodes = nodes.Length;
            statistics.Tasks = taskStarts.Length - 1;
            statistics.TopNodes = topPositions.Length;
            statistics.FactorBlocks = neighbours.Length;
            statistics.FactorScalars = factor.Length;
        }

        /// <summary>
        /// One joint emits several Jacobians between the same two bodies. As separate nodes they would all neighbour each other
        /// anyway, so one node holding all their rows is smaller and loses no structure. A row whose bounds meet joins no node:
        /// it has one possible impulse, so it is a known force on its bodies and not an unknown.
        /// </summary>
        void MergeBlocksByBodyPair(in Problem problem, bool bounded)
        {
            rowNode.Resize(problem.RowCount);
            rowInNode.Resize(problem.RowCount);

            var pairs = new NativeHashMap<long, int>(problem.BlockCount, Allocator.Temp);
            for (int b = 0; b < problem.BlockCount; b++)
            {
                ConstraintBlock block = problem.Blocks[b];
                int lower = math.min(block.BodyA, block.BodyB);
                int higher = math.max(block.BodyA, block.BodyB);
                long key = ((long)lower << 32) | (uint)higher;
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    if (bounded && IsFixed(problem, row))
                    {
                        rowNode[row] = NoNode;
                        continue;
                    }

                    if (!pairs.TryGetValue(key, out int node))
                    {
                        node = nodes.Length;
                        nodes.Add(new Node { Sign = -1.0, BodyA = lower, BodyB = higher });
                        pairs.Add(key, node);
                    }

                    ref Node target = ref nodes.ElementAt(node);
                    rowNode[row] = node;
                    rowInNode[row] = target.Size;
                    target.Size++;
                }
            }

            pairs.Dispose();

            bodyRowStarts.Clear();
            bodyRowStarts.Resize(problem.BodyCount + 1, NativeArrayOptions.ClearMemory);
            for (int pass = 0; pass < 2; pass++)
            {
                for (int b = 0; b < problem.BlockCount; b++)
                {
                    ConstraintBlock block = problem.Blocks[b];
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        for (int side = 0; side < 2; side++)
                        {
                            int body = side == 0 ? block.BodyA : block.BodyB;
                            if (body < 0)
                            {
                                continue;
                            }

                            // The first pass counts, and the second places each row at its body's next free slot.
                            if (pass == 0)
                            {
                                bodyRowStarts[body + 1]++;
                            }
                            else
                            {
                                bodyRows[bodyRowStarts[body]++] = row * 2 + side;
                            }
                        }
                    }
                }

                if (pass == 0)
                {
                    for (int body = 0; body < problem.BodyCount; body++)
                    {
                        bodyRowStarts[body + 1] += bodyRowStarts[body];
                    }

                    bodyRows.Resize(bodyRowStarts[problem.BodyCount]);
                }
            }

            // Placing moved each start to the next body's, so shift them back.
            for (int body = problem.BodyCount; body > 0; body--)
            {
                bodyRowStarts[body] = bodyRowStarts[body - 1];
            }

            bodyRowStarts[0] = 0;
        }

        /// <summary>
        /// Minimum degree: eliminating the node with the fewest neighbours joins the fewest pairs of them. A leaf has one, so a
        /// tree goes from its tips inwards and joins nothing. A body wins a tie, because that divides by a mass and not by a
        /// constraint's much smaller compliance.
        /// </summary>
        void Order()
        {
            int count = nodes.Length;
            order.Clear();
            neighbours.Clear();

            var lists = (UnsafeList<int>*)UnsafeUtility.Malloc(count * sizeof(UnsafeList<int>), UnsafeUtility.AlignOf<UnsafeList<int>>(), Allocator.Temp);
            for (int i = 0; i < count; i++)
            {
                lists[i] = new UnsafeList<int>(4, Allocator.Temp);
            }

            for (int i = 0; i < count; i++)
            {
                Node node = nodes[i];
                if (node.BodyA >= 0)
                {
                    lists[i].Add(node.BodyA);
                    lists[node.BodyA].Add(i);
                }

                if (node.BodyB >= 0)
                {
                    lists[i].Add(node.BodyB);
                    lists[node.BodyB].Add(i);
                }
            }

            var heap = new MinHeap(count, Allocator.Temp);
            for (int i = 0; i < count; i++)
            {
                heap.Push(Priority(lists[i].Length, i));
            }

            var eliminated = new NativeArray<bool>(count, Allocator.Temp);
            while (order.Length < count)
            {
                long entry = heap.Pop();
                int node = (int)(entry & uint.MaxValue);

                // A node is pushed again each time its degree changes, so only its newest entry is real.
                if (eliminated[node] || Priority(lists[node].Length, node) != entry)
                {
                    continue;
                }

                eliminated[node] = true;
                ref Node record = ref nodes.ElementAt(node);
                record.Position = order.Length;
                record.FirstNeighbour = neighbours.Length;
                record.NeighbourCount = lists[node].Length;
                order.Add(node);

                ref UnsafeList<int> around = ref lists[node];
                for (int i = 0; i < around.Length; i++)
                {
                    int other = around[i];
                    neighbours.Add(new Neighbour { Node = other });

                    ref UnsafeList<int> list = ref lists[other];
                    list.RemoveAtSwapBack(IndexOf(list, node));
                    for (int j = 0; j < around.Length; j++)
                    {
                        if (j != i && IndexOf(list, around[j]) < 0)
                        {
                            list.Add(around[j]);
                        }
                    }

                    heap.Push(Priority(list.Length, other));
                }
            }

            for (int i = 0; i < count; i++)
            {
                lists[i].Dispose();
            }

            UnsafeUtility.Free(lists, Allocator.Temp);
            eliminated.Dispose();
            heap.Dispose();
        }

        long Priority(int degree, int node)
        {
            return ((long)degree << 33) | ((nodes[node].Sign < 0.0 ? 1L : 0L) << 32) | (uint)node;
        }

        static int IndexOf(in UnsafeList<int> list, int value)
        {
            for (int i = 0; i < list.Length; i++)
            {
                if (list[i] == value)
                {
                    return i;
                }
            }

            return -1;
        }

        /// <summary>Sorts each column's neighbours by when they are eliminated, then gives every block its place.</summary>
        void LayOut()
        {
            int offset = 0;
            int start = 0;
            for (int position = 0; position < order.Length; position++)
            {
                ref Node node = ref nodes.ElementAt(order[position]);
                node.Start = start;
                start += node.Size;
                node.Diagonal = offset;
                offset += node.Size * node.Size;

                Neighbour* column = neighbours.Ptr + node.FirstNeighbour;
                for (int i = 1; i < node.NeighbourCount; i++)
                {
                    Neighbour moving = column[i];
                    int key = nodes[moving.Node].Position;
                    int j = i - 1;
                    while (j >= 0 && nodes[column[j].Node].Position > key)
                    {
                        column[j + 1] = column[j];
                        j--;
                    }

                    column[j + 1] = moving;
                }

                for (int i = 0; i < node.NeighbourCount; i++)
                {
                    column[i].Offset = offset;
                    offset += nodes[column[i].Node].Size * node.Size;
                }
            }

            factor.Resize(offset);
            right.Resize(start);
            solution.Resize(start);
            correction.Resize(start);
            inverseDiagonal.Resize(start);
        }

        /// <summary>
        /// Splits the elimination tree into subtrees that factor independently. A node's neighbours are all its ancestors in
        /// the tree, whose parent link is a node's first neighbour, so a node only ever updates its own subtree or the nodes
        /// above every subtree. Those top nodes join the subtrees, which on a craft is mostly its core stack. Each subtree is
        /// at most a share of the whole work, so that lanes taking them as they come stay balanced. The split depends on the
        /// structure alone, which keeps the answer the same however many lanes share it.
        /// </summary>
        void Partition()
        {
            int count = order.Length;
            var subtree = new NativeArray<double>(count, Allocator.Temp);
            var parent = new NativeArray<int>(count, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            double total = 0.0;
            for (int position = 0; position < count; position++)
            {
                Node node = nodes[order[position]];
                parent[position] = node.NeighbourCount > 0 ? nodes[neighbours[node.FirstNeighbour].Node].Position : -1;

                // Roughly the multiplications its elimination takes: its own block, one solve per neighbour, and one update
                // per pair of them.
                double size = node.Size;
                double work = size * size * size;
                double below = 0.0;
                for (int i = node.NeighbourCount - 1; i >= 0; i--)
                {
                    double neighbourSize = nodes[neighbours[node.FirstNeighbour + i].Node].Size;
                    below += neighbourSize;
                    work += neighbourSize * size * (size + below);
                }

                subtree[position] += work;
                total += work;
                if (parent[position] >= 0)
                {
                    subtree[parent[position]] += subtree[position];
                }
            }

            // Parents come after their children, so walking back reaches each node after its parent.
            double largestTask = total / TasksPerFactorization;
            int tasks = 0;
            for (int position = count - 1; position >= 0; position--)
            {
                ref Node node = ref nodes.ElementAt(order[position]);
                int above = parent[position] < 0 ? Top : nodes[order[parent[position]]].Task;
                node.Task = above != Top ? above : subtree[position] <= largestTask ? tasks++ : Top;
            }

            taskStarts.Clear();
            taskStarts.Resize(tasks + 1, NativeArrayOptions.ClearMemory);
            topPositions.Clear();
            boundaryPositions.Clear();
            for (int position = 0; position < count; position++)
            {
                ref Node node = ref nodes.ElementAt(order[position]);
                node.OwnNeighbours = node.NeighbourCount;
                if (node.Task == Top)
                {
                    topPositions.Add(position);
                    continue;
                }

                taskStarts[node.Task + 1]++;
                for (int i = 0; i < node.NeighbourCount; i++)
                {
                    if (nodes[neighbours[node.FirstNeighbour + i].Node].Task == Top)
                    {
                        node.OwnNeighbours = i;
                        boundaryPositions.Add(position);
                        break;
                    }
                }
            }

            for (int task = 0; task < tasks; task++)
            {
                taskStarts[task + 1] += taskStarts[task];
            }

            taskPositions.Resize(taskStarts[tasks]);
            var filled = new NativeArray<int>(tasks, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int task = 0; task < tasks; task++)
            {
                filled[task] = taskStarts[task];
            }

            for (int position = 0; position < count; position++)
            {
                int task = nodes[order[position]].Task;
                if (task != Top)
                {
                    taskPositions[filled[task]++] = position;
                }
            }

            var topIndex = new NativeArray<int>(count, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            topUpdateStarts.Clear();
            topUpdateStarts.Resize(topPositions.Length + 1, NativeArrayOptions.ClearMemory);
            for (int i = 0; i < topPositions.Length; i++)
            {
                topIndex[topPositions[i]] = i;
            }

            for (int i = 0; i < boundaryPositions.Length; i++)
            {
                Node node = nodes[order[boundaryPositions[i]]];
                for (int n = node.OwnNeighbours; n < node.NeighbourCount; n++)
                {
                    topUpdateStarts[topIndex[nodes[neighbours[node.FirstNeighbour + n].Node].Position] + 1]++;
                }
            }

            for (int i = 0; i < topPositions.Length; i++)
            {
                topUpdateStarts[i + 1] += topUpdateStarts[i];
            }

            topUpdates.Resize(topUpdateStarts[topPositions.Length]);
            var placed = new NativeArray<int>(topPositions.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int i = 0; i < topPositions.Length; i++)
            {
                placed[i] = topUpdateStarts[i];
            }

            for (int i = 0; i < boundaryPositions.Length; i++)
            {
                Node node = nodes[order[boundaryPositions[i]]];
                for (int n = node.OwnNeighbours; n < node.NeighbourCount; n++)
                {
                    int target = topIndex[nodes[neighbours[node.FirstNeighbour + n].Node].Position];
                    topUpdates[placed[target]++] = new int2(boundaryPositions[i], n);
                }
            }

            placed.Dispose();
            topIndex.Dispose();

            filled.Dispose();
            parent.Dispose();
            subtree.Dispose();
        }

        /// <summary>Where the block between two nodes is, which only the earlier of them holds.</summary>
        double* BlockOf(int earlier, int later)
        {
            Node column = nodes[earlier];
            int laterPosition = nodes[later].Position;
            Neighbour* list = neighbours.Ptr + column.FirstNeighbour;
            int low = 0;
            int high = column.NeighbourCount - 1;
            while (low < high)
            {
                int middle = (low + high) / 2;
                if (nodes[list[middle].Node].Position < laterPosition)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }

            return factor.Ptr + list[low].Offset;
        }

        // Values

        /// <summary>
        /// Fills in the matrix for the current bounds. A row at a bound becomes -z = bound, cut off from the bodies, whose side
        /// of the system takes its fixed impulse instead.
        /// </summary>
        void AssembleMatrix(in Problem problem)
        {
            using ProfilerMarker.AutoScope span = AssembleMarker.Auto();

            stageProblem = problem;
            RunTasks(ClearTask, (factor.Length + ScalarsPerClearTask - 1) / ScalarsPerClearTask, null);

            double* values = factor.Ptr;
            for (int body = 0; body < problem.BodyCount; body++)
            {
                Node node = nodes[body];
                double* diagonal = values + node.Diagonal;
                Body source = problem.Bodies[body];
                for (int i = 0; i < 3; i++)
                {
                    diagonal[i * BodySize + i] = source.Mass;
                    for (int j = 0; j < 3; j++)
                    {
                        diagonal[(i + 3) * BodySize + j + 3] = source.Inertia[j][i];
                    }
                }
            }

            // Each row writes only its own entries.
            RunTasks(AssembleTask, (problem.BlockCount + BlocksPerTask - 1) / BlocksPerTask, null);
        }

        void AssembleBlocks(in Problem problem, int firstBlock, int endBlock)
        {
            double* values = factor.Ptr;
            for (int b = firstBlock; b < endBlock; b++)
            {
                ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    int constraint = rowNode[row];
                    if (constraint == NoNode)
                    {
                        continue;
                    }

                    int index = rowInNode[row];
                    Node node = nodes[constraint];
                    float* jacobian = problem.Jacobian + row * ColumnsPerRow;
                    if (state[row] == Free)
                    {
                        values[node.Diagonal + index * node.Size + index] = -problem.Epsilon[row];
                        WriteJacobian(constraint, index, block.BodyA, jacobian);
                        WriteJacobian(constraint, index, block.BodyB, jacobian + SideColumns);
                    }
                    else
                    {
                        values[node.Diagonal + index * node.Size + index] = -1.0;
                    }
                }
            }
        }

        void AssembleRightHandSide(in Problem problem)
        {
            using ProfilerMarker.AutoScope span = RightHandSideMarker.Auto();

            double* target = right.Ptr;
            UnsafeUtility.MemClear(target, right.Length * sizeof(double));

            for (int b = 0; b < problem.BlockCount; b++)
            {
                ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    int constraint = rowNode[row];
                    byte rowState = state[row];
                    if (constraint != NoNode && rowState == Free)
                    {
                        target[nodes[constraint].Start + rowInNode[row]] = problem.RightHandSide[row];
                        continue;
                    }

                    double bound = rowState == UpperTight ? upper[row] : lower[row];
                    float* jacobian = problem.Jacobian + row * ColumnsPerRow;
                    AddToBody(block.BodyA, jacobian, bound, target);
                    AddToBody(block.BodyB, jacobian + SideColumns, bound, target);
                    if (constraint != NoNode)
                    {
                        target[nodes[constraint].Start + rowInNode[row]] = bound;
                    }
                }
            }
        }

        /// <summary>
        /// Solves the assembled right-hand side with the factorization of an earlier problem, refining until the answer holds
        /// for this one. Returns false if it won't, and the factorization has grown too old.
        /// </summary>
        bool SolveWithOldFactor(in Problem problem)
        {
            double* x = solution.Ptr;
            double* step = correction.Ptr;
            UnsafeUtility.MemCpy(x, right.Ptr, solution.Length * sizeof(double));
            Substitute(x);

            for (int refinement = 0; ; refinement++)
            {
                statistics.Residual = ComputeResidual(problem);
                if (statistics.Residual < RefinementTolerance)
                {
                    return true;
                }

                if (refinement == MaxRefinements)
                {
                    return false;
                }

                statistics.Refinements++;
                Substitute(step);
                for (int i = 0; i < solution.Length; i++)
                {
                    x[i] += step[i];
                }
            }
        }

        /// <summary>
        /// Leaves in <see cref="correction"/> what the current solution misses of the current problem, and returns the largest
        /// miss as a speed: a body's miss is an impulse, which its mass turns into one, and a constraint's already is.
        /// </summary>
        double ComputeResidual(in Problem problem)
        {
            using ProfilerMarker.AutoScope span = ResidualMarker.Auto();

            UnsafeUtility.MemCpy(correction.Ptr, right.Ptr, correction.Length * sizeof(double));

            // A body gathers its rows rather than each row adding to its bodies, so that both split without sharing a value.
            stageProblem = problem;
            int rowTasks = (problem.BlockCount + BlocksPerTask - 1) / BlocksPerTask;
            int bodyTasks = (problem.BodyCount + BodiesPerTask - 1) / BodiesPerTask;
            stageWorst.Resize(rowTasks + bodyTasks);
            RunTasks(RowResidualTask, rowTasks, null);
            RunTasks(BodyResidualTask, bodyTasks, null);

            double worst = 0.0;
            for (int i = 0; i < stageWorst.Length; i++)
            {
                worst = math.max(worst, stageWorst[i]);
            }

            return worst;
        }

        /// <summary>What the rows of some blocks miss, and the largest of it as a speed.</summary>
        double RowResidual(in Problem problem, int firstBlock, int endBlock)
        {
            double* x = solution.Ptr;
            double* residual = correction.Ptr;
            double worst = 0.0;
            for (int b = firstBlock; b < endBlock; b++)
            {
                ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    int constraint = rowNode[row];
                    if (constraint == NoNode)
                    {
                        continue;
                    }

                    int own = nodes[constraint].Start + rowInNode[row];
                    if (state[row] != Free)
                    {
                        residual[own] += x[own];
                        continue;
                    }

                    float* jacobian = problem.Jacobian + row * ColumnsPerRow;
                    residual[own] += problem.Epsilon[row] * x[own]
                        - BodySpeed(block.BodyA, jacobian) - BodySpeed(block.BodyB, jacobian + SideColumns);
                    worst = math.max(worst, math.abs(residual[own]));
                }
            }

            return worst;
        }

        /// <summary>What some bodies miss, from their own momentum and their free rows' impulses, as a speed.</summary>
        double BodyResidual(in Problem problem, int firstBody, int endBody)
        {
            double* x = solution.Ptr;
            double* residual = correction.Ptr;
            double worst = 0.0;
            for (int body = firstBody; body < endBody; body++)
            {
                int start = nodes[body].Start;
                double* speed = x + start;
                double* miss = residual + start;
                for (int i = bodyRowStarts[body]; i < bodyRowStarts[body + 1]; i++)
                {
                    int row = bodyRows[i] >> 1;
                    if (rowNode[row] != NoNode && state[row] == Free)
                    {
                        float* columns = problem.Jacobian + row * ColumnsPerRow + (bodyRows[i] & 1) * SideColumns;
                        double impulse = x[nodes[rowNode[row]].Start + rowInNode[row]];
                        for (int column = 0; column < SideColumns; column++)
                        {
                            miss[column] -= columns[column] * impulse;
                        }
                    }
                }

                Body source = problem.Bodies[body];
                double3 momentum = math.mul(source.Inertia, new double3(speed[3], speed[4], speed[5]));
                double2 scale = bodySpeedScale[body];
                for (int i = 0; i < 3; i++)
                {
                    miss[i] -= source.Mass * speed[i];
                    miss[i + 3] -= momentum[i];
                    worst = math.max(worst, math.abs(miss[i]) * scale.x);
                    worst = math.max(worst, math.abs(miss[i + 3]) * scale.y);
                }
            }

            return worst;
        }

        void WriteJacobian(int constraint, int index, int body, float* columns)
        {
            if (body < 0)
            {
                return;
            }

            Node node = nodes[constraint];
            if (nodes[body].Position < node.Position)
            {
                double* block = BlockOf(body, constraint) + index * BodySize;
                for (int column = 0; column < SideColumns; column++)
                {
                    block[column] = columns[column];
                }
            }
            else
            {
                double* block = BlockOf(constraint, body);
                for (int column = 0; column < SideColumns; column++)
                {
                    block[column * node.Size + index] = columns[column];
                }
            }
        }

        void AddToBody(int body, float* columns, double impulse, double* target)
        {
            if (body < 0)
            {
                return;
            }

            double* own = target + nodes[body].Start;
            for (int column = 0; column < SideColumns; column++)
            {
                own[column] += columns[column] * impulse;
            }
        }

        /// <summary>
        /// Block Cholesky on the signed matrix: A = L S L^t, with S the sign of each node, which elimination preserves because
        /// the matrix is quasi-definite. Each column's blocks are kept as A C^-t, one sign away from L, for the updates to use.
        /// </summary>
        void Factorize()
        {
            using ProfilerMarker.AutoScope span = FactorizeMarker.Auto();

            RunTasks(FactorizeTask, taskStarts.Length - 1, null);

            // The subtrees' updates of the top land in blocks the top holds, which two subtrees can share, so they go by the
            // top node that holds them.
            RunTasks(BoundaryUpdateTask, topPositions.Length, null);

            // The top nodes run in order, but each one's updates of its neighbours land in blocks each neighbour holds.
            for (int i = 0; i < topPositions.Length; i++)
            {
                Node node = nodes[order[topPositions[i]]];
                Eliminate(node);
                if (node.NeighbourCount < SharedUpdateNeighbours)
                {
                    UpdateLaterNodes(node, 0, node.NeighbourCount);
                    continue;
                }

                stageNode = order[topPositions[i]];
                RunTasks(TopUpdateTask, node.NeighbourCount, null);
            }
        }

        /// <summary>Runs a stage's tasks on the team if there is one.</summary>
        void RunTasks(int kind, int tasks, double* x)
        {
            SparseSolveTeam.RunAll(Team, UnsafeUtility.AddressOf(ref this), kind, tasks, x);
        }

        /// <summary>One task of a stage, which touches nothing another task of it does.</summary>
        internal void RunTask(int kind, int task, double* x)
        {
            if (kind == BoundaryUpdateTask)
            {
                for (int i = topUpdateStarts[task]; i < topUpdateStarts[task + 1]; i++)
                {
                    int2 update = topUpdates[i];
                    UpdateLaterNodes(nodes[order[update.x]], update.y, update.y + 1);
                }

                return;
            }

            if (kind == TopUpdateTask)
            {
                UpdateLaterNodes(nodes[stageNode], task, task + 1);
                return;
            }

            if (kind == ClearTask)
            {
                int start = task * ScalarsPerClearTask;
                int length = math.min(ScalarsPerClearTask, factor.Length - start);
                UnsafeUtility.MemClear(factor.Ptr + start, length * sizeof(double));
                return;
            }

            if (kind == AssembleTask)
            {
                AssembleBlocks(stageProblem, task * BlocksPerTask, math.min((task + 1) * BlocksPerTask, stageProblem.BlockCount));
                return;
            }

            if (kind == RowResidualTask)
            {
                int endBlock = math.min((task + 1) * BlocksPerTask, stageProblem.BlockCount);
                stageWorst[task] = RowResidual(stageProblem, task * BlocksPerTask, endBlock);
                return;
            }

            if (kind == BodyResidualTask)
            {
                int rowTasks = (stageProblem.BlockCount + BlocksPerTask - 1) / BlocksPerTask;
                int endBody = math.min((task + 1) * BodiesPerTask, stageProblem.BodyCount);
                stageWorst[rowTasks + task] = BodyResidual(stageProblem, task * BodiesPerTask, endBody);
                return;
            }

            if (kind == BoundaryPushTask)
            {
                PushToTop(task, x);
                return;
            }

            int first = taskStarts[task];
            int end = taskStarts[task + 1];
            if (kind == FactorizeTask)
            {
                for (int i = first; i < end; i++)
                {
                    Node node = nodes[order[taskPositions[i]]];
                    Eliminate(node);
                    UpdateLaterNodes(node, 0, node.OwnNeighbours);
                }
            }
            else if (kind == ForwardTask)
            {
                for (int i = first; i < end; i++)
                {
                    Node node = nodes[order[taskPositions[i]]];
                    SubstituteForward(node, x, node.OwnNeighbours);
                }
            }
            else
            {
                for (int i = end - 1; i >= first; i--)
                {
                    SubstituteBackward(nodes[order[taskPositions[i]]], x);
                }
            }
        }

        /// <summary>Factors a node's diagonal block, which every update from earlier nodes has reached, and its column.</summary>
        void Eliminate(in Node node)
        {
            double* values = factor.Ptr;
            int size = node.Size;
            double* diagonal = values + node.Diagonal;
            CholeskyInPlace(diagonal, size, node.Sign);
            for (int row = 0; row < size; row++)
            {
                inverseDiagonal[node.Start + row] = 1.0 / diagonal[row * size + row];
            }

            Neighbour* column = neighbours.Ptr + node.FirstNeighbour;
            for (int i = 0; i < node.NeighbourCount; i++)
            {
                SolveAgainstTranspose(values + column[i].Offset, nodes[column[i].Node].Size, diagonal, size);
            }
        }

        /// <summary>
        /// Subtracts an eliminated node from the blocks its neighbours from <paramref name="firstNeighbour"/> on hold: each
        /// one's diagonal, and its blocks with every later neighbour.
        /// </summary>
        void UpdateLaterNodes(in Node node, int firstNeighbour, int endNeighbour)
        {
            double* values = factor.Ptr;
            Neighbour* allNeighbours = neighbours.Ptr;
            Neighbour* column = allNeighbours + node.FirstNeighbour;
            int size = node.Size;
            for (int i = firstNeighbour; i < endNeighbour; i++)
            {
                Node first = nodes[column[i].Node];
                double* firstBlock = values + column[i].Offset;
                Update(values + first.Diagonal, first.Size, first.Size, firstBlock, firstBlock, size, node.Sign, true);

                // Elimination gave the first neighbour every later one, in the same order, so one walk finds them all.
                Neighbour* targets = allNeighbours + first.FirstNeighbour;
                int target = 0;
                for (int j = i + 1; j < node.NeighbourCount; j++)
                {
                    while (targets[target].Node != column[j].Node)
                    {
                        target++;
                    }

                    int laterSize = nodes[column[j].Node].Size;
                    Update(values + targets[target].Offset, laterSize, first.Size, values + column[j].Offset, firstBlock, size, node.Sign, false);
                }
            }
        }

        static void CholeskyInPlace(double* a, int size, double sign)
        {
            for (int row = 0; row < size; row++)
            {
                for (int column = 0; column <= row; column++)
                {
                    double sum = sign * a[row * size + column];
                    for (int k = 0; k < column; k++)
                    {
                        sum -= a[row * size + k] * a[column * size + k];
                    }

                    if (row == column)
                    {
                        a[row * size + row] = math.sqrt(math.max(sum, SmallestPivot));
                    }
                    else
                    {
                        a[row * size + column] = sum / a[column * size + column];
                    }
                }
            }
        }

        /// <summary>Replaces each row of the block with itself times C^-t.</summary>
        static void SolveAgainstTranspose(double* block, int rows, double* c, int size)
        {
            for (int row = 0; row < rows; row++)
            {
                double* x = block + row * size;
                for (int column = 0; column < size; column++)
                {
                    double sum = x[column];
                    for (int k = 0; k < column; k++)
                    {
                        sum -= x[k] * c[column * size + k];
                    }

                    x[column] = sum / c[column * size + column];
                }
            }
        }

        /// <summary>target -= sign * left * right^t. The diagonal block's factorization only reads its lower half.</summary>
        static void Update(double* target, int rows, int columns, double* left, double* right, int inner, double sign, bool lowerOnly)
        {
            for (int row = 0; row < rows; row++)
            {
                double* l = left + row * inner;
                int last = lowerOnly ? row + 1 : columns;
                for (int column = 0; column < last; column++)
                {
                    double* r = right + column * inner;
                    double sum = 0.0;
                    for (int k = 0; k < inner; k++)
                    {
                        sum += l[k] * r[k];
                    }

                    target[row * columns + column] -= sign * sum;
                }
            }
        }

        /// <summary>Solves in place: forward through L, across S, and back through L^t.</summary>
        void Substitute(double* x)
        {
            using ProfilerMarker.AutoScope span = SubstituteMarker.Auto();

            RunTasks(ForwardTask, taskStarts.Length - 1, x);

            // Two subtrees can owe the same top node, so what they owe goes by the node it is owed to.
            RunTasks(BoundaryPushTask, topPositions.Length, x);

            for (int i = 0; i < topPositions.Length; i++)
            {
                Node node = nodes[order[topPositions[i]]];
                SubstituteForward(node, x, node.NeighbourCount);
            }

            for (int i = topPositions.Length - 1; i >= 0; i--)
            {
                SubstituteBackward(nodes[order[topPositions[i]]], x);
            }

            RunTasks(BackwardTask, taskStarts.Length - 1, x);
        }

        /// <summary>
        /// Passes one top node what the subtrees owe it, from their boundary nodes' answers, which already carry each node's
        /// sign.
        /// </summary>
        void PushToTop(int top, double* x)
        {
            Node target = nodes[order[topPositions[top]]];
            double* into = x + target.Start;
            for (int i = topUpdateStarts[top]; i < topUpdateStarts[top + 1]; i++)
            {
                int2 update = topUpdates[i];
                Node node = nodes[order[update.x]];
                double* own = x + node.Start;
                double* block = factor.Ptr + neighbours[node.FirstNeighbour + update.y].Offset;
                for (int row = 0; row < target.Size; row++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < node.Size; k++)
                    {
                        sum += block[row * node.Size + k] * own[k];
                    }

                    into[row] -= sum;
                }
            }
        }

        /// <summary>
        /// Solves for a node through L and S, once every earlier node has passed it its share, and passes its own share to its
        /// neighbours before <paramref name="endNeighbour"/>.
        /// </summary>
        void SubstituteForward(in Node node, double* x, int endNeighbour)
        {
            double* values = factor.Ptr;
            int size = node.Size;
            double* c = values + node.Diagonal;
            double* own = x + node.Start;
            double* inverse = inverseDiagonal.Ptr + node.Start;
            for (int row = 0; row < size; row++)
            {
                double sum = own[row];
                for (int k = 0; k < row; k++)
                {
                    sum -= c[row * size + k] * own[k];
                }

                own[row] = sum * inverse[row];
            }

            Neighbour* column = neighbours.Ptr + node.FirstNeighbour;
            for (int i = 0; i < endNeighbour; i++)
            {
                Node other = nodes[column[i].Node];
                double* block = values + column[i].Offset;
                double* target = x + other.Start;
                for (int row = 0; row < other.Size; row++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < size; k++)
                    {
                        sum += block[row * size + k] * own[k];
                    }

                    target[row] -= node.Sign * sum;
                }
            }

            for (int row = 0; row < size; row++)
            {
                own[row] *= node.Sign;
            }
        }

        /// <summary>Solves for a node through L^t, once all its neighbours, which come later, have their answers.</summary>
        void SubstituteBackward(in Node node, double* x)
        {
            double* values = factor.Ptr;
            int size = node.Size;
            double* c = values + node.Diagonal;
            double* own = x + node.Start;

            Neighbour* column = neighbours.Ptr + node.FirstNeighbour;
            for (int i = 0; i < node.NeighbourCount; i++)
            {
                Node other = nodes[column[i].Node];
                double* block = values + column[i].Offset;
                double* source = x + other.Start;
                for (int row = 0; row < other.Size; row++)
                {
                    for (int k = 0; k < size; k++)
                    {
                        own[k] -= node.Sign * block[row * size + k] * source[row];
                    }
                }
            }

            double* inverse = inverseDiagonal.Ptr + node.Start;
            for (int row = size - 1; row >= 0; row--)
            {
                double sum = own[row];
                for (int k = row + 1; k < size; k++)
                {
                    sum -= c[k * size + row] * own[k];
                }

                own[row] = sum * inverse[row];
            }
        }

        void ReadImpulses(in Problem problem, float* impulses)
        {
            double* x = solution.Ptr;
            for (int row = 0; row < problem.RowCount; row++)
            {
                byte rowState = state[row];
                impulses[row] = rowState == Free
                    ? (float)-x[nodes[rowNode[row]].Start + rowInNode[row]]
                    : rowState == LowerTight ? lower[row] : upper[row];
            }
        }

        /// <summary>
        /// Moves rows between free and tight, and returns how many were on the wrong side. A free row is wrong if it passed a
        /// bound, and a tight one if the rest of the system is pulling it back inside: w = J du + E z - b has the wrong sign.
        /// </summary>
        int Pivot(in Problem problem, float* impulses, bool flipAll)
        {
            using ProfilerMarker.AutoScope span = PivotMarker.Auto();

            int infeasible = 0;
            int last = -1;
            byte lastState = Free;
            for (int b = 0; b < problem.BlockCount; b++)
            {
                ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    // A row whose bounds meet has one possible impulse, and it was put tight when they met.
                    if (lower[row] == upper[row])
                    {
                        continue;
                    }

                    byte current = state[row];
                    byte next = current;
                    if (current == Free)
                    {
                        float impulse = impulses[row];
                        if (impulse < lower[row] - Tolerance)
                        {
                            next = LowerTight;
                        }
                        else if (impulse > upper[row] + Tolerance)
                        {
                            next = UpperTight;
                        }
                    }
                    else
                    {
                        double slack = Slack(problem, block, row, impulses[row]);
                        if ((current == LowerTight && slack < -Tolerance) || (current == UpperTight && slack > Tolerance))
                        {
                            next = Free;
                        }
                    }

                    if (next != current)
                    {
                        infeasible++;
                        last = row;
                        lastState = next;
                        if (flipAll)
                        {
                            state[row] = next;
                        }
                    }
                }
            }

            if (!flipAll && last >= 0)
            {
                state[last] = lastState;
            }

            return infeasible;
        }

        double Slack(in Problem problem, in ConstraintBlock block, int row, float impulse)
        {
            float* jacobian = problem.Jacobian + row * ColumnsPerRow;
            return BodySpeed(block.BodyA, jacobian) + BodySpeed(block.BodyB, jacobian + SideColumns)
                + problem.Epsilon[row] * (double)impulse - problem.RightHandSide[row];
        }

        double BodySpeed(int body, float* columns)
        {
            if (body < 0)
            {
                return 0.0;
            }

            double* du = solution.Ptr + nodes[body].Start;
            double sum = 0.0;
            for (int column = 0; column < SideColumns; column++)
            {
                sum += columns[column] * du[column];
            }

            return sum;
        }

        /// <summary>A binary heap of priorities, smallest first.</summary>
        struct MinHeap : IDisposable
        {
            NativeList<long> items;

            public MinHeap(int capacity, Allocator allocator)
            {
                items = new NativeList<long>(capacity, allocator);
            }

            public void Dispose()
            {
                items.Dispose();
            }

            public void Push(long item)
            {
                items.Add(item);
                int child = items.Length - 1;
                while (child > 0)
                {
                    int parent = (child - 1) / 2;
                    if (items[parent] <= items[child])
                    {
                        break;
                    }

                    (items[parent], items[child]) = (items[child], items[parent]);
                    child = parent;
                }
            }

            public long Pop()
            {
                long top = items[0];
                items[0] = items[items.Length - 1];
                items.Length--;
                int parent = 0;
                while (true)
                {
                    int left = parent * 2 + 1;
                    int right = left + 1;
                    int smallest = parent;
                    if (left < items.Length && items[left] < items[smallest])
                    {
                        smallest = left;
                    }

                    if (right < items.Length && items[right] < items[smallest])
                    {
                        smallest = right;
                    }

                    if (smallest == parent)
                    {
                        return top;
                    }

                    (items[parent], items[smallest]) = (items[smallest], items[parent]);
                    parent = smallest;
                }
            }
        }
    }
}
