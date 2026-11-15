using System;
using System.Runtime.CompilerServices;
using System.Threading;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Numerics.Retooled;
using Unity.Profiling;
using Unity.Profiling.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Solves the same constraint problem as the direct solver's dense path, without forming J M^-1 J^t. That matrix
    /// has dense blocks wherever constraints share a body. A part with thirty attachments therefore requires thirty
    /// squared blocks, and the cost of solving a craft grows with the cube of its row count. This solver keeps the
    /// bodies as unknowns instead:
    ///
    ///     | M   J^t | | du |   | 0 |
    ///     | J   -E  | | -z | = | b |
    ///
    /// Eliminating du gives (J M^-1 J^t + E) z = b, so the answer is the same. The graph represents the craft's bodies
    /// and joints. Eliminating a tree from its leaves inward requires no additional nonzero blocks, called fill. Each
    /// strut adds a little fill. With M and E positive, the matrix is quasi-definite. Any symmetric ordering therefore
    /// permits stable factorization without pivoting, and the structure can be determined before numeric values are
    /// available.
    ///
    /// The solver handles bounds on z by block principal pivoting, as <see cref="LCP.SolveMLCP"/> does. Putting a row
    /// at a bound changes only numeric values in this matrix. Each pivoting step therefore requires one numeric
    /// refactorization with the same structure.
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

        /// <summary>
        /// The relative tolerance for changes to coupled bounds. Multiply by the widening from the source impulse, then
        /// use the larger of this value and the absolute tolerance to decide whether to solve again.
        /// </summary>
        const float CouplingTolerance = 1e-3f;
        const double SmallestPivot = 1e-30;
        const int InitialCapacity = 16;

        /// <summary>The allowed residual after refinement, measured in m/s or rad/s at any body or free row.</summary>
        const double RefinementTolerance = 1e-5;
        const int MaxRefinements = 3;

        /// <summary>A body's mass matrix, with its inertia in the same frame as the Jacobian's angular columns.</summary>
        public struct Body
        {
            public double Mass;
            public double3x3 Inertia;
        }

        /// <summary>
        /// Represents consecutive rows acting between two bodies. A body index of -1 represents a body that does not
        /// move.
        /// </summary>
        public struct ConstraintBlock
        {
            public int BodyA;
            public int BodyB;
            public int FirstRow;
            public int RowCount;
        }

        /// <summary>
        /// Describes the problem to solve. Its data must remain valid throughout the call and is only read by the
        /// solver.
        /// </summary>
        public struct Problem
        {
            public int BodyCount;
            public Body* Bodies;

            public int BlockCount;
            public ConstraintBlock* Blocks;

            public int RowCount;

            /// <summary>Contains <see cref="ColumnsPerRow"/> values for each row.</summary>
            public float* Jacobian;

            /// <summary>Specifies the positive regularization value on each row's diagonal.</summary>
            public float* Epsilon;

            public float* RightHandSide;
            public float* Lower;
            public float* Upper;

            /// <summary>
            /// An initial guess for which rows are at a bound, using values from <see cref="LCP.MLCPIndexFlag"/>. Use
            /// null for no guess. A good guess reduces the number of pivoting steps without changing the answer.
            /// </summary>
            public byte* InitialStates;

            public int CouplingCount;

            /// <summary>
            /// Specifies rows whose bounds depend on other rows' impulses, or null if there are none.
            /// </summary>
            public Coupling* Couplings;
        }

        /// <summary>
        /// Describes rows whose problem bounds are widened by <see cref="Factor"/> times the nonnegative total source
        /// impulse. For contact friction, the source is the normal impulse. The total includes the impulses computed
        /// for the source rows plus <see cref="HeldSourceImpulse"/>. Both normal and friction impulses count over the
        /// whole step. A friction row's problem bounds are therefore the negative of its earlier impulse in that step.
        /// If the total normal impulse is zero, the total friction impulse for the step is also zero.
        /// </summary>
        public struct Coupling
        {
            public int FirstRow;
            public int RowCount;
            public int FirstSourceRow;
            public int SourceRowCount;
            public float Factor;

            /// <summary>
            /// The source impulse already applied before this solve, either through warm starting or in an earlier
            /// substep of the same step. This impulse contributes to the coupled bounds along with the impulse computed
            /// by this solve.
            /// </summary>
            public float HeldSourceImpulse;
        }

        /// <summary>
        /// Records the size of the problem, independently of the machine, and the time spent in each stage. Times use
        /// the units of <see cref="ProfilerUnsafeUtility.Timestamp"/>.
        /// </summary>
        public struct Statistics
        {
            public int Nodes;

            /// <summary>
            /// Counts the independent subtrees used for factorization and the nodes above them that are processed
            /// separately.
            /// </summary>
            public int Tasks;
            public int TopNodes;

            public int FactorBlocks;
            public long FactorScalars;
            public int PivotingIterations;
            public int Factorizations;
            public int Refinements;
            public int CouplingIterations;

            /// <summary>
            /// The final residual measured while solving with a reused factorization, in m/s or rad/s.
            /// </summary>
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

        /// <summary>
        /// The target number of subtrees for factorization, chosen to provide enough work for every lane.
        /// </summary>
        const int TasksPerFactorization = 64;

        // The operation performed by each task in a team stage.
        const int FactorizeTask = 0;
        const int SubstituteTask = 1;
        const int TopFactorTask = 2;
        const int ClearTask = 3;
        const int AssembleTask = 4;
        const int RowResidualTask = 5;
        const int BodyResidualTask = 6;

        /// <summary>
        /// Specifies how many factor entries a task clears, or how many blocks or bodies it assembles or checks.
        /// </summary>
        const int ScalarsPerClearTask = 16384;
        const int BlocksPerTask = 64;
        const int BodiesPerTask = 64;

        /// <summary>Represents a graph node: either a body or all the rows between one pair of bodies.</summary>
        struct Node
        {
            public int Size;
            public int Position;
            public double Sign;
            public int BodyA;
            public int BodyB;

            // The starting index of this node's unknowns in the solution.
            public int Start;

            // The node's diagonal block in the factor and its neighbours that occur later in elimination order.
            public int Diagonal;
            public int FirstNeighbour;
            public int NeighbourCount;

            // The subtree in which this node is eliminated, or Top. OwnNeighbours counts the first neighbours in the
            // list that belong to that same subtree.
            public int Task;
            public int OwnNeighbours;
        }

        struct Neighbour
        {
            public int Node;

            // The start of the block in the factor. Its rows belong to this neighbour, and its columns belong to the
            // column node.
            public int Offset;
        }

        UnsafeList<Node> nodes;
        UnsafeList<int> order;
        UnsafeList<Neighbour> neighbours;

        // Subtrees of the elimination tree can be factored and solved independently. Each subtree's ordered positions
        // start at its entry in taskStarts. The top connects the subtrees and is processed node by node. The boundary
        // consists of subtree nodes with a neighbour in the top. Updates from those nodes to the top are applied after
        // the subtrees finish.
        UnsafeList<int> taskStarts;
        UnsafeList<int> taskPositions;
        UnsafeList<int> topPositions;
        UnsafeList<int> boundaryPositions;

        // Updates from boundary nodes to the top, grouped by the top node whose blocks are updated. Each entry
        // identifies a boundary node's position and the index of its neighbour. Each top node's list starts at its
        // entry in topUpdateStarts.
        UnsafeList<int> topUpdateStarts;
        UnsafeList<int2> topUpdates;

        // Blocks within the top, grouped by the top node to which their rows belong. Each entry identifies the node
        // that owns the block's column and the index of its neighbour. Each top node's list starts at its entry in
        // topRowStarts and follows elimination order. Updates are therefore gathered in the same order as in a serial
        // pass.
        UnsafeList<int> topRowStarts;
        UnsafeList<int2> topRows;

        // The last pass in which each node finished. Lanes processing nodes in order wait only for the nodes whose
        // values they read. Each pass uses a new stamp, so these entries do not need to be cleared.
        UnsafeList<int> nodeReady;
        int readyStamp;

        // Each body's rows, starting at its entry in bodyRowStarts. A row is encoded as row * 2, plus 1 for side B.
        UnsafeList<int> bodyRowStarts;
        UnsafeList<int> bodyRows;

        // The problem read during a stage and the largest residual measured by each task.
        Problem stageProblem;
        UnsafeList<double> stageWorst;

        UnsafeList<int> rowNode;
        UnsafeList<int> rowInNode;

        // The index of each row's unknown in the solution, or NoNode if the row has no unknown.
        UnsafeList<int> rowUnknown;
        UnsafeList<double> factor;
        UnsafeList<double> right;
        UnsafeList<double> solution;
        UnsafeList<double> correction;

        // The reciprocal of each diagonal entry of the factor, indexed by unknown. Substitution uses multiplication in
        // place of division.
        UnsafeList<double> inverseDiagonal;

        // The reciprocal of each body's mass and of its smallest diagonal inertia entry, used to express its residual
        // as speeds.
        UnsafeList<double2> bodySpeedScale;

        UnsafeList<byte> state;

        // Bounds for each row. Uncoupled rows use the problem bounds; coupled rows use the last bounds computed from
        // their sources. Bounds for coupled rows are retained between solves of the same shape as an initial guess for
        // the next solve.
        UnsafeList<float> lower;
        UnsafeList<float> upper;
        UnsafeList<bool> coupled;

        // The problem shape for which the structure above was analyzed.
        UnsafeList<ConstraintBlock> analyzedBlocks;
        int analyzedBodies;
        bool analyzedBounded;

        // Whether factor contains a factorization of a recent problem with this shape and the current row states.
        bool factored;

        // Whether solution contains an answer to a recent problem with this shape, for use as an initial guess in the
        // next solve.
        bool answered;

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
            topRowStarts = new UnsafeList<int>(InitialCapacity, allocator);
            topRows = new UnsafeList<int2>(InitialCapacity, allocator);
            nodeReady = new UnsafeList<int>(InitialCapacity, allocator);
            readyStamp = 0;
            bodyRowStarts = new UnsafeList<int>(InitialCapacity, allocator);
            bodyRows = new UnsafeList<int>(InitialCapacity, allocator);
            stageWorst = new UnsafeList<double>(InitialCapacity, allocator);
            Team = null;
            rowNode = new UnsafeList<int>(InitialCapacity, allocator);
            rowInNode = new UnsafeList<int>(InitialCapacity, allocator);
            rowUnknown = new UnsafeList<int>(InitialCapacity, allocator);
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
            answered = false;
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
            topRowStarts.Dispose();
            topRows.Dispose();
            nodeReady.Dispose();
            bodyRowStarts.Dispose();
            bodyRows.Dispose();
            stageWorst.Dispose();
            rowNode.Dispose();
            rowInNode.Dispose();
            rowUnknown.Dispose();
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
        /// The team of lanes used for factorization and substitution. Use null to run only on the caller's thread. The
        /// answer is the same with or without a team.
        /// </summary>
        public SparseSolveTeam* Team;

        public Statistics Stats => statistics;

        /// <summary>
        /// Computes a body's share of J^t z from the impulses of its constraint rows, in the frame of the Jacobian's
        /// columns. Each call reads only that body's rows, so calls for different bodies can run on different threads.
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

        /// <summary>
        /// Returns the bounds used for a row in the last solve, including the final coupled bounds for a coupled row.
        /// </summary>
        public float2 Bounds(int row) => new float2(lower[row], upper[row]);

        /// <summary>
        /// Writes each row's impulse to <paramref name="impulses"/>. With <paramref name="bounded"/> false the bounds are
        /// ignored. Returns false if pivoting ran out of iterations, leaving the last impulses clamped to their bounds.
        ///
        /// Keeping a solver between calls allows it to reuse work. Bodies move little between solves, so the last
        /// factorization is usually close to the next problem of the same shape. Iterative refinement corrects the
        /// difference at a fraction of the cost of a new factorization. The solver checks the answer against the
        /// current problem. If the old factorization is no longer useful, trying it costs some time but does not reduce
        /// accuracy.
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
                answered = false;
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

            // Whether correction contains the residual of the last answer for the current matrix. A successful
            // refinement leaves this residual available until a row changes state.
            bool residualHeld = false;
            for (int iteration = 1; iteration <= MaxPivotingIterations; iteration++)
            {
                statistics.PivotingIterations = iteration;
                started = ProfilerUnsafeUtility.Timestamp;

                // If the matrix is unchanged between rounds, the new residual is the old residual plus the change in
                // the right-hand side.
                bool keepResidual = residualHeld && factored;
                if (keepResidual)
                {
                    AddScaled(correction.Ptr, right.Ptr, -1.0, correction.Length);
                }

                AssembleRightHandSide(problem);
                if (keepResidual)
                {
                    AddScaled(correction.Ptr, right.Ptr, 1.0, correction.Length);
                }

                statistics.AssembleTime += Lap(ref started);

                // The last answer is usually close to the new answer, so refining it often requires fewer substitutions
                // than refining the answer computed with the old factorization. If that fails, try the answer from the
                // old factorization before computing a new one.
                bool solved = factored
                    && (answered && SolveWithOldFactor(problem, true, keepResidual)
                        || SolveWithOldFactor(problem, false, false));
                residualHeld = solved;
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

                answered = true;
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
                    // Each solution updates the normal impulses used to bound friction. If those bounds change enough,
                    // solve again. Most rounds change only the bounds of tight rows, so the factorization can be
                    // reused. At the coupling iteration limit, keep the last bounds, which the current answer
                    // satisfies.
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

                // Changing a row between free and tight changes the matrix.
                factored = false;

                // Changing every infeasible row usually converges in a few steps, but can cycle. If progress stops, use
                // Murty's rule: change only the last infeasible row. That rule guarantees convergence, though this loop
                // still has an iteration limit.
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
        /// Updates <paramref name="impulses"/>, the answer from the last <see cref="Solve"/>, for a small change to the
        /// right-hand side of the free rows. The matrix is unchanged, so the update requires one substitution with the
        /// available factorization. The factorization may come from an earlier problem. Its error affects only this
        /// small correction, so no refinement is performed. Updated free impulses are clamped to their bounds, and
        /// tight rows are unchanged. Returns false without changing anything if no factorization is available.
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
                    // Retained bounds are only an initial guess. Starting a row tight at one of them would apply the
                    // full friction impulse immediately.
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
        /// A row with equal lower and upper problem bounds has one possible impulse. Treat it as a known impulse rather
        /// than an unknown. A coupled row must remain an unknown because its bounds may be equal now and separate
        /// later.
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
                    lower[row] = problem.Lower[row];
                    upper[row] = problem.Upper[row];
                }
            }
        }

        /// <summary>
        /// Updates each coupled row's bounds from the current source impulses and returns whether any bounds changed
        /// beyond tolerance. A tight row follows its bound, and pivoting can free it. If equal bounds separate, the row
        /// is freed; if the widening becomes zero, a free row becomes tight.
        /// </summary>
        bool Couple(in Problem problem, float* impulses)
        {
            bool moved = false;
            for (int i = 0; i < problem.CouplingCount; i++)
            {
                Coupling coupling = problem.Couplings[i];
                float sum = coupling.HeldSourceImpulse;
                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    sum += impulses[row];
                }

                float bound = math.max(sum, 0.0f) * coupling.Factor;
                float tolerance = math.max(Tolerance, CouplingTolerance * bound);
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    float newLower = problem.Lower[row] - bound;
                    float newUpper = problem.Upper[row] + bound;
                    if (math.abs(lower[row] - newLower) <= tolerance && math.abs(upper[row] - newUpper) <= tolerance)
                    {
                        continue;
                    }

                    bool wasFixed = lower[row] == upper[row];
                    lower[row] = newLower;
                    upper[row] = newUpper;
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

        /// <summary>Checks whether the nodes and their ordering from the last analysis match this problem.</summary>
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
            PlaceBlocks();

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
        /// One joint produces several Jacobian rows between the same two bodies. Separate nodes for these rows would
        /// all be neighbours, so grouping the rows in one node reduces the graph without losing structure. When bounds
        /// are enforced, a row with equal uncoupled bounds is omitted from the unknowns because its impulse is known.
        /// Coupled rows remain unknowns because their bounds can separate.
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

                            // The first pass counts rows. The second pass places each row in its body's next free slot.
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

            // Placing the rows advanced each start to the next body's start. Shift the starts back to their original
            // positions.
            for (int body = problem.BodyCount; body > 0; body--)
            {
                bodyRowStarts[body] = bodyRowStarts[body - 1];
            }

            bodyRowStarts[0] = 0;
        }

        /// <summary>
        /// Use minimum degree ordering: eliminating the node with the fewest neighbours connects the fewest pairs of
        /// neighbours. A leaf has one neighbour, so a tree is eliminated from the leaves inward without adding
        /// connections. Prefer a body when degrees are equal, so elimination divides by mass rather than a constraint's
        /// much smaller compliance.
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

                // A new heap entry is added whenever a node's degree changes. Only the newest entry is valid.
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

        /// <summary>
        /// Sorts each column's neighbours by elimination order and assigns solution indices to every node's unknowns.
        /// </summary>
        void LayOut()
        {
            int start = 0;
            for (int position = 0; position < order.Length; position++)
            {
                ref Node node = ref nodes.ElementAt(order[position]);
                node.Start = start;
                start += node.Size;

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
            }

            rowUnknown.Resize(rowNode.Length);
            for (int row = 0; row < rowNode.Length; row++)
            {
                rowUnknown[row] = rowNode[row] == NoNode ? NoNode : nodes[rowNode[row]].Start + rowInNode[row];
            }

            right.Resize(start);
            solution.Resize(start);
            correction.Resize(start);
            inverseDiagonal.Resize(start);
        }

        /// <summary>
        /// Splits the elimination tree into subtrees that can be factored independently. A node's first neighbour is
        /// its parent, and all its neighbours are ancestors in the tree. Updates therefore affect only the same subtree
        /// or the top nodes above the subtrees. On a craft, those top nodes are mostly in the core stack. Each subtree
        /// is limited to a share of the total work, which helps balance lanes as they take tasks. The split depends
        /// only on structure, so the answer does not depend on the number of lanes.
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

                // Estimate the multiplication count for eliminating this node: factor its diagonal block, solve once
                // per neighbour, and update once per pair of neighbours.
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

            // Parents occur after their children in elimination order. Walking backward therefore visits each parent
            // before its children.
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

            // A top node's neighbours occur later in elimination order, so they are also in the top.
            topRowStarts.Clear();
            topRowStarts.Resize(topPositions.Length + 1, NativeArrayOptions.ClearMemory);
            for (int i = 0; i < topPositions.Length; i++)
            {
                Node node = nodes[order[topPositions[i]]];
                for (int n = 0; n < node.NeighbourCount; n++)
                {
                    topRowStarts[topIndex[nodes[neighbours[node.FirstNeighbour + n].Node].Position] + 1]++;
                }
            }

            for (int i = 0; i < topPositions.Length; i++)
            {
                topRowStarts[i + 1] += topRowStarts[i];
                placed[i] = topRowStarts[i];
            }

            topRows.Resize(topRowStarts[topPositions.Length]);
            for (int i = 0; i < topPositions.Length; i++)
            {
                Node node = nodes[order[topPositions[i]]];
                for (int n = 0; n < node.NeighbourCount; n++)
                {
                    Neighbour neighbour = neighbours[node.FirstNeighbour + n];
                    topRows[placed[topIndex[nodes[neighbour.Node].Position]]++] = new int2(order[topPositions[i]], n);
                }
            }

            int oldNodes = nodeReady.Length;
            nodeReady.Resize(nodes.Length);
            for (int i = oldNodes; i < nodes.Length; i++)
            {
                nodeReady[i] = readyStamp;
            }

            placed.Dispose();
            topIndex.Dispose();

            filled.Dispose();
            parent.Dispose();
            subtree.Dispose();
        }

        /// <summary>
        /// Assigns space for every block in the order used by forward substitution. First come each subtree's nodes,
        /// with their diagonal blocks and columns. Each top node's diagonal then follows the blocks whose contributions
        /// it gathers. The factor is too large for L2. Sequential access allows hardware prefetching to reduce the
        /// stalls that scattered reads from L3 would cause. Backward substitution visits these blocks in reverse order.
        /// </summary>
        void PlaceBlocks()
        {
            int offset = 0;
            for (int i = 0; i < taskPositions.Length; i++)
            {
                ref Node node = ref nodes.ElementAt(order[taskPositions[i]]);
                node.Diagonal = offset;
                offset += node.Size * node.Size;

                Neighbour* column = neighbours.Ptr + node.FirstNeighbour;
                for (int n = 0; n < node.OwnNeighbours; n++)
                {
                    column[n].Offset = offset;
                    offset += nodes[column[n].Node].Size * node.Size;
                }
            }

            for (int top = 0; top < topPositions.Length; top++)
            {
                ref Node node = ref nodes.ElementAt(order[topPositions[top]]);
                for (int i = topUpdateStarts[top]; i < topUpdateStarts[top + 1]; i++)
                {
                    int2 update = topUpdates[i];
                    Node source = nodes[order[update.x]];
                    neighbours.ElementAt(source.FirstNeighbour + update.y).Offset = offset;
                    offset += node.Size * source.Size;
                }

                for (int i = topRowStarts[top]; i < topRowStarts[top + 1]; i++)
                {
                    int2 row = topRows[i];
                    Node source = nodes[row.x];
                    neighbours.ElementAt(source.FirstNeighbour + row.y).Offset = offset;
                    offset += node.Size * source.Size;
                }

                node.Diagonal = offset;
                offset += node.Size * node.Size;
            }

            factor.Resize(offset);
        }

        /// <summary>
        /// Returns the block between two nodes. The block is stored only by the node that is eliminated first.
        /// </summary>
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
        /// Assembles the matrix for the current bounds. A tight row enforces z = bound and is disconnected from the
        /// bodies. Its fixed impulse is included in the right-hand side of the body equations instead.
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

            // Each task assembles rows into separate matrix entries.
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
        /// Solves the assembled right-hand side using an earlier factorization, then refines the answer for the current
        /// problem. Starting from the last answer requires correcting only the change between answers. Starting from
        /// the answer computed with the old factorization requires correcting its error across the whole answer.
        /// Returns false if the residual remains too large after the allowed refinements. When <paramref
        /// name="residualKnown"/> is true, correction already contains the residual of the last answer.
        /// </summary>
        bool SolveWithOldFactor(in Problem problem, bool fromLastAnswer, bool residualKnown)
        {
            double* x = solution.Ptr;
            double* step = correction.Ptr;
            int maxRefinements = MaxRefinements;
            if (fromLastAnswer)
            {
                // Computing an answer with the old factorization requires one substitution before the first check.
                // Starting from the last answer allows that substitution to be used for an extra refinement instead.
                maxRefinements++;
            }
            else
            {
                UnsafeUtility.MemCpy(x, right.Ptr, solution.Length * sizeof(double));
                Substitute(x);
            }

            for (int refinement = 0; ; refinement++)
            {
                statistics.Residual = refinement == 0 && residualKnown ? LargestMiss(problem) : ComputeResidual(problem);
                if (statistics.Residual < RefinementTolerance)
                {
                    return true;
                }

                if (refinement == maxRefinements)
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
        /// Stores the residual of the current solution in <see cref="correction"/> and returns its largest component
        /// expressed as a speed. A body residual is an impulse, converted using mass or inertia. A free constraint row
        /// residual is already a speed.
        /// </summary>
        double ComputeResidual(in Problem problem)
        {
            using ProfilerMarker.AutoScope span = ResidualMarker.Auto();

            UnsafeUtility.MemCpy(correction.Ptr, right.Ptr, correction.Length * sizeof(double));

            // Compute each body's residual from its rows, rather than adding each row's contribution to its bodies.
            // This allows both body and row tasks to run without writing to shared entries.
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

        /// <summary>
        /// Returns the largest residual in <see cref="correction"/>, expressed as a speed as in <see
        /// cref="ComputeResidual"/>.
        /// </summary>
        double LargestMiss(in Problem problem)
        {
            double* residual = correction.Ptr;
            double worst = 0.0;
            for (int row = 0; row < problem.RowCount; row++)
            {
                int own = rowUnknown[row];
                if (own != NoNode && state[row] == Free)
                {
                    worst = math.max(worst, math.abs(residual[own]));
                }
            }

            for (int body = 0; body < problem.BodyCount; body++)
            {
                double6 miss = double6.Load(residual + nodes[body].Start);
                double2 scale = bodySpeedScale[body];
                double4 low = math.abs(miss.Low) * new double4(scale.xxx, scale.y);
                double2 high = math.abs(miss.High) * scale.y;
                worst = math.max(worst, math.max(math.cmax(low), math.cmax(high)));
            }

            return worst;
        }

        /// <summary>
        /// Computes residuals for the rows in a range of blocks and returns the largest free row residual as a speed.
        /// </summary>
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
                    int own = rowUnknown[row];
                    if (own == NoNode)
                    {
                        continue;
                    }

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

        /// <summary>
        /// Computes residuals for a range of bodies from their momentum and free row impulses, and returns the largest
        /// as a speed.
        /// </summary>
        double BodyResidual(in Problem problem, int firstBody, int endBody)
        {
            double* x = solution.Ptr;
            double* residual = correction.Ptr;
            double worst = 0.0;
            for (int body = firstBody; body < endBody; body++)
            {
                int start = nodes[body].Start;
                double6 miss = double6.Load(residual + start);
                for (int i = bodyRowStarts[body]; i < bodyRowStarts[body + 1]; i++)
                {
                    int row = bodyRows[i] >> 1;
                    int own = rowUnknown[row];
                    if (own != NoNode && state[row] == Free)
                    {
                        miss -= x[own] * Columns(problem.Jacobian + row * ColumnsPerRow + (bodyRows[i] & 1) * SideColumns);
                    }
                }

                Body source = problem.Bodies[body];
                double6 speed = double6.Load(x + start);
                double3 momentum = math.mul(source.Inertia, new double3(speed.Low.w, speed.High));
                miss -= new double6(new double4(source.Mass * speed.Low.xyz, momentum.x), momentum.yz);
                miss.Store(residual + start);

                double2 scale = bodySpeedScale[body];
                double4 low = math.abs(miss.Low) * new double4(scale.xxx, scale.y);
                double2 high = math.abs(miss.High) * scale.y;
                worst = math.max(worst, math.max(math.cmax(low), math.cmax(high)));
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
        /// Performs block Cholesky factorization of the signed matrix: A = L S L^t. S contains each node's sign, which
        /// is preserved during elimination because the matrix is quasi-definite. Each column block is stored as A C^-t
        /// for use in updates; it differs from L by a sign.
        /// </summary>
        void Factorize()
        {
            using ProfilerMarker.AutoScope span = FactorizeMarker.Auto();

            RunTasks(FactorizeTask, taskStarts.Length - 1, null);

            // Each task gathers the updates for one top node's column, so tasks never write to the same block. It waits
            // only for the nodes that supply those updates. Lanes can therefore process the top together without a
            // separate stage for every node.
            readyStamp++;
            RunTasks(TopFactorTask, topPositions.Length, null);
        }

        /// <summary>
        /// Factors a top node's column after gathering updates to its blocks in serial factorization order. Updates
        /// from the subtrees come first, followed by updates from earlier top nodes as each is eliminated.
        /// </summary>
        void TopFactor(int top)
        {
            for (int i = topUpdateStarts[top]; i < topUpdateStarts[top + 1]; i++)
            {
                int2 update = topUpdates[i];
                UpdateLaterNodes(nodes[order[update.x]], update.y, update.y + 1);
            }

            for (int i = topRowStarts[top]; i < topRowStarts[top + 1]; i++)
            {
                int2 row = topRows[i];
                WaitUntilReady(row.x, readyStamp);
                UpdateLaterNodes(nodes[row.x], row.y, row.y + 1);
            }

            int index = order[topPositions[top]];
            Eliminate(nodes[index]);
            Volatile.Write(ref nodeReady.Ptr[index], readyStamp);
        }

        /// <summary>Runs the tasks of a stage, using the team if one is available.</summary>
        void RunTasks(int kind, int tasks, double* x)
        {
            SparseSolveTeam.RunAll(Team, UnsafeUtility.AddressOf(ref this), kind, tasks, x);
        }

        /// <summary>
        /// Runs one task of a stage. Tasks either access separate data or wait for the earlier tasks whose results they
        /// read.
        /// </summary>
        internal void RunTask(int kind, int task, double* x)
        {
            if (kind == TopFactorTask)
            {
                TopFactor(task);
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

            if (kind == SubstituteTask)
            {
                SubstituteTaskAt(task, x);
                return;
            }

            for (int i = taskStarts[task]; i < taskStarts[task + 1]; i++)
            {
                Node node = nodes[order[taskPositions[i]]];
                Eliminate(node);
                UpdateLaterNodes(node, 0, node.OwnNeighbours);
            }
        }

        /// <summary>
        /// Factors a node's diagonal block and column after all updates from earlier nodes have been applied.
        /// </summary>
        void Eliminate(in Node node)
        {
            double* values = factor.Ptr;
            int size = node.Size;
            double* diagonal = values + node.Diagonal;
            double* inverse = inverseDiagonal.Ptr + node.Start;
            CholeskyInPlace(diagonal, inverse, size, node.Sign);

            Neighbour* column = neighbours.Ptr + node.FirstNeighbour;
            for (int i = 0; i < node.NeighbourCount; i++)
            {
                SolveAgainstTranspose(values + column[i].Offset, nodes[column[i].Node].Size, diagonal, inverse, size);
            }

            if (size == double6x6.Size)
            {
                InvertTriangle(diagonal, inverse);
            }
        }

        /// <summary>
        /// Replaces a factored 6x6 diagonal block with its full inverse. <paramref name="inverse"/> contains the
        /// reciprocals of its diagonal entries. Triangular substitution computes each unknown after the previous one,
        /// causing a chain of dependencies at every node along a tree branch. Multiplication by the full inverse
        /// computes all six together.
        /// </summary>
        static void InvertTriangle(double* c, double* inverse)
        {
            const int size = double6x6.Size;
            double* result = stackalloc double[size * size];
            for (int column = 0; column < size; column++)
            {
                for (int row = 0; row < column; row++)
                {
                    result[row * size + column] = 0.0;
                }

                for (int row = column; row < size; row++)
                {
                    double sum = row == column ? 1.0 : 0.0;
                    for (int j = column; j < row; j++)
                    {
                        sum = MathUtil.Fma(-c[row * size + j], result[j * size + column], sum);
                    }

                    result[row * size + column] = sum * inverse[row];
                }
            }

            UnsafeUtility.MemCpy(c, result, size * size * sizeof(double));
        }

        /// <summary>
        /// Applies an eliminated node's updates to its neighbours, starting at <paramref name="firstNeighbour"/>.
        /// Updates each neighbour's diagonal and the blocks connecting it to later neighbours.
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

                // Elimination connected the first neighbour to every later neighbour in the same order, so a single
                // pass finds all the blocks.
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

        /// <summary>
        /// Factors the lower half of a block in place and writes the reciprocal of each diagonal entry to <paramref
        /// name="inverse"/>. Later columns multiply by these reciprocals. Using division instead would delay the sum
        /// for the next column.
        /// </summary>
        static void CholeskyInPlace(double* a, double* inverse, int size, double sign)
        {
            for (int row = 0; row < size; row++)
            {
                double* r = a + row * size;
                for (int column = 0; column < row; column++)
                {
                    double sum = sign * r[column] - MathUtil.ArrayProductSum(r, a + column * size, column);
                    r[column] = sum * inverse[column];
                }

                double pivot = sign * r[row] - MathUtil.ArrayProductSum(r, r, row);
                r[row] = math.sqrt(math.max(pivot, SmallestPivot));
                inverse[row] = 1.0 / r[row];
            }
        }

        /// <summary>
        /// Multiplies each row of the block by C^-t in place. Uses the reciprocals of C's diagonal entries rather than
        /// division, which would delay the sum for the next column.
        /// </summary>
        static void SolveAgainstTranspose(double* block, int rows, double* c, double* inverse, int size)
        {
            for (int row = 0; row < rows; row++)
            {
                double* x = block + row * size;
                for (int column = 0; column < size; column++)
                {
                    double sum = x[column] - MathUtil.ArrayProductSum(x, c + column * size, column);

                    x[column] = sum * inverse[column];
                }
            }
        }

        /// <summary>
        /// Computes target -= sign * left * right^t, updating four entries of a row at a time. Each entry is a dot
        /// product of two rows, so both rows are read contiguously.
        /// </summary>
        static void Update(double* target, int rows, int columns, double* left, double* right, int inner, double sign, bool lowerOnly)
        {
            for (int row = 0; row < rows; row++)
            {
                double* l = left + row * inner;
                double* into = target + row * columns;
                int last = lowerOnly ? row + 1 : columns;
                int column = 0;

                // A group of entries may extend past the diagonal into the unused upper half. This allows short rows of
                // a diagonal block to use all vector lanes.
                for (; column < last && column + 4 <= columns; column += 4)
                {
                    double* r = right + column * inner;
                    double4 sum = ProductSums(l, r, r + inner, r + 2 * inner, r + 3 * inner, inner);
                    *(double4*)(into + column) -= sign * sum;
                }

                for (; column < last; column++)
                {
                    into[column] -= sign * MathUtil.ArrayProductSum(l, right + column * inner, inner);
                }
            }
        }

        /// <summary>
        /// Computes target -= sign * block * x for a row-major block with the given row count and <paramref
        /// name="inner"/> columns. Processes four rows at a time.
        /// </summary>
        internal static void SubtractProduct(double* target, double* block, double* x, int rows, int inner, double sign)
        {
            int row = 0;
            for (; row + 4 <= rows; row += 4)
            {
                double* b = block + row * inner;
                double4 sum = ProductSums(x, b, b + inner, b + 2 * inner, b + 3 * inner, inner);
                *(double4*)(target + row) -= sign * sum;
            }

            for (; row < rows; row++)
            {
                target[row] -= sign * MathUtil.ArrayProductSum(block + row * inner, x, inner);
            }
        }

        /// <summary>
        /// Computes dot products of <paramref name="shared"/> with four other vectors of the same length, returning one
        /// product in each lane. Reads each vector contiguously and reuses each load of the shared vector in all four
        /// products.
        /// </summary>
        static double4 ProductSums(double* shared, double* b0, double* b1, double* b2, double* b3, int count)
        {
            double4 s0 = 0.0;
            double4 s1 = 0.0;
            double4 s2 = 0.0;
            double4 s3 = 0.0;
            int k = 0;
            for (; k + 4 <= count; k += 4)
            {
                double4 a = *(double4*)(shared + k);
                s0 = MathUtil.Fma(*(double4*)(b0 + k), a, s0);
                s1 = MathUtil.Fma(*(double4*)(b1 + k), a, s1);
                s2 = MathUtil.Fma(*(double4*)(b2 + k), a, s2);
                s3 = MathUtil.Fma(*(double4*)(b3 + k), a, s3);
            }

            // Use a vector for the remaining two entries, which occur with a body's six columns.
            if (k + 2 <= count)
            {
                double2 a = *(double2*)(shared + k);
                s0.xy = MathUtil.Fma(*(double2*)(b0 + k), a, s0.xy);
                s1.xy = MathUtil.Fma(*(double2*)(b1 + k), a, s1.xy);
                s2.xy = MathUtil.Fma(*(double2*)(b2 + k), a, s2.xy);
                s3.xy = MathUtil.Fma(*(double2*)(b3 + k), a, s3.xy);
                k += 2;
            }

            // Each accumulator contains partial sums for one vector. Transpose the accumulators into rows, then add the
            // columns to obtain each vector's total in a separate lane.
            double4x4 partial = math.transpose(new double4x4(s0, s1, s2, s3));
            double4 sum = (partial.c0 + partial.c1) + (partial.c2 + partial.c3);

            if (k < count)
            {
                sum = MathUtil.Fma(new double4(b0[k], b1[k], b2[k], b3[k]), shared[k], sum);
            }

            return sum;
        }

        /// <summary>Computes into += scale * from over <paramref name="count"/> entries.</summary>
        internal static void AddScaled(double* into, double* from, double scale, int count)
        {
            int k = 0;
            for (; k + 4 <= count; k += 4)
            {
                *(double4*)(into + k) = MathUtil.Fma(scale, *(double4*)(from + k), *(double4*)(into + k));
            }

            for (; k < count; k++)
            {
                into[k] = MathUtil.Fma(scale, from[k], into[k]);
            }
        }

        /// <summary>
        /// Solves in place with forward substitution through L, multiplication by S, and backward substitution through
        /// L^t. The solve uses one stage. Tasks process the subtrees forward, the top forward, the top backward from
        /// the root, and finally the subtrees backward. A task waits only for nodes in earlier tasks that lanes have
        /// already taken. Work on the top can therefore overlap work on the subtrees without waiting for the slowest
        /// subtree.
        /// </summary>
        void Substitute(double* x)
        {
            using ProfilerMarker.AutoScope span = SubstituteMarker.Auto();

            // Use one stamp for nodes solved forward and the next stamp for nodes solved backward.
            readyStamp += 2;
            RunTasks(SubstituteTask, 2 * (taskStarts.Length - 1 + topPositions.Length), x);
        }

        void SubstituteTaskAt(int task, double* x)
        {
            int forwardStamp = readyStamp - 1;
            int backwardStamp = readyStamp;
            int subtrees = taskStarts.Length - 1;
            int tops = topPositions.Length;
            if (task < subtrees)
            {
                for (int i = taskStarts[task]; i < taskStarts[task + 1]; i++)
                {
                    int index = order[taskPositions[i]];
                    Node node = nodes[index];
                    SubstituteForward(node, x, node.OwnNeighbours);
                    Volatile.Write(ref nodeReady.Ptr[index], forwardStamp);
                }

                return;
            }

            task -= subtrees;
            if (task < tops)
            {
                TopForward(task, x, forwardStamp);
                return;
            }

            task -= tops;
            if (task < tops)
            {
                TopBackward(tops - 1 - task, x, forwardStamp, backwardStamp);
                return;
            }

            task -= tops;
            int first = taskStarts[task];
            int end = taskStarts[task + 1];

            // The same lane solves the entire subtree forward, finishing with its root.
            WaitUntilReady(order[taskPositions[end - 1]], forwardStamp);
            for (int i = end - 1; i >= first; i--)
            {
                Node node = nodes[order[taskPositions[i]]];

                // The top node answers read here are final. Any top node that read this node's forward answer has
                // already finished its forward pass before starting backward substitution. It is therefore safe to
                // overwrite this node's forward answer.
                for (int n = node.OwnNeighbours; n < node.NeighbourCount; n++)
                {
                    WaitUntilReady(neighbours[node.FirstNeighbour + n].Node, backwardStamp);
                }

                SubstituteBackward(node, x);
            }
        }

        /// <summary>
        /// Solves for a top node through L and S. Gathers contributions from the subtrees and earlier top nodes in the
        /// same order as serial forward substitution.
        /// </summary>
        void TopForward(int top, double* x, int forwardStamp)
        {
            int index = order[topPositions[top]];
            Node node = nodes[index];
            double* into = x + node.Start;

            // These sums are on the dependency chain through the top. For a node with six unknowns, accumulate them in
            // registers and apply the total once.
            bool six = node.Size == double6x6.Size;
            var owed = new double6();

            // First gather contributions from the answers at subtree boundary nodes.
            for (int i = topUpdateStarts[top]; i < topUpdateStarts[top + 1]; i++)
            {
                int2 update = topUpdates[i];
                int sourceIndex = order[update.x];
                WaitUntilReady(sourceIndex, forwardStamp);
                Node source = nodes[sourceIndex];
                double* block = factor.Ptr + neighbours[source.FirstNeighbour + update.y].Offset;
                Owe(ref owed, six, into, block, x + source.Start, node.Size, source.Size);
            }

            // Then gather contributions from earlier top nodes. Their answers already include the node sign that serial
            // substitution would apply when adding each contribution.
            for (int i = topRowStarts[top]; i < topRowStarts[top + 1]; i++)
            {
                int2 row = topRows[i];
                WaitUntilReady(row.x, forwardStamp);
                Node source = nodes[row.x];
                double* block = factor.Ptr + neighbours[source.FirstNeighbour + row.y].Offset;
                Owe(ref owed, six, into, block, x + source.Start, node.Size, source.Size);
            }

            if (six)
            {
                (double6.Load(into) - owed).Store(into);
            }

            SubstituteForward(node, x, 0);
            Volatile.Write(ref nodeReady.Ptr[index], forwardStamp);
        }

        /// <summary>
        /// Computes into -= block * source. If <paramref name="six"/> is true and the block is 6x6, accumulates the
        /// product in <paramref name="owed"/> for the caller to subtract from <paramref name="into"/> at the end.
        /// Otherwise, subtracts the product immediately.
        /// </summary>
        static void Owe(ref double6 owed, bool six, double* into, double* block, double* source, int rows, int inner)
        {
            if (six && inner == double6x6.Size)
            {
                owed += ((double6x6*)block)->Mul(double6.Load(source));
                return;
            }

            SubtractProduct(into, block, source, rows, inner, 1.0);
        }

        /// <summary>
        /// Solves for a top node through L^t after its forward pass and after its later neighbours have been solved.
        /// </summary>
        void TopBackward(int top, double* x, int forwardStamp, int backwardStamp)
        {
            int index = order[topPositions[top]];
            Node node = nodes[index];
            WaitUntilReady(index, forwardStamp);
            for (int i = 0; i < node.NeighbourCount; i++)
            {
                WaitUntilReady(neighbours[node.FirstNeighbour + i].Node, backwardStamp);
            }

            SubstituteBackward(node, x);
            Volatile.Write(ref nodeReady.Ptr[index], backwardStamp);
        }

        /// <summary>Waits for another lane to finish a node in the pass identified by the stamp.</summary>
        void WaitUntilReady(int node, int stamp)
        {
            while (Volatile.Read(ref nodeReady.Ptr[node]) != stamp)
            {
                Common.Pause();
            }
        }

        /// <summary>
        /// Solves for a node through L and S after all contributions from earlier nodes have been applied. Applies its
        /// contributions to neighbours before <paramref name="endNeighbour"/>.
        /// </summary>
        void SubstituteForward(in Node node, double* x, int endNeighbour)
        {
            double* values = factor.Ptr;
            int size = node.Size;
            double* c = values + node.Diagonal;
            double* own = x + node.Start;
            double* inverse = inverseDiagonal.Ptr + node.Start;
            Neighbour* column = neighbours.Ptr + node.FirstNeighbour;
            if (size == double6x6.Size)
            {
                double6 answer = ((double6x6*)c)->Mul(double6.Load(own));
                answer.Store(own);
                for (int i = 0; i < endNeighbour; i++)
                {
                    Node other = nodes[column[i].Node];
                    double* into = x + other.Start;
                    if (other.Size == double6x6.Size)
                    {
                        ref double6x6 block = ref *(double6x6*)(values + column[i].Offset);
                        (double6.Load(into) - node.Sign * block.Mul(answer)).Store(into);
                    }
                    else
                    {
                        SubtractProduct(into, values + column[i].Offset, own, other.Size, size, node.Sign);
                    }
                }

                (node.Sign * answer).Store(own);
                return;
            }

            for (int row = 0; row < size; row++)
            {
                double sum = own[row] - MathUtil.ArrayProductSum(&c[row * size], own, row);

                own[row] = sum * inverse[row];
            }

            for (int i = 0; i < endNeighbour; i++)
            {
                Node other = nodes[column[i].Node];
                SubtractProduct(x + other.Start, values + column[i].Offset, own, other.Size, size, node.Sign);
            }

            for (int row = 0; row < size; row++)
            {
                own[row] *= node.Sign;
            }
        }

        /// <summary>Solves C y = own in place for a diagonal block that was not inverted.</summary>
        static void SolveLower(double* c, double* own, double* inverse, int size)
        {
            for (int row = 0; row < size; row++)
            {
                double sum = own[row];
                for (int j = 0; j < row; j++)
                {
                    sum = MathUtil.Fma(-c[row * size + j], own[j], sum);
                }

                own[row] = sum * inverse[row];
            }
        }

        /// <summary>Solves C^t y = own in place for a diagonal block that was not inverted.</summary>
        static void SolveUpper(double* c, double* own, double* inverse, int size)
        {
            for (int row = size - 1; row >= 0; row--)
            {
                double answer = own[row] * inverse[row];
                own[row] = answer;
                for (int j = 0; j < row; j++)
                {
                    own[j] = MathUtil.Fma(-c[row * size + j], answer, own[j]);
                }
            }
        }

        /// <summary>
        /// Solves for a node through L^t after all its neighbours, which occur later in elimination order, have been
        /// solved.
        /// </summary>
        void SubstituteBackward(in Node node, double* x)
        {
            double* values = factor.Ptr;
            int size = node.Size;
            double* c = values + node.Diagonal;
            double* own = x + node.Start;
            double* inverse = inverseDiagonal.Ptr + node.Start;

            Neighbour* column = neighbours.Ptr + node.FirstNeighbour;
            if (size == double6x6.Size)
            {
                double6 sum = double6.Load(own);
                for (int i = 0; i < node.NeighbourCount; i++)
                {
                    Node other = nodes[column[i].Node];
                    double* block = values + column[i].Offset;
                    double* source = x + other.Start;
                    if (other.Size == double6x6.Size)
                    {
                        sum -= ((double6x6*)block)->TransposeMul(node.Sign * double6.Load(source));
                        continue;
                    }

                    sum.Store(own);
                    for (int row = 0; row < other.Size; row++)
                    {
                        AddScaled(own, block + row * size, -node.Sign * source[row], size);
                    }

                    sum = double6.Load(own);
                }

                ((double6x6*)c)->TransposeMul(sum).Store(own);
                return;
            }

            for (int i = 0; i < node.NeighbourCount; i++)
            {
                Node other = nodes[column[i].Node];
                double* block = values + column[i].Offset;
                double* source = x + other.Start;
                for (int row = 0; row < other.Size; row++)
                {
                    AddScaled(own, block + row * size, -node.Sign * source[row], size);
                }
            }

            // Process one column at a time. Once an answer is known, row `row` of C contains its contributions to
            // earlier unknowns in contiguous entries. Computing a separate sum for each unknown would require reading
            // down columns of C.
            for (int row = size - 1; row >= 0; row--)
            {
                double answer = own[row] * inverse[row];
                own[row] = answer;
                AddScaled(own, c + row * size, -answer, row);
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
        /// Changes rows between free and tight states and returns the number of infeasible rows. A free row is
        /// infeasible if its impulse exceeds a bound. A tight row is infeasible if the system would move it inside its
        /// bounds: w = J du + E z - b has the wrong sign.
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
                    // A row with equal bounds has one possible impulse and was made tight when its bounds became equal.
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

            double6 speed = double6.Load(solution.Ptr + nodes[body].Start);
            double6 jacobian = Columns(columns);
            return math.csum(MathUtil.Fma(speed.Low, jacobian.Low, new double4(speed.High * jacobian.High, 0.0, 0.0)));
        }

        /// <summary>Returns one body's side of a Jacobian row: six columns, linear then angular.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double6 Columns(float* columns)
        {
            return new double6(new double4(*(float4*)columns), new double2(*(float2*)(columns + 4)));
        }

        /// <summary>Stores priorities in a binary heap with the smallest priority first.</summary>
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
