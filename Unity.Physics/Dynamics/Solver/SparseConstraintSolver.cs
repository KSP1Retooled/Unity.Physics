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
        const double SmallestPivot = 1e-30;

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
        }

        /// <summary>
        /// The size of a solve, which doesn't depend on the machine, and its time by stage. Times are in the units of
        /// <see cref="ProfilerUnsafeUtility.Timestamp"/>.
        /// </summary>
        public struct Statistics
        {
            public int Nodes;
            public int FactorBlocks;
            public long FactorScalars;
            public int PivotingIterations;

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

        const byte Free = 0;
        const byte LowerTight = 1;
        const byte UpperTight = 2;

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
        }

        struct Neighbour
        {
            public int Node;

            // Where its block starts in the factor, with this neighbour's rows and the column node's columns.
            public int Offset;
        }

        NativeList<Node> nodes;
        NativeList<int> order;
        NativeList<Neighbour> neighbours;
        NativeList<int> rowNode;
        NativeList<int> rowInNode;
        NativeList<double> factor;
        NativeList<double> solution;
        NativeList<byte> state;
        Statistics statistics;

        public SparseConstraintSolver(Allocator allocator)
        {
            nodes = new NativeList<Node>(allocator);
            order = new NativeList<int>(allocator);
            neighbours = new NativeList<Neighbour>(allocator);
            rowNode = new NativeList<int>(allocator);
            rowInNode = new NativeList<int>(allocator);
            factor = new NativeList<double>(allocator);
            solution = new NativeList<double>(allocator);
            state = new NativeList<byte>(allocator);
            statistics = default;
        }

        public void Dispose()
        {
            nodes.Dispose();
            order.Dispose();
            neighbours.Dispose();
            rowNode.Dispose();
            rowInNode.Dispose();
            factor.Dispose();
            solution.Dispose();
            state.Dispose();
        }

        public Statistics Stats => statistics;

        /// <summary>
        /// Writes each row's impulse to <paramref name="impulses"/>. With <paramref name="bounded"/> false the bounds are
        /// ignored. Returns false if pivoting ran out of iterations, leaving the last impulses clamped to their bounds.
        /// </summary>
        public bool Solve(in Problem problem, float* impulses, bool bounded)
        {
            long started = ProfilerUnsafeUtility.Timestamp;
            Analyze(problem);
            statistics.AnalyzeTime = Lap(ref started);

            state.Clear();
            state.Resize(problem.RowCount, NativeArrayOptions.ClearMemory);
            if (bounded && problem.InitialStates != null)
            {
                UnsafeUtility.MemCpy(state.GetUnsafePtr(), problem.InitialStates, problem.RowCount);
            }

            if (bounded)
            {
                for (int row = 0; row < problem.RowCount; row++)
                {
                    if (problem.Lower[row] == problem.Upper[row])
                    {
                        state[row] = LowerTight;
                    }
                }
            }

            int leastInfeasible = int.MaxValue;
            int stalled = 0;
            bool flipAll = true;
            for (int iteration = 1; iteration <= MaxPivotingIterations; iteration++)
            {
                statistics.PivotingIterations = iteration;
                started = ProfilerUnsafeUtility.Timestamp;
                Assemble(problem);
                statistics.AssembleTime += Lap(ref started);
                Factorize();
                statistics.FactorizeTime += Lap(ref started);
                Substitute();
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
                    return true;
                }

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
                impulses[row] = math.clamp(impulses[row], problem.Lower[row], problem.Upper[row]);
            }

            return false;
        }

        static long Lap(ref long started)
        {
            long now = ProfilerUnsafeUtility.Timestamp;
            long elapsed = now - started;
            started = now;
            return elapsed;
        }

        // Structure

        void Analyze(in Problem problem)
        {
            using ProfilerMarker.AutoScope span = AnalyzeMarker.Auto();

            nodes.Clear();
            for (int body = 0; body < problem.BodyCount; body++)
            {
                nodes.Add(new Node { Size = BodySize, Sign = 1.0, BodyA = -1, BodyB = -1 });
            }

            MergeBlocksByBodyPair(problem);
            Order();
            LayOut();

            statistics.Nodes = nodes.Length;
            statistics.FactorBlocks = neighbours.Length;
            statistics.FactorScalars = factor.Length;
        }

        /// <summary>
        /// One joint emits several Jacobians between the same two bodies. As separate nodes they would all neighbour each other
        /// anyway, so one node holding all their rows is smaller and loses no structure.
        /// </summary>
        void MergeBlocksByBodyPair(in Problem problem)
        {
            rowNode.ResizeUninitialized(problem.RowCount);
            rowInNode.ResizeUninitialized(problem.RowCount);

            var pairs = new NativeHashMap<long, int>(problem.BlockCount, Allocator.Temp);
            for (int b = 0; b < problem.BlockCount; b++)
            {
                ConstraintBlock block = problem.Blocks[b];
                int lower = math.min(block.BodyA, block.BodyB);
                int higher = math.max(block.BodyA, block.BodyB);
                long key = ((long)lower << 32) | (uint)higher;
                if (!pairs.TryGetValue(key, out int node))
                {
                    node = nodes.Length;
                    nodes.Add(new Node { Sign = -1.0, BodyA = lower, BodyB = higher });
                    pairs.Add(key, node);
                }

                ref Node target = ref nodes.ElementAt(node);
                for (int row = 0; row < block.RowCount; row++)
                {
                    rowNode[block.FirstRow + row] = node;
                    rowInNode[block.FirstRow + row] = target.Size + row;
                }

                target.Size += block.RowCount;
            }

            pairs.Dispose();
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

                Neighbour* column = neighbours.GetUnsafePtr() + node.FirstNeighbour;
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

            factor.ResizeUninitialized(offset);
            solution.ResizeUninitialized(start);
        }

        /// <summary>Where the block between two nodes is, which only the earlier of them holds.</summary>
        double* BlockOf(int earlier, int later)
        {
            Node column = nodes[earlier];
            int laterPosition = nodes[later].Position;
            Neighbour* list = neighbours.GetUnsafePtr() + column.FirstNeighbour;
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

            return factor.GetUnsafePtr() + list[low].Offset;
        }

        // Values

        /// <summary>
        /// Fills in the matrix and right-hand side for the current bounds. A row at a bound becomes -z = bound, cut off from
        /// the bodies, whose side of the system takes its fixed impulse instead.
        /// </summary>
        void Assemble(in Problem problem)
        {
            using ProfilerMarker.AutoScope span = AssembleMarker.Auto();

            double* values = factor.GetUnsafePtr();
            double* right = solution.GetUnsafePtr();
            UnsafeUtility.MemClear(values, factor.Length * sizeof(double));
            UnsafeUtility.MemClear(right, solution.Length * sizeof(double));

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

            for (int b = 0; b < problem.BlockCount; b++)
            {
                ConstraintBlock block = problem.Blocks[b];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    int constraint = rowNode[row];
                    int index = rowInNode[row];
                    Node node = nodes[constraint];
                    float* jacobian = problem.Jacobian + row * ColumnsPerRow;
                    byte rowState = state[row];
                    if (rowState == Free)
                    {
                        values[node.Diagonal + index * node.Size + index] = -problem.Epsilon[row];
                        right[node.Start + index] = problem.RightHandSide[row];
                        WriteJacobian(constraint, index, block.BodyA, jacobian);
                        WriteJacobian(constraint, index, block.BodyB, jacobian + SideColumns);
                    }
                    else
                    {
                        double bound = rowState == LowerTight ? problem.Lower[row] : problem.Upper[row];
                        values[node.Diagonal + index * node.Size + index] = -1.0;
                        right[node.Start + index] = bound;
                        AddToBody(block.BodyA, jacobian, bound);
                        AddToBody(block.BodyB, jacobian + SideColumns, bound);
                    }
                }
            }
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

        void AddToBody(int body, float* columns, double bound)
        {
            if (body < 0)
            {
                return;
            }

            double* right = solution.GetUnsafePtr() + nodes[body].Start;
            for (int column = 0; column < SideColumns; column++)
            {
                right[column] += columns[column] * bound;
            }
        }

        /// <summary>
        /// Block Cholesky on the signed matrix: A = L S L^t, with S the sign of each node, which elimination preserves because
        /// the matrix is quasi-definite. Each column's blocks are kept as A C^-t, one sign away from L, for the updates to use.
        /// </summary>
        void Factorize()
        {
            using ProfilerMarker.AutoScope span = FactorizeMarker.Auto();

            double* values = factor.GetUnsafePtr();
            Neighbour* allNeighbours = neighbours.GetUnsafePtr();
            for (int position = 0; position < order.Length; position++)
            {
                Node node = nodes[order[position]];
                int size = node.Size;
                double* diagonal = values + node.Diagonal;
                CholeskyInPlace(diagonal, size, node.Sign);

                Neighbour* column = allNeighbours + node.FirstNeighbour;
                for (int i = 0; i < node.NeighbourCount; i++)
                {
                    SolveAgainstTranspose(values + column[i].Offset, nodes[column[i].Node].Size, diagonal, size);
                }

                for (int i = 0; i < node.NeighbourCount; i++)
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
        void Substitute()
        {
            using ProfilerMarker.AutoScope span = SubstituteMarker.Auto();

            double* values = factor.GetUnsafePtr();
            double* x = solution.GetUnsafePtr();
            Neighbour* allNeighbours = neighbours.GetUnsafePtr();

            for (int position = 0; position < order.Length; position++)
            {
                Node node = nodes[order[position]];
                int size = node.Size;
                double* c = values + node.Diagonal;
                double* own = x + node.Start;
                for (int row = 0; row < size; row++)
                {
                    double sum = own[row];
                    for (int k = 0; k < row; k++)
                    {
                        sum -= c[row * size + k] * own[k];
                    }

                    own[row] = sum / c[row * size + row];
                }

                Neighbour* column = allNeighbours + node.FirstNeighbour;
                for (int i = 0; i < node.NeighbourCount; i++)
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

            for (int position = order.Length - 1; position >= 0; position--)
            {
                Node node = nodes[order[position]];
                int size = node.Size;
                double* c = values + node.Diagonal;
                double* own = x + node.Start;

                Neighbour* column = allNeighbours + node.FirstNeighbour;
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

                for (int row = size - 1; row >= 0; row--)
                {
                    double sum = own[row];
                    for (int k = row + 1; k < size; k++)
                    {
                        sum -= c[k * size + row] * own[k];
                    }

                    own[row] = sum / c[row * size + row];
                }
            }
        }

        void ReadImpulses(in Problem problem, float* impulses)
        {
            double* x = solution.GetUnsafePtr();
            for (int row = 0; row < problem.RowCount; row++)
            {
                byte rowState = state[row];
                impulses[row] = rowState == Free
                    ? (float)-x[nodes[rowNode[row]].Start + rowInNode[row]]
                    : rowState == LowerTight ? problem.Lower[row] : problem.Upper[row];
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
                    // A row whose bounds meet has one possible impulse, and it was tight from the start.
                    if (problem.Lower[row] == problem.Upper[row])
                    {
                        continue;
                    }

                    byte current = state[row];
                    byte next = current;
                    if (current == Free)
                    {
                        float impulse = impulses[row];
                        if (impulse < problem.Lower[row] - Tolerance)
                        {
                            next = LowerTight;
                        }
                        else if (impulse > problem.Upper[row] + Tolerance)
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

            double* du = solution.GetUnsafePtr() + nodes[body].Start;
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
