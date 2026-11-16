using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Numerics.Linear.Solvers.Lowlevel
{
    /// <summary>
    /// Solves a bounded linear complementarity problem by block principal pivoting. The problem has a matrix W that is
    /// dense, symmetric and positive definite, and that is stored row by row. Each unknown z_i has a lower and an upper
    /// bound. In the answer, a free row has W z - q = 0, and a row at a bound has W z - q with the sign that the bound
    /// requires. The bounds of some rows can depend on the values of other rows, as <see cref="BoundCoupling"/>
    /// describes. For example, the bounds of a friction impulse depend on a normal impulse.
    /// <para>
    /// Each round of pivoting splits the rows into free rows and rows at a bound, and solves for the free rows with a
    /// dense Cholesky factor in double precision. Then every row that breaks its conditions changes its state. If the
    /// number of such rows stops going down, only the last such row changes its state instead (Murty's rule). The
    /// coupled bounds lag behind: after each answer that meets every condition, the solver moves the coupled bounds
    /// and solves again. The factor of the free rows carries over from one round to the next, and each round changes it
    /// only by the rows that joined or left the free set, in the way that <see cref="FactorUpdate"/> picks.
    /// </para>
    /// <para>
    /// The vessel solver's forest solve is one user. It solves its boundary block, the dense matrix
    /// W = J_b A^-1 J_b^T + E_b, with this solver. The sparse constraint solver uses the same method on its own sparse
    /// factor.
    /// </para>
    /// </summary>
    public static unsafe class BlockPrincipalPivoting
    {
        const double Tolerance = 1e-4;
        /// <summary>The most rounds of pivoting that a solve without a cap of its own takes.</summary>
        public const int MaxPivotingIterations = 200;
        const int MaxBlockIterationsWithoutProgress = 3;
        const int MaxCouplingIterations = 25;
        const double CouplingTolerance = 1e-3;

        const byte Free = 0;
        const byte LowerTight = 1;
        const byte UpperTight = 2;

        /// <summary>
        /// Solves for <paramref name="z"/>. On a free row, W z - q = 0. On a row at a bound, z equals that bound, and
        /// W z - q has the sign that the bound requires. On input, <paramref name="states"/> holds a first guess at
        /// which rows are at a bound: 0 for a free row, 1 for a row at its lower bound, and 2 for a row at its upper
        /// bound. On output, it holds the states of the answer. If <paramref name="bounded"/> is false, the solver
        /// ignores the bounds and treats every row as free. The function returns false if pivoting ran out of
        /// iterations. In that case, a few sweeps of projected Gauss-Seidel finish the solve, and z is inside its bounds.
        /// </summary>
        public static bool Solve(int n, double* w, double* q, double* problemLower, double* problemUpper, byte* states,
            BoundCoupling* couplings, int couplingCount, bool bounded, double* z)
        {
            return Solve(n, w, q, problemLower, problemUpper, states, couplings, couplingCount, bounded, z,
                MaxPivotingIterations, false, out _);
        }

        /// <summary>
        /// Solves as the overload above does, with two differences. First, the solve takes at most
        /// <paramref name="maxRounds"/> rounds, where one round is one solve for the free rows. If the solve runs out of
        /// rounds after it has found at least one answer that meets every condition, it returns the last such answer,
        /// with that answer's states, and true. If it runs out of rounds before it finds any such answer, it finishes
        /// with a few sweeps of projected Gauss-Seidel and returns false. Either way, every impulse is inside its bounds,
        /// and the bounds of the coupled rows come from the returned answer's own source impulses. Second, if <paramref name="warm"/> is true, the caller's guess at the states also counts for coupled rows.
        /// Before the first round, the solver sets the coupled bounds from the source impulses that were applied before
        /// this solve, as if this solve added nothing to them. Then a coupled row whose bounds have room starts in the
        /// state that the caller guessed for it, instead of starting free. <paramref name="rounds"/> returns the number
        /// of rounds that the solve took.
        /// </summary>
        public static bool Solve(int n, double* w, double* q, double* problemLower, double* problemUpper, byte* states,
            BoundCoupling* couplings, int couplingCount, bool bounded, double* z, int maxRounds, bool warm, out int rounds)
        {
            return Solve(n, w, q, problemLower, problemUpper, states, couplings, couplingCount, bounded, z, maxRounds, warm, true,
                null, out rounds);
        }

        /// <summary>
        /// Solves as the overload above does. If <paramref name="update"/> is true, a round changes the last round's
        /// factor in the default way, <see cref="DefaultUpdate"/>, and a round with the same free rows uses the last
        /// round's factor as it is. If <paramref name="update"/> is false, every round factors W_FF from the start. All
        /// ways give the same answer up to rounding. If <paramref name="stats"/> is not null, the solve adds what it did
        /// to it.
        /// </summary>
        public static bool Solve(int n, double* w, double* q, double* problemLower, double* problemUpper, byte* states,
            BoundCoupling* couplings, int couplingCount, bool bounded, double* z, int maxRounds, bool warm, bool update,
            BlockPrincipalPivotingStats* stats, out int rounds)
        {
            return Solve(n, w, q, problemLower, problemUpper, states, couplings, couplingCount, bounded, z, maxRounds, warm,
                update ? DefaultUpdate : FactorUpdate.Refactor, null, stats, out rounds);
        }

        /// <summary>The way that a round changes the factor of the free rows when the caller doesn't pick one.</summary>
        public const FactorUpdate DefaultUpdate = FactorUpdate.Edit;

        /// <summary>
        /// Solves as the overload above does, with the factor changed from round to round in the given way. If
        /// <paramref name="priority"/> is not null, the first factor takes its free rows in order of falling priority,
        /// and rows of equal priority in their own order. A caller gives the rows that are likely to stay free a high
        /// priority, so that the rows that leave the free set later sit near the end of the factor, where a deletion
        /// costs little. The order changes the answer only by rounding.
        /// </summary>
        public static bool Solve(int n, double* w, double* q, double* problemLower, double* problemUpper, byte* states,
            BoundCoupling* couplings, int couplingCount, bool bounded, double* z, int maxRounds, bool warm, FactorUpdate update,
            byte* priority, BlockPrincipalPivotingStats* stats, out int rounds)
        {
            var lowerArray = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var upperArray = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var coupledArray = new NativeArray<bool>(n, Allocator.Temp);
            var factor = new FreeFactor(n, update, priority, stats);
            var bestArray = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var guessArray = new NativeArray<byte>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var bestStatesArray = new NativeArray<byte>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            double* lower = (double*)lowerArray.GetUnsafePtr();
            double* upper = (double*)upperArray.GetUnsafePtr();
            bool* coupled = (bool*)coupledArray.GetUnsafePtr();
            double* best = (double*)bestArray.GetUnsafePtr();
            byte* guess = (byte*)guessArray.GetUnsafePtr();
            byte* bestStates = (byte*)bestStatesArray.GetUnsafePtr();
            for (int i = 0; i < n; i++)
            {
                lower[i] = problemLower[i];
                upper[i] = problemUpper[i];
                guess[i] = states[i];
            }

            for (int c = 0; bounded && c < couplingCount; c++)
            {
                for (int row = couplings[c].FirstRow; row < couplings[c].FirstRow + couplings[c].RowCount; row++)
                {
                    coupled[row] = true;
                }
            }

            // A warm start sets the coupled bounds from the impulses that were applied before this solve.
            warm &= bounded;
            if (warm)
            {
                for (int i = 0; i < n; i++)
                {
                    z[i] = 0.0;
                }

                Couple(couplings, couplingCount, problemLower, problemUpper, lower, upper, states, z);
            }

            for (int i = 0; i < n; i++)
            {
                byte state = bounded ? guess[i] : Free;
                if (bounded && lower[i] == upper[i])
                {
                    state = LowerTight;
                }
                else if ((coupled[i] && !warm) || state > UpperTight)
                {
                    state = Free;
                }

                states[i] = state;
            }

            bool result = false;
            int leastInfeasible = int.MaxValue;
            int stalled = 0;
            bool flipAll = true;
            int couplingRounds = 0;
            bool found = false;
            rounds = 0;
            for (int iteration = 1; iteration <= maxRounds; iteration++)
            {
                rounds = iteration;
                factor.SolveFree(w, q, lower, upper, states, z);
                if (!bounded)
                {
                    result = true;
                    break;
                }

                long pivotStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                int infeasible = Pivot(n, w, q, lower, upper, states, z, flipAll);
                if (stats != null)
                {
                    stats->PivotTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - pivotStarted;
                }

                if (infeasible == 0)
                {
                    found = true;
                    for (int i = 0; i < n; i++)
                    {
                        best[i] = z[i];
                        bestStates[i] = states[i];
                    }

                    if (couplingRounds == MaxCouplingIterations || !Couple(couplings, couplingCount, problemLower, problemUpper,
                        lower, upper, states, z))
                    {
                        result = true;
                        break;
                    }

                    couplingRounds++;
                    leastInfeasible = int.MaxValue;
                    stalled = 0;
                    flipAll = true;
                    continue;
                }

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

            if (bounded && (result || found))
            {
                // If the solve ran out of rounds, it returns the last answer that met every condition, with that
                // answer's states. Either way, the friction bounds of the answer lag behind its normal impulses by up
                // to one round of coupling, so the solver moves the coupled bounds to the answer's own normal impulses
                // and brings the coupled rows inside them.
                if (!result)
                {
                    for (int i = 0; i < n; i++)
                    {
                        z[i] = best[i];
                        states[i] = bestStates[i];
                    }

                    result = true;
                }

                FitCoupledRows(couplings, couplingCount, problemLower, problemUpper, lower, upper, states, z);
            }
            else if (!result)
            {
                // The solve ran out of rounds before it found any answer that met every condition. The impulses of
                // the last round can be far from the answer, so a few sweeps of projected Gauss-Seidel finish the
                // solve instead. The sweeps keep every impulse inside its bounds, and the function still returns
                // false so that the caller can count the solve.
                FinishBySweeps(n, w, q, couplings, couplingCount, problemLower, problemUpper, lower, upper, states, z);
            }

            if (stats != null)
            {
                stats->Rounds += rounds;
                stats->CouplingRounds += couplingRounds;
            }

            lowerArray.Dispose();
            upperArray.Dispose();
            coupledArray.Dispose();
            factor.Dispose();
            bestArray.Dispose();
            guessArray.Dispose();
            bestStatesArray.Dispose();
            return result;
        }

        /// <summary>
        /// The Cholesky factor L L^T of W_FF, where F is the set of free rows, kept from one round of pivoting to the
        /// next. The factor's rows are the free rows in the order in which they joined the factor, so a row that joins
        /// goes at the end. Each round asks the factor to solve W_FF z_F = q_F - W_FT z_T, where T is the set of rows
        /// at a bound. If the round's free rows are the ones that the factor holds, the factor is used as it is.
        /// Otherwise the factor changes in the way that <see cref="FactorUpdate"/> picks. The factor, its solves and
        /// its changes are the ported faer routines in <see cref="CholeskyLLT"/>. L is stored in column-major order
        /// with a column stride of n.
        /// </summary>
        unsafe struct FreeFactor
        {
            readonly int n;
            readonly FactorUpdate mode;
            readonly byte* priority;
            readonly BlockPrincipalPivotingStats* stats;

            /// <summary>The lower triangle L, in column-major order with a column stride of n.</summary>
            NativeArray<double> factorArray;

            /// <summary>The rows of the factor in its order, and each row's place in the factor, or -1.</summary>
            NativeArray<int> orderArray;
            NativeArray<int> placeArray;
            NativeArray<double> rhsArray;
            NativeArray<double> residualArray;

            /// <summary>For each row, the impulse at a bound that the residual has taken out of q, or zero.</summary>
            NativeArray<double> appliedArray;
            bool residualStarted;

            /// <summary>The new columns that an insertion takes, and the scratch of an insertion or a deletion.</summary>
            NativeArray<double> columnsArray;
            NativeArray<double> scratchArray;
            NativeArray<int> deletedArray;
            int m;
            bool valid;

            public FreeFactor(int n, FactorUpdate mode, byte* priority, BlockPrincipalPivotingStats* stats)
            {
                this.n = n;
                this.mode = mode;
                this.priority = priority;
                this.stats = stats;
                int size = math.max(1, n);
                factorArray = new NativeArray<double>(size * size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                orderArray = new NativeArray<int>(size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                placeArray = new NativeArray<int>(size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                rhsArray = new NativeArray<double>(size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                residualArray = new NativeArray<double>(size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                appliedArray = new NativeArray<double>(size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                residualStarted = false;
                columnsArray = default;
                scratchArray = default;
                deletedArray = default;
                if (mode != FactorUpdate.Refactor)
                {
                    columnsArray = new NativeArray<double>(size * size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    scratchArray = new NativeArray<double>(size * size + 2 * size + 8, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    deletedArray = new NativeArray<int>(size, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                }

                int* place = (int*)placeArray.GetUnsafePtr();
                for (int i = 0; i < n; i++)
                {
                    place[i] = -1;
                }

                m = 0;
                valid = false;
            }

            public void Dispose()
            {
                factorArray.Dispose();
                orderArray.Dispose();
                placeArray.Dispose();
                rhsArray.Dispose();
                residualArray.Dispose();
                appliedArray.Dispose();
                if (columnsArray.IsCreated)
                {
                    columnsArray.Dispose();
                    scratchArray.Dispose();
                    deletedArray.Dispose();
                }
            }

            /// <summary>
            /// Sets each row that is at a bound to that bound. Then solves W_FF z_F = q_F - W_FT z_T for the free rows.
            /// </summary>
            public void SolveFree(double* w, double* q, double* lower, double* upper, byte* states, double* z)
            {
                int* order = (int*)orderArray.GetUnsafePtr();
                int* place = (int*)placeArray.GetUnsafePtr();
                double* rhs = (double*)rhsArray.GetUnsafePtr();

                // Find the rows at a bound, and how the free rows differ from the rows that the factor holds. The rows
                // of the factor before the first row that left the free set can stay.
                bool changing = valid && mode != FactorUpdate.Refactor;
                int added = 0, removed = 0, kept = changing ? m : 0;
                for (int i = 0; i < n; i++)
                {
                    if (states[i] == Free)
                    {
                        added += place[i] < 0 ? 1 : 0;
                    }
                    else
                    {
                        z[i] = states[i] == LowerTight ? lower[i] : upper[i];
                        if (place[i] >= 0)
                        {
                            removed++;
                            kept = math.min(kept, place[i]);
                        }
                    }
                }

                long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                if (valid && stats != null)
                {
                    stats->AddedRows += added;
                    stats->RemovedRows += removed;
                }

                if (changing && added + removed == 0)
                {
                    if (stats != null)
                    {
                        stats->Reuses++;
                    }
                }
                else if (changing && mode == FactorUpdate.Edit)
                {
                    EditRows(w, states, removed);
                    if (stats != null)
                    {
                        stats->Updates++;
                    }
                }
                else
                {
                    // The new order: the rows that stay, in their order, and then the rows that joined. The rows of
                    // the factor before place kept stay as they are, and the rest are inserted after them.
                    int next = kept;
                    for (int a = kept; a < m; a++)
                    {
                        int row = order[a];
                        place[row] = -1;
                        if (states[row] == Free)
                        {
                            order[next++] = row;
                        }
                    }

                    for (int a = kept; a < next; a++)
                    {
                        place[order[a]] = a;
                    }

                    if (!valid && priority != null)
                    {
                        next = OrderByPriority(states, next);
                    }

                    for (int i = 0; i < n; i++)
                    {
                        if (states[i] == Free && place[i] < 0)
                        {
                            place[i] = next;
                            order[next++] = i;
                        }
                    }

                    if (kept == 0)
                    {
                        m = next;
                        FactorAll(w);
                    }
                    else
                    {
                        m = kept;
                        Append(w, next - kept);
                    }

                    if (stats != null)
                    {
                        if (kept > 0)
                        {
                            stats->Updates++;
                        }
                        else
                        {
                            stats->Factorizations++;
                        }
                    }
                }

                valid = true;
                if (stats != null)
                {
                    stats->FactorTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - started;
                    stats->FreeRows += m;
                    stats->MostFreeRows = math.max(stats->MostFreeRows, m);
                    stats->FewestFreeRows = m > 0 && (stats->FewestFreeRows == 0 || m < stats->FewestFreeRows) ? m : stats->FewestFreeRows;
                }

                long solveStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                // The right-hand side q - W_FT z_T. W is symmetric, so column t of W is row t, which is stored in a
                // line. The residual keeps what the last round took out of q, and each row whose impulse at a bound
                // differs from that only takes out the difference, along its row of W. From one round to the next,
                // only the rows that changed state and the coupled rows whose bounds moved differ, and a row at a
                // bound of zero takes out nothing.
                double* residual = (double*)residualArray.GetUnsafePtr();
                double* applied = (double*)appliedArray.GetUnsafePtr();
                if (!residualStarted)
                {
                    UnsafeUtility.MemCpy(residual, q, (long)n * sizeof(double));
                    UnsafeUtility.MemClear(applied, (long)n * sizeof(double));
                    residualStarted = true;
                }

                for (int i = 0; i < n; i++)
                {
                    double atBound = states[i] == Free ? 0.0 : z[i];
                    if (atBound != applied[i])
                    {
                        Blas.Axpy(n, applied[i] - atBound, w + (long)i * n, residual);
                        applied[i] = atBound;
                    }
                }

                for (int a = 0; a < m; a++)
                {
                    rhs[a] = residual[order[a]];
                }

                CholeskyLLT.Solve(m, (double*)factorArray.GetUnsafePtr(), n, rhs);
                for (int a = 0; a < m; a++)
                {
                    z[order[a]] = rhs[a];
                }

                if (stats != null)
                {
                    stats->SolveTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - solveStarted;
                }
            }

            /// <summary>
            /// Puts the free rows into the order from place <paramref name="next"/> on, highest priority first, and rows
            /// of equal priority in their own order. Returns the place after the last one.
            /// </summary>
            int OrderByPriority(byte* states, int next)
            {
                int* order = (int*)orderArray.GetUnsafePtr();
                int* place = (int*)placeArray.GetUnsafePtr();
                int* counts = stackalloc int[257];
                for (int k = 0; k < 257; k++)
                {
                    counts[k] = 0;
                }

                // A counting sort by falling priority: count each priority, turn the counts into starting places, and
                // put each row at its priority's next place.
                for (int i = 0; i < n; i++)
                {
                    if (states[i] == Free && place[i] < 0)
                    {
                        counts[255 - priority[i] + 1]++;
                    }
                }

                counts[0] = next;
                for (int k = 1; k < 257; k++)
                {
                    counts[k] += counts[k - 1];
                }

                int end = counts[256];
                for (int i = 0; i < n; i++)
                {
                    if (states[i] == Free && place[i] < 0)
                    {
                        int at = counts[255 - priority[i]]++;
                        order[at] = i;
                    }
                }

                for (int a = next; a < end; a++)
                {
                    place[order[a]] = a;
                }

                return end;
            }

            /// <summary>Factors W_FF in the factor's order from the start.</summary>
            void FactorAll(double* w)
            {
                long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
                double* l = (double*)factorArray.GetUnsafePtr();
                int* order = (int*)orderArray.GetUnsafePtr();
                for (int j = 0; j < m; j++)
                {
                    double* wj = w + (long)order[j] * n;
                    double* lj = l + (long)j * n;
                    for (int i = j; i < m; i++)
                    {
                        lj[i] = wj[order[i]];
                    }
                }

                CholeskyLLT.Factor(m, l, n, keepUpper: false);
                if (stats != null)
                {
                    stats->FullFactorTicks += Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - started;
                }
            }

            /// <summary>
            /// Inserts the <paramref name="count"/> rows at places m and on of the order at the end of the factor, and
            /// moves m past them. Their new columns come from W, in the factor's order.
            /// </summary>
            void Append(double* w, int count)
            {
                if (count == 0)
                {
                    return;
                }

                if (m == 0)
                {
                    m = count;
                    FactorAll(w);
                    return;
                }

                int* order = (int*)orderArray.GetUnsafePtr();
                double* columns = (double*)columnsArray.GetUnsafePtr();
                int total = m + count;
                for (int c = 0; c < count; c++)
                {
                    double* wc = w + (long)order[m + c] * n;
                    double* bc = columns + (long)c * total;
                    for (int i = 0; i < total; i++)
                    {
                        bc[i] = wc[order[i]];
                    }
                }

                CholeskyLLT.InsertRowsAndColumns(m, (double*)factorArray.GetUnsafePtr(), n, m, count, columns, total,
                    (double*)scratchArray.GetUnsafePtr());
                m = total;
            }

            /// <summary>
            /// Deletes the factor's rows that left the free set, wherever they are, and then inserts the rows that
            /// joined at the end. The rows that stay keep their order.
            /// </summary>
            void EditRows(double* w, byte* states, int removed)
            {
                int* order = (int*)orderArray.GetUnsafePtr();
                int* place = (int*)placeArray.GetUnsafePtr();
                int* deleted = (int*)deletedArray.GetUnsafePtr();
                if (removed > 0)
                {
                    int count = 0, next = 0;
                    for (int a = 0; a < m; a++)
                    {
                        int row = order[a];
                        if (states[row] == Free)
                        {
                            place[row] = next;
                            order[next++] = row;
                        }
                        else
                        {
                            place[row] = -1;
                            deleted[count++] = a;
                        }
                    }

                    if (next > 0)
                    {
                        CholeskyLLT.DeleteRowsAndColumns(m, (double*)factorArray.GetUnsafePtr(), n, deleted, count,
                            (double*)scratchArray.GetUnsafePtr());
                    }

                    m = next;
                }

                int first = m;
                int total = m;
                for (int i = 0; i < n; i++)
                {
                    if (states[i] == Free && place[i] < 0)
                    {
                        place[i] = total;
                        order[total++] = i;
                    }
                }

                Append(w, total - first);
            }
        }

        /// <summary>
        /// Finds the rows that break their conditions. A free row breaks them when it is outside its bounds by more
        /// than the tolerance. A row at a bound breaks them when W z - q has the wrong sign by more than the tolerance.
        /// The routine skips rows whose two bounds are equal. If <paramref name="flipAll"/> is true, the routine
        /// changes the state of every row that breaks its conditions. Otherwise, it changes only the state of the last
        /// such row. Returns how many rows break their conditions.
        /// </summary>
        static int Pivot(int n, double* w, double* q, double* lower, double* upper, byte* states, double* z, bool flipAll)
        {
            int infeasible = 0;
            int last = -1;
            byte lastState = Free;
            for (int i = 0; i < n; i++)
            {
                if (lower[i] == upper[i])
                {
                    continue;
                }

                byte current = states[i];
                byte next = current;
                if (current == Free)
                {
                    if (z[i] < lower[i] - Tolerance)
                    {
                        next = LowerTight;
                    }
                    else if (z[i] > upper[i] + Tolerance)
                    {
                        next = UpperTight;
                    }
                }
                else
                {
                    double slack = Blas.Dot(n, w + (long)i * n, z) - q[i];

                    if ((current == LowerTight && slack < -Tolerance) || (current == UpperTight && slack > Tolerance))
                    {
                        next = Free;
                    }
                }

                if (next != current)
                {
                    infeasible++;
                    last = i;
                    lastState = next;
                    if (flipAll)
                    {
                        states[i] = next;
                    }
                }
            }

            if (!flipAll && last >= 0)
            {
                states[last] = lastState;
            }

            return infeasible;
        }

        /// <summary>
        /// Moves the coupled bounds to match the source impulses in <paramref name="z"/>. When the bounds of a free row
        /// shrink to zero width, the row goes to its lower bound. When a row that had equal bounds gets room to move,
        /// it becomes free. Returns true if any bound moved by more than the tolerance, which means that the caller has
        /// to solve again.
        /// </summary>
        static bool Couple(BoundCoupling* couplings, int couplingCount, double* problemLower,
            double* problemUpper, double* lower, double* upper, byte* states, double* z)
        {
            bool moved = false;
            for (int c = 0; c < couplingCount; c++)
            {
                BoundCoupling coupling = couplings[c];
                double sum = coupling.HeldSourceImpulse;
                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    sum += z[row];
                }

                double bound = math.max(sum, 0.0) * coupling.Factor;
                double tolerance = math.max(Tolerance, CouplingTolerance * bound);
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    double newLower = problemLower[row] - bound;
                    double newUpper = problemUpper[row] + bound;
                    if (math.abs(lower[row] - newLower) <= tolerance && math.abs(upper[row] - newUpper) <= tolerance)
                    {
                        continue;
                    }

                    bool wasFixed = lower[row] == upper[row];
                    lower[row] = newLower;
                    upper[row] = newUpper;
                    moved = true;
                    if (bound == 0.0 && states[row] == Free)
                    {
                        states[row] = LowerTight;
                    }
                    else if (wasFixed && bound > 0.0)
                    {
                        states[row] = Free;
                    }
                }
            }

            return moved;
        }

        /// <summary>
        /// Sets the bounds of every coupled row from the source impulses in <paramref name="z"/>, with no tolerance and
        /// without changing any state.
        /// </summary>
        static void SetCoupledBounds(BoundCoupling* couplings, int couplingCount, double* problemLower, double* problemUpper,
            double* lower, double* upper, double* z)
        {
            for (int c = 0; c < couplingCount; c++)
            {
                BoundCoupling coupling = couplings[c];
                double sum = coupling.HeldSourceImpulse;
                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    sum += z[row];
                }

                double bound = math.max(sum, 0.0) * coupling.Factor;
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    lower[row] = problemLower[row] - bound;
                    upper[row] = problemUpper[row] + bound;
                }
            }
        }

        /// <summary>
        /// Moves the coupled bounds to the source impulses of the answer in <paramref name="z"/>, and brings each coupled
        /// row inside its new bounds. A coupled row that has to move to a bound takes the state of that bound, so that
        /// the states still describe the answer.
        /// </summary>
        static void FitCoupledRows(BoundCoupling* couplings, int couplingCount, double* problemLower, double* problemUpper,
            double* lower, double* upper, byte* states, double* z)
        {
            SetCoupledBounds(couplings, couplingCount, problemLower, problemUpper, lower, upper, z);
            for (int c = 0; c < couplingCount; c++)
            {
                BoundCoupling coupling = couplings[c];
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    if (lower[row] == upper[row] || z[row] <= lower[row])
                    {
                        z[row] = lower[row];
                        states[row] = LowerTight;
                    }
                    else if (z[row] >= upper[row])
                    {
                        z[row] = upper[row];
                        states[row] = UpperTight;
                    }
                    else if (states[row] != Free)
                    {
                        // A row that sat at a bound which has moved away from it stays at that bound.
                        z[row] = states[row] == LowerTight ? lower[row] : upper[row];
                    }
                }
            }
        }

        /// <summary>
        /// The number of projected Gauss-Seidel sweeps that finish a solve that ran out of rounds before it found an
        /// answer that meets every condition. One sweep costs about as much as one check of the conditions, which is far
        /// less than one round of pivoting.
        /// </summary>
        const int FinishingSweeps = 10;

        /// <summary>
        /// Finishes a solve that ran out of rounds without an answer. The routine starts from the impulses of the last
        /// round, clamped to their bounds, or from zero clamped to the bounds if that has the lower value of
        /// 1/2 z^T W z - q^T z. Then it makes <see cref="FinishingSweeps"/> sweeps of projected Gauss-Seidel over the
        /// rows. Each sweep first moves the coupled bounds to the current source impulses. No sweep can raise that value
        /// for fixed bounds. At the end, the routine brings the coupled rows inside the bounds of the final source
        /// impulses, and it sets each row's state from where the row's impulse sits in its bounds.
        /// </summary>
        static void FinishBySweeps(int n, double* w, double* q, BoundCoupling* couplings, int couplingCount,
            double* problemLower, double* problemUpper, double* lower, double* upper, byte* states, double* z)
        {
            var startArray = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            double* start = (double*)startArray.GetUnsafePtr();
            for (int i = 0; i < n; i++)
            {
                z[i] = math.clamp(z[i], lower[i], upper[i]);
                start[i] = math.clamp(0.0, lower[i], upper[i]);
            }

            if (Energy(n, w, q, start) < Energy(n, w, q, z))
            {
                for (int i = 0; i < n; i++)
                {
                    z[i] = start[i];
                }
            }

            startArray.Dispose();
            for (int sweep = 0; sweep < FinishingSweeps; sweep++)
            {
                SetCoupledBounds(couplings, couplingCount, problemLower, problemUpper, lower, upper, z);
                for (int i = 0; i < n; i++)
                {
                    double* wi = w + (long)i * n;
                    if (lower[i] == upper[i] || wi[i] <= 0.0)
                    {
                        z[i] = math.clamp(z[i], lower[i], upper[i]);
                        continue;
                    }

                    double residual = q[i] - Blas.Dot(n, wi, z);
                    z[i] = math.clamp(z[i] + residual / wi[i], lower[i], upper[i]);
                }
            }

            SetCoupledBounds(couplings, couplingCount, problemLower, problemUpper, lower, upper, z);
            for (int i = 0; i < n; i++)
            {
                z[i] = math.clamp(z[i], lower[i], upper[i]);
                states[i] = lower[i] == upper[i] || z[i] <= lower[i] ? LowerTight : z[i] >= upper[i] ? UpperTight : Free;
            }
        }

        /// <summary>Returns 1/2 z^T W z - q^T z.</summary>
        static double Energy(int n, double* w, double* q, double* z)
        {
            double energy = 0.0;
            for (int i = 0; i < n; i++)
            {
                energy += z[i] * (0.5 * Blas.Dot(n, w + (long)i * n, z) - q[i]);
            }

            return energy;
        }
    }
}
