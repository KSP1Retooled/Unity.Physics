using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Numerics.Linear.Solvers.Lowlevel
{
    /// <summary>
    /// Solve bounded complementarity problems by block principal pivoting on a dense symmetric positive definite W.
    /// Use row-major storage. Free rows satisfy W*z - q = 0; bound rows require the corresponding residual sign.
    /// BoundCoupling describes dependent bounds such as contact friction.
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
        /// Solve W*z - q complementarity conditions with optional bounds.
        /// states supplies and receives row states: 0 free, 1 lower, 2 upper. If bounded is false, treat every row
        /// as free.
        /// Return whether a feasible answer was found. Otherwise finish with bounded projected Gauss-Seidel
        /// impulses.
        /// </summary>
        public static bool Solve(int n, double* w, double* q, double* problemLower, double* problemUpper, byte* states,
            BoundCoupling* couplings, int couplingCount, bool bounded, double* z)
        {
            return Solve(n, w, q, problemLower, problemUpper, states, couplings, couplingCount, bounded, z,
                MaxPivotingIterations, false, out _);
        }

        /// <summary>
        /// Solve with at most maxRounds free-set solves and return the round count in rounds.
        /// If capped, retain the last feasible answer and return true, or finish by projected Gauss-Seidel and
        /// return false if none exists.
        /// Fit coupled rows to the returned answer's source impulses. When warm is true, honor initial coupled-row
        /// states too.
        /// </summary>
        public static bool Solve(int n, double* w, double* q, double* problemLower, double* problemUpper, byte* states,
            BoundCoupling* couplings, int couplingCount, bool bounded, double* z, int maxRounds, bool warm, out int rounds)
        {
            return Solve(n, w, q, problemLower, problemUpper, states, couplings, couplingCount, bounded, z, maxRounds, warm, true,
                null, out rounds);
        }

        /// <summary>
        /// Solve with incremental Cholesky updates when update is true; otherwise refactor W_FF every round.
        /// Accumulate counters in stats when supplied. Results agree within rounding.
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
        /// Solve using the selected factor update strategy.
        /// If priority is supplied, order the first free factor by descending priority, breaking ties by row order.
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
                // Fit coupled bounds to the returned answer's own source impulses, which may differ from the
                // last coupling round.
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
                // No feasible answer was found before the cap. Finish with bounded projected Gauss-Seidel
                // sweeps and return false.
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
        /// Persistent column-major Cholesky factor of W_FF, with leading dimension n.
        /// Append newly free rows and update according to FactorUpdate; reuse unchanged factors.
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
                // Maintain q - W_FT*z_T incrementally from changes in bound impulses, using symmetric W's
                // contiguous rows.
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
            /// Append free rows in descending priority, preserving row order for ties. Return the next unused
            /// position.
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
            /// Append count rows from order[m] to the factor and advance m.
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
            /// Delete rows leaving the free set and append newly free rows. Preserve the order of retained rows.
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
        /// Count rows violating bounds or residual signs, ignoring fixed rows.
        /// Change every violating state if flipAll is true, otherwise only the last.
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
        /// Update coupled bounds from z. Fix zero-width free rows at their lower bound and free rows whose fixed
        /// bounds open.
        /// Return whether a bound moved beyond tolerance and another solve is needed.
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
        /// Fit coupled rows to bounds derived from the answer's source impulses and update their states.
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
        /// Projected Gauss-Seidel sweeps used when pivoting reaches its cap without a feasible answer.
        /// </summary>
        const int FinishingSweeps = 10;

        /// <summary>
        /// Finish an unconverged solve with projected Gauss-Seidel.
        /// Start from the clamped last iterate or clamped zero, whichever has the lower quadratic objective.
        /// Update coupled bounds each sweep and fit final impulses and states to their own source impulses.
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
