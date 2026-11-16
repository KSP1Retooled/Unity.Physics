using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics.Numerics.Retooled
{
    /// <summary>
    /// Solves the boundary block of a forest solve. The block is a dense matrix W = J_b A^-1 J_b^T + E_b, which is
    /// symmetric and positive definite. Its unknowns have bounds, and some of those bounds depend on other unknowns,
    /// for example friction bounds that depend on a normal impulse. The solver uses the same method as the sparse
    /// constraint solver, which is block principal pivoting. Each step flips every row that breaks its conditions. If
    /// the number of such rows stops going down, the solver flips only the last such row instead (Murty's rule). The
    /// coupled bounds lag behind: after each feasible answer, the solver moves the coupled bounds and solves again.
    /// Each pivoting step solves for the free rows with a dense Cholesky factorization in double precision.
    /// </summary>
    public static unsafe class BoundarySolver
    {
        const double Tolerance = 1e-4;
        const int MaxPivotingIterations = 200;
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
            var lowerArray = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var upperArray = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var coupledArray = new NativeArray<bool>(n, Allocator.Temp);
            var factor = new NativeArray<double>(n * n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var rhs = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var free = new NativeArray<int>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
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
                SolveFree(n, w, q, lower, upper, states, z, (double*)factor.GetUnsafePtr(), (double*)rhs.GetUnsafePtr(),
                    (int*)free.GetUnsafePtr());
                if (!bounded)
                {
                    result = true;
                    break;
                }

                int infeasible = Pivot(n, w, q, lower, upper, states, z, flipAll);
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

            lowerArray.Dispose();
            upperArray.Dispose();
            coupledArray.Dispose();
            factor.Dispose();
            rhs.Dispose();
            free.Dispose();
            bestArray.Dispose();
            guessArray.Dispose();
            bestStatesArray.Dispose();
            return result;
        }

        /// <summary>
        /// Sets each row that is at a bound to that bound. Then solves W_FF z_F = q_F - W_FT z_T for the free rows,
        /// where F is the set of free rows and T is the set of rows at a bound.
        /// </summary>
        static void SolveFree(int n, double* w, double* q, double* lower, double* upper, byte* states, double* z,
            double* factor, double* rhs, int* free)
        {
            int m = 0;
            for (int i = 0; i < n; i++)
            {
                if (states[i] == Free)
                {
                    free[m++] = i;
                }
                else
                {
                    z[i] = states[i] == LowerTight ? lower[i] : upper[i];
                }
            }

            for (int a = 0; a < m; a++)
            {
                int i = free[a];
                double sum = q[i];
                double* wi = w + (long)i * n;
                for (int t = 0; t < n; t++)
                {
                    if (states[t] != Free)
                    {
                        sum -= wi[t] * z[t];
                    }
                }

                rhs[a] = sum;
                for (int b = 0; b <= a; b++)
                {
                    factor[a * m + b] = wi[free[b]];
                }
            }

            // Factor W_FF in place as L L^T, where L is lower triangular. Then solve with L and with L^T.
            for (int a = 0; a < m; a++)
            {
                double* ra = factor + a * m;
                for (int b = 0; b <= a; b++)
                {
                    double* rb = factor + b * m;
                    double sum = ra[b] - Dot(ra, rb, b);

                    if (a == b)
                    {
                        ra[a] = math.sqrt(math.max(sum, 1e-300));
                    }
                    else
                    {
                        ra[b] = sum / rb[b];
                    }
                }
            }

            for (int a = 0; a < m; a++)
            {
                double sum = rhs[a] - Dot(factor + a * m, rhs, a);
                rhs[a] = sum / factor[a * m + a];
            }

            for (int a = m - 1; a >= 0; a--)
            {
                double sum = rhs[a];
                for (int k = a + 1; k < m; k++)
                {
                    sum -= factor[k * m + a] * rhs[k];
                }

                rhs[a] = sum / factor[a * m + a];
            }

            for (int a = 0; a < m; a++)
            {
                z[free[a]] = rhs[a];
            }
        }

        /// <summary>
        /// Returns the dot product of the first <paramref name="count"/> entries of two arrays. It keeps four separate
        /// sums so that the compiler can use vector instructions, which it does not do for a single running sum.
        /// </summary>
        static double Dot(double* a, double* b, int count)
        {
            double4 sums = 0.0;
            int k = 0;
            for (; k + 4 <= count; k += 4)
            {
                sums += *(double4*)(a + k) * *(double4*)(b + k);
            }

            double sum = (sums.x + sums.y) + (sums.z + sums.w);
            for (; k < count; k++)
            {
                sum += a[k] * b[k];
            }

            return sum;
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
                    double slack = Dot(w + (long)i * n, z, n) - q[i];

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

                    double residual = q[i] - Dot(wi, z, n);
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
                energy += z[i] * (0.5 * Dot(w + (long)i * n, z, n) - q[i]);
            }

            return energy;
        }
    }
}
