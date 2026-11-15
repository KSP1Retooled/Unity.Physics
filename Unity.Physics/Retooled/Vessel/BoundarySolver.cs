using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// The boundary block of a forest solve: a dense, symmetric positive definite W (J_b A^-1 J_b^T + E_b) with bounds on
    /// its unknowns and friction couplings, solved as <see cref="SparseConstraintSolver"/> solves its rows: block principal
    /// pivoting, flipping every infeasible row and falling back to Murty's rule when that stalls, and lagged coupled bounds
    /// that each feasible answer moves. Each pivoting step solves the free rows by a dense Cholesky factorization, in double.
    /// </summary>
    internal static unsafe class BoundarySolver
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
        /// Solves for <paramref name="z"/>, with W z - q = 0 on free rows and the sign conditions of a bounded row at its
        /// bound. <paramref name="states"/> holds a first guess at which rows are tight (the values of
        /// <see cref="LCP.MLCPIndexFlag"/>) and is left with the answer's. Returns false if pivoting ran out of iterations,
        /// leaving z clamped to its bounds.
        /// </summary>
        public static bool Solve(int n, double* w, double* q, double* problemLower, double* problemUpper, byte* states,
            SparseConstraintSolver.Coupling* couplings, int couplingCount, bool bounded, double* z)
        {
            var lowerArray = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var upperArray = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var coupledArray = new NativeArray<bool>(n, Allocator.Temp);
            var factor = new NativeArray<double>(n * n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var rhs = new NativeArray<double>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var free = new NativeArray<int>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            double* lower = (double*)lowerArray.GetUnsafePtr();
            double* upper = (double*)upperArray.GetUnsafePtr();
            bool* coupled = (bool*)coupledArray.GetUnsafePtr();
            for (int i = 0; i < n; i++)
            {
                lower[i] = problemLower[i];
                upper[i] = problemUpper[i];
            }

            for (int c = 0; bounded && c < couplingCount; c++)
            {
                for (int row = couplings[c].FirstRow; row < couplings[c].FirstRow + couplings[c].RowCount; row++)
                {
                    coupled[row] = true;
                }
            }

            for (int i = 0; i < n; i++)
            {
                byte state = bounded ? states[i] : Free;
                if (bounded && lower[i] == upper[i])
                {
                    state = LowerTight;
                }
                else if (coupled[i] || state > UpperTight)
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
            for (int iteration = 1; iteration <= MaxPivotingIterations; iteration++)
            {
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

            if (!result)
            {
                for (int i = 0; i < n; i++)
                {
                    z[i] = math.clamp(z[i], lower[i], upper[i]);
                }
            }

            lowerArray.Dispose();
            upperArray.Dispose();
            coupledArray.Dispose();
            factor.Dispose();
            rhs.Dispose();
            free.Dispose();
            return result;
        }

        /// <summary>Tight rows at their bounds, and the free rows from W_FF z_F = q_F - W_FT z_T.</summary>
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

            // Lower Cholesky in place, then the two triangular solves.
            for (int a = 0; a < m; a++)
            {
                double* ra = factor + a * m;
                for (int b = 0; b <= a; b++)
                {
                    double* rb = factor + b * m;
                    double sum = ra[b];
                    for (int k = 0; k < b; k++)
                    {
                        sum -= ra[k] * rb[k];
                    }

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
                double sum = rhs[a];
                for (int k = 0; k < a; k++)
                {
                    sum -= factor[a * m + k] * rhs[k];
                }

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
                    double slack = -q[i];
                    double* wi = w + (long)i * n;
                    for (int j = 0; j < n; j++)
                    {
                        slack += wi[j] * z[j];
                    }

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

        static bool Couple(SparseConstraintSolver.Coupling* couplings, int couplingCount, double* problemLower,
            double* problemUpper, double* lower, double* upper, byte* states, double* z)
        {
            bool moved = false;
            for (int c = 0; c < couplingCount; c++)
            {
                SparseConstraintSolver.Coupling coupling = couplings[c];
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
    }
}
