using NUnit.Framework;
using Unity.Mathematics;
using Unity.Numerics.Linear.Solvers.Lowlevel;

namespace Unity.Numerics.Linear.Solvers.Lowlevel.Tests
{
    /// <summary>
    /// Test boundary block pivoting, round limits and factor updates on coupled contact problems.
    /// Use W = J*M^-1*J^T plus compliance, with normal rows and friction bounds coupled to normal impulses.
    /// </summary>
    public unsafe class BlockPrincipalPivotingTests
    {
        private const int Contacts = 12;
        private const int RowsPerContact = 5;
        private const int Rows = Contacts * RowsPerContact;
        private const int Dofs = 18;
        private const double Friction = 0.6;
        private const byte LowerTight = 1;
        private const byte UpperTight = 2;

        private struct Problem
        {
            public double[] W;
            public double[] Q;
            public double[] Lower;
            public double[] Upper;
            public BoundCoupling[] Couplings;
        }

        private static Problem Make(uint seed)
        {
            var random = new Unity.Mathematics.Random(seed);
            var jacobian = new double[Rows * Dofs];
            for (int k = 0; k < jacobian.Length; k++)
            {
                jacobian[k] = random.NextDouble(-1.0, 1.0);
            }

            var problem = new Problem
            {
                W = new double[Rows * Rows],
                Q = new double[Rows],
                Lower = new double[Rows],
                Upper = new double[Rows],
                Couplings = new BoundCoupling[Contacts],
            };

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Rows; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < Dofs; k++)
                    {
                        sum += jacobian[i * Dofs + k] * jacobian[j * Dofs + k];
                    }

                    problem.W[i * Rows + j] = sum + (i == j ? 1e-3 : 0.0);
                }
            }

            for (int c = 0; c < Contacts; c++)
            {
                int first = c * RowsPerContact;
                for (int k = 0; k < RowsPerContact; k++)
                {
                    int row = first + k;
                    bool normal = k < 2;
                    problem.Q[row] = normal ? random.NextDouble(-1.0, 3.0) : random.NextDouble(-2.0, 2.0);
                    problem.Lower[row] = 0.0;
                    problem.Upper[row] = normal ? double.PositiveInfinity : 0.0;
                }

                problem.Couplings[c] = new BoundCoupling
                {
                    FirstRow = first + 2,
                    RowCount = 3,
                    FirstSourceRow = first,
                    SourceRowCount = 2,
                    Factor = (float)Friction,
                };
            }

            return problem;
        }

        /// <summary>Solves a copy of the problem cold with a cap on rounds, and returns the answer and its states.</summary>
        private static bool Solve(in Problem problem, int maxRounds, out double[] z, out byte[] states, out int rounds)
        {
            z = new double[Rows];
            states = new byte[Rows];
            var q = (double[])problem.Q.Clone();
            var lower = (double[])problem.Lower.Clone();
            var upper = (double[])problem.Upper.Clone();
            fixed (double* w = problem.W, qp = q, lp = lower, up = upper, zp = z)
            fixed (byte* sp = states)
            fixed (BoundCoupling* cp = problem.Couplings)
            {
                return BlockPrincipalPivoting.Solve(Rows, w, qp, lp, up, sp, cp, Contacts, true, zp, maxRounds, false, out rounds);
            }
        }

        /// <summary>
        /// Solve a copy from cold or supplied states, with incremental factor updates or full refactorization.
        /// </summary>
        private static bool Solve(in Problem problem, byte[] start, bool update, out double[] z, out byte[] states,
            out int rounds, out BlockPrincipalPivotingStats stats)
        {
            z = new double[Rows];
            states = start != null ? (byte[])start.Clone() : new byte[Rows];
            var q = (double[])problem.Q.Clone();
            var lower = (double[])problem.Lower.Clone();
            var upper = (double[])problem.Upper.Clone();
            stats = default;
            fixed (double* w = problem.W, qp = q, lp = lower, up = upper, zp = z)
            fixed (byte* sp = states)
            fixed (BoundCoupling* cp = problem.Couplings)
            fixed (BlockPrincipalPivotingStats* statsPointer = &stats)
            {
                return BlockPrincipalPivoting.Solve(Rows, w, qp, lp, up, sp, cp, Contacts, true, zp, 200, start != null, update,
                    statsPointer, out rounds);
            }
        }

        /// <summary>
        /// Compare incremental factor updates with fresh factorization for cold and warm starts.
        /// Require matching rounds, states and impulses within rounding tolerance.
        /// </summary>
        [Test]
        public void ChangingTheFactorGivesTheSameAnswerAsFactoringAgain()
        {
            long updates = 0, reuses = 0, factorizations = 0;
            double largest = 0.0;
            for (uint seed = 1; seed <= 60; seed++)
            {
                Problem problem = Make(seed);
                Solve(problem, null, false, out _, out byte[] nearby, out _, out _);
                Problem next = problem;
                next.Q = (double[])problem.Q.Clone();
                var random = new Unity.Mathematics.Random(seed + 1000);
                for (int i = 0; i < Rows; i++)
                {
                    next.Q[i] += random.NextDouble(-0.3, 0.3);
                }

                for (int pass = 0; pass < 2; pass++)
                {
                    byte[] start = pass == 0 ? null : nearby;
                    bool solvedAgain = Solve(next, start, false, out double[] again, out byte[] againStates, out int againRounds, out _);
                    bool solvedChanged = Solve(next, start, true, out double[] changed, out byte[] changedStates, out int changedRounds,
                        out BlockPrincipalPivotingStats stats);
                    string what = $"seed {seed}, {(pass == 0 ? "cold" : "warm")}";
                    Assert.That(solvedChanged, Is.EqualTo(solvedAgain), what);
                    Assert.That(changedRounds, Is.EqualTo(againRounds), $"{what}: rounds");
                    Assert.That(changedStates, Is.EqualTo(againStates), $"{what}: states");
                    double scale = 1.0;
                    for (int i = 0; i < Rows; i++)
                    {
                        scale = math.max(scale, math.abs(again[i]));
                    }

                    for (int i = 0; i < Rows; i++)
                    {
                        largest = math.max(largest, math.abs(changed[i] - again[i]) / scale);
                        Assert.That(changed[i], Is.EqualTo(again[i]).Within(1e-9 * scale), $"{what}: row {i}");
                    }

                    updates += stats.Updates;
                    reuses += stats.Reuses;
                    factorizations += stats.Factorizations;
                    Assert.That(stats.Factorizations + stats.Updates + stats.Reuses, Is.EqualTo(changedRounds), $"{what}: every round is counted");
                }
            }

            UnityEngine.Debug.Log($"[BlockPrincipalPivotingTests] factor changes: {factorizations} rounds factored, {updates} changed, " +
                $"{reuses} reused; largest difference {largest:E2} of the largest impulse");
            Assert.That(updates, Is.GreaterThan(0), "some rounds changed the factor");
            Assert.That(reuses, Is.GreaterThan(0), "some rounds reused the factor");
        }

        /// <summary>
        /// Verify that factor ordering and prefix reuse preserve pivot rounds, states and impulses within rounding
        /// tolerance.
        /// </summary>
        [Test]
        public void TheFactorsOrderAndItsUpdatesDoNotChangeTheAnswer()
        {
            for (uint seed = 1; seed <= 40; seed++)
            {
                Problem problem = Make(seed);
                Solve(problem, null, false, out _, out byte[] nearby, out _, out _);
                Problem next = problem;
                next.Q = (double[])problem.Q.Clone();
                var random = new Unity.Mathematics.Random(seed + 2000);
                var priority = new byte[Rows];
                for (int i = 0; i < Rows; i++)
                {
                    next.Q[i] += random.NextDouble(-0.3, 0.3);
                    priority[i] = (byte)random.NextInt(0, 64);
                }

                Solve(next, nearby, FactorUpdate.Edit, null, out double[] plain, out byte[] plainStates, out int plainRounds);
                Solve(next, nearby, FactorUpdate.Edit, priority, out double[] ordered, out byte[] orderedStates, out int orderedRounds);
                Solve(next, nearby, FactorUpdate.KeepPrefix, null, out double[] prefix, out byte[] prefixStates, out int prefixRounds);
                double scale = 1.0;
                for (int i = 0; i < Rows; i++)
                {
                    scale = math.max(scale, math.abs(plain[i]));
                }

                Assert.That(orderedRounds, Is.EqualTo(plainRounds), $"seed {seed}: rounds with the priority");
                Assert.That(prefixRounds, Is.EqualTo(plainRounds), $"seed {seed}: rounds when the factor keeps its prefix");
                Assert.That(orderedStates, Is.EqualTo(plainStates), $"seed {seed}: states with the priority");
                Assert.That(prefixStates, Is.EqualTo(plainStates), $"seed {seed}: states when the factor keeps its prefix");
                for (int i = 0; i < Rows; i++)
                {
                    Assert.That(ordered[i], Is.EqualTo(plain[i]).Within(1e-9 * scale), $"seed {seed}: row {i} with the priority");
                    Assert.That(prefix[i], Is.EqualTo(plain[i]).Within(1e-9 * scale), $"seed {seed}: row {i} when the factor keeps its prefix");
                }
            }
        }

        private static void Solve(in Problem problem, byte[] start, FactorUpdate update, byte[] priority, out double[] z,
            out byte[] states, out int rounds)
        {
            z = new double[Rows];
            states = (byte[])start.Clone();
            var q = (double[])problem.Q.Clone();
            var lower = (double[])problem.Lower.Clone();
            var upper = (double[])problem.Upper.Clone();
            priority ??= new byte[Rows];
            fixed (double* w = problem.W, qp = q, lp = lower, up = upper, zp = z)
            fixed (byte* sp = states, pp = priority)
            fixed (BoundCoupling* cp = problem.Couplings)
            {
                BlockPrincipalPivoting.Solve(Rows, w, qp, lp, up, sp, cp, Contacts, true, zp, 200, true, update, pp, null, out rounds);
            }
        }

        /// <summary>The friction bound of contact c for the normal impulses in z.</summary>
        private static double Bound(double[] z, int c)
        {
            int first = c * RowsPerContact;
            return (float)Friction * math.max(z[first] + z[first + 1], 0.0);
        }

        /// <summary>
        /// Finds a seed whose problem needs more than 8 rounds, and fewer than 100, when it is solved cold without a cap.
        /// </summary>
        private static uint HardSeed(System.Func<Problem, bool> wanted)
        {
            for (uint seed = 1; seed < 400; seed++)
            {
                Problem problem = Make(seed);
                bool solved = Solve(problem, 200, out _, out _, out int rounds);
                if (solved && rounds > 8 && rounds < 100 && wanted(problem))
                {
                    return seed;
                }
            }

            return 0;
        }

        /// <summary>
        /// Check impulse bounds against the answer's own normal impulses and verify matching row states.
        /// </summary>
        private static void CheckAnswer(in Problem problem, double[] z, byte[] states, string what)
        {
            const double tolerance = 1e-6;
            for (int c = 0; c < Contacts; c++)
            {
                int first = c * RowsPerContact;
                double bound = Bound(z, c);
                for (int k = 0; k < RowsPerContact; k++)
                {
                    int row = first + k;
                    bool normal = k < 2;
                    double lower = normal ? problem.Lower[row] : problem.Lower[row] - bound;
                    double upper = normal ? problem.Upper[row] : problem.Upper[row] + bound;
                    Assert.That(z[row], Is.GreaterThanOrEqualTo(lower - tolerance), $"{what}: row {row} is above its lower bound");
                    Assert.That(z[row], Is.LessThanOrEqualTo(upper + tolerance), $"{what}: row {row} is below its upper bound");
                    if (states[row] == LowerTight)
                    {
                        Assert.That(z[row], Is.EqualTo(lower).Within(tolerance), $"{what}: row {row} is at its lower bound, as its state says");
                    }
                    else if (states[row] == UpperTight)
                    {
                        Assert.That(z[row], Is.EqualTo(upper).Within(tolerance), $"{what}: row {row} is at its upper bound, as its state says");
                    }
                }
            }
        }

        /// <summary>The value of 1/2 z^T W z - q^T z, which the answer makes as small as its bounds allow.</summary>
        private static double Energy(in Problem problem, double[] z)
        {
            double energy = 0.0;
            for (int i = 0; i < Rows; i++)
            {
                double wz = 0.0;
                for (int j = 0; j < Rows; j++)
                {
                    wz += problem.W[i * Rows + j] * z[j];
                }

                energy += z[i] * (0.5 * wz - problem.Q[i]);
            }

            return energy;
        }

        /// <summary>
        /// Verify that an eight-round cap reports failure but returns feasible impulses with a negative objective,
        /// rather than excessive energy from simply clamping the last iterate.
        /// </summary>
        [Test]
        public void ASolveThatRunsOutOfRoundsStillReturnsImpulsesInsideTheirBounds()
        {
            uint seed = HardSeed(problem => !Solve(problem, 8, out _, out _, out _));
            Assert.That(seed, Is.Not.EqualTo(0u), "some problem needs more than 8 rounds and has no answer after 8");
            Problem hard = Make(seed);
            Assert.That(Solve(hard, 200, out double[] exact, out _, out int needed), Is.True, "the solve without a cap finds the answer");
            Assert.That(Solve(hard, 8, out double[] z, out byte[] states, out int rounds), Is.False, "the capped solve says that it found no answer");
            Assert.That(rounds, Is.EqualTo(8));
            CheckAnswer(hard, z, states, "capped");

            double best = Energy(hard, exact), capped = Energy(hard, z);
            UnityEngine.Debug.Log($"[BlockPrincipalPivotingTests] seed {seed}: {needed} rounds without a cap; 1/2 z^T W z - q^T z is " +
                $"{best:F4} for the answer and {capped:F4} after 8 rounds");
            Assert.That(capped, Is.LessThan(0.0), "the capped impulses do better than no impulses");
        }

        /// <summary>
        /// Verify that exhausting friction-bound rounds retains the last feasible answer and its states.
        /// </summary>
        [Test]
        public void AnAnswerReturnedAtTheCapKeepsItsFrictionInsideItsOwnBounds()
        {
            uint seed = HardSeed(problem => Solve(problem, 8, out _, out _, out int rounds) && rounds == 8);
            Assert.That(seed, Is.Not.EqualTo(0u), "some problem finds an answer but is still moving its bounds after 8 rounds");
            Problem hard = Make(seed);
            Assert.That(Solve(hard, 8, out double[] z, out byte[] states, out _), Is.True);
            CheckAnswer(hard, z, states, "capped");
        }
    }
}
