using NUnit.Framework;
using Unity.Mathematics;
using Unity.Physics.Numerics.Retooled;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The dense solve of a vessel island's boundary block when it runs out of pivoting rounds. Each problem looks like
    /// a set of contacts between a few bodies: every contact has two normal rows, which may only push, and three
    /// friction rows, whose bounds are the coefficient of friction times the sum of the contact's normal impulses. The
    /// matrix is J M^-1 J^T plus a small compliance, with random Jacobians, so that the rows are strongly coupled and
    /// a cold solve needs many rounds.
    /// </summary>
    public unsafe class BoundarySolverTests
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
            var random = new Random(seed);
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
                return BoundarySolver.Solve(Rows, w, qp, lp, up, sp, cp, Contacts, true, zp, maxRounds, false, out rounds);
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
        /// Checks that every impulse is inside its bounds, where the friction bounds come from the answer's own normal
        /// impulses, and that each row's state matches its impulse.
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
        /// A cold solve that needs more than 8 rounds, with a cap of 8 rounds, finds no answer that meets every
        /// condition. The solver says so, and it still returns impulses that are inside their bounds. The impulses do
        /// better than no impulses at all: they make 1/2 z^T W z - q^T z negative. Impulses that are only clamped to
        /// their bounds, as the solver returned them before, can make that value large and positive, which means that
        /// they push the bodies much harder than the contacts ask for.
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
            UnityEngine.Debug.Log($"[BoundarySolverTests] seed {seed}: {needed} rounds without a cap; 1/2 z^T W z - q^T z is " +
                $"{best:F4} for the answer and {capped:F4} after 8 rounds");
            Assert.That(capped, Is.LessThan(0.0), "the capped impulses do better than no impulses");
        }

        /// <summary>
        /// A cold solve that finds an answer but runs out of rounds while it moves the friction bounds returns that
        /// answer. The answer's friction stays inside the bounds of the answer's own normal impulses, and the states
        /// that the solve returns, which the island remembers for the next solve, are the states of that answer.
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
