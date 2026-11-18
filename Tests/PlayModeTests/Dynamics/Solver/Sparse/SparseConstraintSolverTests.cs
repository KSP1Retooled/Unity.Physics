using System.Collections.Generic;
using NUnit.Framework;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Numerics.Linear;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Numerics.Memory;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Numerics.Linear.Solvers.Lowlevel;

namespace Unity.Physics.Tests.Dynamics.Solver.Sparse
{
    /// <summary>
    /// Compare sparse solving with the dense complementarity solver and independently check row conditions and
    /// factor sizes.
    /// </summary>
    public unsafe class SparseConstraintSolverTests
    {
        private const int Columns = SparseConstraintSolver.ColumnsPerRow;

        /// <summary>Stores a problem in managed lists and can construct the equivalent dense matrix.</summary>
        private sealed class Scenario
        {
            public readonly List<SparseConstraintSolver.Body> Bodies = new List<SparseConstraintSolver.Body>();
            public readonly List<double> InverseMasses = new List<double>();
            public readonly List<double3x3> InverseInertias = new List<double3x3>();
            public readonly List<SparseConstraintSolver.ConstraintBlock> Blocks = new List<SparseConstraintSolver.ConstraintBlock>();
            public readonly List<float> Jacobian = new List<float>();
            public readonly List<float> Epsilon = new List<float>();
            public readonly List<float> RightHandSide = new List<float>();
            public readonly List<float> Lower = new List<float>();
            public readonly List<float> Upper = new List<float>();
            public readonly List<BoundCoupling> Couplings = new List<BoundCoupling>();

            private Random random;

            public Scenario(uint seed)
            {
                random = new Random(seed);
            }

            public int Rows => Epsilon.Count;

            public int AddBody()
            {
                // Use a wide range of masses and inertias to represent heavy fuel tanks attached to very light struts.
                double mass = math.pow(10.0, random.NextDouble(-1.0, 2.0));
                double3 principal = new double3(
                    math.pow(10.0, random.NextDouble(-1.0, 2.0)),
                    math.pow(10.0, random.NextDouble(-1.0, 2.0)),
                    math.pow(10.0, random.NextDouble(-1.0, 2.0)));
                double3x3 turn = new double3x3(new float3x3(random.NextQuaternionRotation()));
                double3x3 inertia = math.mul(math.mul(turn, Diagonal(principal)), math.transpose(turn));
                double3x3 inverse = math.mul(math.mul(turn, Diagonal(1.0 / principal)), math.transpose(turn));

                Bodies.Add(new SparseConstraintSolver.Body { Mass = mass, Inertia = inertia });
                InverseMasses.Add(1.0 / mass);
                InverseInertias.Add(inverse);
                return Bodies.Count - 1;
            }

            public void AddBlock(int bodyA, int bodyB, int rows, float bound = float.MaxValue)
            {
                Blocks.Add(new SparseConstraintSolver.ConstraintBlock { BodyA = bodyA, BodyB = bodyB, FirstRow = Rows, RowCount = rows });
                for (int row = 0; row < rows; row++)
                {
                    for (int column = 0; column < Columns; column++)
                    {
                        bool inUse = column < Columns / 2 ? bodyA >= 0 : bodyB >= 0;
                        Jacobian.Add(inUse ? random.NextFloat(-1.0f, 1.0f) : 0.0f);
                    }

                    Epsilon.Add(math.pow(10.0f, random.NextFloat(-3.0f, -1.0f)));
                    RightHandSide.Add(random.NextFloat(-5.0f, 5.0f));
                    Lower.Add(bound == float.MaxValue ? float.MinValue : -bound);
                    Upper.Add(bound);
                }
            }

            /// <summary>
            /// Add body-world contact rows: push-only normals and three friction rows bounded by their summed
            /// impulse.
            /// </summary>
            public void AddContact(int body, int normals, float friction)
            {
                int first = Rows;
                AddBlock(-1, body, normals + 3);
                float3 normal = random.NextFloat3Direction();
                Math.CalculatePerpendicularNormalized(normal, out float3 tangent0, out float3 tangent1);
                float3 averageArm = float3.zero;
                for (int row = first; row < first + normals; row++)
                {
                    float3 arm = random.NextFloat3(-1.0f, 1.0f);
                    averageArm += arm / normals;
                    SetSide(row, normal, math.cross(arm, normal));
                    RightHandSide[row] = random.NextFloat(0.0f, 5.0f);
                    Lower[row] = 0.0f;
                    Upper[row] = float.MaxValue;
                }

                SetSide(first + normals, tangent0, math.cross(averageArm, tangent0));
                SetSide(first + normals + 1, tangent1, math.cross(averageArm, tangent1));
                SetSide(first + normals + 2, float3.zero, normal);
                for (int row = first + normals; row < Rows; row++)
                {
                    Lower[row] = 0.0f;
                    Upper[row] = 0.0f;
                }

                Couplings.Add(new BoundCoupling
                {
                    FirstRow = first + normals,
                    RowCount = 3,
                    FirstSourceRow = first,
                    SourceRowCount = normals,
                    Factor = friction,
                });
            }

            private void SetSide(int row, float3 linear, float3 angular)
            {
                int start = row * Columns + Columns / 2;
                for (int i = 0; i < 3; i++)
                {
                    Jacobian[start + i] = linear[i];
                    Jacobian[start + 3 + i] = angular[i];
                }
            }

            /// <summary>
            /// Constructs the dense matrix J M^-1 J^t + E directly from the bodies and Jacobian entries.
            /// </summary>
            public double[,] Matrix()
            {
                double[][] jInverseMass = new double[Rows][];
                int[] sideBody = new int[Rows * 2];
                foreach (SparseConstraintSolver.ConstraintBlock block in Blocks)
                {
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        sideBody[row * 2] = block.BodyA;
                        sideBody[row * 2 + 1] = block.BodyB;
                        jInverseMass[row] = new double[Columns];
                        WriteJInverseMass(row, 0, block.BodyA, jInverseMass[row]);
                        WriteJInverseMass(row, 6, block.BodyB, jInverseMass[row]);
                    }
                }

                double[,] a = new double[Rows, Rows];
                for (int i = 0; i < Rows; i++)
                {
                    for (int j = 0; j < Rows; j++)
                    {
                        double sum = i == j ? Epsilon[i] : 0.0;
                        for (int side = 0; side < 2; side++)
                        {
                            int body = sideBody[i * 2 + side];
                            for (int otherSide = 0; otherSide < 2; otherSide++)
                            {
                                if (body >= 0 && body == sideBody[j * 2 + otherSide])
                                {
                                    for (int column = 0; column < 6; column++)
                                    {
                                        sum += jInverseMass[i][side * 6 + column] * Jacobian[j * Columns + otherSide * 6 + column];
                                    }
                                }
                            }
                        }

                        a[i, j] = sum;
                    }
                }

                return a;
            }

            private static double3x3 Diagonal(double3 values)
            {
                return new double3x3(values.x, 0.0, 0.0, 0.0, values.y, 0.0, 0.0, 0.0, values.z);
            }

            private void WriteJInverseMass(int row, int start, int body, double[] target)
            {
                if (body < 0)
                {
                    return;
                }

                double3 linear = new double3(Jacobian[row * Columns + start], Jacobian[row * Columns + start + 1], Jacobian[row * Columns + start + 2]);
                double3 angular = new double3(Jacobian[row * Columns + start + 3], Jacobian[row * Columns + start + 4], Jacobian[row * Columns + start + 5]);
                double3 scaled = linear * InverseMasses[body];
                double3 turned = math.mul(InverseInertias[body], angular);
                target[start] = scaled.x;
                target[start + 1] = scaled.y;
                target[start + 2] = scaled.z;
                target[start + 3] = turned.x;
                target[start + 4] = turned.y;
                target[start + 5] = turned.z;
            }
        }

        private static float[] Solve(Scenario scenario, bool bounded, out SparseConstraintSolver.Statistics statistics, out bool converged, byte[] guess = null)
        {
            var solver = new SparseConstraintSolver(Allocator.Temp);
            float[] impulses = new float[scenario.Rows];
            converged = Solve(ref solver, scenario, bounded, impulses, guess);
            statistics = solver.Stats;
            solver.Dispose();
            return impulses;
        }

        /// <summary>
        /// Uses all lanes of a parallel job to solve one problem. The first lane to run leads and the others help, as
        /// when solving a large direct island.
        /// </summary>
        private struct SharedSolveJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public WorkTeam* Team;

            [NativeDisableUnsafePtrRestriction]
            public SparseConstraintSolver* Solver;

            [NativeDisableUnsafePtrRestriction]
            public SparseConstraintSolver.Problem* Problem;

            [NativeDisableUnsafePtrRestriction]
            public float* Impulses;

            [NativeDisableUnsafePtrRestriction]
            public bool* Converged;

            public bool Bounded;

            public void Execute(int lane)
            {
                if (!Team->TryLead())
                {
                    Team->Help<SparseConstraintSolver.TaskDispatch>();
                    return;
                }

                Solver->Team = Team;
                *Converged = Solver->Solve(*Problem, Impulses, Bounded);
                Solver->Team = null;
                Team->Finish();
            }
        }

        private static bool Solve(ref SparseConstraintSolver solver, Scenario scenario, bool bounded, float[] impulses, byte[] guess = null,
            int lanes = 0)
        {
            SparseConstraintSolver.Body[] bodies = scenario.Bodies.ToArray();
            SparseConstraintSolver.ConstraintBlock[] blocks = scenario.Blocks.ToArray();
            float[] jacobian = scenario.Jacobian.ToArray();
            float[] epsilon = scenario.Epsilon.ToArray();
            float[] rightHandSide = scenario.RightHandSide.ToArray();
            float[] lower = scenario.Lower.ToArray();
            float[] upper = scenario.Upper.ToArray();
            BoundCoupling[] couplings = scenario.Couplings.ToArray();

            fixed (BoundCoupling* couplingsPointer = couplings)
            fixed (SparseConstraintSolver.Body* bodiesPointer = bodies)
            fixed (SparseConstraintSolver.ConstraintBlock* blocksPointer = blocks)
            fixed (float* jacobianPointer = jacobian, epsilonPointer = epsilon, rightPointer = rightHandSide)
            fixed (float* lowerPointer = lower, upperPointer = upper, output = impulses)
            fixed (byte* guessPointer = guess)
            {
                var problem = new SparseConstraintSolver.Problem
                {
                    BodyCount = bodies.Length,
                    Bodies = bodiesPointer,
                    BlockCount = blocks.Length,
                    Blocks = blocksPointer,
                    RowCount = epsilon.Length,
                    Jacobian = jacobianPointer,
                    Epsilon = epsilonPointer,
                    RightHandSide = rightPointer,
                    Lower = lowerPointer,
                    Upper = upperPointer,
                    InitialStates = guessPointer,
                    CouplingCount = couplings.Length,
                    Couplings = couplingsPointer,
                };

                if (lanes == 0)
                {
                    return solver.Solve(problem, output, bounded);
                }

                WorkTeam team = default;
                team.Configure();
                bool converged = false;
                fixed (SparseConstraintSolver* solverPointer = &solver)
                {
                    new SharedSolveJob
                    {
                        Team = &team,
                        Solver = solverPointer,
                        Problem = &problem,
                        Impulses = output,
                        Converged = &converged,
                        Bounded = bounded,
                    }.Schedule(lanes, 1).Complete();
                }

                return converged;
            }
        }

        /// <summary>Solves the same problem with the dense solver from the package.</summary>
        private static float[] SolveDensely(Scenario scenario)
        {
            int rows = scenario.Rows;
            double[,] a = scenario.Matrix();
            float[] answer = new float[rows];

            var heap = new MemoryManager(Allocator.Temp);
            var matrix = heap.Matrix(rows, rows);
            var q = heap.Vector(rows);
            var lower = heap.Vector(rows);
            var upper = heap.Vector(rows);
            var w = heap.Vector(rows);
            var z = heap.Vector(rows);
            var flags = new NativeArray<LCP.MLCPIndexFlag>(rows, Allocator.Temp);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    matrix.Cols[j][i] = (float)a[i, j];
                }

                q[i] = -scenario.RightHandSide[i];
                lower[i] = scenario.Lower[i];
                upper[i] = scenario.Upper[i];
            }

            Assume.That(LCP.SolveMLCP(matrix, q, lower, upper, ref flags, ref w, ref z), Is.True, "the dense solver gave up on this one");
            for (int i = 0; i < rows; i++)
            {
                answer[i] = z[i];
            }

            flags.Dispose();
            z.Dispose();
            w.Dispose();
            upper.Dispose();
            lower.Dispose();
            q.Dispose();
            matrix.Dispose();
            heap.Dispose();
            return answer;
        }

        /// <summary>
        /// The definition of a solution: w = A z - b is zero on a row strictly inside its bounds, and on a row at a bound it
        /// may only push against that bound.
        /// </summary>
        private static void AssertSolves(Scenario scenario, float[] z, double tolerance)
        {
            double[,] a = scenario.Matrix();
            for (int i = 0; i < scenario.Rows; i++)
            {
                double w = -scenario.RightHandSide[i];
                for (int j = 0; j < scenario.Rows; j++)
                {
                    w += a[i, j] * z[j];
                }

                float lower = scenario.Lower[i];
                float upper = scenario.Upper[i];
                Assert.That(z[i], Is.InRange(lower - tolerance, upper + tolerance), $"row {i} is outside its bounds");
                if (lower == upper)
                {
                    continue;
                }

                if (z[i] > lower + tolerance && z[i] < upper - tolerance)
                {
                    Assert.That(w, Is.EqualTo(0.0).Within(tolerance), $"free row {i} isn't satisfied");
                }
                else if (z[i] <= lower + tolerance)
                {
                    Assert.That(w, Is.GreaterThan(-tolerance), $"row {i} is held at its lower bound for no reason");
                }
                else
                {
                    Assert.That(w, Is.LessThan(tolerance), $"row {i} is held at its upper bound for no reason");
                }
            }
        }

        /// <summary>A random tree of joints, and some struts across it, sometimes from a body to the world.</summary>
        private static Scenario Craft(uint seed, int bodies, int struts, float bound = float.MaxValue)
        {
            Scenario scenario = new Scenario(seed);
            Random shape = new Random(seed * 31 + 7);
            for (int i = 0; i < bodies; i++)
            {
                scenario.AddBody();
                if (i > 0)
                {
                    scenario.AddBlock(shape.NextInt(i), i, shape.NextInt(1, 7), bound);
                }
            }

            for (int i = 0; i < struts; i++)
            {
                int first = shape.NextInt(bodies);
                int second = shape.NextBool() ? (first + 1 + shape.NextInt(bodies - 1)) % bodies : -1;
                scenario.AddBlock(second, first, shape.NextInt(1, 4), bound);
            }

            return scenario;
        }

        [Test]
        public void ItSolvesTheDenseSystem([Range(1u, 12u)] uint seed)
        {
            Scenario scenario = Craft(seed, 12, 5);

            float[] impulses = Solve(scenario, false, out _, out bool converged);

            Assert.That(converged, Is.True);
            AssertSolves(scenario, impulses, 2e-3);
        }

        [Test]
        public void ItMatchesTheDenseSolverOnEveryRow([Range(1u, 12u)] uint seed)
        {
            Scenario scenario = Craft(seed, 10, 4);

            float[] sparse = Solve(scenario, false, out _, out _);
            float[] dense = SolveDensely(scenario);

            Assert.That(sparse, Is.EqualTo(dense).Within(0.02f));
        }

        [Test]
        public void ItSolvesTheBoundedProblem([Range(1u, 12u)] uint seed, [Values(0.05f, 0.5f, 3.0f)] float bound)
        {
            Scenario scenario = Craft(seed, 12, 5, bound);

            float[] impulses = Solve(scenario, true, out SparseConstraintSolver.Statistics statistics, out bool converged);

            AssertSolves(scenario, impulses, 5e-3);
            Assert.That(converged, Is.True);
        }

        [Test]
        public void ItMatchesTheDenseSolverWithBounds([Range(1u, 12u)] uint seed, [Values(0.05f, 0.5f, 3.0f)] float bound)
        {
            Scenario scenario = Craft(seed, 10, 4, bound);

            float[] sparse = Solve(scenario, true, out _, out _);
            float[] dense = SolveDensely(scenario);

            Assert.That(sparse, Is.EqualTo(dense).Within(0.02f));
        }

        [Test]
        public void BoundsThatCannotBindTakeOneFactorization()
        {
            Scenario scenario = Craft(3, 40, 10, 1e12f);

            Solve(scenario, true, out SparseConstraintSolver.Statistics statistics, out bool converged);

            Assert.That(converged, Is.True);
            Assert.That(statistics.PivotingIterations, Is.EqualTo(1));
        }

        [Test]
        public void AFirstGuessDoesNotChangeTheAnswer([Range(1u, 6u)] uint seed)
        {
            Scenario scenario = Craft(seed, 12, 5, 0.5f);
            Random random = new Random(seed + 100);
            byte[] guess = new byte[scenario.Rows];
            for (int i = 0; i < guess.Length; i++)
            {
                guess[i] = (byte)random.NextInt(3);
            }

            float[] unguided = Solve(scenario, true, out _, out _);
            float[] guided = Solve(scenario, true, out _, out bool converged, guess);

            Assert.That(converged, Is.True);
            Assert.That(guided, Is.EqualTo(unguided).Within(1e-3f));
        }

        [Test]
        public void RowsWhoseBoundsMeetCostNoPivoting()
        {
            Scenario scenario = new Scenario(12);
            int previous = scenario.AddBody();
            for (int i = 0; i < 20; i++)
            {
                int body = scenario.AddBody();
                scenario.AddBlock(previous, body, 3);
                scenario.AddBlock(previous, body, 1, 0.0f);
                previous = body;
            }

            float[] impulses = Solve(scenario, true, out SparseConstraintSolver.Statistics statistics, out bool converged);

            Assert.That(converged, Is.True);
            Assert.That(statistics.PivotingIterations, Is.EqualTo(1));
            AssertSolves(scenario, impulses, 2e-3);
        }

        [Test]
        public void FrictionSolvesWithinItsShareOfTheNormalImpulse([Range(1u, 6u)] uint seed, [Values(0.2f, 1.0f)] float friction)
        {
            Scenario scenario = Craft(seed, 12, 3);
            Random contacts = new Random(seed + 50);
            for (int i = 0; i < 4; i++)
            {
                scenario.AddContact(contacts.NextInt(scenario.Bodies.Count), contacts.NextInt(1, 4), friction);
            }

            var solver = new SparseConstraintSolver(Allocator.Temp);
            float[] impulses = new float[scenario.Rows];
            bool converged = Solve(ref solver, scenario, true, impulses);
            int couplings = solver.Stats.CouplingIterations;

            // Each final upper bound must equal the friction coefficient times the sum of the normal impulses.
            // The lower bound is its negative. The resulting impulses must solve the problem with those bounds.
            foreach (BoundCoupling coupling in scenario.Couplings)
            {
                float normal = 0.0f;
                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    normal += impulses[row];
                }

                float share = friction * math.max(normal, 0.0f);
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    float2 bounds = solver.Bounds(row);
                    Assert.That(bounds.y, Is.EqualTo(share).Within(1e-3f * share + 1e-4f), $"row {row} has the wrong bound");
                    scenario.Lower[row] = bounds.x;
                    scenario.Upper[row] = bounds.y;
                }
            }

            solver.Dispose();
            Assert.That(converged, Is.True);
            Assert.That(couplings, Is.LessThan(25), "the friction bounds never settled");
            AssertSolves(scenario, impulses, 2e-3);
        }

        [Test]
        public void FrictionCountsWhatTheContactPushedEarlierInTheStep([Range(1u, 4u)] uint seed)
        {
            const float friction = 0.5f;
            Scenario scenario = Craft(seed, 12, 3);
            Random contacts = new Random(seed + 70);
            for (int i = 0; i < 4; i++)
            {
                scenario.AddContact(contacts.NextInt(scenario.Bodies.Count), contacts.NextInt(1, 4), friction);
            }

            // Give each contact a normal impulse from earlier in the step, with part of the allowed friction impulse
            // already applied.
            for (int i = 0; i < scenario.Couplings.Count; i++)
            {
                BoundCoupling coupling = scenario.Couplings[i];
                coupling.HeldSourceImpulse = contacts.NextFloat(1.0f, 4.0f);
                scenario.Couplings[i] = coupling;
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    float spent = contacts.NextFloat(-0.5f, 0.5f) * friction * coupling.HeldSourceImpulse;
                    scenario.Lower[row] = -spent;
                    scenario.Upper[row] = -spent;
                }
            }

            var solver = new SparseConstraintSolver(Allocator.Temp);
            float[] impulses = new float[scenario.Rows];
            bool converged = Solve(ref solver, scenario, true, impulses);

            // Center each row's bounds on the negative of its previously applied friction impulse. The distance to
            // either bound is the friction coefficient times the total normal impulse, including earlier impulses.
            foreach (BoundCoupling coupling in scenario.Couplings)
            {
                float normal = coupling.HeldSourceImpulse;
                for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
                {
                    normal += impulses[row];
                }

                float share = friction * math.max(normal, 0.0f);
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    float2 bounds = solver.Bounds(row);
                    float tolerance = 1e-3f * share + 1e-4f;
                    Assert.That(bounds.x, Is.EqualTo(scenario.Lower[row] - share).Within(tolerance), $"row {row} has the wrong lower bound");
                    Assert.That(bounds.y, Is.EqualTo(scenario.Upper[row] + share).Within(tolerance), $"row {row} has the wrong upper bound");
                }
            }

            foreach (BoundCoupling coupling in scenario.Couplings)
            {
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    float2 bounds = solver.Bounds(row);
                    scenario.Lower[row] = bounds.x;
                    scenario.Upper[row] = bounds.y;
                }
            }

            solver.Dispose();
            Assert.That(converged, Is.True);
            AssertSolves(scenario, impulses, 2e-3);
        }

        [Test]
        public void LanesSharingASolveGetTheSameAnswer([Range(1u, 4u)] uint seed, [Values(2, 8)] int lanes)
        {
            Scenario scenario = Craft(seed, 600, 20, 0.5f);
            Random contacts = new Random(seed + 70);
            for (int i = 0; i < 10; i++)
            {
                scenario.AddContact(contacts.NextInt(scenario.Bodies.Count), contacts.NextInt(1, 4), 0.8f);
            }

            float[] alone = Solve(scenario, true, out SparseConstraintSolver.Statistics statistics, out _);
            var solver = new SparseConstraintSolver(Allocator.TempJob);
            float[] shared = new float[scenario.Rows];
            Solve(ref solver, scenario, true, shared, null, lanes);
            solver.Dispose();

            Assert.That(statistics.Tasks, Is.GreaterThan(lanes), "the craft didn't split into subtrees");
            Assert.That(shared, Is.EqualTo(alone));
        }

        [Test]
        public void ASolverCarriesNothingBetweenSolves()
        {
            Scenario bounded = Craft(5, 12, 5, 0.05f);
            Scenario free = Craft(6, 12, 5);
            var solver = new SparseConstraintSolver(Allocator.Temp);
            float[] first = new float[bounded.Rows];
            float[] second = new float[free.Rows];

            Solve(ref solver, bounded, true, first);
            Solve(ref solver, free, false, second);
            solver.Dispose();

            AssertSolves(free, second, 2e-3);
        }

        [Test]
        public void RowsCanComeFromEitherBodyFirst()
        {
            Scenario scenario = new Scenario(9);
            int a = scenario.AddBody();
            int b = scenario.AddBody();
            scenario.AddBlock(a, b, 3);
            scenario.AddBlock(b, a, 2);
            scenario.AddBlock(a, b, 1);

            float[] impulses = Solve(scenario, false, out SparseConstraintSolver.Statistics statistics, out _);

            Assert.That(statistics.Nodes, Is.EqualTo(3), "two bodies and one pair");
            AssertSolves(scenario, impulses, 2e-3);
        }

        [Test]
        public void ARowAgainstTheWorldMovesOnlyItsBody()
        {
            Scenario scenario = new Scenario(10);
            int body = scenario.AddBody();
            scenario.AddBlock(-1, body, 3);
            scenario.AddBlock(body, -1, 2);

            AssertSolves(scenario, Solve(scenario, false, out _, out _), 2e-3);
        }

        [Test]
        public void ALongChainHoldsAtAHugeMassRatio()
        {
            Scenario scenario = new Scenario(11);
            int previous = -1;
            for (int i = 0; i < 40; i++)
            {
                int body = scenario.AddBody();
                SparseConstraintSolver.Body heavy = scenario.Bodies[body];
                double scale = i % 2 == 0 ? 1e4 : 1e-2;
                heavy.Mass *= scale;
                heavy.Inertia *= scale;
                scenario.Bodies[body] = heavy;
                scenario.InverseMasses[body] /= scale;
                scenario.InverseInertias[body] /= scale;
                if (i > 0)
                {
                    scenario.AddBlock(previous, body, 3);
                }

                previous = body;
            }

            AssertSolves(scenario, Solve(scenario, false, out _, out _), 1e-2);
        }

        [Test]
        public void ATreeFactorsWithoutFill([Values(10, 100, 1000)] int bodies)
        {
            Scenario scenario = Craft(1, bodies, 0);

            Solve(scenario, false, out SparseConstraintSolver.Statistics statistics, out _);

            int joints = bodies - 1;
            Assert.That(statistics.Nodes, Is.EqualTo(bodies + joints));
            Assert.That(statistics.FactorBlocks, Is.EqualTo(joints * 2), "each joint touches two bodies and nothing else");
        }

        [Test]
        public void ManyPartsOnOneTankAddNoFill()
        {
            Scenario scenario = new Scenario(2);
            int tank = scenario.AddBody();
            const int attached = 60;
            for (int i = 0; i < attached; i++)
            {
                scenario.AddBlock(tank, scenario.AddBody(), 6);
            }

            Solve(scenario, false, out SparseConstraintSolver.Statistics statistics, out _);

            Assert.That(statistics.FactorBlocks, Is.EqualTo(attached * 2));
        }

        [Test]
        public void StrutsKeepTheFactorProportionalToTheCraft()
        {
            long small = Ladder(200);
            long large = Ladder(800);

            Assert.That(large, Is.LessThan(small * 5), "four times the craft");
        }

        /// <summary>Moves every Jacobian entry by up to this fraction of itself, as a substep's motion would.</summary>
        private static void Nudge(Scenario scenario, uint seed, float fraction)
        {
            Random random = new Random(seed);
            for (int i = 0; i < scenario.Jacobian.Count; i++)
            {
                scenario.Jacobian[i] *= 1.0f + random.NextFloat(-fraction, fraction);
            }

            for (int i = 0; i < scenario.RightHandSide.Count; i++)
            {
                scenario.RightHandSide[i] += random.NextFloat(-1.0f, 1.0f);
            }
        }

        [Test]
        public void ANearbyProblemReusesTheFactorization([Range(1u, 6u)] uint seed)
        {
            Scenario scenario = Craft(seed, 12, 5);
            var solver = new SparseConstraintSolver(Allocator.Temp);
            float[] impulses = new float[scenario.Rows];
            Solve(ref solver, scenario, false, impulses);

            Nudge(scenario, seed, 1e-4f);
            Solve(ref solver, scenario, false, impulses);

            SparseConstraintSolver.Statistics statistics = solver.Stats;
            solver.Dispose();
            Assert.That(statistics.Factorizations, Is.EqualTo(0));
            AssertSolves(scenario, impulses, 2e-3);
        }

        [Test]
        public void AFarAwayProblemStillGetsTheRightAnswer([Range(1u, 6u)] uint seed)
        {
            Scenario scenario = Craft(seed, 12, 5);
            var solver = new SparseConstraintSolver(Allocator.Temp);
            float[] impulses = new float[scenario.Rows];
            Solve(ref solver, scenario, false, impulses);

            Nudge(scenario, seed, 0.9f);
            Solve(ref solver, scenario, false, impulses);

            SparseConstraintSolver.Statistics statistics = solver.Stats;
            solver.Dispose();
            Assert.That(statistics.Factorizations, Is.EqualTo(1));
            AssertSolves(scenario, impulses, 2e-3);
        }

        [Test]
        public void ADifferentCraftGetsItsOwnStructure()
        {
            var solver = new SparseConstraintSolver(Allocator.Temp);
            Scenario first = Craft(1, 12, 5);
            Scenario second = Craft(2, 20, 3);
            Solve(ref solver, first, false, new float[first.Rows]);

            float[] impulses = new float[second.Rows];
            Solve(ref solver, second, false, impulses);

            solver.Dispose();
            AssertSolves(second, impulses, 2e-3);
        }

        [Test]
        public void ABoundThatStopsHoldingIsNoticedOnReuse()
        {
            Scenario scenario = Craft(3, 12, 5, 0.5f);
            var solver = new SparseConstraintSolver(Allocator.Temp);
            float[] impulses = new float[scenario.Rows];
            Assert.That(Solve(ref solver, scenario, true, impulses), Is.True);

            for (int i = 0; i < scenario.RightHandSide.Count; i++)
            {
                scenario.RightHandSide[i] *= -1.0f;
            }

            Assert.That(Solve(ref solver, scenario, true, impulses), Is.True);
            solver.Dispose();
            AssertSolves(scenario, impulses, 2e-3);
        }

        /// <summary>A stack with a radial on every level and a strut between neighbours, as a braced booster has.</summary>
        private static long Ladder(int levels)
        {
            Scenario scenario = new Scenario(4);
            int previousStack = -1;
            int previousRadial = -1;
            for (int level = 0; level < levels; level++)
            {
                int stack = scenario.AddBody();
                int radial = scenario.AddBody();
                scenario.AddBlock(stack, radial, 6);
                if (level > 0)
                {
                    scenario.AddBlock(previousStack, stack, 6);
                    scenario.AddBlock(previousRadial, radial, 3);
                }

                previousStack = stack;
                previousRadial = radial;
            }

            Solve(scenario, false, out SparseConstraintSolver.Statistics statistics, out _);
            return statistics.FactorScalars;
        }
    }
}
