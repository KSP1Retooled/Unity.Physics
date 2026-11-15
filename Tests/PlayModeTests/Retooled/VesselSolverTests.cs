using System;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The section solver against references it does not share code with: a dense LU of the same system built row by
    /// row, and the same vessel with every edge live and the locked ones made very stiff.
    /// </summary>
    public unsafe class VesselSolverTests
    {
        const double TimeStep = 0.02;

        /// <summary>Solves A X = B in place by LU with partial pivoting; A is n by n and B n by columns, both row by row.</summary>
        [BurstCompile(DisableSafetyChecks = true)]
        struct DenseSolveJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public double* A;
            [NativeDisableUnsafePtrRestriction] public double* B;
            public int N;
            public int Columns;

            public void Execute()
            {
                int n = N, m = Columns;
                for (int k = 0; k < n; k++)
                {
                    int pivot = k;
                    for (int i = k + 1; i < n; i++)
                    {
                        if (math.abs(A[(long)i * n + k]) > math.abs(A[(long)pivot * n + k]))
                        {
                            pivot = i;
                        }
                    }

                    if (pivot != k)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            (A[(long)k * n + j], A[(long)pivot * n + j]) = (A[(long)pivot * n + j], A[(long)k * n + j]);
                        }

                        for (int j = 0; j < m; j++)
                        {
                            (B[(long)k * m + j], B[(long)pivot * m + j]) = (B[(long)pivot * m + j], B[(long)k * m + j]);
                        }
                    }

                    double inverse = 1.0 / A[(long)k * n + k];
                    for (int i = k + 1; i < n; i++)
                    {
                        double factor = A[(long)i * n + k] * inverse;
                        if (factor == 0.0)
                        {
                            continue;
                        }

                        double* rowI = A + (long)i * n;
                        double* rowK = A + (long)k * n;
                        for (int j = k + 1; j < n; j++)
                        {
                            rowI[j] -= factor * rowK[j];
                        }

                        for (int j = 0; j < m; j++)
                        {
                            B[(long)i * m + j] -= factor * B[(long)k * m + j];
                        }
                    }
                }

                for (int k = n - 1; k >= 0; k--)
                {
                    for (int j = 0; j < m; j++)
                    {
                        double sum = B[(long)k * m + j];
                        for (int i = k + 1; i < n; i++)
                        {
                            sum -= A[(long)k * n + i] * B[(long)i * m + j];
                        }

                        B[(long)k * m + j] = sum / A[(long)k * n + k];
                    }
                }
            }
        }

        /// <summary>One row's Jacobian on its child and parent sections, its weight and its position term.</summary>
        static void Row(in VesselRows rows, int r, double[] child, double[] parent, out double weight, out double bias)
        {
            double3 u = r % 3 == 0 ? rows.Axes.c0 : r % 3 == 1 ? rows.Axes.c1 : rows.Axes.c2;
            double3 childAngular, parentAngular;
            if (r < 3)
            {
                childAngular = math.cross(rows.ChildArm, u);
                parentAngular = math.cross(rows.ParentArm, u);
                weight = rows.LinearWeight[r];
                bias = rows.LinearBias[r];
                for (int d = 0; d < 3; d++)
                {
                    child[d] = u[d];
                    parent[d] = -u[d];
                }
            }
            else
            {
                childAngular = u;
                parentAngular = u;
                weight = rows.AngularWeight[r - 3];
                bias = rows.AngularBias[r - 3];
                for (int d = 0; d < 3; d++)
                {
                    child[d] = 0.0;
                    parent[d] = 0.0;
                }
            }

            for (int d = 0; d < 3; d++)
            {
                child[3 + d] = childAngular[d];
                parent[3 + d] = -parentAngular[d];
            }
        }

        /// <summary>
        /// The solver's last system built densely from its sections and its links' rows, as A = M + sum w J^T J and
        /// b = M v + impulse - sum J^T g, with the second column of B a random right-hand side.
        /// </summary>
        static void DenseSystem(VesselSolver solver, NativeArray<double> a, NativeArray<double> b, double[] random)
        {
            int sections = solver.SectionCount, n = 6 * sections;
            for (int s = 0; s < sections; s++)
            {
                VesselSection section = solver.Sections[s];
                int o = 6 * s;
                for (int d = 0; d < 3; d++)
                {
                    a[(o + d) * n + o + d] += section.Mass;
                    for (int e = 0; e < 3; e++)
                    {
                        a[(o + 3 + d) * n + o + 3 + e] += section.Inertia[e][d];
                    }
                }

                double3 linear = section.Mass * section.LinearVelocity + section.LinearImpulse;
                double3 angular = math.mul(section.Inertia, section.AngularVelocity) + section.AngularImpulse;
                for (int d = 0; d < 3; d++)
                {
                    b[2 * (o + d)] = linear[d];
                    b[2 * (o + 3 + d)] = angular[d];
                }
            }

            for (int i = 0; i < n; i++)
            {
                b[2 * i + 1] = random[i];
            }

            var child = new double[6];
            var parent = new double[6];
            var index = new int[12];
            var j = new double[12];
            for (int k = 0; k < solver.LinkCount; k++)
            {
                VesselLink link = solver.Structure.Links[k];
                VesselRows rows = solver.LinkRows[k];
                for (int d = 0; d < 6; d++)
                {
                    index[d] = 6 * link.ChildSection + d;
                    index[6 + d] = 6 * link.ParentSection + d;
                }

                for (int r = 0; r < 6; r++)
                {
                    Row(rows, r, child, parent, out double weight, out double bias);
                    Array.Copy(child, 0, j, 0, 6);
                    Array.Copy(parent, 0, j, 6, 6);
                    for (int p = 0; p < 12; p++)
                    {
                        b[2 * index[p]] -= j[p] * bias;
                        for (int q = 0; q < 12; q++)
                        {
                            a[index[p] * n + index[q]] += weight * j[p] * j[q];
                        }
                    }
                }
            }
        }

        /// <summary>|A x - b| / (|A| |x| + |b|) in the max norm, with A = M + sum w J^T J applied row by row.</summary>
        static double BackwardError(VesselSolver solver, double[] x, double[] rhs, double matrixNorm)
        {
            int n = x.Length;
            var ax = new double[n];
            for (int s = 0; s < solver.SectionCount; s++)
            {
                VesselSection section = solver.Sections[s];
                for (int d = 0; d < 3; d++)
                {
                    ax[6 * s + d] += section.Mass * x[6 * s + d];
                    for (int e = 0; e < 3; e++)
                    {
                        ax[6 * s + 3 + d] += section.Inertia[e][d] * x[6 * s + 3 + e];
                    }
                }
            }

            var child = new double[6];
            var parent = new double[6];
            for (int k = 0; k < solver.LinkCount; k++)
            {
                VesselLink link = solver.Structure.Links[k];
                VesselRows rows = solver.LinkRows[k];
                int c = 6 * link.ChildSection, p = 6 * link.ParentSection;
                for (int r = 0; r < 6; r++)
                {
                    Row(rows, r, child, parent, out double weight, out double bias);
                    double jx = 0.0;
                    for (int d = 0; d < 6; d++)
                    {
                        jx += child[d] * x[c + d] + parent[d] * x[p + d];
                    }

                    for (int d = 0; d < 6; d++)
                    {
                        ax[c + d] += weight * child[d] * jx;
                        ax[p + d] += weight * parent[d] * jx;
                    }
                }
            }

            double residual = 0.0, xNorm = 0.0, bNorm = 0.0;
            for (int i = 0; i < n; i++)
            {
                residual = math.max(residual, math.abs(ax[i] - rhs[i]));
                xNorm = math.max(xNorm, math.abs(x[i]));
                bNorm = math.max(bNorm, math.abs(rhs[i]));
            }

            return residual / (matrixNorm * xNorm + bNorm);
        }

        static VesselSolver MakeSolver(VesselGraph graph, VesselGraph.LockRule rule, double stiffnessScale,
            VesselSolverSettings settings, out NativeArray<VesselPart> parts, out NativeArray<VesselEdge> edges,
            out NativeArray<VesselClosure> closures)
        {
            graph.Build(rule, settings.TimeStep, stiffnessScale, Allocator.Persistent, out parts, out edges, out closures);
            return new VesselSolver(parts, edges, closures, settings);
        }

        [Test]
        public void TheFactorMatchesADenseSolve([Values("behemoth", "pegasus")] string craft, [Values(4, 96, 100000)] int chunkTarget,
            [Values(false, true)] bool threads, [Values(false, true)] bool minimumDegree)
        {
            VesselGraph graph = VesselGraph.Load(craft);
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.ChunkTarget = chunkTarget;
            settings.MinimumDegree = minimumDegree;
            using VesselSolver solver = MakeSolver(graph, VesselGraph.LockRule.AllYielding, 1e3, settings,
                out var parts, out var edges, out var closures);
            try
            {
                solver.Step(threads);
                int n = 6 * solver.SectionCount;
                var a = new NativeArray<double>(n * n, Allocator.Persistent);
                var b = new NativeArray<double>(2 * n, Allocator.Persistent);
                var random = new double[n];
                var generator = new Unity.Mathematics.Random(3);
                for (int i = 0; i < n; i++)
                {
                    random[i] = generator.NextDouble(-1.0, 1.0);
                }

                DenseSystem(solver, a, b, random);
                var rhs = new double[n];
                double matrixNorm = 0.0;
                for (int i = 0; i < n; i++)
                {
                    rhs[i] = b[2 * i];
                    double sum = 0.0;
                    for (int j = 0; j < n; j++)
                    {
                        sum += math.abs(a[i * n + j]);
                    }

                    matrixNorm = math.max(matrixNorm, sum);
                }

                new DenseSolveJob { A = (double*)a.GetUnsafePtr(), B = (double*)b.GetUnsafePtr(), N = n, Columns = 2 }.Run();

                double velocityError = 0.0, velocityScale = 0.0;
                for (int i = 0; i < n; i++)
                {
                    velocityError = math.max(velocityError, math.abs(solver.Solution[i] - b[2 * i]));
                    velocityScale = math.max(velocityScale, math.abs(b[2 * i]));
                }

                // The rows' impulses from the dense velocities.
                double impulseError = 0.0, impulseScale = 0.0;
                var child = new double[6];
                var parent = new double[6];
                for (int k = 0; k < solver.LinkCount; k++)
                {
                    VesselLink link = solver.Structure.Links[k];
                    VesselRows rows = solver.LinkRows[k];
                    for (int r = 0; r < 6; r++)
                    {
                        Row(rows, r, child, parent, out double weight, out double bias);
                        double jv = 0.0;
                        for (int d = 0; d < 6; d++)
                        {
                            jv += child[d] * b[2 * (6 * link.ChildSection + d)] + parent[d] * b[2 * (6 * link.ParentSection + d)];
                        }

                        double expected = -(weight * jv + bias);
                        double actual = r < 3 ? rows.LinearImpulse[r] : rows.AngularImpulse[r - 3];
                        impulseError = math.max(impulseError, math.abs(actual - expected));
                        impulseScale = math.max(impulseScale, math.abs(expected));
                    }
                }

                // Another right-hand side against the kept factor.
                var other = new NativeArray<double>(random, Allocator.Persistent);
                solver.SolveWithFactor(other);
                double otherError = 0.0, otherScale = 0.0;
                for (int i = 0; i < n; i++)
                {
                    otherError = math.max(otherError, math.abs(other[i] - b[2 * i + 1]));
                    otherScale = math.max(otherScale, math.abs(b[2 * i + 1]));
                }

                // Normwise backward error of each answer, |A x - b| / (|A| |x| + |b|), with A applied from the rows.
                var dense = new double[n];
                var mine = new double[n];
                for (int i = 0; i < n; i++)
                {
                    dense[i] = b[2 * i];
                    mine[i] = solver.Solution[i];
                }

                double denseResidual = BackwardError(solver, dense, rhs, matrixNorm);
                double residual = BackwardError(solver, mine, rhs, matrixNorm);

                Debug.Log($"[VesselSolverTests] {craft} chunks {solver.ChunkCount} top {solver.TopCount} sections {solver.SectionCount} " +
                    $"links {solver.LinkCount} hubs {solver.HubCount} fill {solver.FillCount} widest column {solver.MaxStructure}: " +
                    $"velocity {velocityError / velocityScale:E2}, impulses {impulseError / impulseScale:E2}, " +
                    $"second solve {otherError / otherScale:E2}, clamped pivots {solver.ClampedPivots}; backward error " +
                    $"{residual:E2} (dense LU {denseResidual:E2})");
                Assert.Less(residual, 1e-15);
                Assert.Less(velocityError / velocityScale, 1e-8);
                Assert.Less(impulseError / impulseScale, 1e-8);
                Assert.Less(otherError / otherScale, 1e-8);
                Assert.AreEqual(0, solver.ClampedPivots);
                a.Dispose();
                b.Dispose();
                other.Dispose();
            }
            finally
            {
                parts.Dispose();
                edges.Dispose();
                closures.Dispose();
            }
        }

        /// <summary>
        /// Locked edges against the same edges made live and very stiff: the motion and each locked edge's load from
        /// inverse dynamics must converge on the stiff edge's motion and impulses as the stiffness grows.
        ///
        /// A yielding edge's hard rows take their stiffness from the two parts they join, which is soft against a heavy
        /// subtree (the study's finding about today's hard rows), so a stiff stand-in for a locked edge is a closure
        /// between the same two parts at the same joint point, its six rows as stiff as the mass and moment of the lighter
        /// side of the joint call for. The live edges' hard rows stiffen with the same hard period in both solves.
        /// </summary>
        [Test]
        public void LockedEdgesMatchVeryStiffYieldingOnes([Values("pegasus", "behemoth", "tu160")] string craft,
            [Values(VesselGraph.LockRule.LiveJointsOnly, VesselGraph.LockRule.StackLocked)] VesselGraph.LockRule rule)
        {
            VesselGraph graph = VesselGraph.Load(craft);
            var report = new System.Text.StringBuilder($"[VesselSolverTests] {craft} {rule}, locked against stiff, by hard period:");
            double[] periods = { 1e-3, 1e-4, 1e-5 };
            var velocityErrors = new double[periods.Length];
            var loadErrors = new double[periods.Length];
            for (int level = 0; level < periods.Length; level++)
            {
                Compare(graph, rule, periods[level], out velocityErrors[level], out loadErrors[level], report);
            }

            Debug.Log(report.ToString());
            int last = periods.Length - 1;
            // The live edges' hard rows join parts in one solve and sections in the other, so they converge on each other
            // only as both stiffen; where few edges are live (Pegasus, servos only) the loads agree to 1e-8 at 1e-5 s.
            Assert.Less(velocityErrors[last], 5e-3);
            Assert.Less(loadErrors[last], 5e-2);
            Assert.Less(velocityErrors[last], 0.1 * velocityErrors[0]);
            Assert.Less(loadErrors[last], 0.1 * loadErrors[0]);
        }

        static double MeanInverse(double3 inertia) => (1.0 / inertia.x + 1.0 / inertia.y + 1.0 / inertia.z) / 3.0;

        static void Compare(VesselGraph graph, VesselGraph.LockRule rule, double period, out double velocityError,
            out double loadError, System.Text.StringBuilder report)
        {
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.CentripetalLoads = false;
            settings.HardPeriod = period;
            graph.Build(rule, TimeStep, 1.0, Allocator.Persistent, out var lockedParts, out var lockedEdges, out var allClosures);
            graph.Build(VesselGraph.LockRule.AllYielding, TimeStep, 1.0, Allocator.Persistent, out var stiffParts,
                out var stiffEdges, out var unused);
            unused.Dispose();

            // A closure inside a locked section carries nothing in the locked solve, but with every edge live it would share
            // the section's internal loads; keep only the closures between sections.
            var label = new int[graph.Count];
            for (int i = 0; i < graph.Count; i++)
            {
                int p = graph.Parent[i];
                label[i] = p >= 0 && lockedEdges[i].State == VesselEdgeState.Locked ? label[p] : i;
            }

            var kept = new System.Collections.Generic.List<VesselClosure>();
            foreach (VesselClosure closure in allClosures)
            {
                if (label[closure.PartA] != label[closure.PartB])
                {
                    kept.Add(closure);
                }
            }

            allClosures.Dispose();
            var closures = new NativeArray<VesselClosure>(kept.ToArray(), Allocator.Persistent);

            // The stiff stand-ins for locked edges: the tree edge removed and a closure in its place, at the hard period
            // on the lighter side's mass and second moment about the joint, critically damped.
            double omega = 2.0 * math.PI_DBL / period;
            int count = graph.Count;
            var mass = new double[count];
            var first = new double3[count];
            var second = new double[count];
            for (int i = 0; i < count; i++)
            {
                mass[i] = graph.Mass[i];
                first[i] = graph.Mass[i] * graph.Position[i];
                second[i] = graph.Mass[i] * math.lengthsq(graph.Position[i]) + 1.0 / MeanInverse(graph.Inertia[i]);
            }

            for (int i = count - 1; i > 0; i--)
            {
                int p = graph.Parent[i];
                mass[p] += mass[i];
                first[p] += first[i];
                second[p] += second[i];
            }

            var stand = new System.Collections.Generic.List<VesselClosure>();
            var standIn = new int[count];
            for (int i = 0; i < count; i++)
            {
                standIn[i] = -1;
                if (graph.Parent[i] < 0 || lockedEdges[i].State != VesselEdgeState.Locked)
                {
                    continue;
                }

                double3 joint = graph.Joint[i];
                double below = mass[i], above = mass[0] - mass[i];
                double momentBelow = second[i] - 2.0 * math.dot(joint, first[i]) + math.lengthsq(joint) * mass[i];
                double momentAbove = second[0] - second[i] - 2.0 * math.dot(joint, first[0] - first[i])
                    + math.lengthsq(joint) * above;
                double linear = omega * omega * math.min(below, above);
                double angular = omega * omega * math.min(momentBelow, momentAbove);
                VesselEdge edge = stiffEdges[i];
                standIn[i] = kept.Count + stand.Count;
                stand.Add(new VesselClosure
                {
                    PartA = edge.Parent,
                    PartB = i,
                    AnchorA = edge.ParentAnchor,
                    AnchorB = edge.ChildAnchor,
                    Frame = edge.Frame,
                    Rest = edge.Rest,
                    LinearStiffness = linear,
                    AngularStiffness = angular,
                    LinearDamping = 2.0 * linear / omega,
                    AngularDamping = 2.0 * angular / omega,
                });
                edge.State = VesselEdgeState.Broken;
                stiffEdges[i] = edge;
            }

            kept.AddRange(stand);
            var stiffClosures = new NativeArray<VesselClosure>(kept.ToArray(), Allocator.Persistent);

            var before = new double3[graph.Count];
            for (int i = 0; i < graph.Count; i++)
            {
                before[i] = lockedParts[i].Position;
            }

            var locked = new VesselSolver(lockedParts, lockedEdges, closures, settings);
            var stiff = new VesselSolver(stiffParts, stiffEdges, stiffClosures, settings);
            locked.Step(false);
            stiff.Step(false);

            // Velocities at each part's centre, from its section's solution at the pose before the step.
            velocityError = 0.0;
            double velocityScale = 0.0;
            for (int i = 0; i < graph.Count; i++)
            {
                int s = locked.PartSection[i];
                VesselSection section = locked.Sections[s];
                double3 linear = new double3(locked.Solution[6 * s], locked.Solution[6 * s + 1], locked.Solution[6 * s + 2]);
                double3 angular = new double3(locked.Solution[6 * s + 3], locked.Solution[6 * s + 4], locked.Solution[6 * s + 5]);
                double3 v = linear + math.cross(angular, before[i] - section.Center);
                int t = stiff.PartSection[i];
                double3 w = new double3(stiff.Solution[6 * t], stiff.Solution[6 * t + 1], stiff.Solution[6 * t + 2]);
                double3 wa = new double3(stiff.Solution[6 * t + 3], stiff.Solution[6 * t + 4], stiff.Solution[6 * t + 5]);
                velocityError = math.max(velocityError, math.max(math.length(v - w), math.length(angular - wa)));
                velocityScale = math.max(velocityScale, math.max(math.length(w), math.length(wa)));
            }

            velocityError /= velocityScale;
            double forceError = 0.0, forceScale = 0.0, torqueError = 0.0, torqueScale = 0.0;
            double liveError = 0.0, liveScale = 0.0;
            int lockedCount = 0;
            for (int i = 0; i < graph.Count; i++)
            {
                if (graph.Parent[i] < 0)
                {
                    continue;
                }

                VesselEdgeLoad a = locked.EdgeLoads[i];
                VesselEdgeLoad b = standIn[i] >= 0 ? stiff.ClosureLoads[standIn[i]] : stiff.EdgeLoads[i];
                if (lockedEdges[i].State == VesselEdgeState.Locked)
                {
                    lockedCount++;
                    forceError = math.max(forceError, math.length(a.Force - b.Force));
                    forceScale = math.max(forceScale, math.length(b.Force));
                    torqueError = math.max(torqueError, math.length(a.Torque - b.Torque));
                    torqueScale = math.max(torqueScale, math.length(b.Torque));
                }
                else
                {
                    liveError = math.max(liveError, math.max(math.length(a.Force - b.Force), math.length(a.Torque - b.Torque)));
                    liveScale = math.max(liveScale, math.max(math.length(b.Force), math.length(b.Torque)));
                }
            }

            loadError = math.max(forceError / forceScale, torqueError / torqueScale);
            if (!math.isfinite(velocityError) || !math.isfinite(loadError))
            {
                velocityError = loadError = double.PositiveInfinity;
            }

            report.Append($"\n  {period:G2} s: {locked.SectionCount} sections, {lockedCount} locked edges, {closures.Length} " +
                $"live closures; velocity {velocityError:E2}, locked force {forceError / forceScale:E2} of {forceScale:G4} kN, " +
                $"locked torque {torqueError / torqueScale:E2} of {torqueScale:G4} kN m, live loads " +
                $"{liveError / math.max(liveScale, 1e-30):E2}, clamped pivots {locked.ClampedPivots} / {stiff.ClampedPivots}");
            locked.Dispose();
            stiff.Dispose();
            lockedParts.Dispose();
            lockedEdges.Dispose();
            stiffParts.Dispose();
            stiffEdges.Dispose();
            closures.Dispose();
            stiffClosures.Dispose();
        }

        /// <summary>
        /// Inverse dynamics closes each section's momentum balance: what is left at a section's first part, after every
        /// locked edge has passed its subtree's load up and every live row's force is counted, is rounding.
        /// </summary>
        [Test]
        public void EverySectionBalances([Values("pegasus", "behemoth", "tu160", "epsilon", "synthetic4500")] string craft,
            [Values(VesselGraph.LockRule.LiveJointsOnly, VesselGraph.LockRule.StackLocked)] VesselGraph.LockRule rule)
        {
            VesselGraph graph = VesselGraph.Load(craft);
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.CentripetalLoads = false;
            using VesselSolver solver = MakeSolver(graph, rule, 1.0, settings, out var parts, out var edges, out var closures);
            try
            {
                solver.Step(true);
                double residual = 0.0, weight = 0.0, largest = 0.0;
                for (int s = 0; s < solver.SectionCount; s++)
                {
                    residual = math.max(residual, solver.SectionResidual[s]);
                }

                for (int i = 0; i < graph.Count; i++)
                {
                    weight += 9.81 * graph.Mass[i];
                    largest = math.max(largest, math.length(solver.EdgeLoads[i].Force));
                }

                Debug.Log($"[VesselSolverTests] {craft} {rule}: {solver.SectionCount} sections; largest section imbalance " +
                    $"{residual:E2} kN against a weight of {weight:G4} kN and a largest edge force of {largest:G4} kN");
                Assert.IsTrue(math.isfinite(residual));
                Assert.Less(residual / largest, 1e-9);
            }
            finally
            {
                parts.Dispose();
                edges.Dispose();
                closures.Dispose();
            }
        }

        /// <summary>The job threads give the single thread's answer, up to the order of a few sums.</summary>
        [Test]
        public void ThreadsAgreeWithOneThread([Values("tu160", "synthetic4500")] string craft,
            [Values(VesselGraph.LockRule.LiveJointsOnly, VesselGraph.LockRule.AllYielding)] VesselGraph.LockRule rule,
            [Values(false, true)] bool minimumDegree)
        {
            VesselGraph graph = VesselGraph.Load(craft);
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.MinimumDegree = minimumDegree;
            using VesselSolver one = MakeSolver(graph, rule, 1.0, settings, out var parts1, out var edges1, out var closures1);
            using VesselSolver many = MakeSolver(graph, rule, 1.0, settings, out var parts2, out var edges2, out var closures2);
            try
            {
                for (int step = 0; step < 20; step++)
                {
                    one.Step(false);
                    many.Step(true);
                }

                double positionError = 0.0, velocityError = 0.0, velocityScale = 0.0;
                for (int i = 0; i < graph.Count; i++)
                {
                    positionError = math.max(positionError, math.length(parts1[i].Position - parts2[i].Position));
                    velocityError = math.max(velocityError, math.length(parts1[i].LinearVelocity - parts2[i].LinearVelocity));
                    velocityScale = math.max(velocityScale, math.length(parts1[i].LinearVelocity));
                    Assert.IsTrue(math.all(math.isfinite(parts1[i].Position)), $"part {i} position");
                }

                Debug.Log($"[VesselSolverTests] {craft} {rule}: {one.SectionCount} sections, {one.ChunkCount} chunks, " +
                    $"top {one.TopCount}; after 20 steps positions differ by {positionError:E2} m, velocities by " +
                    $"{velocityError / velocityScale:E2}");
                Assert.Less(positionError, 1e-9);
                Assert.Less(velocityError / velocityScale, 1e-9);
            }
            finally
            {
                parts1.Dispose();
                edges1.Dispose();
                closures1.Dispose();
                parts2.Dispose();
                edges2.Dispose();
                closures2.Dispose();
            }
        }

        /// <summary>
        /// A rebuild in place after edges change state gives the answer a solver built fresh with those states gives.
        /// </summary>
        [Test]
        public void ARebuildMatchesAFreshSolver()
        {
            VesselGraph graph = VesselGraph.Load("epsilon");
            VesselSolverSettings settings = VesselSolverSettings.Default;
            using VesselSolver reused = MakeSolver(graph, VesselGraph.LockRule.LiveJointsOnly, 1.0, settings,
                out var parts1, out var edges1, out var closures1);
            try
            {
                reused.Step(true);

                // Unlock every 100th locked edge.
                int unlocked = 0;
                for (int i = 1; i < graph.Count; i += 100)
                {
                    VesselEdge edge = edges1[i];
                    if (edge.State == VesselEdgeState.Locked)
                    {
                        edge.State = VesselEdgeState.Yielding;
                        edges1[i] = edge;
                        unlocked++;
                    }
                }

                var parts2 = new NativeArray<VesselPart>(parts1, Allocator.Persistent);
                var edges2 = new NativeArray<VesselEdge>(edges1, Allocator.Persistent);
                var closures2 = new NativeArray<VesselClosure>(closures1, Allocator.Persistent);
                using var fresh = new VesselSolver(parts2, edges2, closures2, settings);
                int sectionsBefore = reused.SectionCount;
                reused.MarkStructureChanged();
                reused.Step(true);
                fresh.Step(true);
                double error = 0.0, scale = 0.0;
                for (int i = 0; i < graph.Count; i++)
                {
                    error = math.max(error, math.length(parts1[i].LinearVelocity - parts2[i].LinearVelocity));
                    scale = math.max(scale, math.length(parts2[i].LinearVelocity));
                }

                Debug.Log($"[VesselSolverTests] epsilon: {unlocked} edges unlocked, sections {sectionsBefore} -> " +
                    $"{reused.SectionCount}; velocities differ by {error / scale:E2}");
                Assert.Less(error / scale, 1e-9);
                parts2.Dispose();
                edges2.Dispose();
                closures2.Dispose();
            }
            finally
            {
                parts1.Dispose();
                edges1.Dispose();
                closures1.Dispose();
            }
        }

        /// <summary>A vessel that is one section falls freely: every part gains g h and the section stays rigid.</summary>
        [Test]
        public void AOneSectionVesselFallsAsOneBody()
        {
            VesselGraph graph = VesselGraph.Load("pegasus");
            VesselSolverSettings settings = VesselSolverSettings.Default;
            graph.Build(VesselGraph.LockRule.LiveJointsOnly, TimeStep, 1.0, Allocator.Persistent, out var parts, out var edges,
                out var closures);
            try
            {
                for (int i = 0; i < parts.Length; i++)
                {
                    VesselPart part = parts[i];
                    part.LinearVelocity = new double3(1, 2, 3);
                    part.AngularVelocity = 0.0;
                    part.LinearImpulse = new double3(0, -9.81 * part.Mass * TimeStep, 0);
                    part.AngularImpulse = 0.0;
                    parts[i] = part;
                    VesselEdge edge = edges[i];
                    edge.State = edge.Parent >= 0 ? VesselEdgeState.Locked : edge.State;
                    edges[i] = edge;
                }

                using var solver = new VesselSolver(parts, edges, closures, settings);
                solver.Step(false);
                Assert.AreEqual(1, solver.SectionCount);
                double error = 0.0, loads = 0.0;
                for (int i = 0; i < parts.Length; i++)
                {
                    error = math.max(error, math.length(parts[i].LinearVelocity - new double3(1, 2 - 9.81 * TimeStep, 3)));
                    loads = math.max(loads, math.length(solver.EdgeLoads[i].Force));
                }

                Assert.Less(error, 1e-8);
                Assert.Less(loads, 1e-6);
            }
            finally
            {
                parts.Dispose();
                edges.Dispose();
                closures.Dispose();
            }
        }
    }
}
