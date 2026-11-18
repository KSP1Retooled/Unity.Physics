using Unity.Physics.Configuration;
using System;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Dynamics.Vessels.Structure;
using Unity.Physics.Tests.Dynamics.Vessels;
using Debug = UnityEngine.Debug;

namespace Unity.Physics.Tests.Dynamics.Vessels
{
    /// <summary>
    /// Compare the section solver with independent dense LU and stiff live-edge references.
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
        static void DenseSystem(ReferenceVesselSolver solver, NativeArray<double> a, NativeArray<double> b, double[] random)
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
        static double BackwardError(ReferenceVesselSolver solver, double[] x, double[] rhs, double matrixNorm)
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

        static ReferenceVesselSolver MakeSolver(VesselGraph graph, VesselGraph.LockRule rule, double stiffnessScale,
            VesselSolverSettings settings, out NativeArray<VesselPart> parts, out NativeArray<VesselEdge> edges,
            out NativeArray<VesselClosure> closures)
        {
            graph.Build(rule, settings.TimeStep, stiffnessScale, Allocator.Persistent, out parts, out edges, out closures);
            return new ReferenceVesselSolver(parts, edges, closures, settings);
        }

        [Test]
        public void TheFactorMatchesADenseSolve([Values("behemoth", "pegasus")] string craft, [Values(4, 96, 100000)] int chunkTarget,
            [Values(false, true)] bool threads, [Values(false, true)] bool minimumDegree)
        {
            VesselGraph graph = VesselGraph.Load(craft);
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.ChunkTarget = chunkTarget;
            settings.MinimumDegree = minimumDegree;
            using ReferenceVesselSolver solver = MakeSolver(graph, VesselGraph.LockRule.AllYielding, 1e3, settings,
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
        /// Compare locked motion and the loads of the locked edges and of the closures inside the sections with increasingly
        /// stiff live constraints. Closures stand in for the locked edges. On a bridge, an edge that no closure's loop
        /// crosses, the stand-in's stiffness comes from the subtree's mass, so that the reference moves as one rigid body;
        /// the load of a bridge doesn't depend on the stiffnesses. An edge on a closure's loop shares the load with the
        /// closure, so its stand-in has the stiffness that the locked solve gives it: hard rows on the two parts' reduced
        /// mass and inertia.
        /// </summary>
        [Test]
        public void LockedEdgesMatchVeryStiffYieldingOnes([Values("pegasus", "behemoth", "tu160")] string craft,
            [Values(VesselGraph.LockRule.LiveJointsOnly, VesselGraph.LockRule.StackLocked)] VesselGraph.LockRule rule)
        {
            bool wasOn = VesselSwitches.StrutLoadsEnabled;
            VesselSwitches.StrutLoadsEnabled = true;
            try
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
            finally
            {
                VesselSwitches.StrutLoadsEnabled = wasOn;
            }
        }

        static double MeanInverse(double3 inertia) => (1.0 / inertia.x + 1.0 / inertia.y + 1.0 / inertia.z) / 3.0;

        /// <summary>
        /// Marks the tree edges on the loop of each closure whose two parts are in one section, by the label of each part's
        /// section, and returns which closures those are.
        /// </summary>
        static bool[] MarkLoops(VesselGraph graph, int[] label, NativeArray<VesselClosure> closures, bool[] looped)
        {
            int count = graph.Count;
            var depth = new int[count];
            for (int i = 0; i < count; i++)
            {
                depth[i] = graph.Parent[i] >= 0 ? depth[graph.Parent[i]] + 1 : 0;
            }

            var inside = new bool[closures.Length];
            for (int c = 0; c < closures.Length; c++)
            {
                int a = closures[c].PartA, b = closures[c].PartB;
                if (label[a] != label[b] || a == b)
                {
                    continue;
                }

                inside[c] = true;
                while (a != b)
                {
                    if (depth[a] >= depth[b])
                    {
                        looped[a] = true;
                        a = graph.Parent[a];
                    }
                    else
                    {
                        looped[b] = true;
                        b = graph.Parent[b];
                    }
                }
            }

            return inside;
        }

        static void Compare(VesselGraph graph, VesselGraph.LockRule rule, double period, out double velocityError,
            out double loadError, System.Text.StringBuilder report)
        {
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.CentripetalLoads = false;
            settings.HardPeriod = period;
            graph.Build(rule, TimeStep, 1.0, Allocator.Persistent, out var lockedParts, out var lockedEdges, out var closures);
            graph.Build(VesselGraph.LockRule.AllYielding, TimeStep, 1.0, Allocator.Persistent, out var stiffParts,
                out var stiffEdges, out var unused);
            unused.Dispose();

            var label = new int[graph.Count];
            for (int i = 0; i < graph.Count; i++)
            {
                int p = graph.Parent[i];
                label[i] = p >= 0 && lockedEdges[i].State == VesselEdgeState.Locked ? label[p] : i;
            }

            // The closures inside the locked sections are hard, as the game builds its struts, and so are the locked edges on
            // their loops, as the game builds its stack joints. The graphs' springs are finite: against them a section
            // isn't rigid in the live reference, so no locked solve could match it.
            int count = graph.Count;
            var looped = new bool[count];
            bool[] inside = MarkLoops(graph, label, closures, looped);
            int insideCount = 0, loopedCount = 0;
            for (int c = 0; c < closures.Length; c++)
            {
                if (inside[c])
                {
                    VesselClosure closure = closures[c];
                    closure.LinearStiffness = double.PositiveInfinity;
                    closure.AngularStiffness = double.PositiveInfinity;
                    closures[c] = closure;
                    insideCount++;
                }
            }

            for (int i = 0; i < count; i++)
            {
                if (looped[i])
                {
                    VesselEdge edge = lockedEdges[i];
                    edge.TwistStiffness = double.PositiveInfinity;
                    edge.SwingStiffness = double.PositiveInfinity;
                    lockedEdges[i] = edge;
                    loopedCount++;
                }
            }

            // The stiff stand-ins for locked edges: the tree edge removed and a closure in its place. On a bridge it is at the
            // hard period on the lighter side's mass and second moment about the joint, critically damped.
            double omega = 2.0 * math.PI_DBL / period;
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

            var kept = new System.Collections.Generic.List<VesselClosure>(closures);
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
                double linear = looped[i] ? double.PositiveInfinity : omega * omega * math.min(below, above);
                double angular = looped[i] ? double.PositiveInfinity : omega * omega * math.min(momentBelow, momentAbove);
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

            var locked = new ReferenceVesselSolver(lockedParts, lockedEdges, closures, settings);
            var stiff = new ReferenceVesselSolver(stiffParts, stiffEdges, stiffClosures, settings);
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
            double strutError = 0.0, strutScale = 0.0;
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

            // The closures inside the sections count as locked loads: the locked solve takes them from the same static
            // system as the edges on their loops.
            for (int c = 0; c < closures.Length; c++)
            {
                if (!inside[c])
                {
                    continue;
                }

                VesselEdgeLoad a = locked.ClosureLoads[c];
                VesselEdgeLoad b = stiff.ClosureLoads[c];
                forceError = math.max(forceError, math.length(a.Force - b.Force));
                forceScale = math.max(forceScale, math.length(b.Force));
                torqueError = math.max(torqueError, math.length(a.Torque - b.Torque));
                torqueScale = math.max(torqueScale, math.length(b.Torque));
                strutError = math.max(strutError, math.max(math.length(a.Force - b.Force), math.length(a.Torque - b.Torque)));
                strutScale = math.max(strutScale, math.max(math.length(b.Force), math.length(b.Torque)));
            }

            loadError = math.max(forceError / forceScale, torqueError / torqueScale);
            if (!math.isfinite(velocityError) || !math.isfinite(loadError))
            {
                velocityError = loadError = double.PositiveInfinity;
            }

            report.Append($"\n  {period:G2} s: {locked.SectionCount} sections, {lockedCount} locked edges ({loopedCount} on loops), " +
                $"{closures.Length - insideCount} live closures, {insideCount} inside sections; velocity {velocityError:E2}, " +
                $"locked force {forceError / forceScale:E2} of {forceScale:G4} kN, locked torque {torqueError / torqueScale:E2} " +
                $"of {torqueScale:G4} kN m, closures inside {strutError / math.max(strutScale, 1e-30):E2} of {strutScale:G4}, " +
                $"live loads {liveError / math.max(liveScale, 1e-30):E2}, clamped pivots {locked.ClampedPivots} / {stiff.ClampedPivots}");
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
        /// A cantilever of four parts held at its root, with a strut from the root to the third part that closes a
        /// triangle, all locked into one section under gravity. The strut takes the share that the live stiff solve gives
        /// it, and every part's momentum balances.
        /// </summary>
        [Test]
        public void AStrutInALockedCantileverTakesItsShare([Values(1e-4, 1e-5)] double period)
        {
            bool wasOn = VesselSwitches.StrutLoadsEnabled;
            VesselSwitches.StrutLoadsEnabled = true;
            const int count = 4;
            var mass = new double[] { 2.0, 1.0, 1.5, 0.8 };
            var position = new double3[count];
            for (int i = 0; i < count; i++)
            {
                position[i] = new double3(i, 0.0, 0.0);
            }

            // The strut is a weld between the root and part 2, at a point above the beam.
            double3 strutPoint = new double3(1.0, 0.6, 0.0);
            double total = 0.0;
            double3 moment = 0.0;
            for (int i = 0; i < count; i++)
            {
                total += mass[i];
                moment += math.cross(position[i], new double3(0.0, -9.81 * mass[i], 0.0));
            }

            void Build(out NativeArray<VesselPart> parts, out NativeArray<VesselEdge> edges, out NativeArray<VesselClosure> closures,
                bool standIns)
            {
                parts = new NativeArray<VesselPart>(count, Allocator.Persistent);
                edges = new NativeArray<VesselEdge>(count, Allocator.Persistent);
                closures = new NativeArray<VesselClosure>(standIns ? count : 1, Allocator.Persistent);
                for (int i = 0; i < count; i++)
                {
                    var part = new VesselPart
                    {
                        Mass = mass[i],
                        Inertia = new double3(0.1, 0.2 + 0.05 * i, 0.15) * mass[i],
                        Position = position[i],
                        Orientation = new double4(0, 0, 0, 1),
                        LinearImpulse = new double3(0.0, -9.81 * mass[i] * TimeStep, 0.0),
                    };

                    // The root holds the cantilever up: it takes the weight and the moment of the weight, so nothing moves.
                    if (i == 0)
                    {
                        part.LinearImpulse += new double3(0.0, 9.81 * total * TimeStep, 0.0);
                        part.AngularImpulse = -moment * TimeStep;
                    }

                    parts[i] = part;
                    double3 joint = new double3(i - 0.5, 0.0, 0.0);
                    edges[i] = i == 0
                        ? new VesselEdge { Parent = -1, State = VesselEdgeState.Locked, Rest = new double4(0, 0, 0, 1) }
                        : new VesselEdge
                        {
                            Parent = i - 1,
                            State = standIns ? VesselEdgeState.Broken : VesselEdgeState.Locked,
                            MayLock = true,
                            ParentAnchor = joint - position[i - 1],
                            ChildAnchor = joint - position[i],
                            Frame = double3x3.identity,
                            Rest = new double4(0, 0, 0, 1),
                            TwistStiffness = double.PositiveInfinity,
                            SwingStiffness = double.PositiveInfinity,
                            BreakForce = 1e15,
                            BreakTorque = 1e15,
                            YieldFraction = 0.5,
                            RelockFraction = 0.5,
                        };
                    if (standIns && i > 0)
                    {
                        closures[i] = new VesselClosure
                        {
                            PartA = i - 1,
                            PartB = i,
                            AnchorA = joint - position[i - 1],
                            AnchorB = joint - position[i],
                            Frame = double3x3.identity,
                            Rest = new double4(0, 0, 0, 1),
                            LinearStiffness = double.PositiveInfinity,
                            AngularStiffness = double.PositiveInfinity,
                        };
                    }
                }

                closures[0] = new VesselClosure
                {
                    PartA = 0,
                    PartB = 2,
                    AnchorA = strutPoint - position[0],
                    AnchorB = strutPoint - position[2],
                    Frame = double3x3.identity,
                    Rest = new double4(0, 0, 0, 1),
                    LinearStiffness = double.PositiveInfinity,
                    AngularStiffness = double.PositiveInfinity,
                };
            }

            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.CentripetalLoads = false;
            settings.HardPeriod = period;
            Build(out var lockedParts, out var lockedEdges, out var lockedClosures, false);
            Build(out var stiffParts, out var stiffEdges, out var stiffClosures, true);
            var locked = new ReferenceVesselSolver(lockedParts, lockedEdges, lockedClosures, settings);
            var stiff = new ReferenceVesselSolver(stiffParts, stiffEdges, stiffClosures, settings);
            try
            {
                locked.Step(false);
                stiff.Step(false);
                Assert.AreEqual(1, locked.SectionCount, "every edge is locked, so the cantilever is one section");

                // The loads against the stiff solve: the strut, and each edge, whose stand-in is closure i there.
                double error = 0.0, scale = 0.0;
                void Against(VesselEdgeLoad a, VesselEdgeLoad b)
                {
                    error = math.max(error, math.max(math.length(a.Force - b.Force), math.length(a.Torque - b.Torque)));
                    scale = math.max(scale, math.max(math.length(b.Force), math.length(b.Torque)));
                }

                Against(locked.ClosureLoads[0], stiff.ClosureLoads[0]);
                for (int i = 1; i < count; i++)
                {
                    Against(locked.EdgeLoads[i], stiff.ClosureLoads[i]);
                }

                // Each part's momentum: the loads on it from its edges and the strut, about its centre, against what it
                // needs to stay with the section. Nothing moves, so it needs to cancel its own weight, and the root its
                // holding impulse as well.
                double imbalance = 0.0, weight = 9.81 * total;
                var force = new double3[count];
                var torque = new double3[count];
                void Act(int part, double3 point, VesselEdgeLoad load, double sign)
                {
                    force[part] += sign * load.Force;
                    torque[part] += sign * (math.cross(point - position[part], load.Force) + load.Torque);
                }

                for (int i = 1; i < count; i++)
                {
                    double3 joint = new double3(i - 0.5, 0.0, 0.0);
                    Act(i, joint, locked.EdgeLoads[i], 1.0);
                    Act(i - 1, joint, locked.EdgeLoads[i], -1.0);
                }

                Act(2, strutPoint, locked.ClosureLoads[0], 1.0);
                Act(0, strutPoint, locked.ClosureLoads[0], -1.0);
                for (int i = 0; i < count; i++)
                {
                    double3 needForce = -lockedParts[i].LinearImpulse / TimeStep;
                    double3 needTorque = -lockedParts[i].AngularImpulse / TimeStep;
                    imbalance = math.max(imbalance, math.max(math.length(force[i] - needForce), math.length(torque[i] - needTorque)));
                }

                VesselEdgeLoad strut = locked.ClosureLoads[0];
                VesselEdgeLoad cantilever = locked.EdgeLoads[1];
                Debug.Log($"[VesselSolverTests] strut in a cantilever, {period:G2} s: strut force {strut.Force} torque {strut.Torque}, " +
                    $"stiff solve's {stiff.ClosureLoads[0].Force} / {stiff.ClosureLoads[0].Torque}; first edge torque {cantilever.Torque} " +
                    $"(the weight's moment about the root {math.length(moment):G4} kN m); against stiff {error / scale:E2} " +
                    $"of {scale:G4}, largest part imbalance {imbalance:E2} kN against a weight of {weight:G4} kN");
                Assert.That(math.length(strut.Force), Is.GreaterThan(0.1 * weight), "the strut carries a real share");
                Assert.Less(error / scale, period < 5e-5 ? 1e-4 : 1e-3, "the strut and the edges share the load as in the stiff solve");
                Assert.Less(imbalance / weight, 1e-9, "every part's momentum balances");
            }
            finally
            {
                locked.Dispose();
                stiff.Dispose();
                lockedParts.Dispose();
                lockedEdges.Dispose();
                lockedClosures.Dispose();
                stiffParts.Dispose();
                stiffEdges.Dispose();
                stiffClosures.Dispose();
                VesselSwitches.StrutLoadsEnabled = wasOn;
            }
        }

        /// <summary>
        /// Verify inverse dynamics leaves only rounding residual in each section's momentum balance.
        /// </summary>
        [Test]
        public void EverySectionBalances([Values("pegasus", "behemoth", "tu160", "epsilon", "synthetic4500")] string craft,
            [Values(VesselGraph.LockRule.LiveJointsOnly, VesselGraph.LockRule.StackLocked)] VesselGraph.LockRule rule)
        {
            VesselGraph graph = VesselGraph.Load(craft);
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.CentripetalLoads = false;
            using ReferenceVesselSolver solver = MakeSolver(graph, rule, 1.0, settings, out var parts, out var edges, out var closures);
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
            using ReferenceVesselSolver one = MakeSolver(graph, rule, 1.0, settings, out var parts1, out var edges1, out var closures1);
            using ReferenceVesselSolver many = MakeSolver(graph, rule, 1.0, settings, out var parts2, out var edges2, out var closures2);
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
            using ReferenceVesselSolver reused = MakeSolver(graph, VesselGraph.LockRule.LiveJointsOnly, 1.0, settings,
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
                using var fresh = new ReferenceVesselSolver(parts2, edges2, closures2, settings);
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

        /// <summary>
        /// Compare boundary products from elimination-tree path substitutions with full factor solves.
        /// Verify velocity-only steps preserve positions.
        /// </summary>
        [Test]
        public void PathProductsMatchFullSolves([Values("pegasus", "tu160", "synthetic4500")] string craft,
            [Values(false, true)] bool minimumDegree)
        {
            VesselGraph graph = VesselGraph.Load(craft);
            VesselSolverSettings settings = VesselSolverSettings.Default;
            settings.MinimumDegree = minimumDegree;
            using ReferenceVesselSolver solver = MakeSolver(graph, VesselGraph.LockRule.AllYielding, 1.0, settings,
                out var parts, out var edges, out var closures);
            var before = new NativeArray<VesselPart>(parts, Allocator.Persistent);
            const int rowCount = 24;
            var seeds = new NativeArray<int>(2, Allocator.Persistent);
            NativeArray<double> full = default, y = default;
            NativeArray<int> path = default, mark = default;
            try
            {
                solver.VelocityOnly = true;
                solver.Step(false);
                for (int i = 0; i < parts.Length; i++)
                {
                    Assert.AreEqual(before[i].Position, parts[i].Position);
                    Assert.AreEqual(before[i].LinearVelocity, parts[i].LinearVelocity);
                }

                int sections = solver.SectionCount;
                full = new NativeArray<double>(6 * sections * rowCount, Allocator.Persistent);
                y = new NativeArray<double>(6 * sections, Allocator.Persistent);
                path = new NativeArray<int>(sections, Allocator.Persistent);
                mark = new NativeArray<int>(sections, Allocator.Persistent);
                for (int s = 0; s < sections; s++)
                {
                    mark[s] = -1;
                }

                // Each row touches one or two random sections with random columns, as a contact or a joint would.
                var generator = new Unity.Mathematics.Random(11);
                var rowSections = new int[rowCount, 2];
                var rowColumns = new double[rowCount, 12];
                for (int r = 0; r < rowCount; r++)
                {
                    rowSections[r, 0] = generator.NextInt(sections);
                    rowSections[r, 1] = r % 3 == 0 ? -1 : generator.NextInt(sections);
                    for (int c = 0; c < 12; c++)
                    {
                        rowColumns[r, c] = generator.NextDouble(-1.0, 1.0);
                    }
                }

                // Full solves: A^-1 b for every row.
                var one = new NativeArray<double>(6 * sections, Allocator.Persistent);
                for (int r = 0; r < rowCount; r++)
                {
                    for (int i = 0; i < one.Length; i++)
                    {
                        one[i] = 0.0;
                    }

                    for (int side = 0; side < 2; side++)
                    {
                        int s = rowSections[r, side];
                        for (int c = 0; s >= 0 && c < 6; c++)
                        {
                            one[6 * s + c] += rowColumns[r, 6 * side + c];
                        }
                    }

                    solver.SolveWithFactor(one);
                    for (int i = 0; i < one.Length; i++)
                    {
                        full[r * 6 * sections + i] = one[i];
                    }
                }

                one.Dispose();

                // Path forward solves, kept as the section, y and z = inv(D) y per path entry.
                var pathSections = new System.Collections.Generic.List<int>[rowCount];
                var pathY = new System.Collections.Generic.List<double6>[rowCount];
                var pathZ = new System.Collections.Generic.List<double6>[rowCount];
                int longest = 0;
                double* yPtr = (double*)y.GetUnsafePtr();
                for (int r = 0; r < rowCount; r++)
                {
                    int seedCount = 0;
                    for (int side = 0; side < 2; side++)
                    {
                        int s = rowSections[r, side];
                        if (s < 0)
                        {
                            continue;
                        }

                        seeds[seedCount++] = s;
                        for (int c = 0; c < 6; c++)
                        {
                            yPtr[6 * s + c] += rowColumns[r, 6 * side + c];
                        }
                    }

                    int count = solver.Context.CollectPath((int*)seeds.GetUnsafePtr(), seedCount, (int*)path.GetUnsafePtr(),
                        (int*)mark.GetUnsafePtr(), r);
                    longest = math.max(longest, count);
                    solver.Context.ForwardPath(yPtr, (int*)path.GetUnsafePtr(), count);
                    pathSections[r] = new System.Collections.Generic.List<int>();
                    pathY[r] = new System.Collections.Generic.List<double6>();
                    pathZ[r] = new System.Collections.Generic.List<double6>();
                    for (int k = 0; k < count; k++)
                    {
                        int s = path[k];
                        double6 ys = double6.Load(yPtr + 6 * s);
                        pathSections[r].Add(s);
                        pathY[r].Add(ys);
                        pathZ[r].Add(solver.Context.InverseDiagonal(s, ys));
                        for (int c = 0; c < 6; c++)
                        {
                            yPtr[6 * s + c] = 0.0;
                        }
                    }
                }

                double error = 0.0, scale = 0.0;
                for (int a = 0; a < rowCount; a++)
                {
                    for (int b = 0; b < rowCount; b++)
                    {
                        // Reference: b_a^T (A^-1 b_b).
                        double reference = 0.0;
                        for (int side = 0; side < 2; side++)
                        {
                            int s = rowSections[a, side];
                            for (int c = 0; s >= 0 && c < 6; c++)
                            {
                                reference += rowColumns[a, 6 * side + c] * full[b * 6 * sections + 6 * s + c];
                            }
                        }

                        double product = 0.0;
                        for (int i = 0; i < pathSections[a].Count; i++)
                        {
                            for (int j = 0; j < pathSections[b].Count; j++)
                            {
                                if (pathSections[a][i] == pathSections[b][j])
                                {
                                    product += double6.Dot(pathY[a][i], pathZ[b][j]);
                                }
                            }
                        }

                        error = math.max(error, math.abs(product - reference));
                        scale = math.max(scale, math.abs(reference));
                    }
                }

                Debug.Log($"[VesselSolverTests] {craft} md {minimumDegree}: {sections} sections, longest path {longest}; " +
                    $"path products differ from full solves by {error / scale:E2}");
                Assert.Less(error / scale, 1e-10);
            }
            finally
            {
                before.Dispose();
                seeds.Dispose();
                if (full.IsCreated)
                {
                    full.Dispose();
                    y.Dispose();
                    path.Dispose();
                    mark.Dispose();
                }

                parts.Dispose();
                edges.Dispose();
                closures.Dispose();
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

                using var solver = new ReferenceVesselSolver(parts, edges, closures, settings);
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
