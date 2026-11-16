using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Jobs.LowLevel.Unsafe;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Whole physics steps of a large craft built from a vessel graph (Tests/VesselGraphs~) as Rigidbodies and stock-style
    /// ConfigurableJoints, on the sparse path, the forest path and the forest path with locking, on one thread and on eight,
    /// warm (RETOOLED_BENCH_CRAFT, _SCENE, _THREADS and _PATHS pick part of the table). The "vessel" path hands the step a
    /// vessel set that the bench builds itself, and the "compiled" path turns vessels on and lets the backend compile the
    /// craft's joints into vessels. The craft floats
    /// (every row a forest row) or rests on the ground (contacts make a boundary). Parts never collide with each other.
    /// Reports the machine's CPU load when it starts. Results go to the log and to VesselForestBench.txt in Logs.
    /// </summary>
    [Explicit("A benchmark, minutes long")]
    [Category("Slow")]
    public class VesselForestBench : PhysicsScene
    {
        private const int PartLayer = 30;
        private const int WarmUp = 20;
        private const int Measured = 60;

        private struct Graph
        {
            public string Name;
            public int[] Parent;
            public float[] Mass;
            public Vector3[] Inertia;
            public Vector3[] Position;
            public Vector3[] Joint;
            public float[] Swing;
            public int[] ClosureA;
            public int[] ClosureB;
        }

        // The fused job's settings before the bench changes them for a path.
        private static int fusedParts = Unity.Physics.Solver.DefaultFusedVesselParts;
        private static bool fusedOn = true;

        private static Graph Load(string name)
        {
            string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "io.github.ksp1retooled.physics",
                "Tests", "VesselGraphs~"));
            using var file = File.OpenRead(Path.Combine(folder, name + ".txt.gz"));
            using var unzip = new GZipStream(file, CompressionMode.Decompress);
            using var reader = new StreamReader(unzip);
            var graph = new Graph { Name = reader.ReadLine().Substring("vessel ".Length) };
            int count = int.Parse(reader.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
            graph.Parent = new int[count];
            graph.Mass = new float[count];
            graph.Inertia = new Vector3[count];
            graph.Position = new Vector3[count];
            graph.Joint = new Vector3[count];
            graph.Swing = new float[count];
            for (int i = 0; i < count; i++)
            {
                string[] f = reader.ReadLine().Split(' ');
                float F(int k) => float.Parse(f[k], CultureInfo.InvariantCulture);
                graph.Parent[i] = int.Parse(f[0], CultureInfo.InvariantCulture);
                graph.Mass[i] = F(2);
                graph.Inertia[i] = new Vector3(F(3), F(4), F(5));
                graph.Position[i] = new Vector3(F(6), F(7), F(8));
                graph.Joint[i] = new Vector3(F(9), F(10), F(11));
                graph.Swing[i] = F(15);
            }

            int closures = int.Parse(reader.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
            graph.ClosureA = new int[closures];
            graph.ClosureB = new int[closures];
            for (int c = 0; c < closures; c++)
            {
                string[] f = reader.ReadLine().Split(' ');
                graph.ClosureA[c] = int.Parse(f[0], CultureInfo.InvariantCulture);
                graph.ClosureB[c] = int.Parse(f[1], CultureInfo.InvariantCulture);
            }

            return graph;
        }

        /// <summary>
        /// Cuts a craft down to its first RETOOLED_BENCH_PARTS parts and lays out RETOOLED_BENCH_COPIES copies of it side by
        /// side, for a scene of several smaller crafts. The parts come parents first, so the first parts of a craft are
        /// one connected tree. Without either setting, returns the craft as it is.
        /// </summary>
        private static Graph Fleet(Graph graph)
        {
            int parts = int.TryParse(System.Environment.GetEnvironmentVariable("RETOOLED_BENCH_PARTS"), out int p) && p > 0
                ? System.Math.Min(p, graph.Parent.Length) : graph.Parent.Length;
            int copies = int.TryParse(System.Environment.GetEnvironmentVariable("RETOOLED_BENCH_COPIES"), out int c) && c > 0 ? c : 1;
            if (parts == graph.Parent.Length && copies == 1)
            {
                return graph;
            }

            float left = float.MaxValue, right = float.MinValue;
            for (int i = 0; i < parts; i++)
            {
                left = Mathf.Min(left, graph.Position[i].x);
                right = Mathf.Max(right, graph.Position[i].x);
            }

            float spacing = right - left + 10.0f;
            var closures = new List<int>();
            for (int k = 0; k < graph.ClosureA.Length; k++)
            {
                if (graph.ClosureA[k] < parts && graph.ClosureB[k] < parts)
                {
                    closures.Add(k);
                }
            }

            var fleet = new Graph
            {
                Name = $"{graph.Name} {copies}x{parts}",
                Parent = new int[parts * copies],
                Mass = new float[parts * copies],
                Inertia = new Vector3[parts * copies],
                Position = new Vector3[parts * copies],
                Joint = new Vector3[parts * copies],
                Swing = new float[parts * copies],
                ClosureA = new int[closures.Count * copies],
                ClosureB = new int[closures.Count * copies],
            };

            for (int copy = 0; copy < copies; copy++)
            {
                var offset = new Vector3(copy * spacing, 0.0f, 0.0f);
                for (int i = 0; i < parts; i++)
                {
                    int to = copy * parts + i;
                    fleet.Parent[to] = graph.Parent[i] >= 0 ? copy * parts + graph.Parent[i] : -1;
                    fleet.Mass[to] = graph.Mass[i];
                    fleet.Inertia[to] = graph.Inertia[i];
                    fleet.Position[to] = graph.Position[i] + offset;
                    fleet.Joint[to] = graph.Joint[i] + offset;
                    fleet.Swing[to] = graph.Swing[i];
                }

                for (int k = 0; k < closures.Count; k++)
                {
                    fleet.ClosureA[copy * closures.Count + k] = copy * parts + graph.ClosureA[closures[k]];
                    fleet.ClosureB[copy * closures.Count + k] = copy * parts + graph.ClosureB[closures[k]];
                }
            }

            return fleet;
        }

        /// <summary>Total CPU load in percent over three seconds, from the Windows performance counter.</summary>
        private static double SampleLoad()
        {
            var start = new ProcessStartInfo("powershell.exe",
                "-NoProfile -Command \"(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 3)." +
                "CounterSamples | Measure-Object -Property CookedValue -Average | ForEach-Object { $_.Average }\"")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };
            using Process process = Process.Start(start);
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            return double.TryParse(output.Trim(), NumberStyles.Float, CultureInfo.CurrentCulture, out double load) ? load
                : double.TryParse(output.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out load) ? load : -1.0;
        }

        private static ConfigurableJoint Joint(Rigidbody host, Rigidbody target, Vector3 anchor, float spring, bool locked)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(anchor);
            joint.connectedAnchor = target.transform.InverseTransformPoint(anchor);
            if (locked)
            {
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
                return joint;
            }

            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
            joint.angularXMotion = ConfigurableJointMotion.Locked;
            joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 1.0f };
            joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
            joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
            joint.xDrive = joint.yDrive = joint.zDrive = new JointDrive { positionSpring = 1e20f, maximumForce = 1e20f };
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularYZDrive = new JointDrive { positionSpring = spring, positionDamper = 0.02f * spring, maximumForce = 1e20f };
            return joint;
        }

        private List<Rigidbody> Build(Graph graph, bool ground, bool joints = true)
        {
            var bodies = new List<Rigidbody>(graph.Parent.Length);
            float lowest = float.MaxValue;
            for (int i = 0; i < graph.Parent.Length; i++)
            {
                float mass = Mathf.Max(graph.Mass[i], 1e-3f);
                float inertia = Mathf.Max((graph.Inertia[i].x + graph.Inertia[i].y + graph.Inertia[i].z) / 3.0f, 1e-6f);
                float side = Mathf.Clamp(Mathf.Sqrt(6.0f * inertia / mass), 0.05f, 5.0f);
                GameObject part = Box(graph.Position[i], Vector3.one * side);
                part.layer = PartLayer;
                Rigidbody body = part.AddComponent<Rigidbody>();
                body.mass = mass;
                body.sleepThreshold = 0.0f;
                bodies.Add(body);
                lowest = Mathf.Min(lowest, graph.Position[i].y - 0.5f * side);
            }

            for (int i = 0; i < graph.Parent.Length && joints; i++)
            {
                int parent = graph.Parent[i];
                if (parent >= 0)
                {
                    Joint(bodies[i], bodies[parent], graph.Joint[i], graph.Swing[i] > 0.0f ? graph.Swing[i] : 1e5f, false);
                }
            }

            for (int c = 0; c < graph.ClosureA.Length && joints; c++)
            {
                int a = graph.ClosureA[c], b = graph.ClosureB[c];
                Joint(bodies[b], bodies[a], 0.5f * (graph.Position[a] + graph.Position[b]), 0.0f, true);
            }

            if (ground)
            {
                Box(new Vector3(0.0f, lowest - 0.5f, 0.0f), new Vector3(400.0f, 1.0f, 400.0f));
            }

            return bodies;
        }

        /// <summary>
        /// The craft as one vessel for the "vessel" path: one part per body in parent-first order, an edge for every graph
        /// edge with the bench joint's swing spring and a hard twist, and a hard closure for every strut. The parts' world
        /// indices are found once, from the first world the backend builds, since the bench adds no bodies afterwards.
        /// </summary>
        private sealed unsafe class BenchVesselSet : IDisposable
        {
            private NativeArray<VesselRecord> vessels;
            private NativeArray<int> partBodies;
            private NativeArray<VesselEdge> edges;
            private NativeArray<VesselClosure> closures;
            private NativeArray<VesselEdgeLoad> edgeLoads;
            private NativeArray<VesselEdgeLoad> closureLoads;
            private NativeArray<bool> edgeLocked;

            public BenchVesselSet(Graph graph, List<Rigidbody> bodies)
            {
                int count = graph.Parent.Length;

                // Parent-first order: each root, then breadth first along the children.
                var children = new List<int>[count];
                var order = new List<int>(count);
                for (int i = 0; i < count; i++)
                {
                    children[i] = new List<int>();
                }

                for (int i = 0; i < count; i++)
                {
                    if (graph.Parent[i] >= 0)
                    {
                        children[graph.Parent[i]].Add(i);
                    }
                }

                var partOf = new int[count];
                var roots = new List<int>();
                for (int i = 0; i < count; i++)
                {
                    if (graph.Parent[i] < 0)
                    {
                        int first = order.Count;
                        roots.Add(first);
                        order.Add(i);
                        for (int k = first; k < order.Count; k++)
                        {
                            order.AddRange(children[order[k]]);
                        }
                    }
                }

                for (int k = 0; k < count; k++)
                {
                    partOf[order[k]] = k;
                }

                // Every tree of the graph is a vessel of its own.
                vessels = new NativeArray<VesselRecord>(roots.Count, Allocator.Persistent);
                for (int v = 0; v < roots.Count; v++)
                {
                    int end = v + 1 < roots.Count ? roots[v + 1] : count;
                    vessels[v] = new VesselRecord
                    {
                        Id = (ulong)(5000 + v),
                        StructureVersion = 1,
                        FirstPart = roots[v],
                        PartCount = end - roots[v],
                        FirstClosure = 0,
                        ClosureCount = v == 0 ? graph.ClosureA.Length : 0,
                    };
                }

                PhysicsWorld world = StepKernel.Current->World;
                NativeArray<RigidBody> dynamic = world.DynamicBodies;
                partBodies = new NativeArray<int>(count, Allocator.Persistent);
                edges = new NativeArray<VesselEdge>(count, Allocator.Persistent);
                for (int k = 0; k < count; k++)
                {
                    int i = order[k];
                    Rigidbody body = bodies[i];
                    float best = float.MaxValue;
                    for (int b = 0; b < dynamic.Length; b++)
                    {
                        float d = math.distancesq(dynamic[b].WorldFromBody.pos, (float3)body.position);
                        if (d < best)
                        {
                            best = d;
                            partBodies[k] = b;
                        }
                    }

                    int parent = graph.Parent[i];
                    if (parent < 0)
                    {
                        edges[k] = new VesselEdge { Parent = -1, State = VesselEdgeState.Yielding };
                        continue;
                    }

                    int vessel = 0;
                    while (vessel + 1 < roots.Count && roots[vessel + 1] <= k)
                    {
                        vessel++;
                    }

                    double spring = graph.Swing[i] > 0.0f ? graph.Swing[i] : 1e5;
                    Vector3 point = graph.Joint[i];
                    edges[k] = new VesselEdge
                    {
                        Parent = partOf[parent] - roots[vessel],
                        State = VesselEdgeState.Yielding,
                        MayLock = true,
                        ChildAnchor = (float3)(point - bodies[i].worldCenterOfMass),
                        ParentAnchor = (float3)(point - bodies[parent].worldCenterOfMass),
                        Frame = double3x3.identity,
                        Rest = new double4(0.0, 0.0, 0.0, 1.0),
                        TwistStiffness = double.PositiveInfinity,
                        SwingStiffness = spring,
                        SwingDamping = 0.02 * spring,
                        BreakForce = double.PositiveInfinity,
                        BreakTorque = double.PositiveInfinity,
                    };
                }

                // The closures belong to the first vessel; a strut between two vessels would leave both in one island.
                closures = new NativeArray<VesselClosure>(System.Math.Max(1, graph.ClosureA.Length), Allocator.Persistent);
                for (int c = 0; c < graph.ClosureA.Length; c++)
                {
                    int a = graph.ClosureA[c], b = graph.ClosureB[c];
                    Vector3 point = 0.5f * (graph.Position[a] + graph.Position[b]);
                    closures[c] = new VesselClosure
                    {
                        PartA = partOf[a],
                        PartB = partOf[b],
                        AnchorA = (float3)(point - bodies[a].worldCenterOfMass),
                        AnchorB = (float3)(point - bodies[b].worldCenterOfMass),
                        Frame = double3x3.identity,
                        Rest = new double4(0.0, 0.0, 0.0, 1.0),
                        LinearStiffness = double.PositiveInfinity,
                        AngularStiffness = double.PositiveInfinity,
                    };
                }

                edgeLoads = new NativeArray<VesselEdgeLoad>(count, Allocator.Persistent);
                closureLoads = new NativeArray<VesselEdgeLoad>(System.Math.Max(1, graph.ClosureA.Length), Allocator.Persistent);
                edgeLocked = new NativeArray<bool>(count, Allocator.Persistent);
                VesselSet.TestSet = new VesselSet
                {
                    VesselCount = vessels.Length,
                    Vessels = (VesselRecord*)vessels.GetUnsafePtr(),
                    PartBodies = (int*)partBodies.GetUnsafePtr(),
                    Edges = (VesselEdge*)edges.GetUnsafePtr(),
                    Closures = (VesselClosure*)closures.GetUnsafePtr(),
                    EdgeLoads = (VesselEdgeLoad*)edgeLoads.GetUnsafePtr(),
                    ClosureLoads = (VesselEdgeLoad*)closureLoads.GetUnsafePtr(),
                    EdgeLocked = (bool*)edgeLocked.GetUnsafePtr(),
                };
            }

            public void Dispose()
            {
                VesselSet.TestSet = default;
                vessels.Dispose();
                partBodies.Dispose();
                edges.Dispose();
                closures.Dispose();
                edgeLoads.Dispose();
                closureLoads.Dispose();
                edgeLocked.Dispose();
            }
        }

        private string lastTimes;

        private (double min, double median, long solves, long fallbacks) Session(Graph graph, bool ground, string path, int threads,
            int workers)
        {
            TearDownScene();
            SetUpScene();
            // A compiled path can name settings of the fused vessel job after a colon: "compiled:parts=64" sends vessel
            // islands of 64 parts or more to the fused job, "compiled:nocontacts" keeps the islands with contacts or
            // joints out of it, and "compiled:off" turns it off.
            string[] options = path.Split(':');
            Unity.Physics.Solver.FusedVesselParts = fusedParts;
            Unity.Physics.Solver.FusedContacts = true;
            Unity.Physics.Solver.VesselFused = fusedOn;
            for (int i = 1; i < options.Length; i++)
            {
                if (options[i].StartsWith("parts="))
                {
                    Unity.Physics.Solver.FusedVesselParts = int.Parse(options[i].Substring("parts=".Length), CultureInfo.InvariantCulture);
                }

                Unity.Physics.Solver.FusedContacts &= options[i] != "nocontacts";
                Unity.Physics.Solver.VesselFused &= options[i] != "off";
            }

            path = options[0];
            VesselForest.Enabled = path != "sparse";
            VesselPipeline.Enabled = path == "compiled";
            VesselForest.ForgetIslands();
            VesselForest.Locking = path == "locked";
            string cap = Environment.GetEnvironmentVariable("RETOOLED_BENCH_CAP");
            VesselForest.BoundaryRowCap = string.IsNullOrEmpty(cap) ? VesselForest.DefaultBoundaryRowCap : int.Parse(cap, CultureInfo.InvariantCulture);
            string hold = Environment.GetEnvironmentVariable("RETOOLED_BENCH_HOLD");
            VesselForest.MinimumHoldSteps = string.IsNullOrEmpty(hold) ? -1 : int.Parse(hold, CultureInfo.InvariantCulture);

            // RETOOLED_BENCH_BOUNDARY=cold starts every boundary solve cold, and =uncapped also lifts the caps.
            string boundaryMode = Environment.GetEnvironmentVariable("RETOOLED_BENCH_BOUNDARY");
            VesselForest.Caps = boundaryMode == "uncapped"
                ? new BoundaryCaps { OutsideContactPoints = -1, PointsPerManifold = -1, SelfContactManifolds = -1, PivotRounds = -1, ColdStart = true }
                : new BoundaryCaps { ColdStart = boundaryMode == "cold" };
            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
            List<Rigidbody> built = Build(graph, ground, path != "vessel");
            BenchVesselSet vesselSet = null;
            if (path == "vessel")
            {
                Simulate();
                vesselSet = new BenchVesselSet(graph, built);
            }

            Simulate(WarmUp);
            VesselForest.GetCounts(out long solvesBefore, out long fallbacksBefore, out _);
            VesselForest.GetTimes(out double s0, out double a0, out double b0, out double c0, out double f0, out double w0, out double d0);
            VesselForest.GetSectionCounts(out long sections0, out long locked0, out long builds0, out long changes0, out double rebuild0,
                out _, out _, out _);
            double settle0 = VesselForest.SettleMilliseconds;
            double[] serial0 = VesselForest.SerialMilliseconds();
            BoundaryCounters.Reset();
            var samples = new List<double>(Measured);
            for (int i = 0; i < Measured; i++)
            {
                long start = Stopwatch.GetTimestamp();
                UnityEngine.Physics.Simulate(Step);
                samples.Add((Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
            }

            VesselForest.GetCounts(out long solves, out long fallbacks, out _);
            VesselForest.GetTimes(out double s1, out double a1, out double b1, out double c1, out double f1, out double w1, out double d1);
            VesselForest.GetSectionCounts(out long sections1, out long locked1, out long builds1, out long changes1, out double rebuild1,
                out double residual, out _, out _);
            long forestSolves = System.Math.Max(1, solves - solvesBefore);
            lastTimes = path == "sparse" ? $"sparse {(s1 - s0) / Measured:F2}"
                : $"start {(a1 - a0) / Measured:F2} (factor {(f1 - f0) / Measured:F2}, boundary {(w1 - w0) / Measured:F2}), " +
                $"solve {(b1 - b0) / Measured:F2}, finish {(c1 - c0) / Measured:F2} (settle {(VesselForest.SettleMilliseconds - settle0) / Measured:F2}, " +
                $"drift aim {(d1 - d0) / Measured:F2}), " +
                $"fallback sparse {(s1 - s0) / Measured:F2}";
            if (path != "sparse")
            {
                double[] serial1 = VesselForest.SerialMilliseconds();
                lastTimes += $"; one lane: classify {(serial1[0] - serial0[0]) / Measured:F2}, decide {(serial1[1] - serial0[1]) / Measured:F2}, " +
                    $"section sums {(serial1[2] - serial0[2]) / Measured:F2}, body impulses {(serial1[3] - serial0[3]) / Measured:F2}, " +
                    $"subtree {(serial1[4] - serial0[4]) / Measured:F2}";
            }

            BoundaryCounters boundary = BoundaryCounters.Current;
            if (boundary.Solves > 0)
            {
                lastTimes += $"; vessel boundary: {boundary.Rows / (double)boundary.Solves:F0} rows and " +
                    $"{boundary.Rounds / (double)boundary.Solves:F2} pivoting rounds a solve ({boundary.CappedSolves} solves at the cap), " +
                    $"{boundary.HeldPoints / (double)boundary.Solves:F0} points held a solve, " +
                    $"{boundary.WarmRows / (double)boundary.Solves:F0} rows warm a solve";
            }

            if (path == "locked")
            {
                lastTimes += $"; sections {(sections1 - sections0) / (double)forestSolves:F0}, locked pairs {(locked1 - locked0) / (double)forestSolves:F0} " +
                    $"per solve, structure rebuilds {builds1 - builds0} ({(rebuild1 - rebuild0) / System.Math.Max(1, builds1 - builds0):F2} ms each), " +
                    $"lock changes {changes1 - changes0}, largest residual {residual:E1} N s";
            }
            JobsUtility.JobWorkerCount = workers;
            vesselSet?.Dispose();
            Unity.Physics.Solver.FusedVesselParts = fusedParts;
            Unity.Physics.Solver.FusedContacts = true;
            Unity.Physics.Solver.VesselFused = fusedOn;
            samples.Sort();
            return (samples[0], samples[samples.Count / 2], solves - solvesBefore, fallbacks - fallbacksBefore);
        }

        private struct SweepRun
        {
            public double Median;
            public Vector3[] Positions;
            public Vector3[] Forces;
            public Vector3[] Torques;
            public long Changes, ContactChanges, Rebuilds, FallbackBoundary, Fallbacks;
            public double RebuildMs, Sections, Locked;
            public long[] All, LockedBins;
        }

        private SweepRun Sweep(Graph graph, bool ground, bool locking, float lockAngle, float yieldAngle, int hold, int smoothing,
            int threads, int workers)
        {
            TearDownScene();
            SetUpScene();
            VesselForest.Enabled = true;
            VesselForest.ForgetIslands();
            string cap = Environment.GetEnvironmentVariable("RETOOLED_BENCH_CAP");
            VesselForest.BoundaryRowCap = string.IsNullOrEmpty(cap) ? VesselForest.DefaultBoundaryRowCap : int.Parse(cap, CultureInfo.InvariantCulture);
            VesselForest.Locking = locking;
            VesselForest.LockAngle = lockAngle;
            VesselForest.YieldAngle = yieldAngle;
            VesselForest.MinimumHoldSteps = hold;
            VesselForest.DeflectionSmoothingSteps = smoothing;
            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
            List<Rigidbody> bodies = Build(graph, ground);
            var joints = new List<ConfigurableJoint>();
            foreach (Rigidbody body in bodies)
            {
                joints.AddRange(body.GetComponents<ConfigurableJoint>());
            }

            Simulate(WarmUp);
            VesselForest.GetCounts(out long solves0, out long fallbacks0, out _);
            VesselForest.GetSectionCounts(out long sections0, out long locked0, out long builds0, out long changes0, out double rebuild0,
                out _, out _, out _);
            var all = new long[10];
            var lockedBins = new long[10];
            VesselForest.GetLockDiagnostics(out long contact0, out _, all, lockedBins);
            var samples = new List<double>(Measured);
            var forces = new Vector3[joints.Count];
            var torques = new Vector3[joints.Count];
            const int averaged = Measured / 2;
            for (int i = 0; i < Measured; i++)
            {
                long start = Stopwatch.GetTimestamp();
                UnityEngine.Physics.Simulate(Step);
                samples.Add((Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);

                // Loads averaged over the last half, outside the timing: a single step's are mostly vibration.
                for (int j = 0; i >= Measured - averaged && j < joints.Count; j++)
                {
                    forces[j] += joints[j].currentForce / averaged;
                    torques[j] += joints[j].currentTorque / averaged;
                }
            }

            JobsUtility.JobWorkerCount = workers;
            VesselForest.GetCounts(out long solves1, out long fallbacks1, out _);
            VesselForest.GetSectionCounts(out long sections1, out long locked1, out long builds1, out long changes1, out double rebuild1,
                out _, out _, out _);
            VesselForest.GetLockDiagnostics(out long contact1, out long fallbackBoundary, all, lockedBins);
            samples.Sort();
            long forestSolves = System.Math.Max(1, solves1 - solves0);
            return new SweepRun
            {
                Median = samples[samples.Count / 2],
                Positions = bodies.Select(b => b.position).ToArray(),
                Forces = forces,
                Torques = torques,
                Changes = changes1 - changes0,
                ContactChanges = contact1 - contact0,
                Rebuilds = builds1 - builds0,
                RebuildMs = (rebuild1 - rebuild0) / Measured,
                Sections = (sections1 - sections0) / (double)forestSolves,
                Locked = (locked1 - locked0) / (double)forestSolves,
                Fallbacks = fallbacks1 - fallbacks0,
                FallbackBoundary = fallbackBoundary,
                All = all,
                LockedBins = lockedBins,
            };
        }

        /// <summary>Summed difference over summed size, and the largest difference over the largest size.</summary>
        private static (double relative, double largest) Compare(Vector3[] run, Vector3[] reference)
        {
            double difference = 0.0, total = 0.0, largest = 0.0, scale = 0.0;
            for (int i = 0; i < run.Length; i++)
            {
                double d = (run[i] - reference[i]).magnitude;
                difference += d;
                total += reference[i].magnitude;
                largest = System.Math.Max(largest, d);
                scale = System.Math.Max(scale, reference[i].magnitude);
            }

            return (total > 0.0 ? difference / total : 0.0, scale > 0.0 ? largest / scale : largest);
        }

        private static string Bins(long[] all, long[] locked, int first)
        {
            return string.Join(" ", Enumerable.Range(first, 5).Select(i => $"{all[i]}({locked[i]})"));
        }

        /// <summary>
        /// The lock policy's settings against churn, cost and accuracy: for each craft and scene, the forest path without
        /// locking as the reference, then each setting of lock angle, yield angle, minimum hold and deflection smoothing
        /// (RETOOLED_SWEEP lists them as lock:yield:hold:smoothing in mrad and steps, separated by commas). Reports lock
        /// changes and rebuilds per step, the step's median, how far poses and joint loads end from the reference, and the
        /// tree pairs by deflection. Results go to the log and to VesselLockSweep.txt in Logs.
        /// </summary>
        [Test]
        [Timeout(7200000)]
        public void SweepLocking()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The forest is part of the Unity Physics backend.");
            }

            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            bool wasEnabled = VesselForest.Enabled, wasLocking = VesselForest.Locking;
            float wasLock = VesselForest.LockAngle, wasYield = VesselForest.YieldAngle;
            int wasHold = VesselForest.MinimumHoldSteps, wasSmoothing = VesselForest.DeflectionSmoothingSteps;
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(PartLayer, PartLayer);
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, true);
            bool jobDebugger = JobsUtility.JobDebuggerEnabled;
            JobsUtility.JobDebuggerEnabled = false;
            Unity.Collections.NativeLeakDetection.Mode = Unity.Collections.NativeLeakDetectionMode.Disabled;
            int workers = JobsUtility.JobWorkerCount;
            int threads = int.Parse(Choice("RETOOLED_BENCH_THREADS", "8")[0], CultureInfo.InvariantCulture);
            string[] settings = Choice("RETOOLED_SWEEP",
                "0.5:2:10:0,0.5:2:0:0,0.5:2:30:0,0.25:2:10:0,0.5:4:10:0,0.25:4:10:0,0.5:2:0:10,0.5:2:10:10,0.25:4:30:10");
            var report = new StringBuilder($"[VesselLockSweep] {threads} threads, {Measured} steps after {WarmUp}; " +
                $"CPU load at start {SampleLoad():F1}%" + Environment.NewLine);
            report.Append("craft | scene | lock/yield mrad | hold | smooth | changes/step (contact) | rebuilds/step, ms/step | median ms | " +
                "sections, locked pairs | pose drift rel / max m | force drift rel / max | torque drift rel / max | fallbacks (max boundary) | " +
                "pairs by deflection <L/2 <L <Y <2Y more, no contact (locked) | contact (locked)" + Environment.NewLine);
            try
            {
                foreach (string craft in Choice("RETOOLED_BENCH_CRAFT", "tu160,synthetic4500"))
                {
                    Graph graph = Load(craft);
                    foreach (bool ground in Choice("RETOOLED_BENCH_SCENE", "ground,floating").Select(s => s == "ground"))
                    {
                        string scene = ground ? "ground" : "floating";
                        SweepRun reference = Sweep(graph, ground, false, 0.5e-3f, 2e-3f, 10, 0, threads, workers);
                        report.Append($"{craft} | {scene} | unlocked | | | | | {reference.Median:F2} | | | " +
                            $"mean |F| {reference.Forces.Average(v => v.magnitude):F1} N | mean |T| {reference.Torques.Average(v => v.magnitude):F1} N m | " +
                            $"{reference.Fallbacks} ({reference.FallbackBoundary}) | |" + Environment.NewLine);
                        foreach (string setting in settings)
                        {
                            string[] f = setting.Split(':');
                            float lockAngle = float.Parse(f[0], CultureInfo.InvariantCulture) * 1e-3f;
                            float yieldAngle = float.Parse(f[1], CultureInfo.InvariantCulture) * 1e-3f;
                            int hold = int.Parse(f[2], CultureInfo.InvariantCulture);
                            int smoothing = int.Parse(f[3], CultureInfo.InvariantCulture);
                            SweepRun run = Sweep(graph, ground, true, lockAngle, yieldAngle, hold, smoothing, threads, workers);
                            double maxPose = 0.0;
                            for (int i = 0; i < run.Positions.Length; i++)
                            {
                                maxPose = System.Math.Max(maxPose, (run.Positions[i] - reference.Positions[i]).magnitude);
                            }

                            var pose = Compare(run.Positions, reference.Positions);
                            var force = Compare(run.Forces, reference.Forces);
                            var torque = Compare(run.Torques, reference.Torques);
                            report.Append($"{craft} | {scene} | {f[0]}/{f[1]} | {hold} | {smoothing} | " +
                                $"{run.Changes / (double)Measured:F1} ({run.ContactChanges / (double)Measured:F1}) | " +
                                $"{run.Rebuilds / (double)Measured:F2}, {run.RebuildMs:F2} | {run.Median:F2} | {run.Sections:F0}, {run.Locked:F0} | " +
                                $"{pose.relative:E1} / {maxPose:F4} | {force.relative:F3} / {force.largest:F3} | " +
                                $"{torque.relative:F3} / {torque.largest:F3} | {run.Fallbacks} ({run.FallbackBoundary}) | " +
                                $"{Bins(run.All, run.LockedBins, 0)} | {Bins(run.All, run.LockedBins, 5)}" + Environment.NewLine);
                        }
                    }
                }
            }
            finally
            {
                JobsUtility.JobWorkerCount = workers;
                JobsUtility.JobDebuggerEnabled = jobDebugger;
                VesselForest.Enabled = wasEnabled;
                VesselForest.Locking = wasLocking;
                VesselForest.LockAngle = wasLock;
                VesselForest.YieldAngle = wasYield;
                VesselForest.MinimumHoldSteps = wasHold;
                VesselForest.DeflectionSmoothingSteps = wasSmoothing;
                VesselForest.BoundaryRowCap = VesselForest.DefaultBoundaryRowCap;
                UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, !collided);
            }

            Debug.Log(report.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "VesselLockSweep.txt"), report.ToString());
        }

        /// <summary>A comma-separated list from the environment, to bench a part of the table, or the default.</summary>
        private static string[] Choice(string variable, string fallback)
        {
            string value = System.Environment.GetEnvironmentVariable(variable);
            return (string.IsNullOrEmpty(value) ? fallback : value).Split(',');
        }

        [Test]
        [Timeout(7200000)]
        public void Measure()
        {
            bool wasVessels = VesselPipeline.Enabled;
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The forest is part of the Unity Physics backend.");
            }

            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            bool wasEnabled = VesselForest.Enabled;
            bool wasLocking = VesselForest.Locking;
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(PartLayer, PartLayer);
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, true);
            bool jobDebugger = JobsUtility.JobDebuggerEnabled;
            JobsUtility.JobDebuggerEnabled = false;
            Unity.Collections.NativeLeakDetection.Mode = Unity.Collections.NativeLeakDetectionMode.Disabled;
            int workers = JobsUtility.JobWorkerCount;
            VesselPipeline.Initialize();
            fusedParts = Unity.Physics.Solver.FusedVesselParts;
            fusedOn = Unity.Physics.Solver.VesselFused;
            double load = SampleLoad();
            var report = new StringBuilder($"[VesselForestBench] whole Physics.Simulate step, ms min / median over {Measured} warm steps; " +
                $"CPU load at start {load:F1}%, {SystemInfo.processorCount} logical processors, {workers} job workers by default\n");
            string[] paths = Choice("RETOOLED_BENCH_PATHS", "sparse,forest,locked");
            report.Append($"craft | bodies | scene | threads | {string.Join(" | ", paths)} (ms min / median) | " +
                "island solve ms per step and counts, by path\n");
            try
            {
                foreach (string craft in Choice("RETOOLED_BENCH_CRAFT", "tu160,synthetic4500"))
                {
                    Graph graph = Fleet(Load(craft));
                    foreach (bool ground in Choice("RETOOLED_BENCH_SCENE", "floating,ground").Select(s => s == "ground"))
                    {
                        foreach (int threads in Choice("RETOOLED_BENCH_THREADS", "1,8").Select(int.Parse))
                        {
                            var cells = new List<string>();
                            var notes = new List<string>();
                            foreach (string path in paths)
                            {
                                var result = Session(graph, ground, path, threads, workers);
                                cells.Add($"{result.min:F2} / {result.median:F2}");
                                notes.Add(path == "sparse" ? $"{path}: {lastTimes}"
                                    : $"{path}: {result.solves} forest solves / {result.fallbacks} fallbacks, {lastTimes}");
                            }

                            report.Append($"{graph.Name} | {graph.Parent.Length} | {(ground ? "ground" : "floating")} | {threads} | " +
                                $"{string.Join(" | ", cells)} | {string.Join("; ", notes)}\n");
                        }
                    }
                }
            }
            finally
            {
                JobsUtility.JobWorkerCount = workers;
                JobsUtility.JobDebuggerEnabled = jobDebugger;
                VesselForest.Enabled = wasEnabled;
                VesselForest.Locking = wasLocking;
                VesselPipeline.Enabled = wasVessels;
                UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, !collided);
            }

            Debug.Log(report.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "VesselForestBench.txt"), report.ToString());
        }
    }
}
