using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Diagnostics;
using KSP.Retooled.Physics.Stepping;
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
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Jobs.LowLevel.Unsafe;
using UnityEngine;
using BoxCollider = UnityEngine.BoxCollider;
using Collider = UnityEngine.Collider;
using Debug = UnityEngine.Debug;
using Mesh = UnityEngine.Mesh;
using MeshCollider = UnityEngine.MeshCollider;
using Unity.Physics.Configuration;

namespace KSP.Retooled.Physics.Tests.Benchmarks
{
    /// <summary>
    /// Measures whole steps with compiled or supplied vessels, with optional locking.
    /// Select configurations with RETOOLED_BENCH_CRAFT, _SCENE, _THREADS and _PATHS.
    /// The paths are compiled, locked and vessel. Use parts=n after a colon to change the minimum size for fusing.
    /// Write timings and CPU load to Logs/VesselForestBench.txt.
    /// </summary>
    [Explicit("A benchmark, minutes long")]
    [Category("Slow")]
    public class VesselForestBench : PhysicsScene
    {
        internal const int PartLayer = 30;
        private const int WarmUp = 20;
        /// <summary>
        /// Measured steps per configuration, from RETOOLED_BENCH_STEPS, default 60.
        /// </summary>
        private static int Measured =>
            int.TryParse(Environment.GetEnvironmentVariable("RETOOLED_BENCH_STEPS"), out int steps) && steps > 0 ? steps : 60;

        internal struct Graph
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
        private static int fusedParts = SolverSwitches.DefaultFusedVesselParts;

        /// <summary>
        /// Load vessel graph scenes. Use tu160x4 for copies, tu160:128x16 for copies of a parent-first prefix,
        /// and + to combine scenes. Timeline accepts the same names.
        /// </summary>
        internal static Graph Load(string name)
        {
            string[] scenes = name.Split('+');
            if (scenes.Length > 1)
            {
                Graph joined = Load(scenes[0]);
                for (int i = 1; i < scenes.Length; i++)
                {
                    joined = Join(joined, Load(scenes[i]));
                }

                joined.Name = name;
                return joined;
            }

            System.Text.RegularExpressions.Match cut = System.Text.RegularExpressions.Regex.Match(name, @"^(.+):(\d+)x(\d+)$");
            if (cut.Success)
            {
                Graph whole = LoadFile(cut.Groups[1].Value);
                int parts = System.Math.Min(int.Parse(cut.Groups[2].Value, CultureInfo.InvariantCulture), whole.Parent.Length);
                return Copies(whole, parts, int.Parse(cut.Groups[3].Value, CultureInfo.InvariantCulture));
            }

            System.Text.RegularExpressions.Match copies = System.Text.RegularExpressions.Regex.Match(name, @"^(.+)x(\d+)$");
            if (copies.Success)
            {
                Graph single = LoadFile(copies.Groups[1].Value);
                return Copies(single, single.Parent.Length, int.Parse(copies.Groups[2].Value, CultureInfo.InvariantCulture));
            }

            return LoadFile(name);
        }

        /// <summary>
        /// Puts the scene <paramref name="right"/> beside the scene <paramref name="left"/> along x, 10 m past the
        /// right edge of the left scene, and returns the two as one scene.
        /// </summary>
        private static Graph Join(Graph left, Graph right)
        {
            float edge = left.Position.Max(p => p.x);
            float start = right.Position.Min(p => p.x);
            var offset = new Vector3(edge + 10.0f - start, 0.0f, 0.0f);
            int count = left.Parent.Length;
            return new Graph
            {
                Name = left.Name + " + " + right.Name,
                Parent = left.Parent.Concat(right.Parent.Select(p => p >= 0 ? p + count : -1)).ToArray(),
                Mass = left.Mass.Concat(right.Mass).ToArray(),
                Inertia = left.Inertia.Concat(right.Inertia).ToArray(),
                Position = left.Position.Concat(right.Position.Select(p => p + offset)).ToArray(),
                Joint = left.Joint.Concat(right.Joint.Select(p => p + offset)).ToArray(),
                Swing = left.Swing.Concat(right.Swing).ToArray(),
                ClosureA = left.ClosureA.Concat(right.ClosureA.Select(a => a + count)).ToArray(),
                ClosureB = left.ClosureB.Concat(right.ClosureB.Select(b => b + count)).ToArray(),
            };
        }

        private static Graph LoadFile(string name)
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
        /// Apply RETOOLED_BENCH_PARTS and RETOOLED_BENCH_COPIES to form a fleet from a connected parent-first
        /// prefix.
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

            return Copies(graph, parts, copies);
        }

        /// <summary>
        /// Lays out <paramref name="copies"/> copies of the first <paramref name="parts"/> parts of a craft side by side
        /// along x, 10 m apart.
        /// </summary>
        private static Graph Copies(Graph graph, int parts, int copies)
        {
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

        private static ConfigurableJoint Joint(Rigidbody host, Rigidbody target, Vector3 anchor, float spring, bool locked, float? flexLimit = null)
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
            if (flexLimit is float angle)
            {
                joint.angularXMotion = ConfigurableJointMotion.Limited;
                joint.lowAngularXLimit = new SoftJointLimit { limit = -angle };
                joint.highAngularXLimit = joint.angularYLimit = joint.angularZLimit = new SoftJointLimit { limit = angle };
                joint.angularXDrive = joint.angularYZDrive;
            }

            return joint;
        }

        private List<Rigidbody> Build(Graph graph, bool ground, bool joints = true)
        {
            return Build(graph, ground, Box, joints);
        }

        internal static List<Rigidbody> Build(Graph graph, bool ground, Func<Vector3, Vector3, GameObject> box, bool joints = true)
        {
            var bodies = new List<Rigidbody>(graph.Parent.Length);
            float lowest = float.MaxValue;
            for (int i = 0; i < graph.Parent.Length; i++)
            {
                float mass = Mathf.Max(graph.Mass[i], 1e-3f);
                float inertia = Mathf.Max((graph.Inertia[i].x + graph.Inertia[i].y + graph.Inertia[i].z) / 3.0f, 1e-6f);
                float side = Mathf.Clamp(Mathf.Sqrt(6.0f * inertia / mass), 0.05f, 5.0f);
                GameObject part = box(graph.Position[i], Vector3.one * side);
                part.layer = PartLayer;
                Rigidbody body = part.AddComponent<Rigidbody>();
                body.mass = mass;
                body.sleepThreshold = 0.0f;
                bodies.Add(body);
                lowest = Mathf.Min(lowest, graph.Position[i].y - 0.5f * side);
            }

            float? flexLimit = float.TryParse(Environment.GetEnvironmentVariable("RETOOLED_BENCH_FLEX_LIMIT"),
                NumberStyles.Float, CultureInfo.InvariantCulture, out float angle) ? angle : null;
            for (int i = 0; i < graph.Parent.Length && joints; i++)
            {
                int parent = graph.Parent[i];
                if (parent >= 0)
                {
                    Joint(bodies[i], bodies[parent], graph.Joint[i], graph.Swing[i] > 0.0f ? graph.Swing[i] : 1e5f, false, flexLimit);
                }
            }

            for (int c = 0; c < graph.ClosureA.Length && joints; c++)
            {
                int a = graph.ClosureA[c], b = graph.ClosureB[c];
                Joint(bodies[b], bodies[a], 0.5f * (graph.Position[a] + graph.Position[b]), 0.0f, true);
            }

            if (ground)
            {
                // The ground is 400 m square under the origin. A fleet of copies that reaches past it gets a ground
                // that covers the whole fleet with 50 m to spare.
                float left = float.MaxValue, right = float.MinValue;
                for (int i = 0; i < graph.Parent.Length; i++)
                {
                    left = Mathf.Min(left, graph.Position[i].x);
                    right = Mathf.Max(right, graph.Position[i].x);
                }

                bool fits = left > -190.0f && right < 190.0f;
                float center = fits ? 0.0f : 0.5f * (left + right);
                float width = fits ? 400.0f : right - left + 100.0f;
                box(new Vector3(center, lowest - 0.5f, 0.0f), new Vector3(width, 1.0f, 400.0f));
            }

            return bodies;
        }

        /// <summary>
        /// Scene name for a grounded craft among unchanged static colliders, used by the benchmark and profiler.
        /// </summary>
        internal const string StaticsScene = "statics";

        /// <summary>The number of static colliders that <see cref="StaticField"/> makes.</summary>
        internal const int StaticFieldCount = 5000;

        private static Mesh hullMesh;

        /// <summary>
        /// A mesh with twelve vertices, a stretched icosahedron, for the static convex hulls of <see cref="StaticField"/>.
        /// All of the hulls share it.
        /// </summary>
        internal static Mesh HullMesh()
        {
            if (hullMesh != null)
            {
                return hullMesh;
            }

            float g = 0.5f * (1.0f + Mathf.Sqrt(5.0f));
            var vertices = new[]
            {
                new Vector3(-1, g, 0), new Vector3(1, g, 0), new Vector3(-1, -g, 0), new Vector3(1, -g, 0),
                new Vector3(0, -1, g), new Vector3(0, 1, g), new Vector3(0, -1, -g), new Vector3(0, 1, -g),
                new Vector3(g, 0, -1), new Vector3(g, 0, 1), new Vector3(-g, 0, -1), new Vector3(-g, 0, 1),
            };

            for (int i = 0; i < vertices.Length; i++)
            {
                vertices[i] = Vector3.Scale(vertices[i], new Vector3(0.9f, 0.6f, 0.7f));
            }

            hullMesh = new Mesh
            {
                vertices = vertices,
                triangles = new[]
                {
                    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                    3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
                },
                name = "bench hull",
            };
            return hullMesh;
        }

        /// <summary>
        /// Create a grid of alternating static boxes and hulls, leaving space around the craft.
        /// groundTop sets the surface height; track registers objects for teardown.
        /// </summary>
        internal static List<Collider> StaticField(Graph graph, float groundTop, Func<GameObject, GameObject> track)
        {
            float left = graph.Position.Min(p => p.x) - 10.0f, right = graph.Position.Max(p => p.x) + 10.0f;
            float back = graph.Position.Min(p => p.z) - 10.0f, front = graph.Position.Max(p => p.z) + 10.0f;
            var colliders = new List<Collider>(StaticFieldCount);
            const float Spacing = 5.0f;
            int side = (int)Mathf.Ceil(Mathf.Sqrt(StaticFieldCount * 1.5f));
            for (int cell = 0; colliders.Count < StaticFieldCount; cell++)
            {
                float x = Spacing * (cell % side - side / 2);
                float z = Spacing * (cell / side - side / 2);
                if (x > left && x < right && z > back && z < front)
                {
                    continue;
                }

                GameObject item = track(new GameObject($"static {colliders.Count}"));
                item.transform.rotation = Quaternion.Euler(0.0f, 37.0f * colliders.Count, 0.0f);
                if (colliders.Count % 2 == 0)
                {
                    item.transform.position = new Vector3(x, groundTop + 1.0f, z);
                    BoxCollider box = item.AddComponent<BoxCollider>();
                    box.size = new Vector3(2.0f, 2.0f, 3.0f);
                    colliders.Add(box);
                }
                else
                {
                    item.transform.position = new Vector3(x, groundTop + 0.6f * 1.618034f, z);
                    MeshCollider hull = item.AddComponent<MeshCollider>();
                    hull.sharedMesh = HullMesh();
                    hull.convex = true;
                    colliders.Add(hull);
                }
            }

            return colliders;
        }

        /// <summary>
        /// The height of the top face of the ground that <see cref="Build"/> makes under the craft.
        /// </summary>
        internal static float GroundTop(Graph graph)
        {
            float lowest = float.MaxValue;
            for (int i = 0; i < graph.Parent.Length; i++)
            {
                float mass = Mathf.Max(graph.Mass[i], 1e-3f);
                float inertia = Mathf.Max((graph.Inertia[i].x + graph.Inertia[i].y + graph.Inertia[i].z) / 3.0f, 1e-6f);
                float side = Mathf.Clamp(Mathf.Sqrt(6.0f * inertia / mass), 0.05f, 5.0f);
                lowest = Mathf.Min(lowest, graph.Position[i].y - 0.5f * side);
            }

            return lowest;
        }

        /// <summary>
        /// Supplied vessel set with parent-first parts, spring edges and rigid strut closures.
        /// Resolve body indices once, since the benchmark adds no bodies after construction.
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

                PhysicsWorld world = StepEntryPoints.Current->World;
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

        [System.Runtime.InteropServices.DllImport("kernel32.dll")]
        private static extern bool GetSystemTimes(out long idle, out long kernel, out long user);

        /// <summary>
        /// System and process CPU times in 100 ns units, summed over logical processors.
        /// </summary>
        internal struct CpuReading
        {
            public long Busy;
            public long Total;
            public long Own;

            public static CpuReading Take()
            {
                try
                {
                    if (!GetSystemTimes(out long idle, out long kernel, out long user))
                    {
                        return default;
                    }

                    // The kernel time that Windows reports includes the idle time.
                    using Process self = Process.GetCurrentProcess();
                    return new CpuReading { Busy = kernel + user - idle, Total = kernel + user, Own = self.TotalProcessorTime.Ticks };
                }
                catch (Exception)
                {
                    return default;
                }
            }

            /// <summary>
            /// The share of all the logical processors, in percent, that processes other than this one kept busy between
            /// <paramref name="start"/> and this reading, or -1 when the readings failed.
            /// </summary>
            public double OthersLoadSince(CpuReading start)
            {
                long total = Total - start.Total;
                if (total <= 0 || start.Total == 0)
                {
                    return -1.0;
                }

                return System.Math.Max(0.0, 100.0 * ((Busy - start.Busy) - (Own - start.Own)) / total);
            }
        }

        /// <summary>The result of measuring one scene on one path with one number of threads.</summary>
        private struct SessionResult
        {
            public double Min;
            public double Median;
            public long Solves;
            public long Fallbacks;

            /// <summary>The island solve's times and counts, for the notes column.</summary>
            public string Times;

            /// <summary>Where the session's time went before and during the measured steps, and the load of other processes.</summary>
            public string SetUp;
        }

        /// <summary>
        /// Number of measured step blocks per scene, from RETOOLED_BENCH_BLOCKS, default one.
        /// Report the minimum and median per block.
        /// </summary>
        private static int Blocks =>
            int.TryParse(Environment.GetEnvironmentVariable("RETOOLED_BENCH_BLOCKS"), out int blocks) && blocks > 0 ? blocks : 1;

        private string lastTimes;

        /// <summary>
        /// Builds the scene once and measures it with each number of threads in <paramref name="threadCounts"/>, in that
        /// order, after warm-up steps with that number of threads. Returns one result for each number of threads.
        /// </summary>
        private List<SessionResult> Session(Graph graph, string scene, string path, int[] threadCounts, int workers)
        {
            bool ground = scene != "floating";
            CpuReading setUpStart = CpuReading.Take();
            long clock = Stopwatch.GetTimestamp();
            TearDownScene();
            SetUpScene();
            double clear = Since(clock);
            // The parts option sets the minimum island size for the fused job.
            string[] options = path.Split(':');
            path = options[0];
            Assert.That(path, Is.EqualTo("compiled").Or.EqualTo("locked").Or.EqualTo("vessel"), "the benchmark path");
            SolverSwitches.FusedVesselParts = fusedParts;
            for (int i = 1; i < options.Length; i++)
            {
                Assert.That(options[i], Does.StartWith("parts="), "the benchmark option");
                SolverSwitches.FusedVesselParts = int.Parse(options[i].Substring("parts=".Length), CultureInfo.InvariantCulture);
            }

            ForestDiagnostics.ForgetIslands();
            ForestSwitches.Locking = path == "locked";
            string cap = Environment.GetEnvironmentVariable("RETOOLED_BENCH_CAP");
            ForestSwitches.BoundaryRowCap = string.IsNullOrEmpty(cap) ? ForestSwitches.DefaultBoundaryRowCap : int.Parse(cap, CultureInfo.InvariantCulture);
            string hold = Environment.GetEnvironmentVariable("RETOOLED_BENCH_HOLD");
            ForestSwitches.MinimumHoldSteps = string.IsNullOrEmpty(hold) ? -1 : int.Parse(hold, CultureInfo.InvariantCulture);

            // Boundary modes: cold, uncapped, refactor, prefix or unordered.
            // The uncapped mode still limits self-contact to 25 manifolds.
            string boundaryMode = Environment.GetEnvironmentVariable("RETOOLED_BENCH_BOUNDARY");
            ForestSwitches.Caps = boundaryMode == "uncapped"
                ? new BoundaryCaps { OutsideContactPoints = -1, PointsPerManifold = -1, SelfContactManifolds = -1, PivotRounds = -1, ColdStart = true }
                : new BoundaryCaps { ColdStart = boundaryMode == "cold", RefactorEveryRound = boundaryMode == "refactor", KeepFactorPrefix = boundaryMode == "prefix", UnorderedFactor = boundaryMode == "unordered" };
            JobsUtility.JobWorkerCount = threadCounts[0] > 1 ? threadCounts[0] - 1 : 0;
            clock = Stopwatch.GetTimestamp();
            List<Rigidbody> built = Build(graph, ground, path != "vessel");
            if (scene == StaticsScene)
            {
                StaticField(graph, GroundTop(graph), Track);
            }

            double build = Since(clock);

            // The first step builds the world from the new bodies, colliders and joints, and compiles the craft's vessels.
            // The vessel path takes one step more than the others, since it builds its vessel set after this step.
            clock = Stopwatch.GetTimestamp();
            Simulate();
            double first = Since(clock);
            BenchVesselSet vesselSet = null;
            if (path == "vessel")
            {
                vesselSet = new BenchVesselSet(graph, built);
            }

            double setUpLoad = CpuReading.Take().OthersLoadSince(setUpStart);
            var results = new List<SessionResult>(threadCounts.Length);
            for (int run = 0; run < threadCounts.Length; run++)
            {
                results.Add(MeasureSession(path, threadCounts[run], run == 0 && path != "vessel" ? WarmUp - 1 : WarmUp,
                    run == 0 ? $"clear {clear:F0} ms, build {build:F0} ms, first step {first:F0} ms, other processes' load {setUpLoad:F0}%; "
                        : "scene shared with the first number of threads; "));
            }

            JobsUtility.JobWorkerCount = workers;
            vesselSet?.Dispose();
            SolverSwitches.FusedVesselParts = fusedParts;
            return results;
        }

        /// <summary>
        /// Takes <paramref name="warmUp"/> steps with <paramref name="threads"/> threads, then measures the scene's steps
        /// in <see cref="Blocks"/> blocks of <see cref="Measured"/> steps.
        /// </summary>
        private SessionResult MeasureSession(string path, int threads, int warmUp, string setUp)
        {
            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
            long clock = Stopwatch.GetTimestamp();
            Simulate(warmUp);
            double warm = Since(clock);
            int steps = Measured * Blocks;
            ForestDiagnostics.GetCounts(out long solvesBefore, out long fallbacksBefore, out _);
            ForestDiagnostics.GetTimes(out double s0, out double a0, out double b0, out double c0, out double f0, out double w0, out double d0);
            ForestDiagnostics.GetSectionCounts(out long sections0, out long locked0, out long builds0, out long changes0, out double rebuild0,
                out _, out _, out _);
            double settle0 = ForestDiagnostics.SettleMilliseconds;
            double[] serial0 = ForestDiagnostics.SerialMilliseconds();
            BoundaryCounters.Reset();
            WorkTeamCounters.Reset();

            // RETOOLED_BENCH_HISTOGRAM=1 counts the sizes of the blocks that the sparse solver factors. Counting costs
            // time on every lane, so the times of such a run are not comparable with others.
            SparseBlockCounters.Enabled = Environment.GetEnvironmentVariable("RETOOLED_BENCH_HISTOGRAM") == "1";
            SparseBlockCounters.Reset();
            var samples = new List<double>(steps);
            var blockMedians = new List<string>(Blocks);
            CpuReading measureStart = CpuReading.Take();
            clock = Stopwatch.GetTimestamp();
            for (int block = 0; block < Blocks; block++)
            {
                var blockSamples = new List<double>(Measured);
                for (int i = 0; i < Measured; i++)
                {
                    long start = Stopwatch.GetTimestamp();
                    UnityEngine.Physics.Simulate(Step);
                    blockSamples.Add((Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
                }

                samples.AddRange(blockSamples);
                blockSamples.Sort();
                blockMedians.Add(blockSamples[blockSamples.Count / 2].ToString("F2", CultureInfo.InvariantCulture));
            }

            double measured = Since(clock);
            double measureLoad = CpuReading.Take().OthersLoadSince(measureStart);
            setUp += $"warm-up {warm:F0} ms ({warmUp} steps), measured {measured:F0} ms ({steps} steps), " +
                $"other processes' load {measureLoad:F0}%; block medians {string.Join(" ", blockMedians)}";
            ForestDiagnostics.GetCounts(out long solves, out long fallbacks, out _);
            ForestDiagnostics.GetTimes(out double s1, out double a1, out double b1, out double c1, out double f1, out double w1, out double d1);
            ForestDiagnostics.GetSectionCounts(out long sections1, out long locked1, out long builds1, out long changes1, out double rebuild1,
                out double residual, out _, out _);
            long forestSolves = System.Math.Max(1, solves - solvesBefore);
            lastTimes = $"start {(a1 - a0) / steps:F2} (factor {(f1 - f0) / steps:F2}, boundary {(w1 - w0) / steps:F2}), " +
                $"solve {(b1 - b0) / steps:F2}, finish {(c1 - c0) / steps:F2} (settle {(ForestDiagnostics.SettleMilliseconds - settle0) / steps:F2}, " +
                $"drift aim {(d1 - d0) / steps:F2}), " +
                $"fallback sparse {(s1 - s0) / steps:F2}";
            double[] serial1 = ForestDiagnostics.SerialMilliseconds();
            lastTimes += $"; one lane: classify {(serial1[0] - serial0[0]) / steps:F2}, decide {(serial1[1] - serial0[1]) / steps:F2}, " +
                $"section sums {(serial1[2] - serial0[2]) / steps:F2}, body impulses {(serial1[3] - serial0[3]) / steps:F2}, " +
                $"subtree {(serial1[4] - serial0[4]) / steps:F2}";

            // How fast the teams' helpers reached the stages that the leaders published.
            WorkTeamCounters handOff = WorkTeamCounters.Current;
            if (handOff.Stages > 0)
            {
                double tickUs = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                    / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-3;
                lastTimes += $"; teams: {handOff.Stages / (double)steps:F1} stages published a step, " +
                    $"{100.0 * handOff.HelpedStages / handOff.Stages:F0}% of them helped, first helper after " +
                    $"{handOff.ArrivalTicks * tickUs / System.Math.Max(1, handOff.HelpedStages):F1} us on average";
            }

            if (SparseBlockCounters.Enabled)
            {
                SparseBlockCounters.Enabled = false;
                lastTimes += "; " + SparseHistogram(SparseBlockCounters.Current, steps);
            }

            BoundaryCounters boundary = BoundaryCounters.Current;
            if (boundary.Solves > 0)
            {
                lastTimes += $"; vessel boundary: {boundary.Rows / (double)boundary.Solves:F0} rows and " +
                    $"{boundary.Rounds / (double)boundary.Solves:F2} pivoting rounds a solve ({boundary.CappedSolves} solves at the cap, " +
                    $"{boundary.UnsettledSolves} of them without an answer), " +
                    $"{boundary.HeldPoints / (double)boundary.Solves:F0} points held a solve " +
                    $"({boundary.EmptiedOutsideManifolds} manifolds left with no point, {boundary.HeldReleases} held rows gave back " +
                    $"their warm start, {boundary.HeldSupports} held rows pushed, {boundary.HeldPushes} held rows pushed out), " +
                    $"{boundary.DormantRows / (double)boundary.Solves:F1} dormant rows a solve, " +
                    $"{boundary.WarmRows / (double)boundary.Solves:F0} rows warm a solve";
                double rowSamples = System.Math.Max(1, boundary.RowKindSamples);
                lastTimes += $"; rows per build: {boundary.ContactRows / rowSamples:F1} contact, " +
                    $"{boundary.AngularStopRows / rowSamples:F1} angular stops, {boundary.OtherRows / rowSamples:F1} other, " +
                    $"{boundary.StopRetries} stop retries";
                double tickMs = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Numerator
                    / (double)Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio.Denominator * 1e-6;
                Unity.Numerics.Linear.Solvers.Lowlevel.BlockPrincipalPivotingStats f = boundary.Factor;
                long laterSolves = System.Math.Max(1, boundary.Solves - boundary.FirstSubstepSolves);
                lastTimes += $"; dense solve {boundary.SolveTicks * tickMs / steps:F3} ms a step (factor {f.FactorTicks * tickMs / steps:F3}, of it from the start {f.FullFactorTicks * tickMs / steps:F3}, solves {f.SolveTicks * tickMs / steps:F3}, checks {f.PivotTicks * tickMs / steps:F3}); " +
                    $"rounds a solve: {f.Factorizations / (double)boundary.Solves:F2} factored, {f.Updates / (double)boundary.Solves:F2} updated, " +
                    $"{f.Reuses / (double)boundary.Solves:F2} reused, {f.CouplingRounds / (double)boundary.Solves:F2} coupling rounds; " +
                    $"free rows a round {f.FreeRows / (double)System.Math.Max(1, f.Rounds):F1} ({f.FewestFreeRows} to {f.MostFreeRows}); " +
                    $"rows that joined / left the free set a solve {f.AddedRows / (double)boundary.Solves:F1} / {f.RemovedRows / (double)boundary.Solves:F1}; " +
                    $"warm rows that changed state a solve: first substep {boundary.FirstSubstepChanges / (double)System.Math.Max(1, boundary.FirstSubstepSolves):F1}, " +
                    $"later substeps {boundary.LaterSubstepChanges / (double)laterSolves:F1}; " +
                    $"build ms a step: choose rows {boundary.ChooseTicks * tickMs / steps:F3}, paths {boundary.PathTicks * tickMs / steps:F3}, " +
                    $"fill W {boundary.FillTicks * tickMs / steps:F3}; apply {boundary.ApplyTicks * tickMs / steps:F3}, " +
                    $"push-out {boundary.PushOutTicks * tickMs / steps:F3} ({boundary.PushSolves / (double)steps:F1} solves a step, " +
                    $"{boundary.PushRounds / (double)System.Math.Max(1, boundary.PushSolves):F1} rounds each, dense {boundary.PushTicks * tickMs / steps:F3})";
            }

            if (path == "locked" || path == "compiled")
            {
                lastTimes += $"; sections {(sections1 - sections0) / (double)forestSolves:F0}, locked pairs {(locked1 - locked0) / (double)forestSolves:F0} " +
                    $"per solve, structure rebuilds {builds1 - builds0} ({(rebuild1 - rebuild0) / System.Math.Max(1, builds1 - builds0):F2} ms each), " +
                    $"lock changes {changes1 - changes0}, largest residual {residual:E1} N s";
            }

            samples.Sort();
            return new SessionResult
            {
                Min = samples[0],
                Median = samples[samples.Count / 2],
                Solves = solves - solvesBefore,
                Fallbacks = fallbacks - fallbacksBefore,
                Times = lastTimes,
                SetUp = setUp,
            };
        }

        /// <summary>
        /// Format sparse block histograms by node unknowns, later neighbours and off-diagonal block dimensions.
        /// The final bin includes all sizes at least SparseBlockCounters.Largest - 1.
        /// </summary>
        private static unsafe string SparseHistogram(SparseBlockCounters counts, int steps)
        {
            int last = SparseBlockCounters.Largest - 1;
            string Bin(int i) => i == last ? $"{i}+" : $"{i}";
            var sizes = new List<string>();
            var neighbours = new List<string>();
            var shapes = new List<string>();
            for (int i = 0; i <= last; i++)
            {
                if (counts.NodeSizes[i] > 0)
                {
                    sizes.Add($"{Bin(i)}: {counts.NodeSizes[i] / (double)steps:F0}");
                }

                if (counts.NeighbourCounts[i] > 0)
                {
                    neighbours.Add($"{Bin(i)}: {counts.NeighbourCounts[i] / (double)steps:F0}");
                }

                for (int j = 0; j <= last; j++)
                {
                    long count = counts.BlockShapes[i * SparseBlockCounters.Largest + j];
                    if (count > 0)
                    {
                        shapes.Add($"{Bin(i)}x{Bin(j)}: {count / (double)steps:F0}");
                    }
                }
            }

            return $"sparse factors a step {counts.Factorizations / (double)steps:F1}; nodes a step by size {{{string.Join(", ", sizes)}}}, " +
                $"by later neighbours {{{string.Join(", ", neighbours)}}}; blocks below the diagonal a step by rows x columns {{{string.Join(", ", shapes)}}}";
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
            ForestDiagnostics.ForgetIslands();
            string cap = Environment.GetEnvironmentVariable("RETOOLED_BENCH_CAP");
            ForestSwitches.BoundaryRowCap = string.IsNullOrEmpty(cap) ? ForestSwitches.DefaultBoundaryRowCap : int.Parse(cap, CultureInfo.InvariantCulture);
            ForestSwitches.Locking = locking;
            ForestSwitches.LockAngle = lockAngle;
            ForestSwitches.YieldAngle = yieldAngle;
            ForestSwitches.MinimumHoldSteps = hold;
            ForestSwitches.DeflectionSmoothingSteps = smoothing;
            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
            List<Rigidbody> bodies = Build(graph, ground);
            var joints = new List<ConfigurableJoint>();
            foreach (Rigidbody body in bodies)
            {
                joints.AddRange(body.GetComponents<ConfigurableJoint>());
            }

            Simulate(WarmUp);
            ForestDiagnostics.GetCounts(out long solves0, out long fallbacks0, out _);
            ForestDiagnostics.GetSectionCounts(out long sections0, out long locked0, out long builds0, out long changes0, out double rebuild0,
                out _, out _, out _);
            var all = new long[10];
            var lockedBins = new long[10];
            ForestDiagnostics.GetLockDiagnostics(out long contact0, out _, all, lockedBins);
            var samples = new List<double>(Measured);
            var forces = new Vector3[joints.Count];
            var torques = new Vector3[joints.Count];
            int averaged = Measured / 2;
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
            ForestDiagnostics.GetCounts(out long solves1, out long fallbacks1, out _);
            ForestDiagnostics.GetSectionCounts(out long sections1, out long locked1, out long builds1, out long changes1, out double rebuild1,
                out _, out _, out _);
            ForestDiagnostics.GetLockDiagnostics(out long contact1, out long fallbackBoundary, all, lockedBins);
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
        /// Sweep lock/yield angles, hold duration and smoothing from RETOOLED_SWEEP (lock:yield:hold:smoothing).
        /// Compare cost, lock churn, poses and loads with unlocked solving. Write Logs/VesselLockSweep.txt.
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
            bool wasLocking = ForestSwitches.Locking;
            float wasLock = ForestSwitches.LockAngle, wasYield = ForestSwitches.YieldAngle;
            int wasHold = ForestSwitches.MinimumHoldSteps, wasSmoothing = ForestSwitches.DeflectionSmoothingSteps;
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
                ForestSwitches.Locking = wasLocking;
                ForestSwitches.LockAngle = wasLock;
                ForestSwitches.YieldAngle = wasYield;
                ForestSwitches.MinimumHoldSteps = wasHold;
                ForestSwitches.DeflectionSmoothingSteps = wasSmoothing;
                ForestSwitches.BoundaryRowCap = ForestSwitches.DefaultBoundaryRowCap;
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

        private static string MinMedian(List<double> samples)
        {
            samples.Sort();
            return $"{samples[0]:F3} / {samples[samples.Count / 2]:F3}";
        }

        private static double Since(long start) => (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;

        /// <summary>
        /// Time Unity Prepare, Run and Publish stages separately and compare with Physics.Simulate.
        /// Configure children, craft, scene and threads through RETOOLED_BENCH_* variables.
        /// Write Logs/VesselStageBench.txt.
        /// </summary>
        [Test]
        [Timeout(7200000)]
        public void MeasureStages()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The bench measures the Unity Physics backend.");
            }

            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            bool wasLocking = ForestSwitches.Locking;
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(PartLayer, PartLayer);
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, true);
            bool jobDebugger = JobsUtility.JobDebuggerEnabled;
            JobsUtility.JobDebuggerEnabled = false;
            Unity.Collections.NativeLeakDetection.Mode = Unity.Collections.NativeLeakDetectionMode.Disabled;
            int workers = JobsUtility.JobWorkerCount;
            int children = int.Parse(Choice("RETOOLED_BENCH_CHILDREN", "0")[0], CultureInfo.InvariantCulture);
            var report = new StringBuilder($"[VesselStageBench] compiled path, ms min / median over {Measured} steps after {WarmUp}, " +
                $"{children} child Transforms a part; CPU load at start {SampleLoad():F1}%\n");
            report.Append("craft | bodies | scene | threads | prepare | run | publish | whole step\n");
            try
            {
                ForestSwitches.Locking = false;
                foreach (string craft in Choice("RETOOLED_BENCH_CRAFT", "tu160,tu160x4"))
                {
                    Graph graph = Fleet(Load(craft));
                    foreach (bool ground in Choice("RETOOLED_BENCH_SCENE", "floating").Select(s => s == "ground"))
                    {
                        foreach (int threads in Choice("RETOOLED_BENCH_THREADS", "8,1").Select(int.Parse))
                        {
                            TearDownScene();
                            SetUpScene();
                            UnityEngine.Physics.autoSyncTransforms = false;
                            ForestDiagnostics.ForgetIslands();
                            JobsUtility.JobWorkerCount = threads > 1 ? threads - 1 : 0;
                            List<Rigidbody> bodies = Build(graph, ground);
                            foreach (Rigidbody body in bodies)
                            {
                                Transform parent = body.transform;
                                for (int c = 0; c < children; c++)
                                {
                                    var child = new GameObject("model").transform;
                                    child.SetParent(parent, false);
                                    parent = c == 0 ? child : parent;
                                }
                            }

                            Simulate(WarmUp);
                            UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
                            var prepare = new List<double>();
                            var run = new List<double>();
                            var publish = new List<double>();
                            var whole = new List<double>();
                            for (int i = 0; i < Measured; i++)
                            {
                                long start = Stopwatch.GetTimestamp();
                                scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation, SimulationOption.All);
                                prepare.Add(Since(start));
                                start = Stopwatch.GetTimestamp();
                                scene.RunSimulationStages(Step, SimulationStage.RunSimulation, SimulationOption.All);
                                run.Add(Since(start));
                                start = Stopwatch.GetTimestamp();
                                scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
                                publish.Add(Since(start));
                            }

                            for (int i = 0; i < Measured; i++)
                            {
                                long start = Stopwatch.GetTimestamp();
                                UnityEngine.Physics.Simulate(Step);
                                whole.Add(Since(start));
                            }

                            report.Append($"{graph.Name} | {graph.Parent.Length} | {(ground ? "ground" : "floating")} | {threads} | " +
                                $"{MinMedian(prepare)} | {MinMedian(run)} | {MinMedian(publish)} | {MinMedian(whole)}\n");
                        }
                    }
                }

                report.Append($"CPU load at the end {SampleLoad():F1}%\n");
            }
            finally
            {
                JobsUtility.JobWorkerCount = workers;
                JobsUtility.JobDebuggerEnabled = jobDebugger;
                ForestSwitches.Locking = wasLocking;
                UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, !collided);
            }

            Debug.Log(report.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "VesselStageBench.txt"), report.ToString());
        }

        /// <summary>
        /// Time whole steps across craft, scene, thread and path configurations.
        /// RETOOLED_BENCH_SHARE=1 reuses scenes across thread counts; RETOOLED_BENCH_BLOCKS sets repeated
        /// measurements.
        /// Report setup costs and competing CPU load alongside timings.
        /// </summary>
        [Test]
        [Timeout(7200000)]
        public void Measure()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The forest is part of the Unity Physics backend.");
            }

            Assert.IsTrue(BurstCompiler.IsEnabled, "Burst is disabled");
            long benchStart = Stopwatch.GetTimestamp();
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            double kernelWait = Since(benchStart);
            bool wasLocking = ForestSwitches.Locking;
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(PartLayer, PartLayer);
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, true);
            bool jobDebugger = JobsUtility.JobDebuggerEnabled;
            JobsUtility.JobDebuggerEnabled = false;
            Unity.Collections.NativeLeakDetection.Mode = Unity.Collections.NativeLeakDetectionMode.Disabled;
            int workers = JobsUtility.JobWorkerCount;
            EnvironmentSwitches.Read();
            fusedParts = SolverSwitches.FusedVesselParts;
            double load = SampleLoad();
            bool share = Environment.GetEnvironmentVariable("RETOOLED_BENCH_SHARE") == "1";
            var report = new StringBuilder($"[VesselForestBench] whole Physics.Simulate step, ms min / median over {Measured} warm steps; " +
                $"CPU load at start {load:F1}%, {SystemInfo.processorCount} logical processors, {workers} job workers by default; " +
                $"{kernelWait / 1000.0:F1} s waiting for the compiled kernels, {Blocks} blocks of measured steps, " +
                $"{(share ? "one scene for every number of threads" : "a new scene for each number of threads")}\n");
            string[] paths = Choice("RETOOLED_BENCH_PATHS", "compiled");
            report.Append($"craft | bodies | scene | threads | {string.Join(" | ", paths)} (ms min / median) | " +
                "island solve ms per step and counts, by path | set-up times and other processes' load, by path\n");
            int[] threadCounts = Choice("RETOOLED_BENCH_THREADS", "1,8").Select(int.Parse).ToArray();
            try
            {
                foreach (string craft in Choice("RETOOLED_BENCH_CRAFT", "tu160,synthetic4500"))
                {
                    long loadStart = Stopwatch.GetTimestamp();
                    Graph graph = Fleet(Load(craft));
                    double loadTime = Since(loadStart);
                    // RETOOLED_BENCH_SCENE can also name StaticsScene, the ground among thousands of static colliders.
                    foreach (string scene in Choice("RETOOLED_BENCH_SCENE", "floating,ground"))
                    {
                        // results[t][p] is the result for the number of threads threadCounts[t] on the path paths[p].
                        var results = new SessionResult[threadCounts.Length][];
                        for (int t = 0; t < threadCounts.Length; t++)
                        {
                            results[t] = new SessionResult[paths.Length];
                        }

                        if (share)
                        {
                            for (int p = 0; p < paths.Length; p++)
                            {
                                List<SessionResult> runs = Session(graph, scene, paths[p], threadCounts, workers);
                                for (int t = 0; t < threadCounts.Length; t++)
                                {
                                    results[t][p] = runs[t];
                                }
                            }
                        }
                        else
                        {
                            for (int t = 0; t < threadCounts.Length; t++)
                            {
                                for (int p = 0; p < paths.Length; p++)
                                {
                                    results[t][p] = Session(graph, scene, paths[p], new[] { threadCounts[t] }, workers)[0];
                                }
                            }
                        }

                        for (int t = 0; t < threadCounts.Length; t++)
                        {
                            var cells = new List<string>();
                            var notes = new List<string>();
                            var setUps = new List<string>();
                            for (int p = 0; p < paths.Length; p++)
                            {
                                SessionResult result = results[t][p];
                                cells.Add($"{result.Min:F2} / {result.Median:F2}");
                                notes.Add($"{paths[p]}: {result.Solves} forest solves / {result.Fallbacks} fallbacks, {result.Times}");
                                setUps.Add($"{paths[p]}: {result.SetUp}");
                            }

                            report.Append($"{graph.Name} | {graph.Parent.Length} | {scene} | {threadCounts[t]} | " +
                                $"{string.Join(" | ", cells)} | {string.Join("; ", notes)} | " +
                                $"load the craft {loadTime:F0} ms; {string.Join(" / ", setUps)}\n");
                        }
                    }
                }

                report.Append($"The bench took {Since(benchStart) / 1000.0:F1} s in all\n");
            }
            finally
            {
                JobsUtility.JobWorkerCount = workers;
                JobsUtility.JobDebuggerEnabled = jobDebugger;
                ForestSwitches.Locking = wasLocking;
                UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, !collided);
            }

            Debug.Log(report.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "VesselForestBench.txt"), report.ToString());
        }
    }
}
