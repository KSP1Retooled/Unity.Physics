using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Physics;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Load a vessel graph from Tests/VesselGraphs~ and prepare solver inputs under a lock rule, gravity and
    /// thrust.
    /// </summary>
    public sealed class VesselGraph
    {
        /// <summary>Which edges are live, as in the section solver spec's bench cases.</summary>
        public enum LockRule
        {
            /// <summary>Case 1: only servos, docking ports and decouplers live.</summary>
            LiveJointsOnly,

            /// <summary>Case 2: stack edges locked, every surface attachment yielding (rigid attachment stays locked).</summary>
            StackLocked,

            /// <summary>Case 3: every edge yielding, servos free about their axis.</summary>
            AllYielding,
        }

        public string Name;
        public int[] Parent;
        public string[] Kind;
        public double[] Mass;
        public double3[] Inertia;
        public double3[] Position;
        public double3[] Joint;
        public double3[] Axis;
        public double[] Swing;
        public bool[] Rigid;
        public int[] ClosureA;
        public int[] ClosureB;

        public int Count => Parent.Length;

        public static string Folder => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages",
            "io.github.ksp1retooled.physics", "Tests", "VesselGraphs~"));

        public static VesselGraph Load(string name)
        {
            using var file = File.OpenRead(Path.Combine(Folder, name + ".txt.gz"));
            using var unzip = new GZipStream(file, CompressionMode.Decompress);
            using var reader = new StreamReader(unzip);
            var graph = new VesselGraph();
            graph.Name = reader.ReadLine().Substring("vessel ".Length);
            int count = int.Parse(reader.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
            graph.Parent = new int[count];
            graph.Kind = new string[count];
            graph.Mass = new double[count];
            graph.Inertia = new double3[count];
            graph.Position = new double3[count];
            graph.Joint = new double3[count];
            graph.Axis = new double3[count];
            graph.Swing = new double[count];
            graph.Rigid = new bool[count];
            for (int i = 0; i < count; i++)
            {
                string[] f = reader.ReadLine().Split(' ');
                double D(int k) => double.Parse(f[k], CultureInfo.InvariantCulture);
                graph.Parent[i] = int.Parse(f[0], CultureInfo.InvariantCulture);
                graph.Kind[i] = f[1];
                graph.Mass[i] = D(2);
                graph.Inertia[i] = new double3(D(3), D(4), D(5));
                graph.Position[i] = new double3(D(6), D(7), D(8));
                graph.Joint[i] = new double3(D(9), D(10), D(11));
                graph.Axis[i] = new double3(D(12), D(13), D(14));
                graph.Swing[i] = D(15);
                graph.Rigid[i] = f[16] == "1";
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

        public VesselEdgeState StateFor(int i, LockRule rule)
        {
            string kind = Kind[i];
            if (Parent[i] < 0)
            {
                return VesselEdgeState.Locked;
            }

            if (kind == "servo")
            {
                return VesselEdgeState.FreeAxis;
            }

            switch (rule)
            {
                case LockRule.LiveJointsOnly:
                    return kind == "docking" || kind == "decoupler" ? VesselEdgeState.Yielding : VesselEdgeState.Locked;
                case LockRule.StackLocked:
                    return kind == "srf" && !Rigid[i] ? VesselEdgeState.Yielding : VesselEdgeState.Locked;
                default:
                    return VesselEdgeState.Yielding;
            }
        }

        /// <summary>Two unit vectors completing a right-handed frame with the axis, as matrix columns after it.</summary>
        public static double3x3 FrameFor(double3 axis)
        {
            axis = math.normalize(axis);
            double3 helper = math.abs(axis.x) < 0.9 ? new double3(1, 0, 0) : new double3(0, 1, 0);
            double3 u = math.normalize(math.cross(axis, helper));
            return new double3x3(axis, u, math.cross(axis, u));
        }

        /// <summary>
        /// Build rest-state solver inputs with gravity and root thrust of 1.5 times vessel weight plus a small
        /// torque.
        /// Scale springs by stiffnessScale; dampers are 2% of their spring.
        /// </summary>
        public void Build(LockRule rule, double timeStep, double stiffnessScale, Allocator allocator,
            out NativeArray<VesselPart> parts, out NativeArray<VesselEdge> edges, out NativeArray<VesselClosure> closures,
            ulong seed = 1)
        {
            int count = Count;
            parts = new NativeArray<VesselPart>(count, allocator);
            edges = new NativeArray<VesselEdge>(count, allocator);
            closures = new NativeArray<VesselClosure>(ClosureA.Length, allocator);
            var random = new Unity.Mathematics.Random((uint)seed);
            double total = 0.0;
            for (int i = 0; i < count; i++)
            {
                total += Mass[i];
            }

            for (int i = 0; i < count; i++)
            {
                var part = new VesselPart
                {
                    Mass = Mass[i],
                    Inertia = Inertia[i],
                    Position = Position[i],
                    Orientation = new double4(0, 0, 0, 1),
                    LinearVelocity = 0.01 * (random.NextDouble3() - 0.5),
                    AngularVelocity = 0.01 * (random.NextDouble3() - 0.5),
                    LinearImpulse = new double3(0, -9.81 * Mass[i] * timeStep, 0),
                };
                if (i == 0)
                {
                    part.LinearImpulse += new double3(0.02, 1.5 * 9.81 * total, 0.01) * timeStep;
                    part.AngularImpulse = new double3(0.01 * total, 0.0, 0.02 * total) * timeStep;
                }

                parts[i] = part;
                int parent = Parent[i];
                if (parent < 0)
                {
                    edges[i] = new VesselEdge { Parent = -1, State = VesselEdgeState.Locked, Rest = new double4(0, 0, 0, 1) };
                    continue;
                }

                double rate = (Swing[i] > 0.0 ? Swing[i] : 1e5) * stiffnessScale;
                edges[i] = new VesselEdge
                {
                    Parent = parent,
                    State = StateFor(i, rule),
                    MayLock = Kind[i] == "stack" || Kind[i] == "srf",
                    ParentAnchor = Joint[i] - Position[parent],
                    ChildAnchor = Joint[i] - Position[i],
                    Frame = FrameFor(Axis[i]),
                    Rest = new double4(0, 0, 0, 1),
                    TwistStiffness = rate,
                    SwingStiffness = rate,
                    TwistDamping = 0.02 * rate,
                    SwingDamping = 0.02 * rate,
                    BreakForce = 1e15,
                    BreakTorque = 1e15,
                    YieldFraction = 0.5,
                    RelockFraction = 0.5,
                };
            }

            for (int c = 0; c < ClosureA.Length; c++)
            {
                int a = ClosureA[c], b = ClosureB[c];
                double3 middle = 0.5 * (Position[a] + Position[b]);
                double3 along = Position[b] - Position[a];
                closures[c] = new VesselClosure
                {
                    PartA = a,
                    PartB = b,
                    AnchorA = middle - Position[a],
                    AnchorB = middle - Position[b],
                    Frame = FrameFor(math.length(along) > 1e-6 ? along : new double3(0, 1, 0)),
                    Rest = new double4(0, 0, 0, 1),
                    LinearStiffness = 1e6 * stiffnessScale,
                    AngularStiffness = 1e5 * stiffnessScale,
                    LinearDamping = 2e4 * stiffnessScale,
                    AngularDamping = 2e3 * stiffnessScale,
                };
            }
        }
    }
}
