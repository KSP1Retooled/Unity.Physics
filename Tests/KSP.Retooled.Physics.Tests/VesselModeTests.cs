using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The direct solver with vessels: the tests build the vessel set themselves and hand it to the step through
    /// <see cref="VesselSet.TestSet"/>, as the backend will once it compiles vessels from the store's joints. The parts of a
    /// vessel have no Unity joints between them; the vessel kernel's rows hold them together.
    /// </summary>
    public class VesselModeTests : PhysicsScene
    {
        private const float PartMass = 4.0f;

        /// <summary>
        /// A vessel set that a test fills and keeps alive while it steps. The parts' world indices are looked up again
        /// before every step, from the last world the backend built.
        /// </summary>
        private sealed unsafe class TestVessels : IDisposable
        {
            public readonly List<Rigidbody> Bodies = new List<Rigidbody>();
            public readonly List<VesselEdge> Edges = new List<VesselEdge>();
            public readonly List<(int first, int count)> Ranges = new List<(int, int)>();
            public readonly List<VesselClosure> Closures = new List<VesselClosure>();

            private NativeArray<VesselRecord> vessels;
            private NativeArray<int> partBodies;
            private NativeArray<VesselEdge> edges;
            private NativeArray<VesselClosure> closures;
            private NativeArray<VesselEdgeLoad> edgeLoads;
            private NativeArray<VesselEdgeLoad> closureLoads;
            private NativeArray<bool> edgeLocked;
            private static ulong nextId = 1000;

            /// <summary>Adds a vessel of the given parts, the first one its root, each part's parent given by index into it.</summary>
            public void AddVessel(IList<Rigidbody> parts, IList<int> parents, Func<int, VesselEdge> edge)
            {
                int first = Bodies.Count;
                for (int i = 0; i < parts.Count; i++)
                {
                    Bodies.Add(parts[i]);
                    VesselEdge e = i == 0 ? new VesselEdge { Parent = -1, State = VesselEdgeState.Yielding } : edge(i);
                    e.Parent = parents[i];
                    Edges.Add(e);
                }

                Ranges.Add((first, parts.Count));
            }

            public VesselEdgeLoad Load(int part) => edgeLoads[part];

            public VesselEdgeLoad ClosureLoad(int closure) => closureLoads[closure];

            public bool Locked(int part) => edgeLocked[part];

            /// <summary>Makes the arrays and hands the set to the step, with every part's body as the last world has it.</summary>
            public void Publish()
            {
                Free();
                vessels = new NativeArray<VesselRecord>(Ranges.Count, Allocator.Persistent);
                for (int v = 0; v < Ranges.Count; v++)
                {
                    vessels[v] = new VesselRecord
                    {
                        Id = nextId++,
                        StructureVersion = 1,
                        FirstPart = Ranges[v].first,
                        PartCount = Ranges[v].count,
                        FirstClosure = 0,
                        ClosureCount = v == 0 ? Closures.Count : 0,
                    };
                }

                partBodies = new NativeArray<int>(Bodies.Count, Allocator.Persistent);
                edges = new NativeArray<VesselEdge>(Edges.ToArray(), Allocator.Persistent);
                closures = new NativeArray<VesselClosure>(System.Math.Max(1, Closures.Count), Allocator.Persistent);
                for (int c = 0; c < Closures.Count; c++)
                {
                    closures[c] = Closures[c];
                }

                edgeLoads = new NativeArray<VesselEdgeLoad>(Bodies.Count, Allocator.Persistent);
                closureLoads = new NativeArray<VesselEdgeLoad>(System.Math.Max(1, Closures.Count), Allocator.Persistent);
                edgeLocked = new NativeArray<bool>(Bodies.Count, Allocator.Persistent);
                Refresh();
            }

            /// <summary>Looks each part's body up in the last world the backend built, and hands the set to the step.</summary>
            public void Refresh()
            {
                // The step returns while the refit moves the built bodies to their solved poses.
                StepKernel.Current->FinishPendingRefit();
                PhysicsWorld world = StepKernel.Current->World;
                NativeArray<RigidBody> dynamic = world.DynamicBodies;
                for (int i = 0; i < Bodies.Count; i++)
                {
                    Vector3 position = Bodies[i].position;
                    int found = -1;
                    float best = float.MaxValue;
                    for (int b = 0; b < dynamic.Length; b++)
                    {
                        float d = math.distancesq(dynamic[b].WorldFromBody.pos, (float3)position);
                        if (d < best)
                        {
                            best = d;
                            found = b;
                        }
                    }

                    partBodies[i] = found;
                }

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

            public void Step(int steps = 1)
            {
                for (int i = 0; i < steps; i++)
                {
                    Refresh();
                    UnityEngine.Physics.Simulate(PhysicsScene.Step);
                }
            }

            private void Free()
            {
                if (vessels.IsCreated)
                {
                    vessels.Dispose();
                    partBodies.Dispose();
                    edges.Dispose();
                    closures.Dispose();
                    edgeLoads.Dispose();
                    closureLoads.Dispose();
                    edgeLocked.Dispose();
                }
            }

            public void Dispose()
            {
                VesselSet.TestSet = default;
                Free();
            }
        }

        private TestVessels vessels;
        private bool wasEnabled;
        private int wasFusedParts;

        /// <summary>
        /// The number of parts at which a vessel island goes to the fused vessel job in these tests, or 0 to keep the
        /// default, which sends every vessel island to the fused job. These tests keep the vessels in the substeps'
        /// jobs, and <see cref="FusedVesselModeTests"/> runs the same tests with the default.
        /// </summary>
        protected virtual int FusedParts => int.MaxValue;

        [SetUp]
        public void Begin()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("Vessels are part of the Unity Physics backend.");
            }

            VesselForest.Initialize();
            wasEnabled = VesselForest.Enabled;
            VesselForest.Enabled = true;
            wasFusedParts = Unity.Physics.Solver.FusedVesselParts;
            if (FusedParts > 0)
            {
                Unity.Physics.Solver.FusedVesselParts = FusedParts;
            }

            vessels = new TestVessels();
        }

        [TearDown]
        public void End()
        {
            vessels?.Dispose();
            vessels = null;
            VesselForest.Enabled = wasEnabled;
            Unity.Physics.Solver.FusedVesselParts = wasFusedParts;
        }

        /// <summary>
        /// Steps once without gravity, so that the backend builds a world to look the parts up in, and then hands the
        /// vessels to the step. With gravity, the parts would fall apart for that one step, before any edge holds them.
        /// </summary>
        private static void Begin(TestVessels set)
        {
            Vector3 gravity = UnityEngine.Physics.gravity;
            UnityEngine.Physics.gravity = Vector3.zero;
            Simulate();
            UnityEngine.Physics.gravity = gravity;
            set.Publish();
        }

        /// <summary>
        /// A rigid edge from part i to its parent at a point in world space, at rest where the parts are now. Every axis is
        /// hard.
        /// </summary>
        private static VesselEdge RigidEdge(Rigidbody child, Rigidbody parent, Vector3 point, double breakForce = double.PositiveInfinity)
        {
            return new VesselEdge
            {
                State = VesselEdgeState.Yielding,
                MayLock = true,
                ChildAnchor = (float3)(Quaternion.Inverse(child.rotation) * (point - child.worldCenterOfMass)),
                ParentAnchor = (float3)(Quaternion.Inverse(parent.rotation) * (point - parent.worldCenterOfMass)),
                Frame = double3x3.identity,
                Rest = ((quaternion)(Quaternion.Inverse(parent.rotation) * child.rotation)).value,
                TwistStiffness = double.PositiveInfinity,
                SwingStiffness = double.PositiveInfinity,
                BreakForce = breakForce,
                BreakTorque = double.PositiveInfinity,
            };
        }

        /// <summary>Boxes in a column, each one a part of a vessel, joined by rigid edges. The parts don't collide.</summary>
        private List<Rigidbody> Column(int count, float bottom, double breakForce = double.PositiveInfinity, float x = 0.0f)
        {
            var bodies = new List<Rigidbody>();
            var parents = new List<int>();
            for (int i = 0; i < count; i++)
            {
                Rigidbody body = Box(new Vector3(x, bottom + 1.0f * (count - 1 - i) + 0.5f, 0.0f), new Vector3(0.6f, 1.0f, 0.6f))
                    .AddComponent<Rigidbody>();
                body.mass = PartMass;

                // The backend puts a vessel's parts to sleep together through the store's joints, which these parts don't
                // have, so here they stay awake.
                body.sleepThreshold = 0.0f;
                bodies.Add(body);
                parents.Add(i - 1);
                for (int j = 0; j < i; j++)
                {
                    UnityEngine.Physics.IgnoreCollision(body.GetComponent<UnityEngine.Collider>(), bodies[j].GetComponent<UnityEngine.Collider>());
                }
            }

            vessels.AddVessel(bodies, parents, i => RigidEdge(bodies[i], bodies[i - 1],
                new Vector3(x, bottom + 1.0f * (count - i), 0.0f), breakForce));
            return bodies;
        }

        [Test]
        public void AHangingColumnCarriesItsWeightThroughItsEdges()
        {
            const int count = 5;
            List<Rigidbody> bodies = Column(count, 5.0f);
            ConfigurableJoint hook = bodies[0].gameObject.AddComponent<ConfigurableJoint>();
            hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
            hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;
            Begin(vessels);
            Vector3 top = bodies[0].position, bottom = bodies[count - 1].position;
            vessels.Step(50);

            for (int i = 1; i < count; i++)
            {
                float weight = (count - i) * PartMass * -Gravity;
                VesselEdgeLoad load = vessels.Load(i);
                Debug.Log($"[VesselModeTests] hanging: edge {i} force {(float3)load.Force}, weight below {weight:F2}, locked {vessels.Locked(i)}");
                Assert.That((float)math.length(load.Force), Is.EqualTo(weight).Within(0.02f * weight), $"edge {i} carries what hangs below it");
                Assert.That(vessels.Locked(i), Is.True, "a rigid edge under a light load stays locked");
            }

            Debug.Log($"[VesselModeTests] hanging: top moved {(bodies[0].position - top).magnitude:E2} m, bottom {(bodies[count - 1].position - bottom).magnitude:E2} m");
            Assert.That((bodies[count - 1].position - bottom).magnitude, Is.LessThan(0.01f), "the column hangs where it was");
        }

        /// <summary>
        /// An arm of one part held out sideways from a part hooked to the world. With a swing spring that may not lock, the
        /// arm droops by its moment over the spring. With an edge that may lock, it hardly droops at all.
        /// </summary>
        [Test]
        public void ALockedArmBarelyDroopsWhereASpringDroopsByItsLoad()
        {
            const double spring = 2000.0;
            var droop = new float[2];
            for (int pass = 0; pass < 2; pass++)
            {
                TearDownScene();
                SetUpScene();
                VesselForest.Enabled = true;
                vessels.Dispose();
                vessels = new TestVessels();
                Rigidbody root = Box(new Vector3(0.0f, 5.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                Rigidbody arm = Box(new Vector3(1.0f, 5.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                root.mass = arm.mass = PartMass;
                root.sleepThreshold = arm.sleepThreshold = 0.0f;
                UnityEngine.Physics.IgnoreCollision(root.GetComponent<UnityEngine.Collider>(), arm.GetComponent<UnityEngine.Collider>());
                ConfigurableJoint hook = root.gameObject.AddComponent<ConfigurableJoint>();
                hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
                hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;
                bool mayLock = pass == 1;
                vessels.AddVessel(new[] { root, arm }, new[] { -1, 0 }, i =>
                {
                    VesselEdge edge = RigidEdge(arm, root, new Vector3(0.5f, 5.0f, 0.0f));
                    edge.SwingStiffness = spring;
                    edge.SwingDamping = 0.1 * spring;
                    edge.MayLock = mayLock;
                    return edge;
                });
                Begin(vessels);
                vessels.Step(100);
                // The hook to the world is a Unity joint, which gives a little too, so the droop is measured in the frame
                // of the root part.
                droop[pass] = -root.transform.InverseTransformPoint(arm.position).y;
            }

            // The weight of the arm acts half a metre from the joint and turns the arm about it. That lowers the centre of
            // the arm by the same half metre times the angle.
            float expected = (float)(PartMass * -Gravity * 0.5 / spring * 0.5);
            Debug.Log($"[VesselModeTests] arm: droop with a spring {droop[0] * 1e3f:F3} mm (expected {expected * 1e3f:F3}), locked {droop[1] * 1e3f:F4} mm");
            Assert.That(droop[0], Is.EqualTo(expected).Within(0.2f * expected), "the spring takes the moment of the arm");
            Assert.That(droop[1], Is.LessThan(0.02f * expected), "the locked edge holds the arm up");
        }

        /// <summary>
        /// A column whose edges are springs that may not lock, with a rigid strut from the top part to the bottom one. A
        /// side force at the bottom bends the springs, and the strut, which is stiffer, takes a share of the load.
        /// </summary>
        [Test]
        public void AStrutAcrossSpringsCarriesLoad()
        {
            List<Rigidbody> bodies = Column(3, 5.0f);
            for (int i = 1; i < 3; i++)
            {
                VesselEdge edge = vessels.Edges[i];
                edge.MayLock = false;
                edge.SwingStiffness = 500.0;
                edge.SwingDamping = 50.0;
                vessels.Edges[i] = edge;
            }

            ConfigurableJoint hook = bodies[0].gameObject.AddComponent<ConfigurableJoint>();
            hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
            hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;
            Vector3 point = 0.5f * (bodies[0].position + bodies[2].position) + new Vector3(0.3f, 0.0f, 0.0f);
            vessels.Closures.Add(new VesselClosure
            {
                PartA = 0,
                PartB = 2,
                AnchorA = (float3)(point - bodies[0].worldCenterOfMass),
                AnchorB = (float3)(point - bodies[2].worldCenterOfMass),
                Frame = double3x3.identity,
                Rest = new double4(0.0, 0.0, 0.0, 1.0),
                LinearStiffness = double.PositiveInfinity,
                AngularStiffness = double.PositiveInfinity,
            });
            Begin(vessels);
            for (int i = 0; i < 50; i++)
            {
                bodies[2].AddForce(new Vector3(20.0f, 0.0f, 0.0f));
                vessels.Step();
            }

            VesselEdgeLoad strut = vessels.ClosureLoad(0);
            Debug.Log($"[VesselModeTests] strut: force {(float3)strut.Force}, torque {(float3)strut.Torque}; edge 1 {(float3)vessels.Load(1).Force}, edge 2 {(float3)vessels.Load(2).Force}");
            Assert.That(math.length(strut.Force) + math.length(strut.Torque), Is.GreaterThan(1.0), "the strut carries load");
        }

        /// <summary>A hanging column split into two vessels at its middle: the lower one falls away in one piece.</summary>
        [Test]
        public void AColumnSplitInTwoComesApart()
        {
            List<Rigidbody> bodies = Column(4, 5.0f);
            ConfigurableJoint hook = bodies[0].gameObject.AddComponent<ConfigurableJoint>();
            hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
            hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;
            Begin(vessels);
            vessels.Step(10);
            Vector3 lower = bodies[2].position;

            // Part 2 loses its edge and becomes the root of a vessel of its own, with part 3 under it.
            VesselEdge cut = vessels.Edges[2];
            cut.Parent = -1;
            vessels.Edges[2] = cut;
            VesselEdge below = vessels.Edges[3];
            below.Parent = 0;
            vessels.Edges[3] = below;
            vessels.Ranges.Clear();
            vessels.Ranges.Add((0, 2));
            vessels.Ranges.Add((2, 2));
            vessels.Publish();
            vessels.Step(25);
            float fall = lower.y - bodies[2].position.y;
            Debug.Log($"[VesselModeTests] split: the lower vessel fell {fall:F3} m");
            Assert.That(fall, Is.GreaterThan(0.5f), "the lower vessel falls");
            Assert.That((bodies[3].position - bodies[2].position).magnitude, Is.EqualTo(1.0f).Within(1e-3f), "and stays in one piece");
        }

        [Test]
        public void AnEdgePastItsBreakForceReadsAsBroken()
        {
            List<Rigidbody> bodies = Column(2, 5.0f, 20.0);
            ConfigurableJoint hook = bodies[0].gameObject.AddComponent<ConfigurableJoint>();
            hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
            hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;
            Begin(vessels);
            vessels.Step(10);
            VesselEdgeLoad load = vessels.Load(1);
            Debug.Log($"[VesselModeTests] break: ratio {load.Ratio:F3}, force {math.length(load.Force):F2}");
            Assert.That(load.Ratio, Is.EqualTo(PartMass * -Gravity / 20.0).Within(0.05), "the ratio is the load over the break force");
            Assert.That(load.Ratio, Is.GreaterThan(1.0));
        }

        /// <summary>
        /// A floating column pulled apart from both ends at 70% of its edges' break force. Its edges yield after a
        /// few steps. Then the column leaves the fused job for one step and comes back. The edges' lock states go with
        /// the column from the fused job's forest to the island's forest and back, so the edges stay yielding. Before,
        /// each forest started the column again with every edge locked.
        /// </summary>
        [Test]
        public void LockStatesFollowAVesselInAndOutOfTheFusedJob()
        {
            const float pull = 30.0f;
            List<Rigidbody> bodies = Column(4, 5.0f, pull / 0.7);
            foreach (Rigidbody body in bodies)
            {
                body.useGravity = false;
            }

            bool wasFused = Unity.Physics.Solver.VesselFused;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            Unity.Physics.Solver.FusedVesselParts = 2;
            Unity.Physics.Solver.VesselFused = true;
            try
            {
                Begin(vessels);
                void Pull(string when)
                {
                    bodies[0].AddForce(new Vector3(0.0f, pull, 0.0f));
                    bodies[3].AddForce(new Vector3(0.0f, -pull, 0.0f));
                    vessels.Step();
                    for (int i = 1; i < 4; i++)
                    {
                        Assert.That(vessels.Locked(i), Is.False, $"edge {i} stays yielding {when} (ratio {vessels.Load(i).Ratio:F3})");
                    }
                }

                for (int i = 0; i < 5; i++)
                {
                    bodies[0].AddForce(new Vector3(0.0f, pull, 0.0f));
                    bodies[3].AddForce(new Vector3(0.0f, -pull, 0.0f));
                    vessels.Step();
                }

                Pull("in the fused job");
                Unity.Physics.Solver.VesselFused = false;
                Pull("in the step that leaves the fused job");
                Pull("outside the fused job");
                Unity.Physics.Solver.VesselFused = true;
                Pull("in the step that comes back to the fused job");
                Pull("back in the fused job");
            }
            finally
            {
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
            }
        }

        /// <summary>
        /// A column dropped onto the ground at a slant, once with the island's jobs of each substep and once with the
        /// fused vessel job, which takes the ground contacts into its boundary block. The two land the column in the
        /// same place with the same loads.
        /// </summary>
        [Test]
        public void TheFusedJobLandsAColumnAsTheIslandPathDoes()
        {
            bool wasFused = Unity.Physics.Solver.VesselFused;
            Unity.Physics.Solver.VesselFused = true;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            var positions = new Vector3[2][];
            var loads = new float[2];
            try
            {
                for (int pass = 0; pass < 2; pass++)
                {
                    TearDownScene();
                    SetUpScene();
                    VesselForest.Enabled = true;
                    vessels.Dispose();
                    vessels = new TestVessels();
                    Unity.Physics.Solver.FusedVesselParts = pass == 0 ? 1000 : 2;
                    Ground();
                    List<Rigidbody> bodies = Column(4, 0.6f);
                    foreach (Rigidbody body in bodies)
                    {
                        body.angularVelocity = new Vector3(0.0f, 0.0f, 0.3f);
                    }

                    Begin(vessels);
                    long before = Unity.Physics.Solver.FusedIslandSolves;
                    vessels.Step(80);
                    long fused = Unity.Physics.Solver.FusedIslandSolves - before;
                    Assert.That(fused, pass == 0 ? Is.EqualTo(0) : Is.GreaterThanOrEqualTo(80), "the column goes to the fused job only in the second pass");
                    positions[pass] = bodies.ConvertAll(b => b.position).ToArray();
                    loads[pass] = (float)math.length(vessels.Load(3).Force);
                }
            }
            finally
            {
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
            }

            for (int i = 0; i < positions[0].Length; i++)
            {
                Debug.Log($"[VesselModeTests] fused landing: part {i} at {positions[0][i]:F5} by the island's jobs, {positions[1][i]:F5} by the fused job");
                Assert.That((positions[1][i] - positions[0][i]).magnitude, Is.LessThan(1e-3f), $"part {i} lands in the same place");
            }

            Debug.Log($"[VesselModeTests] fused landing: bottom edge force {loads[0]:F3} by the island's jobs, {loads[1]:F3} by the fused job");
            Assert.That(loads[1], Is.EqualTo(loads[0]).Within(0.01f * loads[0] + 0.01f), "the bottom edge carries the same load");
        }

        /// <summary>
        /// A column whose top part is kinematic, moved sideways and up by hand in every step. The fused job moves a
        /// kinematic part of its island at the end of each substep, and the substeps' jobs move it when the island is
        /// not fused. The scene runs once with the island in the substeps' jobs and once with it in the fused job. The
        /// kinematic part ends in the same place to the last bit, and the parts below it end in the same places within
        /// what the two solvers differ by.
        /// <para>
        /// The game never hands the step a vessel with a kinematic part: VesselPipeline leaves a vessel out of the step
        /// while any of its parts is kinematic, and <see cref="VesselSet"/> says that all vessel bodies are dynamic.
        /// In this scene the parts below the kinematic part don't follow it, in either path, so the test only checks
        /// that the two paths agree.
        /// </para>
        /// </summary>
        [Test]
        public void AKinematicPartMovesTheSameInTheFusedJobAsInTheSubsteps()
        {
            bool wasFused = Unity.Physics.Solver.VesselFused;
            Unity.Physics.Solver.VesselFused = true;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            var positions = new Vector3[2][];
            var rotations = new Quaternion[2][];
            const int steps = 60;
            try
            {
                for (int pass = 0; pass < 2; pass++)
                {
                    TearDownScene();
                    SetUpScene();
                    VesselForest.Enabled = true;
                    vessels.Dispose();
                    vessels = new TestVessels();
                    Unity.Physics.Solver.FusedVesselParts = pass == 0 ? 1000 : 2;
                    List<Rigidbody> bodies = Column(4, 5.0f);
                    Rigidbody top = bodies[0];
                    top.isKinematic = true;
                    Vector3 start = top.position;
                    Begin(vessels);
                    long before = Unity.Physics.Solver.FusedIslandSolves;
                    for (int i = 1; i <= steps; i++)
                    {
                        top.MovePosition(start + new Vector3(0.02f * i, 0.005f * i, 0.0f));
                        top.MoveRotation(Quaternion.Euler(0.0f, 0.5f * i, 0.0f));
                        vessels.Step();
                    }

                    long fused = Unity.Physics.Solver.FusedIslandSolves - before;
                    Assert.That(fused, pass == 0 ? Is.EqualTo(0) : Is.GreaterThanOrEqualTo(steps), "the column goes to the fused job only in the second pass");
                    positions[pass] = bodies.ConvertAll(b => b.position).ToArray();
                    rotations[pass] = bodies.ConvertAll(b => b.rotation).ToArray();
                }
            }
            finally
            {
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
            }

            for (int i = 0; i < positions[0].Length; i++)
            {
                Debug.Log($"[VesselModeTests] kinematic top: part {i} at {positions[0][i]:F6} by the island's jobs, {positions[1][i]:F6} by the fused job");
            }

            Assert.That(positions[1][0], Is.EqualTo(positions[0][0]), "the kinematic part ends in the same place");
            Assert.That(rotations[1][0], Is.EqualTo(rotations[0][0]), "and turned the same");
            Assert.That(positions[0][0].x, Is.EqualTo(0.02f * steps).Within(1e-3f), "the kinematic part went where it was moved");
            for (int i = 1; i < positions[0].Length; i++)
            {
                Assert.That((positions[1][i] - positions[0][i]).magnitude, Is.LessThan(1e-3f), $"part {i} ends in the same place");
            }
        }

        /// <summary>
        /// Four columns of different heights fall onto the ground at a slant, each one a fused island of its own. The
        /// fused job solves the four islands at the same time on several threads. The scene runs once on one thread and
        /// once on the default job workers, and the two runs end with the same poses and loads, to the last bit.
        /// </summary>
        [Test]
        public void SeveralFusedIslandsEndTheSameOnOneThreadAndOnMany()
        {
            bool wasFused = Unity.Physics.Solver.VesselFused;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            int workers = Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount;
            int[] heights = { 6, 12, 9, 16 };
            var positions = new Vector3[2][];
            var rotations = new Quaternion[2][];
            var loads = new double3[2][];
            try
            {
                Unity.Physics.Solver.VesselFused = true;
                for (int pass = 0; pass < 2; pass++)
                {
                    Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount = pass == 0 ? 0 : System.Math.Max(workers, 3);
                    TearDownScene();
                    SetUpScene();
                    VesselForest.Enabled = true;
                    VesselForest.ForgetIslands();
                    vessels.Dispose();
                    vessels = new TestVessels();
                    Unity.Physics.Solver.FusedVesselParts = 2;
                    Ground();
                    var bodies = new List<Rigidbody>();
                    for (int c = 0; c < heights.Length; c++)
                    {
                        foreach (Rigidbody body in Column(heights[c], 0.6f, double.PositiveInfinity, 3.0f * c))
                        {
                            body.angularVelocity = new Vector3(0.0f, 0.1f * c, 0.3f);
                            bodies.Add(body);
                        }
                    }

                    Begin(vessels);
                    long before = Unity.Physics.Solver.FusedIslandSolves;
                    const int steps = 60;
                    vessels.Step(steps);
                    Assert.That(Unity.Physics.Solver.FusedIslandSolves - before, Is.EqualTo(heights.Length * steps),
                        "every column is a fused island in every step");
                    positions[pass] = bodies.ConvertAll(b => b.position).ToArray();
                    rotations[pass] = bodies.ConvertAll(b => b.rotation).ToArray();
                    loads[pass] = new double3[bodies.Count];
                    for (int i = 0; i < bodies.Count; i++)
                    {
                        loads[pass][i] = vessels.Load(i).Force;
                    }
                }
            }
            finally
            {
                Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount = workers;
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
            }

            for (int i = 0; i < positions[0].Length; i++)
            {
                Assert.That(positions[1][i], Is.EqualTo(positions[0][i]), $"part {i} ends in the same place");
                Assert.That(rotations[1][i], Is.EqualTo(rotations[0][i]), $"part {i} ends in the same orientation");
                Assert.That(loads[1][i], Is.EqualTo(loads[0][i]), $"part {i}'s edge carries the same load");
            }

            Debug.Log($"[VesselModeTests] fused islands on one thread and on many: top of the tallest column at {positions[0][positions[0].Length - 16]:F5}");
        }

        /// <summary>
        /// By default, every vessel island goes to the fused job, however small it is and however many there are: one
        /// floating column of 64 parts, and two of them. Before, two such columns stayed with the substeps' jobs.
        /// </summary>
        [Test]
        public void SmallVesselsGoToTheFusedJobHoweverManyThereAre()
        {
            bool wasFused = Unity.Physics.Solver.VesselFused;
            Unity.Physics.Solver.VesselFused = true;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            Unity.Physics.Solver.FusedVesselParts = Unity.Physics.Solver.DefaultFusedVesselParts;
            try
            {
                for (int columns = 1; columns <= 2; columns++)
                {
                    TearDownScene();
                    SetUpScene();
                    VesselForest.Enabled = true;
                    vessels.Dispose();
                    vessels = new TestVessels();
                    for (int c = 0; c < columns; c++)
                    {
                        foreach (Rigidbody body in Column(64, 5.0f, double.PositiveInfinity, 3.0f * c))
                        {
                            body.useGravity = false;
                        }
                    }

                    Begin(vessels);
                    long before = Unity.Physics.Solver.FusedIslandSolves;
                    vessels.Step(5);
                    long fused = Unity.Physics.Solver.FusedIslandSolves - before;
                    Assert.That(fused, Is.EqualTo(5 * columns), $"fused island solves with {columns} columns");
                }
            }
            finally
            {
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
            }
        }

        /// <summary>
        /// A fleet of twelve small vessels, columns of 3 to 8 parts, falls onto the ground at a slant. The scene runs
        /// twice: once with the substeps' jobs, and once with the fused vessel job, which the default rule sends every
        /// column to. The fused job solves several columns at once. It lands the columns where the substeps' jobs land
        /// them, with the same loads, within the tolerance of <see cref="TheFusedJobLandsAColumnAsTheIslandPathDoes"/>.
        /// </summary>
        [Test]
        public void AFleetOfSmallVesselsLandsTheSameInTheFusedJobAsInTheSubsteps()
        {
            bool wasFused = Unity.Physics.Solver.VesselFused;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            const int columns = 12;
            const int steps = 60;
            var positions = new Vector3[2][];
            var loads = new double3[2][];
            try
            {
                Unity.Physics.Solver.VesselFused = true;
                for (int pass = 0; pass < 2; pass++)
                {
                    TearDownScene();
                    SetUpScene();
                    VesselForest.Enabled = true;
                    VesselForest.ForgetIslands();
                    vessels.Dispose();
                    vessels = new TestVessels();
                    Unity.Physics.Solver.FusedVesselParts = pass == 0 ? 1000 : Unity.Physics.Solver.DefaultFusedVesselParts;
                    Ground();
                    var bodies = new List<Rigidbody>();
                    for (int c = 0; c < columns; c++)
                    {
                        foreach (Rigidbody body in Column(3 + c % 6, 0.6f, double.PositiveInfinity, 3.0f * c))
                        {
                            body.angularVelocity = new Vector3(0.05f * c, 0.0f, 0.3f);
                            bodies.Add(body);
                        }
                    }

                    Begin(vessels);
                    long before = Unity.Physics.Solver.FusedIslandSolves;
                    vessels.Step(steps);
                    long fused = Unity.Physics.Solver.FusedIslandSolves - before;
                    Assert.That(fused, pass == 0 ? Is.EqualTo(0) : Is.EqualTo(columns * steps),
                        $"in pass {pass}, the columns go to the fused job only in the second pass, and then every column in every step");
                    positions[pass] = bodies.ConvertAll(b => b.position).ToArray();
                    loads[pass] = new double3[bodies.Count];
                    for (int i = 0; i < bodies.Count; i++)
                    {
                        loads[pass][i] = vessels.Load(i).Force;
                    }
                }
            }
            finally
            {
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
            }

            double largestPose = 0.0, largestLoad = 0.0;
            for (int i = 0; i < positions[0].Length; i++)
            {
                float pose = (positions[1][i] - positions[0][i]).magnitude;
                float load = (float)math.length(loads[1][i] - loads[0][i]);
                float size = (float)math.length(loads[0][i]);
                largestPose = System.Math.Max(largestPose, pose);
                largestLoad = System.Math.Max(largestLoad, load);
                Assert.That(pose, Is.LessThan(1e-3f), $"part {i} lands in the same place");
                Assert.That(load, Is.LessThan(0.01f * size + 0.01f), $"part {i}'s edge carries the same load");
            }

            Debug.Log($"[VesselModeTests] small fleet: largest difference from the substeps' jobs {largestPose:E2} m in pose, {largestLoad:E2} N in load");
        }

        /// <summary>
        /// Forty floating vessels of two parts each, every one a fused island. The fused job keeps the state of every
        /// one of them from step to step, so that no island builds its structure again. With room for only 32 islands,
        /// eight of them built their structure again in every step.
        /// </summary>
        [Test]
        public void FortyFusedIslandsKeepTheirStructuresFromStepToStep()
        {
            ManyFusedIslandsKeepTheirStructures(40);
        }

        /// <summary>
        /// Three hundred floating vessels of two parts each: more fused islands than the table of fused slots starts
        /// with. The table grows, and no island builds its structure again. When the table held 256 islands at most,
        /// 44 of them built their structure again in every step.
        /// </summary>
        [Test]
        public void MoreFusedIslandsThanTheTableStartsWithKeepTheirStructures()
        {
            ManyFusedIslandsKeepTheirStructures(300);
        }

        private void ManyFusedIslandsKeepTheirStructures(int columns)
        {
            bool wasFused = Unity.Physics.Solver.VesselFused;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            try
            {
                Unity.Physics.Solver.VesselFused = true;
                Unity.Physics.Solver.FusedVesselParts = 2;
                for (int c = 0; c < columns; c++)
                {
                    foreach (Rigidbody body in Column(2, 5.0f, double.PositiveInfinity, 3.0f * c))
                    {
                        body.useGravity = false;
                    }
                }

                Begin(vessels);
                vessels.Step(3);
                VesselForest.GetSectionCounts(out _, out _, out long builds0, out _, out _, out _, out _, out _);
                long before = Unity.Physics.Solver.FusedIslandSolves;
                vessels.Step(5);
                VesselForest.GetSectionCounts(out _, out _, out long builds1, out _, out _, out _, out _, out _);
                Assert.That(Unity.Physics.Solver.FusedIslandSolves - before, Is.EqualTo(columns * 5), "every vessel is a fused island in every step");
                Assert.That(builds1 - builds0, Is.EqualTo(0), "no fused island builds its structure again");
            }
            finally
            {
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
            }
        }

        /// <summary>
        /// A hundred floating vessels of two parts each, which the fused job doesn't take, so that the substeps solve
        /// them: more islands than the 64 slots that the substeps' cache held before. The cache's table grows, and no
        /// island builds its structure again. With 64 slots, the 36 islands past them built their structures again in
        /// every substep. The test runs with the substeps' chain of jobs and with one thread doing the substeps.
        /// </summary>
        [Test]
        public void MoreUnfusedIslandsThanTheOldCacheHeldKeepTheirStructures([Values] bool folded)
        {
            const int columns = 100;
            bool wasFused = Unity.Physics.Solver.VesselFused;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            Unity.Physics.Solver.SubstepFolding folding = Unity.Physics.Solver.UnfusedFolding;
            try
            {
                Unity.Physics.Solver.VesselFused = true;
                Unity.Physics.Solver.FusedVesselParts = int.MaxValue;
                Unity.Physics.Solver.UnfusedFolding = folded ? Unity.Physics.Solver.SubstepFolding.Fold : Unity.Physics.Solver.SubstepFolding.Chain;
                for (int c = 0; c < columns; c++)
                {
                    foreach (Rigidbody body in Column(2, 5.0f, double.PositiveInfinity, 3.0f * c))
                    {
                        body.useGravity = false;
                    }
                }

                Begin(vessels);
                vessels.Step(3);
                VesselForest.GetSectionCounts(out _, out _, out long builds0, out _, out _, out _, out _, out _);
                long fusedBefore = Unity.Physics.Solver.FusedIslandSolves;
                long foldedBefore = Unity.Physics.Solver.UnfusedFoldedSteps;
                vessels.Step(5);
                VesselForest.GetSectionCounts(out _, out _, out long builds1, out _, out _, out _, out _, out _);
                Assert.That(Unity.Physics.Solver.FusedIslandSolves - fusedBefore, Is.EqualTo(0), "the substeps solve every vessel");
                Assert.That(Unity.Physics.Solver.UnfusedFoldedSteps - foldedBefore, Is.EqualTo(folded ? 5 : 0),
                    "one thread does the substeps' work in every step only when the test asks for it");
                Assert.That(builds1 - builds0, Is.EqualTo(0), "no island builds its structure again");
            }
            finally
            {
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
                Unity.Physics.Solver.UnfusedFolding = folding;
            }
        }

        /// <summary>
        /// Columns of 2 to 9 parts fall onto the ground at a slant. A loose box falls onto two of them, and two plain
        /// bodies swing on a hinge, so that some islands can't be fused. The scene runs with the substeps' work done
        /// by the chain of jobs, and with that work done on one thread: in a lane of the fused job when the fused job
        /// takes the columns of 5 parts or more, and in a job of its own when the fused job is off. Both ways end with
        /// the same poses and loads, to the last bit, because each island's work is the same and in the same order.
        /// </summary>
        [Test]
        public void TheSubstepsEndTheSameOnOneThreadAsInTheirChainOfJobs([Values] bool fusedJob)
        {
            bool wasFused = Unity.Physics.Solver.VesselFused;
            int fusedParts = Unity.Physics.Solver.FusedVesselParts;
            Unity.Physics.Solver.SubstepFolding folding = Unity.Physics.Solver.UnfusedFolding;
            const int columns = 8;
            const int steps = 60;
            var positions = new Vector3[2][];
            var rotations = new Quaternion[2][];
            var loads = new double3[2][];
            var folded = new long[2];
            int partCount = 0;
            try
            {
                Unity.Physics.Solver.VesselFused = fusedJob;
                for (int pass = 0; pass < 2; pass++)
                {
                    TearDownScene();
                    SetUpScene();
                    VesselForest.Enabled = true;
                    VesselForest.ForgetIslands();
                    vessels.Dispose();
                    vessels = new TestVessels();
                    Unity.Physics.Solver.FusedVesselParts = 5;
                    Unity.Physics.Solver.UnfusedFolding = pass == 0 ? Unity.Physics.Solver.SubstepFolding.Chain : Unity.Physics.Solver.SubstepFolding.Fold;
                    Ground();
                    var bodies = new List<Rigidbody>();
                    for (int c = 0; c < columns; c++)
                    {
                        foreach (Rigidbody body in Column(2 + c, 0.6f, double.PositiveInfinity, 3.0f * c))
                        {
                            body.angularVelocity = new Vector3(0.05f * c, 0.0f, 0.3f);
                            bodies.Add(body);
                        }
                    }

                    partCount = bodies.Count;
                    foreach (int c in new[] { 1, 6 })
                    {
                        Rigidbody box = Box(new Vector3(3.0f * c, 0.6f + (2 + c) + 1.0f, 0.0f), new Vector3(0.5f, 0.5f, 0.5f)).AddComponent<Rigidbody>();
                        box.mass = 1.0f;
                        box.sleepThreshold = 0.0f;
                        bodies.Add(box);
                    }

                    Rigidbody hung = Box(new Vector3(-6.0f, 4.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                    Rigidbody swung = Box(new Vector3(-4.5f, 4.0f, 0.0f), Vector3.one).AddComponent<Rigidbody>();
                    hung.sleepThreshold = swung.sleepThreshold = 0.0f;
                    UnityEngine.Physics.IgnoreCollision(hung.GetComponent<UnityEngine.Collider>(), swung.GetComponent<UnityEngine.Collider>());
                    ConfigurableJoint hook = hung.gameObject.AddComponent<ConfigurableJoint>();
                    hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
                    hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;
                    HingeJoint hinge = swung.gameObject.AddComponent<HingeJoint>();
                    hinge.connectedBody = hung;
                    hinge.anchor = new Vector3(-0.75f, 0.0f, 0.0f);
                    hinge.axis = new Vector3(0.0f, 0.0f, 1.0f);
                    bodies.Add(hung);
                    bodies.Add(swung);

                    Begin(vessels);
                    long before = Unity.Physics.Solver.UnfusedFoldedSteps;
                    vessels.Step(steps);
                    folded[pass] = Unity.Physics.Solver.UnfusedFoldedSteps - before;
                    positions[pass] = bodies.ConvertAll(b => b.position).ToArray();
                    rotations[pass] = bodies.ConvertAll(b => b.rotation).ToArray();
                    loads[pass] = new double3[partCount];
                    for (int i = 0; i < partCount; i++)
                    {
                        loads[pass][i] = vessels.Load(i).Force;
                    }
                }
            }
            finally
            {
                Unity.Physics.Solver.VesselFused = wasFused;
                Unity.Physics.Solver.FusedVesselParts = fusedParts;
                Unity.Physics.Solver.UnfusedFolding = folding;
            }

            Assert.That(folded[0], Is.EqualTo(0), "the first pass runs the chain of jobs");
            Assert.That(folded[1], Is.EqualTo(steps), "the second pass does the substeps on one thread in every step");
            for (int i = 0; i < positions[0].Length; i++)
            {
                Assert.That(positions[1][i], Is.EqualTo(positions[0][i]), $"body {i} ends in the same place");
                Assert.That(rotations[1][i], Is.EqualTo(rotations[0][i]), $"body {i} ends in the same orientation");
            }

            for (int i = 0; i < loads[0].Length; i++)
            {
                Assert.That(loads[1][i], Is.EqualTo(loads[0][i]), $"part {i}'s edge carries the same load");
            }

            Debug.Log($"[VesselModeTests] substeps on one thread and in a chain of jobs, fused job {fusedJob}: the tallest column's top at {positions[0][partCount - (columns + 1)]:F5}");
        }

        [Test]
        public void AFloatingVesselKeepsItsMomentum()
        {
            List<Rigidbody> bodies = Column(4, 5.0f);
            foreach (Rigidbody body in bodies)
            {
                body.useGravity = false;
            }

            bodies[0].linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            bodies[3].linearVelocity = new Vector3(0.0f, 0.0f, -2.0f);
            bodies[1].angularVelocity = new Vector3(0.0f, 3.0f, 0.0f);
            Begin(vessels);

            Vector3 Momentum()
            {
                Vector3 sum = Vector3.zero;
                foreach (Rigidbody body in bodies)
                {
                    sum += body.mass * body.linearVelocity;
                }

                return sum;
            }

            vessels.Step(1);
            Vector3 before = Momentum();
            Vector3 gap = bodies[0].position - bodies[1].position;
            vessels.Step(100);
            Vector3 after = Momentum();
            Debug.Log($"[VesselModeTests] floating: momentum {before:F4} -> {after:F4}; gap {gap.magnitude:F4} -> {(bodies[0].position - bodies[1].position).magnitude:F4}");
            Assert.That((after - before).magnitude, Is.LessThan(1e-3f * before.magnitude + 1e-4f), "momentum is kept");
            Assert.That((bodies[0].position - bodies[1].position).magnitude, Is.EqualTo(gap.magnitude).Within(1e-3f), "the vessel stays rigid");
        }

        [Test]
        public void AColumnLandsOnTheGroundAndSettles()
        {
            Ground();
            List<Rigidbody> bodies = Column(4, 0.5f);
            Begin(vessels);
            vessels.Step(150);
            float speed = 0.0f;
            foreach (Rigidbody body in bodies)
            {
                speed = Mathf.Max(speed, body.linearVelocity.magnitude);
            }

            VesselEdgeLoad load = vessels.Load(3);
            Debug.Log($"[VesselModeTests] landing: bottom at {bodies[3].position:F3}, fastest part {speed:E2} m/s, bottom edge force {math.length(load.Force):F2}");
            Assert.That(bodies[3].position.y, Is.EqualTo(0.5f).Within(0.02f), "the column stands on the ground");
            Assert.That(speed, Is.LessThan(0.02f), "and comes to rest");
            Assert.That((float)math.length(load.Force), Is.EqualTo(3 * PartMass * -Gravity).Within(0.05f * 3 * PartMass * -Gravity),
                "the bottom part's edge carries the three parts above it");
        }

        [Test]
        public void TwoVesselsJoinedByAHingeSolveAsOne()
        {
            List<Rigidbody> left = Column(3, 5.0f);
            var rightBodies = new List<Rigidbody>();
            for (int i = 0; i < 2; i++)
            {
                Rigidbody body = Box(new Vector3(1.0f + i, 7.5f, 0.0f), new Vector3(1.0f, 0.6f, 0.6f)).AddComponent<Rigidbody>();
                body.mass = PartMass;
                body.sleepThreshold = 0.0f;
                rightBodies.Add(body);
            }

            UnityEngine.Physics.IgnoreCollision(rightBodies[0].GetComponent<UnityEngine.Collider>(), rightBodies[1].GetComponent<UnityEngine.Collider>());
            vessels.AddVessel(rightBodies, new[] { -1, 0 }, i => RigidEdge(rightBodies[1], rightBodies[0], new Vector3(1.5f, 7.5f, 0.0f)));
            foreach (Rigidbody body in left)
            {
                UnityEngine.Physics.IgnoreCollision(body.GetComponent<UnityEngine.Collider>(), rightBodies[0].GetComponent<UnityEngine.Collider>());
            }

            ConfigurableJoint hook = left[0].gameObject.AddComponent<ConfigurableJoint>();
            hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
            hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;

            HingeJoint hinge = rightBodies[0].gameObject.AddComponent<HingeJoint>();
            hinge.connectedBody = left[0];
            hinge.anchor = new Vector3(-0.5f, 0.0f, 0.0f);
            hinge.axis = new Vector3(0.0f, 0.0f, 1.0f);
            Begin(vessels);
            Vector3 gap = rightBodies[1].position - rightBodies[0].position;
            vessels.Step(60);
            float hingeGap = (rightBodies[0].transform.TransformPoint(hinge.anchor) - left[0].transform.TransformPoint(hinge.connectedAnchor)).magnitude;
            Debug.Log($"[VesselModeTests] hinge: right vessel turned to {rightBodies[1].position - rightBodies[0].position:F3}, hinge gap {hingeGap:E2} m");
            Assert.That(rightBodies[1].position.y, Is.LessThan(7.0f), "the right vessel swings down about the hinge");
            Assert.That(hingeGap, Is.LessThan(0.01f), "the hinge holds");
            Assert.That((rightBodies[1].position - rightBodies[0].position).magnitude, Is.EqualTo(gap.magnitude).Within(1e-3f), "each vessel stays rigid");
        }
    }

    /// <summary>
    /// The tests of <see cref="VesselModeTests"/> with the fused vessel job taking every vessel island, as it does by
    /// default, so that the fused job's results are checked against the same expectations as the substeps' jobs'.
    /// </summary>
    public class FusedVesselModeTests : VesselModeTests
    {
        protected override int FusedParts => 0;
    }
}
