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
            vessels = new TestVessels();
        }

        [TearDown]
        public void End()
        {
            vessels?.Dispose();
            vessels = null;
            VesselForest.Enabled = wasEnabled;
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
        private List<Rigidbody> Column(int count, float bottom, double breakForce = double.PositiveInfinity)
        {
            var bodies = new List<Rigidbody>();
            var parents = new List<int>();
            for (int i = 0; i < count; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, bottom + 1.0f * (count - 1 - i) + 0.5f, 0.0f), new Vector3(0.6f, 1.0f, 0.6f))
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
                new Vector3(0.0f, bottom + 1.0f * (count - i), 0.0f), breakForce));
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
            Simulate();
            vessels.Publish();
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

        [Test]
        public void AnEdgePastItsBreakForceReadsAsBroken()
        {
            List<Rigidbody> bodies = Column(2, 5.0f, 20.0);
            ConfigurableJoint hook = bodies[0].gameObject.AddComponent<ConfigurableJoint>();
            hook.xMotion = hook.yMotion = hook.zMotion = ConfigurableJointMotion.Locked;
            hook.angularXMotion = hook.angularYMotion = hook.angularZMotion = ConfigurableJointMotion.Locked;
            Simulate();
            vessels.Publish();
            vessels.Step(10);
            VesselEdgeLoad load = vessels.Load(1);
            Debug.Log($"[VesselModeTests] break: ratio {load.Ratio:F3}, force {math.length(load.Force):F2}");
            Assert.That(load.Ratio, Is.EqualTo(PartMass * -Gravity / 20.0).Within(0.05), "the ratio is the load over the break force");
            Assert.That(load.Ratio, Is.GreaterThan(1.0));
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
            Simulate();
            vessels.Publish();

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
            Simulate();
            vessels.Publish();
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
            Simulate();
            vessels.Publish();
            Vector3 gap = rightBodies[1].position - rightBodies[0].position;
            vessels.Step(60);
            float hingeGap = (rightBodies[0].transform.TransformPoint(hinge.anchor) - left[0].transform.TransformPoint(hinge.connectedAnchor)).magnitude;
            Debug.Log($"[VesselModeTests] hinge: right vessel turned to {rightBodies[1].position - rightBodies[0].position:F3}, hinge gap {hingeGap:E2} m");
            Assert.That(rightBodies[1].position.y, Is.LessThan(7.0f), "the right vessel swings down about the hinge");
            Assert.That(hingeGap, Is.LessThan(0.01f), "the hinge holds");
            Assert.That((rightBodies[1].position - rightBodies[0].position).magnitude, Is.EqualTo(gap.magnitude).Within(1e-3f), "each vessel stays rigid");
        }
    }
}
