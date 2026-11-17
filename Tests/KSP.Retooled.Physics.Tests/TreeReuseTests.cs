using KSP.Retooled.Physics.Diagnostics;
using KSP.Retooled.Physics.Stepping;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Compare refitted and rebuilt dynamic trees through scene changes, requiring identical motion, query hits and
    /// events.
    /// </summary>
    public unsafe class TreeReuseTests : PhysicsScene
    {
        // A layer that no other test uses, which the test makes ignore the ground's layer.
        private const int FallingLayer = 29;

        private const int Steps = 130;

        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
        }

        private sealed class Run
        {
            public readonly List<string> Steps = new List<string>();
            public readonly List<Vector3> Positions = new List<Vector3>();
            public readonly List<Quaternion> Rotations = new List<Quaternion>();

            // For each step, whether the step's build built the dynamic tree from scratch.
            public readonly List<bool> Built = new List<bool>();
            public long Refits;
            public long FilterRefits;
        }

        /// <summary>
        /// Run settling boxes through body creation/removal, movement, collider/layer changes, kinematic changes
        /// and sleep/wake.
        /// Record query hits and collision hashes after each step.
        /// </summary>
        private Run RunScene(bool reuse)
        {
            TearDownScene();
            SetUpScene();
            StepState.TreeReuseEnabled = reuse;
            ForestDiagnostics.ForgetIslands();
            var run = new Run();
            GameObject ground = Ground();
            var names = new Dictionary<Collider, string> { [ground.GetComponent<Collider>()] = "ground" };
            var bodies = new List<Rigidbody>();

            Rigidbody Add(Vector3 position)
            {
                Rigidbody body = DynamicBox(position);
                body.gameObject.AddComponent<JointLoadTrace.ContactHash>();
                names[body.GetComponent<Collider>()] = $"body {bodies.Count}";
                bodies.Add(body);
                return body;
            }

            for (int i = 0; i < 40; i++)
            {
                Add(new Vector3(1.5f * (i % 8) - 5.0f, 0.5f + 1.02f * (i / 8), 0.3f * (i % 3)));
            }

            // The first step makes the world's state, whose counts the test reads.
            Simulate();
            JointLoadTrace.ContactHash.Hash = 14695981039346656037UL;
            JointLoadTrace.ContactHash.Messages = 0;
            long builds = StepEntryPoints.Current->TreeBuildCount;
            long refits = StepEntryPoints.Current->TreeRefitCount;
            long filterRefits = StepEntryPoints.Current->TreeFilterRefitCount;
            for (int step = 0; step < Steps; step++)
            {
                switch (step)
                {
                    case 30:
                        Add(new Vector3(0.2f, 7.0f, 0.0f));
                        break;
                    case 40:
                        Object.DestroyImmediate(bodies[5].gameObject);
                        break;
                    case 50:
                        bodies[10].position += new Vector3(0.3f, 0.0f, 0.0f);
                        break;
                    case 55:
                        bodies[11].position = new Vector3(400.0f, 0.5f, 0.0f);
                        break;
                    case 60:
                        bodies[11].position = new Vector3(-8.0f, 0.5f, 4.0f);
                        break;
                    case 65:
                        bodies[12].GetComponent<BoxCollider>().size = new Vector3(1.4f, 1.0f, 1.0f);
                        break;
                    case 70:
                        bodies[13].gameObject.layer = FallingLayer;
                        break;
                    case 80:
                        bodies[14].isKinematic = true;
                        break;
                    case 85:
                        bodies[14].isKinematic = false;
                        break;
                    case 90:
                        bodies[15].Sleep();
                        break;
                    case 100:
                        bodies[15].WakeUp();
                        break;
                }

                Simulate();
                long built = StepEntryPoints.Current->TreeBuildCount;
                run.Built.Add(built != builds);
                builds = built;

                var found = new List<string>();
                for (int ray = 0; ray < 6; ray++)
                {
                    var origin = new Vector3(-10.0f, 0.5f + ray, 0.1f * ray);
                    foreach (RaycastHit hit in UnityEngine.Physics.RaycastAll(origin, Vector3.right, 20.0f))
                    {
                        found.Add($"ray {ray} {names[hit.collider]} {hit.point.x:R} {hit.point.y:R} {hit.point.z:R} {hit.distance:R}");
                    }
                }

                foreach (Collider collider in UnityEngine.Physics.OverlapBox(new Vector3(0.0f, 2.0f, 0.0f), new Vector3(3.0f, 2.0f, 1.0f)))
                {
                    found.Add($"box {names[collider]}");
                }

                found.Sort(string.CompareOrdinal);
                run.Steps.Add($"step {step}: contacts {JointLoadTrace.ContactHash.Hash:x16} in {JointLoadTrace.ContactHash.Messages} messages; {string.Join(", ", found)}");
            }

            run.Refits = StepEntryPoints.Current->TreeRefitCount - refits;
            run.FilterRefits = StepEntryPoints.Current->TreeFilterRefitCount - filterRefits;
            foreach (Rigidbody body in bodies)
            {
                if (body != null)
                {
                    run.Positions.Add(body.position);
                    run.Rotations.Add(body.rotation);
                }
            }

            return run;
        }

        [Test]
        public void ARefittedTreeGivesTheResultsOfATreeBuiltFromScratch()
        {
            // RETOOLED_TREE_REUSE=0 turns the reuse off for a whole run, so give the later tests the switch as it was.
            bool wasEnabled = StepState.TreeReuseEnabled;
            bool wasAlways = StepState.TreeAlwaysRefitted;
            bool ignored = UnityEngine.Physics.GetIgnoreLayerCollision(FallingLayer, 0);
            UnityEngine.Physics.IgnoreLayerCollision(FallingLayer, 0, true);
            try
            {
                // The first scene of this shape can land a few ulps from every later one.
                RunScene(false);
                Run refitted = RunScene(true);
                Run built = RunScene(false);

                Debug.Log($"[TreeReuseTests] {refitted.Refits} refits in the builds of {Steps} steps and of their queries, {refitted.FilterRefits} of them with new filters; the steps that built the tree from scratch: "
                    + string.Join(" ", Enumerable(refitted.Built)));
                Assert.That(built.Refits, Is.Zero, "with the reuse off, every build builds the tree");
                Assert.That(built.Built, Is.All.True);
                Assert.That(refitted.Refits, Is.GreaterThan(Steps / 2), "most builds refit the tree");
                Assert.That(refitted.Built[30], Is.True, "adding a body builds the tree");
                Assert.That(refitted.Built[40], Is.True, "removing a body builds the tree");
                Assert.That(refitted.Built[56], Is.True, "the tree is built again after a body moved 400 m away");
                Assert.That(refitted.FilterRefits, Is.GreaterThan(0), "a refit built the nodes' filters again for the box on the new layer");

                Assert.That(refitted.Positions.Count, Is.EqualTo(built.Positions.Count));
                for (int i = 0; i < built.Positions.Count; i++)
                {
                    Assert.That(refitted.Positions[i], Is.EqualTo(built.Positions[i]), $"body {i}'s position");
                    Assert.That(refitted.Rotations[i], Is.EqualTo(built.Rotations[i]), $"body {i}'s rotation");
                }

                for (int step = 0; step < Steps; step++)
                {
                    Assert.That(refitted.Steps[step], Is.EqualTo(built.Steps[step]));
                }

                // The box on the new layer fell through the ground, so the filter reached the pairs.
                Assert.That(built.Steps[0], Does.Contain("body 13 "), "a ray finds the box at the start");
                Assert.That(built.Steps[Steps - 1], Does.Not.Contain("body 13 ").And.Not.Contain("body 13,").And.Not.EndsWith("body 13"),
                    "the box on the layer that ignores the ground fell through it");
            }
            finally
            {
                StepState.TreeReuseEnabled = wasEnabled;
                if (wasAlways)
                {
                    StepState.TreeAlwaysRefitted = true;
                }

                UnityEngine.Physics.IgnoreLayerCollision(FallingLayer, 0, ignored);
            }
        }

        /// <summary>
        /// Verify equal-distance queries choose the later-created box regardless of refitted or rebuilt tree
        /// traversal order.
        /// </summary>
        [Test]
        public void AClosestHitAtEqualDistancesDoesNotDependOnTheTree()
        {
            bool wasEnabled = StepState.TreeReuseEnabled;
            bool wasAlways = StepState.TreeAlwaysRefitted;
            try
            {
                string built = TieScene(refitted: false);
                string refitted = TieScene(refitted: true);
                Debug.Log($"[TreeReuseTests] ties with a tree built from scratch: {built}");
                Assert.That(refitted, Is.EqualTo(built));
                Assert.That(built, Does.Not.Contain("first"), "every query reports the box created later, as PhysX does");
            }
            finally
            {
                StepState.TreeReuseEnabled = wasEnabled;
                if (wasAlways)
                {
                    StepState.TreeAlwaysRefitted = true;
                }
            }
        }

        private string TieScene(bool refitted)
        {
            TearDownScene();
            SetUpScene();
            ForestDiagnostics.ForgetIslands();
            Rigidbody Kinematic(string name, Vector3 position)
            {
                Rigidbody body = DynamicBox(position);
                body.isKinematic = true;
                body.name = name;
                return body;
            }

            Rigidbody first = Kinematic("first", new Vector3(-0.25f, 0.0f, 0.0f));
            Rigidbody second = Kinematic("second", new Vector3(0.25f, 0.0f, 0.0f));
            for (int i = 0; i < 20; i++)
            {
                Kinematic($"other {i}", new Vector3(3.0f + 1.5f * i, 0.3f * (i % 4), 2.0f * (i % 3)));
            }

            // The first step makes the world's state, which reads the switches from the environment, so the test sets
            // them after that step. The first step builds the tree from scratch either way.
            Simulate();
            StepState.TreeReuseEnabled = refitted;
            if (refitted)
            {
                StepState.TreeAlwaysRefitted = true;
            }

            first.position = new Vector3(0.25f, 0.0f, 0.0f);
            second.position = new Vector3(-0.25f, 0.0f, 0.0f);
            first.transform.position = first.position;
            second.transform.position = second.position;
            Simulate(2);

            long builds = StepEntryPoints.Current->TreeBuildCount;
            Simulate();
            Assert.That(StepEntryPoints.Current->TreeBuildCount != builds, Is.EqualTo(!refitted), refitted ? "the step refits the tree" : "the step builds the tree");

            var found = new List<string>();
            var origin = new Vector3(0.0f, 5.0f, 0.0f);
            if (UnityEngine.Physics.Raycast(origin, Vector3.down, out RaycastHit ray, 20.0f))
            {
                found.Add($"ray {ray.collider.name} {ray.distance:R} {ray.point.x:R} {ray.point.y:R} {ray.point.z:R} {ray.normal.y:R}");
            }

            if (UnityEngine.Physics.SphereCast(origin, 0.1f, Vector3.down, out RaycastHit sphere, 20.0f))
            {
                found.Add($"sphere {sphere.collider.name} {sphere.distance:R} {sphere.point.x:R} {sphere.point.y:R} {sphere.point.z:R}");
            }

            if (UnityEngine.Physics.BoxCast(origin, new Vector3(0.1f, 0.1f, 0.1f), Vector3.down, out RaycastHit box, Quaternion.identity, 20.0f))
            {
                found.Add($"box {box.collider.name} {box.distance:R} {box.point.x:R} {box.point.y:R} {box.point.z:R}");
            }

            Assert.That(found.Count, Is.EqualTo(3), "every query hits a box");
            return string.Join("; ", found);
        }

        private static IEnumerable<int> Enumerable(List<bool> built)
        {
            for (int step = 0; step < built.Count; step++)
            {
                if (built[step])
                {
                    yield return step;
                }
            }
        }
    }
}
