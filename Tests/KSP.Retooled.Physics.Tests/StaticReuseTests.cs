using KSP.Retooled.Physics.Diagnostics;
using KSP.Retooled.Physics.Stepping;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using Unity.Physics;
using UnityEngine;
using BoxCollider = UnityEngine.BoxCollider;
using Collider = UnityEngine.Collider;
using MeshCollider = UnityEngine.MeshCollider;
using RaycastHit = UnityEngine.RaycastHit;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Verify retained static rows through body changes, comparing motion, queries and events with full rebuilds.
    /// </summary>
    public unsafe class StaticReuseTests : PhysicsScene
    {
        // A layer that no other test uses, which the test makes ignore the default layer.
        private const int IgnoringLayer = 29;

        private const int Steps = 110;

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

            // For each step, whether the step's build kept the static rows.
            public readonly List<bool> Kept = new List<bool>();
        }

        /// <summary>
        /// Compare built static collider, pose and entity fields with freshly built store rows. The anchor must
        /// have no collider.
        /// </summary>
        private static void AssertStaticRowsMatchTheStore(int step)
        {
            StepEntryPoints.Current->FinishPendingRefit();
            PhysicsWorld world = StepEntryPoints.Current->World;
            WorldData* store = StepEntryPoints.Current->Store;
            Unity.Collections.NativeList<int> built = StepEntryPoints.Current->BuiltIndices;
            int statics = 0;
            for (int row = 0; row < (int)store->bodyCount; row++)
            {
                ref BodyData body = ref store->bodies[row];
                if ((body.flags & BodyFlags.Static) == 0 || (body.flags & BodyFlags.Simulated) == 0)
                {
                    continue;
                }

                statics++;
                int index = built[row];
                Assert.That(index, Is.GreaterThanOrEqualTo(world.NumDynamicBodies), $"step {step}: row {row} is among the static bodies");
                RigidBody rigidBody = world.Bodies[index];
                RigidTransform expected = StoreBodies.BuildBodiesJob.WorldFromBody(ref body);
                Assert.That((long)rigidBody.Collider.GetUnsafePtr(), Is.EqualTo((long)body.Collider()->Value.GetUnsafePtr()), $"step {step}: the collider of row {row}");
                Assert.That(math.asuint(rigidBody.WorldFromBody.pos), Is.EqualTo(math.asuint(expected.pos)), $"step {step}: the position of row {row}");
                Assert.That(math.asuint(rigidBody.WorldFromBody.rot.value), Is.EqualTo(math.asuint(expected.rot.value)), $"step {step}: the rotation of row {row}");
                Assert.That(rigidBody.Entity.Index, Is.EqualTo(row), $"step {step}: the entity of row {row}");
                Assert.That(rigidBody.Entity.Version, Is.EqualTo((int)body.serial), $"step {step}: the serial number of row {row}");
            }

            Assert.That(world.NumStaticBodies, Is.EqualTo(statics + 1), $"step {step}: the static bodies and the anchor");
            Assert.That(world.Bodies[world.NumBodies - 1].Collider.IsCreated, Is.False, $"step {step}: the anchor has no collider");
        }

        /// <summary>
        /// Run static pose, shape, layer, material and membership changes, including dynamic count changes that
        /// move retained rows.
        /// Record queries and collision hashes after each step.
        /// </summary>
        private Run RunScene(bool reuse)
        {
            TearDownScene();
            SetUpScene();
            StepState.StaticReuseEnabled = reuse;
            ForestDiagnostics.ForgetIslands();
            var run = new Run();
            GameObject ground = Ground();
            var names = new Dictionary<Collider, string> { [ground.GetComponent<Collider>()] = "ground" };
            var statics = new List<Collider>();
            var bodies = new List<Rigidbody>();
            PhysicsMaterial slippery = new PhysicsMaterial { dynamicFriction = 0.1f, staticFriction = 0.1f };

            Collider AddStatic(Vector3 position, bool hull)
            {
                GameObject item = Track(new GameObject($"static {statics.Count}"));
                item.transform.SetPositionAndRotation(position, Quaternion.Euler(0.0f, 13.0f * statics.Count, 0.0f));
                Collider collider;
                if (hull)
                {
                    MeshCollider mesh = item.AddComponent<MeshCollider>();
                    mesh.sharedMesh = VesselForestBench.HullMesh();
                    mesh.convex = true;
                    collider = mesh;
                }
                else
                {
                    collider = item.AddComponent<BoxCollider>();
                    ((BoxCollider)collider).size = new Vector3(1.6f, 1.0f, 1.6f);
                }

                names[collider] = item.name;
                statics.Add(collider);
                return collider;
            }

            Rigidbody AddBody(Vector3 position)
            {
                Rigidbody body = DynamicBox(position);
                body.gameObject.AddComponent<JointLoadTrace.ContactHash>();
                names[body.GetComponent<Collider>()] = $"body {bodies.Count}";
                bodies.Add(body);
                return body;
            }

            for (int i = 0; i < 25; i++)
            {
                bool hull = i % 2 == 1;
                AddStatic(new Vector3(3.0f * (i % 5) - 6.0f, hull ? 0.971f : 0.5f, 3.0f * (i / 5) - 6.0f), hull);
            }

            for (int i = 0; i < 12; i++)
            {
                AddBody(statics[i].transform.position + new Vector3(0.1f, 2.5f, 0.05f * i));
            }

            Rigidbody extra = null;

            // The first step makes the world's state, whose counts the test reads.
            Simulate();
            JointLoadTrace.ContactHash.Hash = 14695981039346656037UL;
            JointLoadTrace.ContactHash.Messages = 0;
            var found = new List<string>();
            void Look(string when)
            {
                for (int ray = 0; ray < 5; ray++)
                {
                    var origin = new Vector3(-9.0f, 0.3f + 0.4f * ray, 3.0f * ray - 6.0f + 0.2f);
                    foreach (RaycastHit hit in UnityEngine.Physics.RaycastAll(origin, Vector3.right, 18.0f))
                    {
                        found.Add($"{when} ray {ray} {names[hit.collider]} {hit.point.x:R} {hit.point.y:R} {hit.point.z:R} {hit.distance:R}");
                    }
                }

                foreach (Collider collider in UnityEngine.Physics.OverlapBox(new Vector3(0.0f, 0.5f, 0.0f), new Vector3(4.0f, 0.6f, 4.0f)))
                {
                    found.Add($"{when} box {names[collider]}");
                }
            }

            for (int step = 0; step < Steps; step++)
            {
                found.Clear();
                switch (step)
                {
                    case 10:
                        statics[0].transform.position += new Vector3(0.3f, 0.0f, 0.0f);
                        break;
                    case 15:
                        statics[1].transform.rotation = Quaternion.Euler(0.0f, 20.0f, 0.0f);
                        break;
                    case 20:
                        ((BoxCollider)statics[2]).size = new Vector3(2.0f, 1.0f, 2.0f);
                        break;
                    case 25:
                        statics[4].gameObject.layer = IgnoringLayer;
                        break;
                    case 30:
                        statics[6].sharedMaterial = slippery;
                        break;
                    case 35:
                        slippery.dynamicFriction = 0.02f;
                        break;
                    case 40:
                        statics[8].enabled = false;
                        break;
                    case 45:
                        statics[8].enabled = true;
                        break;
                    case 50:
                        names.Remove(statics[10]);
                        Object.DestroyImmediate(statics[10].gameObject);
                        break;
                    case 55:
                        AddStatic(new Vector3(-7.5f, 0.5f, 7.5f), false);
                        break;
                    case 60:
                        statics[12].gameObject.AddComponent<Rigidbody>();
                        break;
                    case 65:
                        Object.DestroyImmediate(statics[12].GetComponent<Rigidbody>());
                        break;
                    case 70:
                        statics[14].gameObject.AddComponent<Rigidbody>().isKinematic = true;
                        break;
                    case 75:
                        statics[16].isTrigger = true;
                        break;
                    case 80:
                        // A collider without a Rigidbody would make a static body first, so the Rigidbody comes first.
                        GameObject added = Track(new GameObject("extra"));
                        added.transform.position = new Vector3(7.5f, 3.0f, -7.5f);
                        extra = added.AddComponent<Rigidbody>();
                        names[added.AddComponent<BoxCollider>()] = "extra";
                        added.AddComponent<JointLoadTrace.ContactHash>();
                        bodies.Add(extra);
                        break;
                    case 85:
                        bodies.Remove(extra);
                        names.Remove(extra.GetComponent<Collider>());
                        Object.DestroyImmediate(extra.gameObject);
                        break;
                    case 90:
                        // The query builds the world, which takes in the move, before the step does.
                        statics[18].transform.position += new Vector3(0.0f, 0.0f, 0.4f);
                        Look("before");
                        break;
                }

                long kept = StepEntryPoints.Current->KeptStaticRowsCount;
                Simulate();
                run.Kept.Add(StepEntryPoints.Current->KeptStaticRowsCount != kept);
                AssertStaticRowsMatchTheStore(step);
                Look("after");
                found.Sort(string.CompareOrdinal);
                run.Steps.Add($"step {step}: contacts {JointLoadTrace.ContactHash.Hash:x16} in {JointLoadTrace.ContactHash.Messages} messages; {string.Join(", ", found)}");
            }

            foreach (Rigidbody body in bodies)
            {
                run.Positions.Add(body.position);
                run.Rotations.Add(body.rotation);
            }

            Object.DestroyImmediate(slippery);
            return run;
        }

        [Test]
        public void KeptStaticRowsGiveTheResultsOfRowsWrittenAgain()
        {
            // RETOOLED_STATIC_REUSE=0 turns the reuse off for a whole run, so give the later tests the switch as it was.
            bool wasEnabled = StepState.StaticReuseEnabled;
            bool ignored = UnityEngine.Physics.GetIgnoreLayerCollision(IgnoringLayer, 0);
            UnityEngine.Physics.IgnoreLayerCollision(IgnoringLayer, 0, true);
            try
            {
                // The first scene of this shape can land a few ulps from every later one.
                RunScene(false);
                Run kept = RunScene(true);
                Run written = RunScene(false);

                Assert.That(written.Kept, Is.All.False, "with the reuse off, every build writes the static rows");
                int[] changes = { 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75 };
                foreach (int step in changes)
                {
                    Assert.That(kept.Kept[step], Is.False, $"the build after the change at step {step} writes the static rows");
                }

                Assert.That(kept.Kept[5], Is.True, "a build in which nothing changed keeps the static rows");
                Assert.That(kept.Kept[80], Is.True, "adding a dynamic body keeps the static rows, which move along");
                Assert.That(kept.Kept[85], Is.True, "removing the last dynamic body keeps the static rows, which move back");
                Assert.That(kept.Kept[90], Is.True, "the query's build took in the move, so the step keeps the rows that it wrote");
                int keptSteps = kept.Kept.FindAll(k => k).Count;
                Assert.That(keptSteps, Is.GreaterThan(Steps - changes.Length - 2), "every other build keeps the static rows");

                Assert.That(kept.Positions.Count, Is.EqualTo(written.Positions.Count));
                for (int i = 0; i < written.Positions.Count; i++)
                {
                    Assert.That(kept.Positions[i], Is.EqualTo(written.Positions[i]), $"body {i}'s position");
                    Assert.That(kept.Rotations[i], Is.EqualTo(written.Rotations[i]), $"body {i}'s rotation");
                }

                for (int step = 0; step < Steps; step++)
                {
                    Assert.That(kept.Steps[step], Is.EqualTo(written.Steps[step]));
                }

                // The box that rested on the hull that moved to the layer that ignores the boxes fell through it onto
                // the ground, so the new filter reached the pairs.
                Assert.That(written.Positions[4].y, Is.LessThan(1.0f), "body 4 fell through static 4 onto the ground");
            }
            finally
            {
                StepState.StaticReuseEnabled = wasEnabled;
                UnityEngine.Physics.IgnoreLayerCollision(IgnoringLayer, 0, ignored);
            }
        }
    }
}
