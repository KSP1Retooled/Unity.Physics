using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using KSP.Retooled.Physics.Stepping;
using KSP.Retooled.Physics.Tests.Benchmarks;
using NUnit.Framework;
using Unity.Physics.Configuration;
using Unity.Physics.Stepping.Recording;
using UnityEngine;
using Object = UnityEngine.Object;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;

namespace KSP.Retooled.Physics.Tests.Tools
{
    public unsafe class FlightRecorderTests : PhysicsScene
    {
        private readonly List<Object> assets = new List<Object>();
        private string directory;
        private bool enabled;
        private int window;
        private int postSteps;
        private int maxFiles;
        private string output;
        private RecordedPhysicsSettings settings;

        [SetUp]
        public void SetUpRecorder()
        {
            Assume.That(BackendInfo.IsRetooled);
            EnvironmentSwitches.Read();
            settings = RecordedPhysicsSettings.Read();
            VesselSwitches.MinimumParts = 2;
            SolverSwitches.FusedVesselParts = 1;
            enabled = FlightRecorderSwitches.Enabled;
            window = FlightRecorderSwitches.WindowSteps;
            postSteps = FlightRecorderSwitches.PostTriggerSteps;
            maxFiles = FlightRecorderSwitches.MaxFiles;
            output = FlightRecorderSwitches.OutputDirectory;
            directory = Path.Combine(Application.dataPath, "..", "Logs", "FlightRecorder", Guid.NewGuid().ToString("N"));
            FlightRecorderSwitches.Enabled = false;
            FlightRecorderSwitches.WindowSteps = 4;
            FlightRecorderSwitches.PostTriggerSteps = 2;
            FlightRecorderSwitches.MaxFiles = int.MaxValue;
            FlightRecorderSwitches.OutputDirectory = directory;
        }

        [TearDown]
        public void TearDownRecorder()
        {
            if (directory == null)
            {
                return;
            }

            DestroyAll();
            foreach (Object asset in assets)
            {
                Object.DestroyImmediate(asset);
            }

            assets.Clear();
            FlightRecorderSwitches.Enabled = false;
            FlightRecorderSwitches.Enabled = enabled;
            FlightRecorderSwitches.OutputDirectory = output;
            FlightRecorderSwitches.MaxFiles = maxFiles;
            FlightRecorderSwitches.WindowSteps = window;
            FlightRecorderSwitches.PostTriggerSteps = postSteps;
            settings.Apply();
            directory = null;
        }

        [Test]
        public void ManualSaveReplaysRollingWindow()
        {
            GameObject ground = Ground();
            Rigidbody root = WeightlessBox(new Vector3(0, 0.49f, 0));
            Rigidbody child = WeightlessBox(new Vector3(0, 1.49f, 0));
            StackJoint(child, root);
            Rigidbody probe = DynamicBox(new Vector3(3, 1f, 0));
            root.sleepThreshold = child.sleepThreshold = 0f;
            Simulate(8);
            Assert.That(StepEntryPoints.Current->VesselSet.VesselCount > 0, Is.True);

            FlightRecorderSwitches.Enabled = true;
            Simulate(3);
            probe.AddForce(new Vector3(3, 0, 1), ForceMode.Impulse);
            child.GetComponent<ConfigurableJoint>().targetAngularVelocity = new Vector3(0, 0.4f, 0.1f);
            child.mass *= 1.25f;
            UnityEngine.Physics.IgnoreCollision(probe.GetComponent<Collider>(), ground.GetComponent<Collider>());
            Rigidbody removed = DynamicBox(new Vector3(7, 1, 0));
            Simulate(2);
            Object.DestroyImmediate(removed.gameObject);
            Simulate();

            string path = WaitForRecording(FlightRecorderSwitches.SaveWithReason("rolling-window test"));
            using var summary = new StringWriter();
            Assert.That(FlightReplay.Run(path, long.MaxValue, summary), Is.EqualTo(4), summary.ToString());
            var numbers = new List<long>();
            Assert.That(FlightReplay.Run(path, 4, TextWriter.Null,
                (step, _) => numbers.Add(step)), Is.EqualTo(2));
            Assert.That(numbers, Is.EqualTo(new long[] { 3, 4 }));
        }

        /// <summary>
        /// A save gives its slot back when the file is written, so a recorder that allows one queued save can save
        /// again and again.
        /// </summary>
        [Test]
        public void CompletedSaveFreesItsSlot()
        {
            WeightlessBox(new Vector3(0, 5, 0));
            FlightRecorderSwitches.Enabled = true;
            Simulate();
            FlightRecorderSwitches.MaxFiles = 1;

            for (int i = 0; i < 5; i++)
            {
                string path = FlightRecorderSwitches.Save();
                Assert.That(path, Is.Not.Null, FlightRecorderSwitches.GetStatus());
                WaitForRecording(path);
                WaitForWriter();
            }

            Assert.That(Directory.GetFiles(directory, "*.rfr"), Has.Length.EqualTo(5));
        }

        /// <summary>
        /// A save that fails to write also gives its slot back, so the next save still goes through.
        /// </summary>
        [Test]
        public void FailedSaveFreesItsSlot()
        {
            WeightlessBox(new Vector3(0, 5, 0));
            FlightRecorderSwitches.Enabled = true;
            Simulate();
            FlightRecorderSwitches.MaxFiles = 1;
            Directory.CreateDirectory(directory);
            string blockedDirectory = Path.Combine(directory, "file");
            File.WriteAllText(blockedDirectory, "The writer cannot create a directory here.");
            FlightRecorderSwitches.OutputDirectory = blockedDirectory;

            Assert.That(FlightRecorderSwitches.Save(), Is.Not.Null, FlightRecorderSwitches.GetStatus());
            WaitForWriter();

            FlightRecorderSwitches.OutputDirectory = directory;
            string path = FlightRecorderSwitches.Save();
            Assert.That(path, Is.Not.Null, FlightRecorderSwitches.GetStatus());
            WaitForRecording(path);
            WaitForWriter();
        }

        private static void WaitForWriter()
        {
            Assert.That(SpinWait.SpinUntil(() => FlightRecorderSwitches.GetStatus().Contains("SavesQueued=0,"),
                TimeSpan.FromSeconds(2)), Is.True, FlightRecorderSwitches.GetStatus());
        }

        /// <summary>
        /// The replay holds the same ignored collider pairs as the recorded world, not pairs made from the table's
        /// internal slot contents.
        /// </summary>
        [Test]
        public void ReplayKeepsIgnoredPairs()
        {
            Ground();
            Rigidbody first = DynamicBox(new Vector3(0, 0.5f, 0));
            Rigidbody second = DynamicBox(new Vector3(3, 0.5f, 0));
            Rigidbody third = DynamicBox(new Vector3(6, 0.5f, 0));
            UnityEngine.Physics.IgnoreCollision(first.GetComponent<Collider>(), second.GetComponent<Collider>());
            UnityEngine.Physics.IgnoreCollision(second.GetComponent<Collider>(), third.GetComponent<Collider>());
            FlightRecorderSwitches.Enabled = true;
            Simulate(3);

            string path = WaitForRecording(FlightRecorderSwitches.SaveWithReason("ignored pairs test"));
            var counts = new List<int>();
            FlightReplay.Run(path, long.MaxValue, TextWriter.Null, null, (step, world) =>
            {
                var store = world->Store;
                int ignored = 0;
                for (uint a = 0; a < store->bodyCount; a++)
                {
                    for (uint b = a + 1; b < store->bodyCount; b++)
                    {
                        for (uint i = 0; i < store->bodies[a].shapeCount; i++)
                        {
                            for (uint j = 0; j < store->bodies[b].shapeCount; j++)
                            {
                                ignored += Unity.Physics.Collision.Filter.Groups.IgnoreTable.Contains(world->Globals,
                                    store->bodies[a].shapes[i]->ignorePairId, store->bodies[b].shapes[j]->ignorePairId) ? 1 : 0;
                            }
                        }
                    }
                }

                counts.Add(ignored);
            });

            Assert.That(counts, Is.Not.Empty);
            Assert.That(counts, Is.All.EqualTo(2));
        }

        /// <summary>
        /// The replay keeps the crash speed that the game gave a collider. The bottom box of a falling two-box vessel
        /// hits the ground above its crash speed, so its contact may take only the box's own momentum in its first step.
        /// A replay without the crash speed would stop the whole vessel in that step and differ from the recording.
        /// </summary>
        [Test]
        public void ReplayKeepsCrashSpeeds()
        {
            Ground();
            Rigidbody bottom = DynamicBox(new Vector3(0, 1.5f, 0));
            Rigidbody top = DynamicBox(new Vector3(0, 2.5f, 0));
            StackJoint(top, bottom);
            Assert.That(CrashContacts.SetCrashSpeed(bottom.GetComponent<Collider>(), 5.0f), Is.True, "crash speed set");
            Simulate();
            bottom.linearVelocity = top.linearVelocity = new Vector3(0, -30, 0);
            FlightRecorderSwitches.Enabled = true;
            Simulate(4);

            string path = WaitForRecording(FlightRecorderSwitches.SaveWithReason("crash speed test"));
            var crashShapes = new List<int>();
            using var summary = new StringWriter();
            FlightReplay.Run(path, long.MaxValue, summary, after: (step, world) =>
            {
                var store = world->Store;
                int count = 0;
                for (uint row = 0; row < store->bodyCount; row++)
                {
                    for (uint i = 0; i < store->bodies[row].shapeCount; i++)
                    {
                        float speed = Unity.Physics.Collision.Filter.Groups.GroupTable.CrashSpeedOf(world->Globals, store->bodies[row].shapes[i]);
                        count += speed == 5.0f ? 1 : 0;
                    }
                }

                crashShapes.Add(count);
            });

            Assert.That(crashShapes, Is.Not.Empty);
            Assert.That(crashShapes, Is.All.EqualTo(1), summary.ToString());
        }

        [Test]
        public void VelocityWriteTriggersAndReplays()
        {
            Rigidbody body = WeightlessBox(new Vector3(17, 5, 3));
            Rigidbody neighbor = WeightlessBox(new Vector3(17, 6, 3));
            StackJoint(neighbor, body);
            FlightRecorderSwitches.Enabled = true;
            Simulate(4);
            uint serial = FindSerial(body.position);

            body.linearVelocity = new Vector3(450f, 0f, 0f);
            Simulate(2);
            StringAssert.Contains("PostTriggerStepsRemaining=1", FlightRecorderSwitches.GetStatus());
            Assert.That(Directory.Exists(directory), Is.False);
            Simulate();
            string path = WaitForRecording(null);
            using var summary = new StringWriter();
            Assert.That(FlightReplay.Run(path, long.MaxValue, summary), Is.EqualTo(6), summary.ToString());
            StringAssert.Contains($"linear velocity change, body {serial},", summary.ToString());
            StringAssert.Contains("Joint ", summary.ToString());
            StringAssert.Contains("FusedVessel", summary.ToString());
        }

        [Test]
        public void NonFiniteMotionRestoredByWriteBackTriggersAndReplays()
        {
            Rigidbody body = WeightlessBox(new Vector3(2, 5, 3));
            body.sleepThreshold = 0f;
            Simulate();
            uint serial = FindSerial(body.position);
            int recoveries = StepEntryPoints.Current->Counters.NonFiniteBodies;
            FlightRecorderSwitches.Enabled = true;
            Simulate();

            var store = StepEntryPoints.Current->Store;
            for (uint row = 0; row < store->bodyCount; row++)
            {
                if (store->bodies[row].serial == serial)
                {
                    store->bodies[row].forces->AngularVelocityChange = new Unity.Mathematics.float3(float.NaN, 0, 0);
                }
            }

            Simulate(3);
            Assert.That(StepEntryPoints.Current->Counters.NonFiniteBodies, Is.EqualTo(recoveries + 1));
            Assert.That(body.angularVelocity, Is.EqualTo(Vector3.zero));

            string path = WaitForRecording(null);
            using var summary = new StringWriter();
            int replayedRecoveries = 0;
            Assert.That(FlightReplay.Run(path, long.MaxValue, summary,
                after: (_, stepper) => replayedRecoveries += stepper->Counters.NonFiniteBodies), Is.EqualTo(4));
            Assert.That(replayedRecoveries, Is.EqualTo(1));
            StringAssert.Contains($"write-back restored non-finite motion, body {serial},", summary.ToString());
        }

        [Test]
        public void LargeTriangleMeshesAndHeightFieldReplay()
        {
            TerrainData terrain = TerrainAndMeshes();
            FlightRecorderSwitches.Enabled = true;
            Simulate(2);
            terrain.SetHeights(20, 20, new[,] { { 0.02f, 0.02f }, { 0.02f, 0.02f } });
            Simulate(3);
            string path = WaitForRecording(FlightRecorderSwitches.Save());
            Assert.That(FlightReplay.Run(path, long.MaxValue, TextWriter.Null), Is.EqualTo(4));
            TestContext.WriteLine($"Terrain recording {new FileInfo(path).Length:N0} bytes");
        }

        [Test]
        public void SixSecondWindowReplaysFromAWorldAlreadyInMotion()
        {
            FlightRecorderSwitches.WindowSteps = 300;
            Ground();
            Rigidbody root = DynamicBox(new Vector3(0, 0.55f, 0));
            Rigidbody child = DynamicBox(new Vector3(0, 1.55f, 0));
            StackJoint(child, root);
            Rigidbody probe = DynamicBox(new Vector3(4, 1, 0));
            Simulate(24);
            FlightRecorderSwitches.Enabled = true;
            for (int i = 0; i < 320; i++)
            {
                if (i % 50 == 0)
                {
                    root.AddForce(new Vector3(0.3f, 0.6f, 0.02f), ForceMode.Impulse);
                    probe.AddForce(new Vector3(-0.2f, 2f, 0.3f), ForceMode.Impulse);
                }

                Simulate();
            }

            string path = WaitForRecording(FlightRecorderSwitches.Save());
            Assert.That(FlightReplay.Run(path, long.MaxValue, TextWriter.Null), Is.EqualTo(300));
            TestContext.WriteLine($"Six-second recording {new FileInfo(path).Length:N0} bytes");
        }

        [Test]
        public void LockedForestReplaysWithOrdinaryJointRows()
        {
            VesselSwitches.MinimumParts = 3;
            ForestSwitches.Locking = true;
            Ground();
            Rigidbody root = WeightlessBox(new Vector3(0, 0.499f, 0));
            Rigidbody child = WeightlessBox(new Vector3(0, 1.499f, 0));
            root.sleepThreshold = child.sleepThreshold = 0f;
            StackJoint(child, root);
            Simulate(24);
            Assert.That(StepEntryPoints.Current->VesselSet.VesselCount, Is.Zero);
            FlightRecorderSwitches.Enabled = true;
            child.AddForce(new Vector3(0.15f, -0.04f, 0.1f), ForceMode.Impulse);
            Simulate(8);
            string path = WaitForRecording(FlightRecorderSwitches.Save());
            Assert.That(FlightReplay.Run(path, long.MaxValue, TextWriter.Null), Is.EqualTo(4));
        }

        [Test]
        public void ReplayPreservesEmptyVehicleSlots()
        {
            Ground();
            Rigidbody first = DynamicBox(new Vector3(-2, 0.95f, 0), 100f);
            Rigidbody second = DynamicBox(new Vector3(2, 0.95f, 0), 100f);
            AddWheel(first);
            WheelCollider survivingWheel = AddWheel(second);
            Simulate(14);
            FlightRecorderSwitches.Enabled = true;
            Simulate(2);
            Object.DestroyImmediate(first.gameObject);
            Simulate(3);

            var world = StepEntryPoints.Current->Store;
            bool emptySlot = false;
            for (uint i = 0; i < world->vehicleCount; i++)
            {
                emptySlot |= world->vehicles[i] == null;
            }

            Assert.That(emptySlot, Is.True);
            Assert.That(survivingWheel.isGrounded, Is.True);
            string path = WaitForRecording(FlightRecorderSwitches.Save());
            Assert.That(FlightReplay.Run(path, long.MaxValue, TextWriter.Null), Is.EqualTo(4));
        }

        private WheelCollider AddWheel(Rigidbody body)
        {
            body.freezeRotation = true;
            var wheel = Track(new GameObject("wheel"));
            wheel.transform.SetParent(body.transform, false);
            wheel.transform.localPosition = new Vector3(0, -0.2f, 0);
            return wheel.AddComponent<WheelCollider>();
        }

        [Test, Explicit("Measures recording overhead on a large vessel and a terrain scene.")]
        public void Measure()
        {
            FlightRecorderSwitches.WindowSteps = 300;
            var graph = VesselForestBench.Load("tu160:128x8");
            Assert.That(graph.Parent.Length, Is.EqualTo(1024));
            int layer = VesselForestBench.PartLayer;
            bool ignored = UnityEngine.Physics.GetIgnoreLayerCollision(layer, layer);
            UnityEngine.Physics.IgnoreLayerCollision(layer, layer, true);
            try
            {
                VesselForestBench.Build(graph, false, Box);
                Simulate();
                Assert.That(StepEntryPoints.Current->VesselSet.VesselCount, Is.GreaterThan(0));
                MeasureScene("1024 vessel parts");
            }
            finally
            {
                UnityEngine.Physics.IgnoreLayerCollision(layer, layer, ignored);
            }

            DestroyAll();
            TerrainAndMeshes();
            MeasureScene("three 32,768-triangle meshes and a 257 × 257 height field");
        }

        private void MeasureScene(string name)
        {
            const int passes = 10;
            const int measuredSteps = 300;
            for (int pass = 0; pass < passes; pass++)
            {
                bool recording = pass % 2 != 0;
                FlightRecorderSwitches.Enabled = recording;
                Simulate(320);
                var samples = new double[measuredSteps];
                var startLoad = VesselForestBench.CpuReading.Take();
                for (int i = 0; i < samples.Length; i++)
                {
                    long start = System.Diagnostics.Stopwatch.GetTimestamp();
                    Simulate();
                    samples[i] = (System.Diagnostics.Stopwatch.GetTimestamp() - start)
                        * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                }
                Array.Sort(samples);
                double median = (samples[149] + samples[150]) / 2;
                double p95 = samples[284];
                TestContext.WriteLine($"{name}, pass {pass + 1}, recording {recording}, median {median:F4} ms, "
                    + $"p95 {p95:F4} ms, other processes' CPU "
                    + $"{VesselForestBench.CpuReading.Take().OthersLoadSince(startLoad):F1}%, "
                    + FlightRecorderSwitches.GetStatus());
                if (recording && pass == passes - 1)
                {
                    string path = WaitForRecording(FlightRecorderSwitches.SaveWithReason(name));
                    TestContext.WriteLine($"Recording file {new FileInfo(path).Length:N0} bytes at {path}");
                }
            }
        }

        private static void StackJoint(Rigidbody body, Rigidbody parent, Vector3? anchor = null)
        {
            Vector3 center = anchor ?? (body.position + parent.position) * 0.5f;
            for (int i = 0; i < 3; i++)
            {
                Vector3 node = center + Quaternion.AngleAxis(i * 120, Vector3.up) * Vector3.right * 0.3f;
                var joint = body.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = parent;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = body.transform.InverseTransformPoint(node);
                joint.connectedAnchor = parent.transform.InverseTransformPoint(node);
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = ConfigurableJointMotion.Locked;
                joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
                joint.angularYLimit = new SoftJointLimit { limit = 177 };
                joint.angularZLimit = new SoftJointLimit { limit = 177 };
                joint.angularYZDrive = new JointDrive { positionSpring = 20000f, positionDamper = 200f, maximumForce = 1e20f };
            }
        }

        private TerrainData TerrainAndMeshes()
        {
            const int side = 129;
            for (int part = 0; part < 3; part++)
            {
                var vertices = new Vector3[side * side];
                var indices = new int[(side - 1) * (side - 1) * 6];
                int next = 0;
                for (int i = 0; i < vertices.Length; i++)
                {
                    int x = i % side;
                    int z = i / side;
                    vertices[i] = new Vector3(x, Mathf.Sin((x + part * 17) * 0.1f) * 0.15f, z);
                    if (x + 1 < side && z + 1 < side)
                    {
                        indices[next++] = i;
                        indices[next++] = i + side;
                        indices[next++] = i + 1;
                        indices[next++] = i + 1;
                        indices[next++] = i + side;
                        indices[next++] = i + side + 1;
                    }
                }

                var mesh = new Mesh { vertices = vertices, triangles = indices };
                assets.Add(mesh);
                GameObject ground = Track(new GameObject("terrain mesh"));
                ground.transform.position = new Vector3(part * 140, 0, 0);
                ground.AddComponent<MeshCollider>().sharedMesh = mesh;
                for (int i = 1; i <= 6; i++)
                {
                    DynamicBox(ground.transform.position + new Vector3(i * 5, 0.55f, i * 5)).sleepThreshold = 0f;
                }
            }

            var terrain = new TerrainData { heightmapResolution = 257, size = new Vector3(128, 16, 128) };
            terrain.SetHeights(0, 0, new float[257, 257]);
            assets.Add(terrain);
            GameObject field = Track(new GameObject("height field"));
            field.transform.position = new Vector3(-140, 0, 0);
            field.AddComponent<TerrainCollider>().terrainData = terrain;
            DynamicBox(field.transform.position + new Vector3(20, 0.51f, 20)).sleepThreshold = 0f;
            return terrain;
        }

        private static uint FindSerial(Vector3 position)
        {
            var world = StepEntryPoints.Current->Store;
            for (uint i = 0; i < world->bodyCount; i++)
            {
                if ((Vector3)world->bodies[i].pose->Position == position)
                {
                    return world->bodies[i].serial;
                }
            }

            throw new InvalidOperationException("The triggering body was not in the store.");
        }

        private string WaitForRecording(string path)
        {
            var timeout = System.Diagnostics.Stopwatch.StartNew();
            while (timeout.Elapsed < TimeSpan.FromSeconds(30))
            {
                if (path != null && File.Exists(path))
                {
                    return path;
                }

                if (path == null && Directory.Exists(directory))
                {
                    string[] files = Directory.GetFiles(directory, "*.rfr");
                    if (files.Length != 0)
                    {
                        return files[0];
                    }
                }

                Thread.Sleep(10);
            }

            Assert.Fail($"Timed out waiting for the flight recording. {FlightRecorderSwitches.GetStatus()}");
            return null;
        }
    }
}
