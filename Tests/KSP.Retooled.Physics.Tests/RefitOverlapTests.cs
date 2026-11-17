using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Physics.Collision.World.Trees;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Run queries and commands immediately after steps to verify pending refits complete without deadlocks or
    /// stale results.
    /// </summary>
    public unsafe class RefitOverlapTests : PhysicsScene
    {
        private const int Count = 1500;

        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
        }

        /// <summary>
        /// Boxes in a grid that fall fast, so that a box's hit at the pose of the start of the step is 0.2 m off the
        /// hit at its solved pose.
        /// </summary>
        private List<Rigidbody> FallingBoxes(int count)
        {
            UnityEngine.Physics.autoSyncTransforms = false;
            var bodies = new List<Rigidbody>(count);
            int side = (int)Math.Ceiling(Math.Sqrt(count));
            for (int i = 0; i < count; i++)
            {
                Rigidbody body = WeightlessBox(new Vector3(2.0f * (i % side), 0.0f, 2.0f * (i / side)));
                body.sleepThreshold = 0.0f;
                body.linearVelocity = new Vector3(0.0f, -10.0f, 0.0f);
                bodies.Add(body);
            }

            Simulate(2);
            return bodies;
        }

        /// <summary>One ray straight down onto each box, from 5 m above where the box is now.</summary>
        private static NativeArray<RaycastCommand> RaysOnto(List<Rigidbody> bodies)
        {
            var rays = new NativeArray<RaycastCommand>(bodies.Count, Allocator.TempJob);
            for (int i = 0; i < bodies.Count; i++)
            {
                rays[i] = new RaycastCommand(bodies[i].position + new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, QueryParameters.Default, 10.0f);
            }

            return rays;
        }

        private static bool RefitPending => StepKernel.Current->Store->treeReady == 0;

        /// <summary>Checks that each ray hit its own box's top face, at the box's solved pose.</summary>
        private static void AssertHitsOnTop(List<Rigidbody> bodies, NativeArray<RaycastHit> hits, string when)
        {
            for (int i = 0; i < bodies.Count; i++)
            {
                Assert.That(hits[i].collider, Is.Not.Null, $"{when}: ray {i} hit nothing");
                Assert.That(hits[i].collider.attachedRigidbody, Is.SameAs(bodies[i]), $"{when}: ray {i}");
                Assert.That(hits[i].point.y, Is.EqualTo(bodies[i].position.y + 0.5f).Within(1e-3f), $"{when}: ray {i}");
            }
        }

        [TearDown]
        public void LetTheJobsWorkAgain()
        {
            SharedDynamicTreeRefit.JobsLeaveTheWork = false;
            SharedDynamicTreeRefit.ThrowForTestsEnabled = false;
            LogAssert.ignoreFailingMessages = false;
        }

        /// <summary>
        /// Verify that a step throwing after refit scheduling waits for solved motions before queries can help the
        /// refit.
        /// Requires Burst disabled for the injected exception.
        /// </summary>
        [Test]
        [Timeout(120000)]
        public void AStepThatThrowsAfterItSchedulesTheRefitWaitsForTheSolvedMotions()
        {
            SharedDynamicTreeRefit.JobsLeaveTheWork = true;
            List<Rigidbody> bodies = FallingBoxes(Count);
            int throws = StepState.ThrowsAfterRefitScheduled;
            LogAssert.ignoreFailingMessages = true;
            try
            {
                StepState.ThrowAfterRefitScheduledForTests = true;
                try
                {
                    Simulate(1);
                }
                catch (InvalidOperationException)
                {
                    // The system that runs the step may report the exception in the log or let it through.
                }
            }
            finally
            {
                StepState.ThrowAfterRefitScheduledForTests = false;
            }

            Assume.That(StepState.ThrowsAfterRefitScheduled, Is.GreaterThan(throws), "the step ran in Mono and threw");
            Assert.That(StepKernel.Current->RefitPending, Is.True, "the refit is pending");
            Assert.That(StepKernel.Current->RefitInputsComplete, Is.True, "the solved motions are written before a query can help");
            using (NativeArray<RaycastCommand> rays = RaysOnto(bodies))
            using (var hits = new NativeArray<RaycastHit>(Count, Allocator.TempJob))
            {
                RaycastCommand.ScheduleBatch(rays, hits, 16).Complete();
            }

            // The next steps run as before.
            SharedDynamicTreeRefit.JobsLeaveTheWork = false;
            Simulate(2);
            LogAssert.ignoreFailingMessages = false;
            using NativeArray<RaycastCommand> after = RaysOnto(bodies);
            using var afterHits = new NativeArray<RaycastHit>(Count, Allocator.TempJob);
            BackendHost.SyncQueryWorlds();
            RaycastCommand.ScheduleBatch(after, afterHits, 16).Complete();
            AssertHitsOnTop(bodies, afterHits, "after the step that threw");
        }

        /// <summary>
        /// Verify that detach completes pending refits before removing the query helper, avoiding query-lock
        /// deadlocks.
        /// </summary>
        [Test]
        [Timeout(120000)]
        public void ADetachWhileWorkerQueriesWaitForTheRefitFinishesIt()
        {
            int defaultWorkers = JobsUtility.JobWorkerCount;
            try
            {
                JobsUtility.JobWorkerCount = 1;
                SharedDynamicTreeRefit.JobsLeaveTheWork = true;
                List<Rigidbody> bodies = FallingBoxes(Count);
                for (int round = 0; round < 5; round++)
                {
                    Simulate(1);
                    Assert.That(RefitPending, Is.True, $"round {round}: the refit is pending");
                    using NativeArray<RaycastCommand> rays = RaysOnto(bodies);
                    using var hits = new NativeArray<RaycastHit>(Count, Allocator.TempJob);

                    // One command per job, so that the worker keeps starting queries while the main thread detaches.
                    JobHandle queries = RaycastCommand.ScheduleBatch(rays, hits, 1);
                    JobHandle.ScheduleBatchedJobs();
                    BackendHost.Detach();
                    queries.Complete();
                    BackendHost.Attach();
                }

                SharedDynamicTreeRefit.JobsLeaveTheWork = false;
                Simulate(1);
                using NativeArray<RaycastCommand> after = RaysOnto(bodies);
                using var afterHits = new NativeArray<RaycastHit>(Count, Allocator.TempJob);
                BackendHost.SyncQueryWorlds();
                RaycastCommand.ScheduleBatch(after, afterHits, 16).Complete();
                AssertHitsOnTop(bodies, afterHits, "after the detaches");
            }
            finally
            {
                JobsUtility.JobWorkerCount = defaultWorkers;
            }
        }

        /// <summary>
        /// Verify that a refit exception releases waiters and the next main-thread query rebuilds the world.
        /// Requires Burst disabled for the injected exception.
        /// </summary>
        [TestCase(false)]
        [TestCase(true)]
        [Timeout(120000)]
        public void AThrowInTheRefitEndsTheWaitAndTheNextQueryBuildsTheWorld(bool queriesDoTheWork)
        {
            SharedDynamicTreeRefit.JobsLeaveTheWork = queriesDoTheWork;
            List<Rigidbody> bodies = FallingBoxes(Count);
            UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
            int throws = SharedDynamicTreeRefit.ThrowsForTests;

            // The thread that throws reports the exception in the log.
            LogAssert.ignoreFailingMessages = true;
            SharedDynamicTreeRefit.ThrowForTestsEnabled = true;
            scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation | SimulationStage.RunSimulation, SimulationOption.All);
            using (NativeArray<RaycastCommand> rays = RaysOnto(bodies))
            using (var hits = new NativeArray<RaycastHit>(Count, Allocator.TempJob))
            {
                // The queries on workers end, though one piece of the refit was never done. Their hits can be at the
                // poses before the step, so the test doesn't check them.
                RaycastCommand.ScheduleBatch(rays, hits, 16).Complete();
            }

            scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
            SharedDynamicTreeRefit.ThrowForTestsEnabled = false;
            Assume.That(SharedDynamicTreeRefit.ThrowsForTests, Is.GreaterThan(throws), "the refit ran in Mono and threw");

            // A query on the main thread finishes the failed refit and builds the world again.
            for (int i = 0; i < Count; i += 7)
            {
                Vector3 above = bodies[i].position + new Vector3(0.0f, 5.0f, 0.0f);
                Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, out RaycastHit hit, 10.0f), Is.True, $"ray {i}");
                Assert.That(hit.rigidbody, Is.SameAs(bodies[i]), $"ray {i}");
                Assert.That(hit.point.y, Is.EqualTo(bodies[i].position.y + 0.5f).Within(1e-3f), $"ray {i}");
            }

            // The next steps refit the tree again.
            LogAssert.ignoreFailingMessages = false;
            for (int step = 0; step < 3; step++)
            {
                Simulate(1);
                using NativeArray<RaycastCommand> rays = RaysOnto(bodies);
                using var hits = new NativeArray<RaycastHit>(Count, Allocator.TempJob);
                RaycastCommand.ScheduleBatch(rays, hits, 16).Complete();
                AssertHitsOnTop(bodies, hits, $"step {step} after the failed refit");
            }
        }

        /// <param name="queriesDoTheWork">
        /// Whether the refit's jobs leave all of the work to the threads that need the tree, so that the refit is
        /// surely pending when the queries start.
        /// </param>
        [TestCase(false)]
        [TestCase(true)]
        public void AWorkerQueryRightAfterAStepFindsTheRefittedHits(bool queriesDoTheWork)
        {
            SharedDynamicTreeRefit.JobsLeaveTheWork = queriesDoTheWork;
            List<Rigidbody> bodies = FallingBoxes(Count);
            UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
            int pending = 0;
            for (int step = 0; step < 10; step++)
            {
                // Without Unity's read-back, the queries start as soon as the backend's step returns.
                scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation | SimulationStage.RunSimulation, SimulationOption.All);
                pending += RefitPending ? 1 : 0;
                using NativeArray<RaycastCommand> rays = RaysOnto(bodies);
                using var hits = new NativeArray<RaycastHit>(Count, Allocator.TempJob);
                RaycastCommand.ScheduleBatch(rays, hits, 16).Complete();
                AssertHitsOnTop(bodies, hits, $"step {step}, right after the step");

                // The same queries once the refit has finished.
                scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
                StepKernel.Current->FinishPendingRefit();
                using var after = new NativeArray<RaycastHit>(Count, Allocator.TempJob);
                RaycastCommand.ScheduleBatch(rays, after, 16).Complete();
                for (int i = 0; i < Count; i++)
                {
                    Assert.That(after[i].point, Is.EqualTo(hits[i].point), $"step {step}, ray {i}");
                    Assert.That(after[i].collider, Is.SameAs(hits[i].collider), $"step {step}, ray {i}");
                }
            }

            Debug.Log($"[RefitOverlapTests] queries do the work {queriesDoTheWork}: the refit was pending after {pending} of 10 steps");
            if (queriesDoTheWork)
            {
                Assert.That(pending, Is.EqualTo(10), "the refit was pending after every step");
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void AMainThreadQueryRightAfterAStepFindsTheRefittedHits(bool queriesDoTheWork)
        {
            SharedDynamicTreeRefit.JobsLeaveTheWork = queriesDoTheWork;
            List<Rigidbody> bodies = FallingBoxes(Count);
            UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
            for (int step = 0; step < 5; step++)
            {
                scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation | SimulationStage.RunSimulation, SimulationOption.All);
                for (int i = 0; i < Count; i += 7)
                {
                    Vector3 above = bodies[i].position + new Vector3(0.0f, 5.0f, 0.0f);
                    Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, out RaycastHit hit, 10.0f), Is.True, $"step {step}, ray {i}");
                    Assert.That(hit.rigidbody, Is.SameAs(bodies[i]), $"step {step}, ray {i}");
                    Assert.That(hit.point.y, Is.EqualTo(bodies[i].position.y + 0.5f).Within(1e-3f), $"step {step}, ray {i}");
                }

                scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
            }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void DestroyingABodyRightAfterAStepLeavesTheOthersToTheRefit(bool queriesDoTheWork)
        {
            SharedDynamicTreeRefit.JobsLeaveTheWork = queriesDoTheWork;
            List<Rigidbody> bodies = FallingBoxes(Count);
            UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
            for (int step = 0; step < 5; step++)
            {
                scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation | SimulationStage.RunSimulation, SimulationOption.All);
                Rigidbody gone = bodies[step * 31];
                bodies.RemoveAt(step * 31);
                UnityEngine.Object.DestroyImmediate(gone.gameObject);
                scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
            }

            Simulate(1);
            using NativeArray<RaycastCommand> rays = RaysOnto(bodies);
            using var hits = new NativeArray<RaycastHit>(bodies.Count, Allocator.TempJob);
            BackendHost.SyncQueryWorlds();
            RaycastCommand.ScheduleBatch(rays, hits, 16).Complete();
            AssertHitsOnTop(bodies, hits, "after the destroys");
        }

        [TestCase(false)]
        [TestCase(true)]
        public void DestroyingAWorldRightAfterItsStep(bool queriesDoTheWork)
        {
            SharedDynamicTreeRefit.JobsLeaveTheWork = queriesDoTheWork;
            for (int round = 0; round < 5; round++)
            {
                Scene preview = EditorSceneManager.NewPreviewScene();
                try
                {
                    UnityEngine.PhysicsScene physics = preview.GetPhysicsScene();
                    Assume.That(physics.IsValid(), "the preview scene has its own physics scene");
                    for (int i = 0; i < 500; i++)
                    {
                        var box = new GameObject("box");
                        box.transform.position = new Vector3(2.0f * (i % 25), 0.0f, 2.0f * (i / 25));
                        box.AddComponent<BoxCollider>();
                        Rigidbody body = box.AddComponent<Rigidbody>();
                        body.useGravity = false;
                        body.linearVelocity = new Vector3(0.0f, -10.0f, 0.0f);
                        SceneManager.MoveGameObjectToScene(box, preview);
                    }

                    physics.Simulate(Step);
                    physics.Simulate(Step);
                }
                finally
                {
                    EditorSceneManager.ClosePreviewScene(preview);
                }
            }

            // The default world still steps and answers queries.
            Rigidbody kept = WeightlessBox(Vector3.zero);
            Simulate(1);
            Assert.That(UnityEngine.Physics.Raycast(kept.position + Vector3.up * 5.0f, Vector3.down, out RaycastHit hit, 10.0f), Is.True);
            Assert.That(hit.rigidbody, Is.SameAs(kept));
        }

        /// <summary>
        /// Run query jobs after each step, including with no workers, to verify they can finish pending refits
        /// without completing job handles.
        /// </summary>
        [TestCase(-1, false)]
        [TestCase(-1, true)]
        [TestCase(0, false)]
        [TestCase(0, true)]
        [Timeout(120000)]
        public void QueriesOnEveryThreadWhileTheRefitIsPendingDoNotDeadlock(int workers, bool queriesDoTheWork)
        {
            SharedDynamicTreeRefit.JobsLeaveTheWork = queriesDoTheWork;
            int defaultWorkers = JobsUtility.JobWorkerCount;
            try
            {
                if (workers >= 0)
                {
                    JobsUtility.JobWorkerCount = workers;
                }

                List<Rigidbody> bodies = FallingBoxes(Count);
                UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
                int pending = 0;
                for (int step = 0; step < 40; step++)
                {
                    scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation | SimulationStage.RunSimulation, SimulationOption.All);
                    pending += RefitPending ? 1 : 0;
                    using NativeArray<RaycastCommand> rays = RaysOnto(bodies);
                    using var hits = new NativeArray<RaycastHit>(Count, Allocator.TempJob);

                    // One command per job, so that there are far more jobs than threads.
                    JobHandle queries = RaycastCommand.ScheduleBatch(rays, hits, 1);
                    JobHandle.ScheduleBatchedJobs();
                    queries.Complete();
                    AssertHitsOnTop(bodies, hits, $"step {step}");
                    scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
                }

                Debug.Log($"[RefitOverlapTests] {JobsUtility.JobWorkerCount} workers, queries do the work {queriesDoTheWork}: the refit was pending after {pending} of 40 steps");
                if (queriesDoTheWork)
                {
                    Assert.That(pending, Is.EqualTo(40), "the queries ran while the refit was pending");
                }
            }
            finally
            {
                JobsUtility.JobWorkerCount = defaultWorkers;
            }
        }
    }
}
