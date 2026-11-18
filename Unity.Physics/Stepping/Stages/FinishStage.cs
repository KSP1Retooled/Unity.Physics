using Unity.Physics.Configuration;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Dynamics.Sleep;
using Unity.Physics.Stepping.State;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Store.World;
using Unity.Profiling;

namespace Unity.Physics.Stepping.Stages
{
    /// <summary>
    /// Schedules write-back, refit, and joint loads. Publishes queries and finishes sleep after contact reporting.
    /// </summary>
    internal static unsafe class FinishStage
    {
        private static readonly ProfilerMarker StepPublishQueryWorldMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.PublishQueryWorld");
        private static readonly ProfilerMarker StepReadImpulsesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadImpulses");
        private static readonly ProfilerMarker StepReadJointForcesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadJointForces");
        private static readonly ProfilerMarker StepSleepMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Sleep");
        private static readonly ProfilerMarker StepWaitForSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForSolve");
        private static readonly ProfilerMarker StepWriteBackMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WriteBack");

        private const int ParallelBatchSize = 64;

        /// <summary>
        /// Read impulse and vessel loads, then break overloaded joints.
        /// Wait for body pose write-back before converting impulse events to world space.
        /// </summary>
        private static JobHandle ReadJointForces(WorldData* store, StoreGlobals* globals,
            ref JointBuildState jointState, ref VesselState vessels, NativeStream events, float deltaTime,
            JobHandle vesselsSettled)
        {
            const int groupsPerBatch = 16;

            // ScheduleVesselLoads has already sized the joint loads.
            int jointCount = jointState.jointLoads.Length;
            jointState.groupLoads.ResizeUninitialized(jointState.jointGroups.Length);
            UnsafeUtility.MemClear(jointState.groupLoads.GetUnsafePtr(), (long)jointState.groupLoads.Length * sizeof(JointLoad));

            bool listed = vessels.vesselSet.IsCreated;
            int seededJoints = listed ? jointState.otherJoints.Length : jointCount;
            int sharedGroups = listed ? jointState.otherGroups.Length : jointState.jointGroups.Length;
            JobHandle loaded = new JointLoads.SeedJointLoadsJob
            {
                World = store,
                States = jointState.jointStates->Ptr,
                Loads = jointState.jointLoads.AsArray(),
                Joints = jointState.otherJoints.AsArray(),
                Listed = listed,
            }.Schedule(seededJoints, ParallelBatchSize);
            if (events.IsCreated)
            {
                JobHandle rotated = new JointLoads.RotateJointImpulsesJob
                {
                    Events = events.AsReader(),
                    World = store,
                    Groups = jointState.jointGroups.AsArray(),
                    DeltaTime = deltaTime,
                }.ScheduleUnsafe(events, 1, default);

                loaded = new JointLoads.SumJointImpulsesJob
                {
                    Events = events.AsReader(),
                    JointLoads = jointState.jointLoads.AsArray(),
                    GroupLoads = jointState.groupLoads.AsArray(),
                }.Schedule(JobHandle.CombineDependencies(loaded, rotated));
            }

            JobHandle shared = new JointLoads.ShareGroupLoadsJob
            {
                World = store,
                Groups = jointState.jointGroups.AsArray(),
                Members = jointState.jointGroupMembers.AsArray(),
                GroupLoads = jointState.groupLoads.AsArray(),
                JointLoads = jointState.jointLoads.AsArray(),
                ListedGroups = jointState.otherGroups.AsArray(),
                Listed = listed,
            }.Schedule(sharedGroups, groupsPerBatch, loaded);

            JobHandle settled = new JointLoads.SettleJointLoadsJob
            {
                World = store,
                States = jointState.jointStates->Ptr,
                Loads = jointState.jointLoads.AsArray(),
                Overloaded = jointState.jointsOverloaded.AsArray(),
                Joints = jointState.otherJoints.AsArray(),
                Listed = listed,
                DeltaTime = deltaTime,
            }.Schedule(seededJoints, ParallelBatchSize, shared);

            settled = JobHandle.CombineDependencies(settled, vesselsSettled);
            return new JointBreaking.BreakJointsJob
            {
                World = store,
                Globals = globals,
                States = jointState.jointStates->Ptr,
                Overloaded = jointState.jointsOverloaded.AsArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(settled);
        }

        /// <summary>
        /// Convert test-supplied vessel loads to joint loads and process breaking as at step completion.
        /// </summary>
        internal static void ReadVesselLoads(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref BodyPartition partition, ref JointBuildState jointState, ref VesselState vessels, float deltaTime)
        {
            JobHandle vesselsSettled = SolveStage.ScheduleVesselLoads(store, ref world, ref partition, ref jointState, ref vessels, deltaTime, default);
            ReadJointForces(store, globals, ref jointState, ref vessels, default, deltaTime, vesselsSettled).Complete();
        }

        internal static void SchedulePostSolve(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref BodyPartition partition, ref JointBuildState jointState, ref VesselState vessels, ref TreeState trees,
            ref QuerySnapshotState queries, JobHandle warmStartsIndexed, ref Simulation simulation, float deltaTime,
            out JobHandle writtenBack, out JobHandle vesselsSettled, out JobHandle snapshot)
        {
            // Schedule post-solve work now. Include manifold and warm-start dependencies even when nothing is
            // solved.
            JobHandle solved = JobHandle.CombineDependencies(simulation.FinalJobHandle, warmStartsIndexed, vessels.vesselsFilled);

            // The jobs below read only the solved motions, so they don't wait for the jobs that free the solve's
            // buffers and export the events of its direct islands. Those jobs only read the motions too.
            JobHandle motionsSolved = JobHandle.CombineDependencies(simulation.MotionsSolvedHandle, warmStartsIndexed, vessels.vesselsFilled);

            // Schedule write-back first because the main thread waits for it first.
            writtenBack = new StoreBodies.WriteBackJob
            {
                Store = store->bodies,
                DynamicIndices = partition.dynamicIndices.AsArray(),
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
            }.Schedule(partition.dynamicIndices.Length, ParallelBatchSize, motionsSolved);

            // Vessel loads read solved motions, so they can run alongside pose write-back.
            vesselsSettled = SolveStage.ScheduleVesselLoads(store, ref world, ref partition, ref jointState, ref vessels, deltaTime, motionsSolved);

            // Refit the dynamic tree to solved poses alongside write-back. The step returns without waiting for
            // the refit.
            // Clear treeReady until completion. Later world access must finish the refit, and queries can help it.
            // Snapshot copying reads only Static flags, which these concurrent jobs do not change.
            trees.SchedulePendingRefit(store, ref world, deltaTime, motionsSolved);

            // Queries that help the refit read the solved motions. If the step throws, wait for those motions before returning.
            bool waitedForSolve = false;
            try
            {
                StoreStepper.ThrowAfterRefitScheduledIfRequested();

                snapshot = queries.ScheduleQuerySnapshot(store, globals, ref world, ref partition, motionsSolved);
                JobHandle.ScheduleBatchedJobs();

                using (StepWaitForSolveMarker.Auto())
                {
                    solved.Complete();
                }

                waitedForSolve = true;
            }
            finally
            {
                if (!waitedForSolve)
                {
                    motionsSolved.Complete();
                }
            }

            // The completed build wrote each body's current collider to its bounds cache entry. The refit writes it again.
            trees.MarkBoundsCurrent();

            if (BroadphaseSwitches.BoundsCheckEnabled)
            {
                TreeState.LogBoundsCheck();
            }
        }

        internal static void CompleteBeforeContacts(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref JointBuildState jointState, ref VesselState vessels, ref ContactState contactState, ref SleepState sleepState,
            ref Simulation simulation, float deltaTime, JobHandle writtenBack, JobHandle vesselsSettled,
            out JobHandle contactsJoined, out JobHandle sleep)
        {
            // FallAsleep.SleepContactsJob must read the manifolds before the contact report sorts them, so schedule it now.
            contactsJoined = new FallAsleep.SleepContactsJob
            {
                World = store,
                Records = contactState.manifolds,
                Parents = sleepState.sleepParents,
                Contacts = sleepState.sleepContacts,
                TouchesStill = sleepState.sleepTouchesStill,
            }.Schedule();
            JobHandle.ScheduleBatchedJobs();

            using (StepWriteBackMarker.Auto())
            {
                writtenBack.Complete();
            }

            JobHandle forces;
            using (StepReadJointForcesMarker.Auto())
            {
                forces = ReadJointForces(store, globals, ref jointState, ref vessels, simulation.ImpulseEvents.EventDataStream, deltaTime, vesselsSettled);
            }

            // Wake bodies affected by shape removal after joint load reads but before sleep and contact
            // reporting.
            // Sleep grouping must also wait for joint breaking.
            contactState.ForgetRemovedShapes(store, ref forces);

            JobHandle counted = new SleepCounters.UpdateWakeCountersJob
            {
                World = store,
                Contacts = sleepState.sleepContacts,
                TouchesStill = sleepState.sleepTouchesStill,
                RunOut = (int*)sleepState.sleepRunOut.GetUnsafePtr(),
                DeltaTime = deltaTime,
            }.Schedule((int)store->bodyCount, ParallelBatchSize, contactsJoined);
            sleep = new FallAsleep.SleepJob
            {
                World = store,
                Parents = sleepState.sleepParents,
                Falling = sleepState.fallingAsleep,
                RunOut = (int*)sleepState.sleepRunOut.GetUnsafePtr(),
            }.Schedule(JobHandle.CombineDependencies(counted, forces));
            JobHandle.ScheduleBatchedJobs();

            // ImpulseMemory.ReadImpulsesJob reads only the motions, which the rebuild of the tree doesn't change.
            using (StepReadImpulsesMarker.Auto())
            {
                if (world.NumDynamicBodies != 0)
                {
                    new ImpulseMemory.ReadImpulsesJob { Records = contactState.manifolds, Events = simulation.CollisionEvents, Built = world }.Run();
                }

                new ImpulseMemory.RememberImpulsesJob { Records = contactState.manifolds, LastImpulses = contactState.lastImpulses }.Run();
            }
        }

        /// <summary>Publish queries and finish sleep after the caller has run the contact report.</summary>
        internal static void FinishAfterContacts(WorldData* store, QueryWorld* query, ref PhysicsWorld world,
            ref QuerySnapshotState queries, ref TreeState trees, ref BodyPartition partition,
            ref SleepState sleepState, JobHandle snapshot, JobHandle sleep)
        {
            // Publish as current unless write-back changed poses. Queries finish the pending refit before
            // reading.
            using (StepPublishQueryWorldMarker.Auto())
            {
                snapshot.Complete();
                queries.FinishQuerySnapshot();
                queries.PublishAfterStep(store, query, ref world, ref partition, ref trees);
            }

            // FallAsleep.FallAsleepJob sets the Sleeping flag. The snapshot reads only the Static flag, so it doesn't need to finish
            // first, although it has.
            using (StepSleepMarker.Auto())
            {
                sleep.Complete();
                new FallAsleep.FallAsleepJob { World = store, Falling = sleepState.fallingAsleep }.Run();
            }
        }
    }
}
