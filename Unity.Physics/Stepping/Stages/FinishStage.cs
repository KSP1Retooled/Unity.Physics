using Unity.Burst;
using Unity.Physics.Configuration;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Dynamics.Articulations;
using Unity.Physics.Dynamics.Joint.Grouping;
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
        private static readonly ProfilerMarker StepPublishQueryWorldMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Step.PublishQueryWorld");
        private static readonly ProfilerMarker StepReadJointForcesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Step.ReadJointForces");

        private const int ParallelBatchSize = 64;

        /// <summary>
        /// Read impulse and vessel loads, then break overloaded joints.
        /// Wait for body pose write-back before converting impulse events to world space.
        /// </summary>
        private static JobHandle ReadJointForces(WorldData* store, StoreGlobals* globals,
            ref PhysicsWorld world, ref BodyPartition partition, ref JointBuildState jointState, ref VesselState vessels,
            NativeStream events, float deltaTime, JobHandle motionsSolved, JobHandle solved, JobHandle writtenBack, ref JobHandle pending)
        {
            const int groupsPerBatch = 16;

            SolveStage.ResizeJointLoads(store, ref jointState);
            int jointCount = jointState.jointLoads.Length;
            JobHandle loaded = Track(ref pending, new ClearJointLoadsJob
            {
                JointLoads = jointState.jointLoads.AsArray(),
                GroupLoads = jointState.groupLoads,
                Groups = jointState.jointGroups,
            }.Schedule(motionsSolved));

            bool listed = vessels.vesselSet.IsCreated;
            if (events.IsCreated)
            {
                JobHandle rotated = Track(ref pending, new JointLoads.RotateJointImpulsesJob
                {
                    Events = events.AsReader(),
                    World = store,
                    Groups = jointState.jointGroups.AsDeferredJobArray(),
                    DeltaTime = deltaTime,
                }.ScheduleUnsafe(events, 1, JobHandle.CombineDependencies(solved, writtenBack)));

                loaded = Track(ref pending, new JointLoads.SumJointImpulsesJob
                {
                    Events = events.AsReader(),
                    JointLoads = jointState.jointLoads.AsArray(),
                    GroupLoads = jointState.groupLoads.AsDeferredJobArray(),
                }.Schedule(JobHandle.CombineDependencies(loaded, rotated)));
            }

            loaded = JobHandle.CombineDependencies(loaded, writtenBack);
            JobHandle vesselsSettled = Track(ref pending, SolveStage.ScheduleVesselLoads(store, ref world, ref partition,
                ref jointState, ref vessels, deltaTime, loaded));

            var share = new JointLoads.ShareGroupLoadsJob
            {
                World = store,
                Groups = jointState.jointGroups.AsDeferredJobArray(),
                Members = jointState.jointGroupMembers.AsDeferredJobArray(),
                GroupLoads = jointState.groupLoads.AsDeferredJobArray(),
                JointLoads = jointState.jointLoads.AsArray(),
                ListedGroups = jointState.otherGroups.AsDeferredJobArray(),
                Listed = listed,
            };
            JobHandle shared = Track(ref pending, listed
                ? share.Schedule(jointState.otherGroups, groupsPerBatch, loaded)
                : share.Schedule(jointState.jointGroups, groupsPerBatch, loaded));

            var settle = new JointLoads.SettleJointLoadsJob
            {
                World = store,
                States = jointState.jointStates->Ptr,
                Loads = jointState.jointLoads.AsArray(),
                Overloaded = jointState.jointsOverloaded.AsArray(),
                Joints = jointState.otherJoints.AsDeferredJobArray(),
                Listed = listed,
                DeltaTime = deltaTime,
            };
            JobHandle settled = Track(ref pending, listed
                ? settle.Schedule(jointState.otherJoints, ParallelBatchSize, shared)
                : settle.Schedule(jointCount, ParallelBatchSize, shared));

            settled = JobHandle.CombineDependencies(settled, vesselsSettled);
            return Track(ref pending, new JointBreaking.BreakJointsJob
            {
                World = store,
                Globals = globals,
                States = jointState.jointStates->Ptr,
                Overloaded = jointState.jointsOverloaded.AsArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(settled));
        }

        /// <summary>
        /// Convert test-supplied vessel loads to joint loads and process breaking as at step completion.
        /// </summary>
        internal static void ReadVesselLoads(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref BodyPartition partition, ref JointBuildState jointState, ref VesselState vessels, float deltaTime)
        {
            JobHandle pending = default;
            try
            {
                ReadJointForces(store, globals, ref world, ref partition, ref jointState, ref vessels, default, deltaTime,
                    default, default, default, ref pending);
            }
            finally
            {
                pending.Complete();
            }
        }

        internal static JobHandle ScheduleBeforeContacts(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref BodyPartition partition, ref JointBuildState jointState, ref VesselState vessels, ref ContactState contacts,
            ref SleepState sleepState, ref TreeState trees, ref QuerySnapshotState queries, JobHandle warmStartsIndexed,
            NativeReference<StoreStepperCounters> counters, byte* restoredMotion, ref Simulation simulation, float deltaTime,
            ref JobHandle pending, out JobHandle snapshot, out JobHandle sleep)
        {
            JobHandle solved = Track(ref pending,
                JobHandle.CombineDependencies(simulation.FinalJobHandle, warmStartsIndexed, vessels.vesselsFilled));
            JobHandle motionsSolved = JobHandle.CombineDependencies(simulation.MotionsSolvedHandle, warmStartsIndexed, vessels.vesselsFilled);

            // The event exporter reads solved motions. Wait for it before write-back can repair them.
            JobHandle canRepair = JobHandle.CombineDependencies(simulation.FinalSimulationJobHandle, motionsSolved);
            JobHandle writtenBack = Track(ref pending, new StoreBodies.WriteBackJob
            {
                Store = store->bodies,
                Counters = (StoreStepperCounters*)counters.GetUnsafePtr(),
                RestoredMotion = restoredMotion,
                DynamicIndices = partition.dynamicIndices.AsArray(),
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
            }.Schedule(partition.dynamicIndices.Length, ParallelBatchSize, canRepair));

            if (store->articulations != null)
            {
                writtenBack = Track(ref pending, new ArticulationSolver.FinishJob
                {
                    World = store,
                    Indices = partition.builtIndices.AsArray(),
                    MotionData = world.MotionDatas,
                    MotionVelocity = world.MotionVelocities,
                    DeltaTime = deltaTime,
                    Contacts = contacts.manifolds,
                    Points = contacts.manifoldPoints,
                }.Schedule(writtenBack));
            }

            // Queries finish the refit when they need the tree. The step waits for its inputs.
            trees.SchedulePendingRefit(store, ref world, deltaTime, writtenBack);
            StoreStepper.ThrowAfterRefitScheduledIfRequested();

            snapshot = Track(ref pending, queries.ScheduleQuerySnapshot(store, globals, ref world, ref partition, writtenBack));

            // Write-back can clear a flag on kinematic bodies. Read those flags before removed shapes can wake bodies.
            JobHandle contactsJoined = Track(ref pending, new FallAsleep.SleepContactsJob
            {
                World = store,
                Records = contacts.manifolds,
                Parents = sleepState.sleepParents,
                Contacts = sleepState.sleepContacts,
                TouchesStill = sleepState.sleepTouchesStill,
            }.Schedule(writtenBack));

            JobHandle forces;
            using (StepReadJointForcesMarker.Auto())
            {
                forces = ReadJointForces(store, globals, ref world, ref partition, ref jointState, ref vessels,
                    simulation.ImpulseEvents.EventDataStream, deltaTime, motionsSolved, solved, writtenBack, ref pending);
            }

            JobHandle joinedAndBroken = JobHandle.CombineDependencies(contactsJoined, forces);
            JobHandle shapesForgotten = Track(ref pending, contacts.ScheduleForgetRemovedShapes(store, JobHandle.CombineDependencies(joinedAndBroken, snapshot)));
            JobHandle counted = Track(ref pending, new SleepCounters.UpdateWakeCountersJob
            {
                World = store,
                Contacts = sleepState.sleepContacts,
                TouchesStill = sleepState.sleepTouchesStill,
                RunOut = (int*)sleepState.sleepRunOut.GetUnsafePtr(),
                DeltaTime = deltaTime,
            }.Schedule((int)store->bodyCount, ParallelBatchSize, shapesForgotten));
            sleep = Track(ref pending, new FallAsleep.SleepJob
            {
                World = store,
                Parents = sleepState.sleepParents,
                Falling = sleepState.fallingAsleep,
                RunOut = (int*)sleepState.sleepRunOut.GetUnsafePtr(),
            }.Schedule(counted));

            // The report sorts the manifolds after the sleep and impulse jobs finish reading them.
            JobHandle impulsesRead = JobHandle.CombineDependencies(solved, contactsJoined);
            if (world.NumDynamicBodies != 0)
            {
                impulsesRead = Track(ref pending, new ImpulseMemory.ReadImpulsesJob
                {
                    Records = contacts.manifolds,
                    Events = simulation.CollisionEvents,
                    Built = world,
                }.Schedule(impulsesRead));
            }

            JobHandle remembered = Track(ref pending, new ImpulseMemory.RememberImpulsesJob
            {
                Records = contacts.manifolds,
                LastImpulses = contacts.lastImpulses,
            }.Schedule(impulsesRead));

            return JobHandle.CombineDependencies(remembered, shapesForgotten);
        }

        internal static JobHandle ScheduleSleep(WorldData* store, ref SleepState sleep, JobHandle dependency)
        {
            return new FallAsleep.FallAsleepJob { World = store, Falling = sleep.fallingAsleep }.Schedule(dependency);
        }

        internal static void PublishQueries(WorldData* store, QueryWorld* query, ref PhysicsWorld world,
            ref QuerySnapshotState queries, ref TreeState trees, ref BodyPartition partition)
        {
            trees.MarkBoundsCurrent();
            if (BroadphaseSwitches.BoundsCheckEnabled)
            {
                TreeState.LogBoundsCheck();
            }

            using (StepPublishQueryWorldMarker.Auto())
            {
                queries.FinishQuerySnapshot();
                queries.PublishAfterStep(store, query, ref world, ref partition, ref trees);
            }
        }

        // Keep every scheduled job reachable if a later Schedule call throws.
        private static JobHandle Track(ref JobHandle pending, JobHandle scheduled)
        {
            pending = JobHandle.CombineDependencies(pending, scheduled);
            return scheduled;
        }

        [BurstCompile]
        private struct ClearJointLoadsJob : IJob
        {
            public NativeArray<JointLoad> JointLoads;
            public NativeList<JointLoad> GroupLoads;

            [ReadOnly]
            public NativeList<JointGroup> Groups;

            public void Execute()
            {
                GroupLoads.ResizeUninitialized(Groups.Length);
                UnsafeUtility.MemClear(GroupLoads.GetUnsafePtr(), (long)GroupLoads.Length * sizeof(JointLoad));
                UnsafeUtility.MemClear(JointLoads.GetUnsafePtr(), (long)JointLoads.Length * sizeof(JointLoad));
            }
        }
    }
}
