using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Base.Threading;
using Unity.Physics.Collision.World.Trees;
using Unity.Physics.Dynamics.Simulation.Islands;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Configuration;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Dynamics.Vessels.Detection;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Dynamics.Sleep;
using Unity.Physics.Dynamics.Vehicles;
using Unity.Physics.Dynamics.Bodies;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;
using Unity.Profiling;
using Unity.Physics.Stepping.Stages;
using Unity.Physics.Stepping.State;

namespace Unity.Physics.Stepping
{
    /// <summary>
    /// Unmanaged simulation state for one store world, used by the Burst step entry points.
    /// </summary>
    internal unsafe partial struct StoreStepper
    {
        private static readonly ProfilerMarker StepPublishQueryWorldMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.PublishQueryWorld");
        private static readonly ProfilerMarker StepReadImpulsesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadImpulses");
        private static readonly ProfilerMarker StepReadJointForcesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadJointForces");
        private static readonly ProfilerMarker StepScheduleMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Schedule");
        private static readonly ProfilerMarker StepSleepMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Sleep");
        private static readonly ProfilerMarker StepWaitForSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForSolve");
        private static readonly ProfilerMarker StepWriteBackMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WriteBack");

        private const int ParallelBatchSize = 64;

        private readonly WorldData* store;
        private readonly StoreGlobals* globals;
        private readonly QueryWorld* query;

        private PhysicsWorld world;
        internal StepSolverCaches SolverCaches;
        internal ColliderState Colliders;
        internal QuerySnapshotState Queries;
        internal SleepState Sleep;
        internal ContactState Contacts;
        internal VesselState VesselData;
        internal JointBuildState Joints;
        internal TreeState Trees;
        internal StaticRows StaticBodies;
        internal BodyPartition Partition;

        // Prepared simulation, or null since the last FinishStep.
        // Avoid reading the singleton here, which would wait for solver cleanup before scheduling write-back.
        internal Simulation* preparedSimulation;

        // Pre-solve jobs that FinishStepSystem completes before FinishStep reads their lists.
        internal JobHandle preparedInputs;

        // The job that ApplyForces leaves running. BuildWorld makes the build wait for it.
        private JobHandle forcesApplied;
        private JobHandle warmStartsIndexed;
        internal StoreStepper(WorldData* store, StoreGlobals* globals, QueryWorld* query)
        {
            this = default;
            VesselData.InitializeTombstones();
            this.store = store;
            this.globals = globals;
            this.query = query;
            world = new PhysicsWorld(0, 0, 0);
            Trees.InitializeStaticTree();
            Partition.InitializeRows();
            StaticBodies.Initialize();
            Partition.Initialize();
            Joints.Initialize(store);
            SolverCaches.InitializeDirectSolver();
            VesselData.Initialize();
            SolverCaches.Initialize();
            Queries.Initialize();
            Contacts.Initialize();
            Sleep.Initialize();
            Contacts.InitializeImpulseMemory();
            Colliders.Initialize();
            Trees.Initialize(store);
            Contacts.InitializePairCache();
        }

        internal void Dispose()
        {
            // The refit writes the world that is freed below.
            FinishPendingRefit();
            store->queryWorld = null;
            store->refit = null;
            Trees.DisposeRefit();

            forcesApplied.Complete();
            Colliders.collidersBuilt.Complete();

            // A step that threw before FinishStep leaves the fill jobs unfinished, and they write the vessel lists.
            VesselData.vesselsFilled.Complete();
            Colliders.ReleaseWorldColliders(store);

            store->queryDirty = 1;
            world.Dispose();
            Trees.DisposeStaticTree();
            Partition.DisposeRows();
            StaticBodies.Dispose();
            Partition.Dispose();
            Joints.Dispose(store);
            SolverCaches.DisposeDirectSolver();
            VesselData.Dispose();
            SolverCaches.Dispose();
            Queries.Dispose();
            Contacts.Dispose();
            Sleep.Dispose();
            Contacts.DisposeImpulseMemory();
            Colliders.Dispose();
            Trees.Dispose();
            Contacts.DisposePairCache();
        }

        /// <summary>The direct solver's cache, including the counts of solves that used the forest.</summary>
        internal readonly IslandSolverCache IslandSolverCache => SolverCaches.directSolverCache;

        /// <summary>The world built for the simulation systems to advance.</summary>
        internal readonly PhysicsWorld World => world;

        /// <summary>The index of each store body in the built world.</summary>
        internal readonly NativeList<int> BuiltIndices => Partition.builtIndices;

        internal readonly int StaticsChanged => StaticBodies.staticsChanged ? 1 : 0;

        /// <summary>The settings that the simulation systems use to build their input for a step.</summary>
        internal readonly PhysicsStep Settings
        {
            get
            {
                PhysicsStep settings = PhysicsStep.Default;
                settings.Gravity = float3.zero;
                settings.DirectIslandSubsteps = SolverConstants.DirectIslandSubsteps;
                settings.SolverIterationCount = SolverConstants.SolverIterations;
                settings.MaxDynamicDepenetrationVelocity = SolverConstants.MaxDepenetrationVelocity;
                settings.MaxStaticDepenetrationVelocity = SolverConstants.MaxDepenetrationVelocity;

                // FinishStep synchronizes the collision world while reading the results of the step.
                settings.SynchronizeCollisionWorld = 0;
                return settings;
            }
        }

        /// <summary>
        /// Complete outstanding work and invalidate queries before the caller prepares body pointers for jobs.
        /// </summary>
        internal void BeginForces()
        {
            ForcesStage.BeginForces(store, ref Trees, ref forcesApplied);
        }

        /// <summary>
        /// Wake bodies through joints, then schedule force integration.
        /// BuildWorld waits for the force jobs. The main thread may read flags while they run.
        /// </summary>
        internal void ApplyForces(float deltaTime)
        {
            ForcesStage.ApplyForces(store, ref Contacts, ref forcesApplied, deltaTime);
        }

        /// <summary>Builds the bodies, joints and broadphase for the simulation systems to advance.</summary>
        internal JobHandle BuildWorld(float deltaTime, JobHandle dependency)
        {
            return BuildStage.BuildWorld(store, globals, ref world, ref Partition, ref StaticBodies, ref Trees,
                ref Joints, ref VesselData, ref Colliders, ref forcesApplied, ref Queries.builtWhole, ref Contacts.stepCount, deltaTime, dependency);
        }

        /// <summary>The store updated by this simulation state. Used by tests.</summary>
        internal readonly WorldData* Store => store;

        /// <summary>The manifolds of the last step, with the impulses that they started with. Used by tests.</summary>
        internal readonly NativeList<ManifoldRecord> Manifolds => Contacts.manifolds;

        internal readonly StoreGlobals* Globals => globals;

        internal readonly NativeHashMap<ShapePairKey, ShapePairState> Touching => Contacts.touching;

        internal readonly NativeList<ManifoldPoint> ManifoldPoints => Contacts.manifoldPoints;

        internal readonly NativeList<ShapePairKey> RemovedTriggerPairs => Contacts.removedTriggerPairs;

        internal bool CanWriteContainers() => Partition.CanWriteContainers();

        internal uint NextContactStep() => ++Contacts.stepCount;

        /// <summary>The vessels compiled during the last build. Used by tests.</summary>
        internal readonly VesselBuild Vessels => VesselData.vesselBuild;

        /// <summary>The set of vessels supplied to the step during the last build. Used by tests.</summary>
        internal readonly VesselSet VesselSet => VesselData.vesselSet;

        /// <summary>
        /// Remove unsolved overlaps before island finding and supply this step's vessels.
        /// </summary>
        internal JobHandle SortIslands(SimulationSingleton singleton, ref SimulationStepInput input, JobHandle dependency)
        {
            return IslandStage.SortIslands(store, globals, ref world, ref Contacts, ref Colliders, ref Joints,
                ref VesselData, ref SolverCaches, singleton, ref input, dependency);
        }

        /// <summary>Reads the contacts produced by the narrow phase before the Jacobians are built from them.</summary>
        internal JobHandle ReadContacts(SimulationSingleton singleton, float deltaTime, JobHandle dependency)
        {
            return ContactStage.ReadContacts(store, globals, ref world, ref Partition, ref Contacts,
                ref warmStartsIndexed, singleton, deltaTime, dependency);
        }

        /// <summary>Warm starts the Jacobians.</summary>
        internal JobHandle PrepareSolve(SimulationSingleton singleton, ref SimulationStepInput input, JobHandle dependency)
        {
            using var span = StepScheduleMarker.Auto();

            SolverCaches.directSolverCache.Forest = new ForestSettings
            {
                Enabled = ForestSwitches.Enabled || VesselSwitches.Enabled,
                BoundaryRowCap = ForestSwitches.BoundaryRowCap,
                Locking = ForestSwitches.Locking,
                LockAngle = ForestSwitches.LockAngle,
                YieldAngle = ForestSwitches.YieldAngle,
                MinimumHoldSteps = ForestSwitches.MinimumHoldSteps,
                DeflectionSmoothingSteps = ForestSwitches.DeflectionSmoothingSteps,
                Caps = ForestSwitches.Caps,
            };
            input.IslandSolverCache = SolverCaches.directSolverCache;
            input.VesselSet = VesselData.vesselSet;
            preparedSimulation = singleton.AsSimulationPtr();
            preparedInputs = JobHandle.CombineDependencies(dependency, warmStartsIndexed, VesselData.vesselsFilled);
            return new WarmStart.WarmStartJob
            {
                WarmStarts = Contacts.warmStarts,
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
                MotionCount = Partition.dynamicIndices.Length,
            }.Schedule(singleton, ref world, JobHandle.CombineDependencies(dependency, warmStartsIndexed, VesselData.vesselsFilled));
        }

        /// <summary>
        /// Read impulse and vessel loads, then break overloaded joints.
        /// Wait for body pose write-back before converting impulse events to world space.
        /// </summary>
        private JobHandle ReadJointForces(NativeStream events, float deltaTime, JobHandle vesselsSettled)
        {
            const int groupsPerBatch = 16;

            // ScheduleVesselLoads has already sized the joint loads.
            int jointCount = Joints.jointLoads.Length;
            Joints.groupLoads.ResizeUninitialized(Joints.jointGroups.Length);
            UnsafeUtility.MemClear(Joints.groupLoads.GetUnsafePtr(), (long)Joints.groupLoads.Length * sizeof(JointLoad));

            bool listed = VesselData.vesselSet.IsCreated;
            int seededJoints = listed ? Joints.otherJoints.Length : jointCount;
            int sharedGroups = listed ? Joints.otherGroups.Length : Joints.jointGroups.Length;
            JobHandle loaded = new JointLoads.SeedJointLoadsJob
            {
                World = store,
                States = Joints.jointStates->Ptr,
                Loads = Joints.jointLoads.AsArray(),
                Joints = Joints.otherJoints.AsArray(),
                Listed = listed,
            }.Schedule(seededJoints, ParallelBatchSize);
            if (events.IsCreated)
            {
                JobHandle rotated = new JointLoads.RotateJointImpulsesJob
                {
                    Events = events.AsReader(),
                    World = store,
                    Groups = Joints.jointGroups.AsArray(),
                    DeltaTime = deltaTime,
                }.ScheduleUnsafe(events, 1, default);

                loaded = new JointLoads.SumJointImpulsesJob
                {
                    Events = events.AsReader(),
                    JointLoads = Joints.jointLoads.AsArray(),
                    GroupLoads = Joints.groupLoads.AsArray(),
                }.Schedule(JobHandle.CombineDependencies(loaded, rotated));
            }

            JobHandle shared = new JointLoads.ShareGroupLoadsJob
            {
                World = store,
                Groups = Joints.jointGroups.AsArray(),
                Members = Joints.jointGroupMembers.AsArray(),
                GroupLoads = Joints.groupLoads.AsArray(),
                JointLoads = Joints.jointLoads.AsArray(),
                ListedGroups = Joints.otherGroups.AsArray(),
                Listed = listed,
            }.Schedule(sharedGroups, groupsPerBatch, loaded);

            JobHandle settled = new JointLoads.SettleJointLoadsJob
            {
                World = store,
                States = Joints.jointStates->Ptr,
                Loads = Joints.jointLoads.AsArray(),
                Overloaded = Joints.jointsOverloaded.AsArray(),
                Joints = Joints.otherJoints.AsArray(),
                Listed = listed,
                DeltaTime = deltaTime,
            }.Schedule(seededJoints, ParallelBatchSize, shared);

            settled = JobHandle.CombineDependencies(settled, vesselsSettled);
            return new JointBreaking.BreakJointsJob
            {
                World = store,
                Globals = globals,
                States = Joints.jointStates->Ptr,
                Overloaded = Joints.jointsOverloaded.AsArray(),
                EngineObjectIdOffset = globals->engineObjectIdOffset,
            }.Schedule(settled);
        }

        /// <summary>
        /// Convert test-supplied vessel loads to joint loads and process breaking as at step completion.
        /// </summary>
        internal void ReadVesselLoads(float deltaTime)
        {
            JobHandle vesselsSettled = ScheduleVesselLoads(deltaTime, default);
            ReadJointForces(default, deltaTime, vesselsSettled).Complete();
        }

        /// <summary>
        /// Schedule vessel joint load extraction after solving. Read solved motion poses without waiting for
        /// write-back.
        /// </summary>
        private JobHandle ScheduleVesselLoads(float deltaTime, JobHandle dependency)
        {
            const int vesselLoadsPerBatch = 32;

            // The jobs write the joint loads through arrays that the safety system doesn't track, so the list must not
            // change size after this point in the step.
            Joints.jointLoads.ResizeUninitialized((int)store->jointCount);
            Joints.jointsOverloaded.ResizeUninitialized((int)store->jointCount);
            if (Joints.jointStates->Length != (int)store->jointCount)
            {
                throw new InvalidOperationException("the joints' load states and the store's joints differ in number");
            }

            if (!VesselData.vesselSet.IsCreated)
            {
                return default;
            }

            return new VesselLoads.VesselLoadsJob
            {
                World = store,
                States = Joints.jointStates->Ptr,
                Units = Joints.jointUnits.AsArray(),
                Groups = Joints.jointGroups.AsArray(),
                Members = Joints.jointGroupMembers.AsArray(),
                MotionDatas = world.MotionDatas,
                BuiltIndices = Partition.builtIndices.AsArray(),
                DeltaTime = deltaTime,
                PartUnits = VesselData.vesselBuild.PartUnits.AsArray(),
                ClosureUnits = VesselData.vesselBuild.ClosureUnits.AsArray(),
                PartIsBody0 = VesselData.vesselBuild.PartIsBody0.AsArray(),
                UnitVessels = VesselData.vesselBuild.UnitVessels.AsArray(),
                Active = VesselData.vesselBuild.Active.AsArray(),
                EdgeLoads = VesselData.vesselBuild.EdgeLoads.AsArray(),
                ClosureLoads = VesselData.vesselBuild.ClosureLoads.AsArray(),
                JointLoads = Joints.jointLoads.AsArray(),
                Overloaded = Joints.jointsOverloaded.AsArray(),
            }.Schedule(VesselData.vesselBuild.PartUnits.Length + VesselData.vesselBuild.ClosureUnits.Length, vesselLoadsPerBatch, dependency);
        }

        /// <summary>
        /// Writes back the solved world and prepares contact reports, query publication and sleep.
        /// </summary>
        internal void FinishBeforeContacts(Simulation simulation, float deltaTime,
            out JobHandle contactsJoined, out JobHandle snapshot, out JobHandle sleep)
        {
            // Schedule post-solve work now. Include manifold and warm-start dependencies even when nothing is
            // solved.
            JobHandle solved = JobHandle.CombineDependencies(simulation.FinalJobHandle, warmStartsIndexed, VesselData.vesselsFilled);

            // The jobs below read only the solved motions, so they don't wait for the jobs that free the solve's
            // buffers and export the events of its direct islands. Those jobs only read the motions too.
            JobHandle motionsSolved = JobHandle.CombineDependencies(simulation.MotionsSolvedHandle, warmStartsIndexed, VesselData.vesselsFilled);

            // Schedule write-back first because the main thread waits for it first.
            JobHandle writtenBack = new StoreBodies.WriteBackJob
            {
                Store = store->bodies,
                DynamicIndices = Partition.dynamicIndices.AsArray(),
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
            }.Schedule(Partition.dynamicIndices.Length, ParallelBatchSize, motionsSolved);

            // Vessel loads read solved motions, so they can run alongside pose write-back.
            JobHandle vesselsSettled = ScheduleVesselLoads(deltaTime, motionsSolved);

            // Refit the dynamic tree to solved poses alongside write-back. The step returns without waiting for
            // the refit.
            // Clear treeReady until completion; later world access must finish the refit, and queries can help
            // it.
            // Snapshot copying reads only Static flags, which these concurrent jobs do not change.
            Trees.refit = SharedDynamicTreeRefit.Schedule(Trees.sharedRefit, ref world, deltaTime, float3.zero, motionsSolved,
                Trees.boundsCacheBuilt ? Trees.boundsCache : default, &store->treeReady, JobsUtility.JobWorkerCount + 1);
            Trees.refitPending = true;
            Trees.refitInputs = motionsSolved;

            // If the step throws, wait for solved motions before returning: queries helping the pending refit
            // read them.
            bool waitedForSolve = false;
            try
            {
                if (s_ThrowAfterRefitScheduled.Data != 0)
                {
                    ThrowAfterRefitScheduled();
                }

                snapshot = Queries.ScheduleQuerySnapshot(store, globals, ref world, ref Partition, motionsSolved);
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

            // The build of this step has completed, so every entry of the bounds cache that a body could take holds
            // the body's current collider: the build wrote it, and the refit writes it again now.
            if (Trees.boundsCacheBuilt)
            {
                Trees.boundsCache.MarkCurrent();
            }

            if (DynamicBoundsCache.CheckEnabled)
            {
                TreeState.LogBoundsCheck();
            }

            // FallAsleep.SleepContactsJob must read the manifolds before the contact report sorts them, so schedule it now.
            contactsJoined = new FallAsleep.SleepContactsJob
            {
                World = store,
                Records = Contacts.manifolds,
                Parents = Sleep.sleepParents,
                Contacts = Sleep.sleepContacts,
                TouchesStill = Sleep.sleepTouchesStill,
            }.Schedule();
            JobHandle.ScheduleBatchedJobs();

            using (StepWriteBackMarker.Auto())
            {
                writtenBack.Complete();
            }

            JobHandle forces;
            using (StepReadJointForcesMarker.Auto())
            {
                forces = ReadJointForces(simulation.ImpulseEvents.EventDataStream, deltaTime, vesselsSettled);
            }

            // Wake bodies affected by shape removal after joint load reads but before sleep and contact
            // reporting.
            // Sleep grouping must also wait for joint breaking.
            if (store->removedShapeCount != 0)
            {
                forces.Complete();
                new TouchingPairs.ForgetRemovedShapesJob { World = store, Touching = Contacts.touching, RemovedTriggerPairs = Contacts.removedTriggerPairs }.Run();
            }

            JobHandle counted = new SleepCounters.UpdateWakeCountersJob
            {
                World = store,
                Contacts = Sleep.sleepContacts,
                TouchesStill = Sleep.sleepTouchesStill,
                RunOut = (int*)Sleep.sleepRunOut.GetUnsafePtr(),
                DeltaTime = deltaTime,
            }.Schedule((int)store->bodyCount, ParallelBatchSize, contactsJoined);
            sleep = new FallAsleep.SleepJob
            {
                World = store,
                Parents = Sleep.sleepParents,
                Falling = Sleep.fallingAsleep,
                RunOut = (int*)Sleep.sleepRunOut.GetUnsafePtr(),
            }.Schedule(JobHandle.CombineDependencies(counted, forces));
            JobHandle.ScheduleBatchedJobs();

            // ImpulseMemory.ReadImpulsesJob reads only the motions, which the rebuild of the tree doesn't change.
            using (StepReadImpulsesMarker.Auto())
            {
                if (world.NumDynamicBodies != 0)
                {
                    new ImpulseMemory.ReadImpulsesJob { Records = Contacts.manifolds, Events = simulation.CollisionEvents, Built = world }.Run();
                }

                new ImpulseMemory.RememberImpulsesJob { Records = Contacts.manifolds, LastImpulses = Contacts.lastImpulses }.Run();
            }
        }

        /// <summary>Publish queries and finish sleep after the caller has run the contact report.</summary>
        internal void FinishAfterContacts(JobHandle snapshot, JobHandle sleep)
        {
            // Publish as current unless write-back changed poses. Queries finish the pending refit before
            // reading.
            using (StepPublishQueryWorldMarker.Auto())
            {
                snapshot.Complete();
                Queries.FinishQuerySnapshot();
                // A refit that failed has set the ready word too, and leaves the world stale.
                bool treeReady = store->treeReady != 0;
                store->queryDirty = !Partition.motionsMatchStore || (treeReady && Trees.sharedRefit->Failed) ? StoreMemory.QueryStale
                    : !treeReady ? StoreMemory.QueryTreePending : 0u;
                Queries.PublishQueryWorld(store, query, ref world);
            }

            // FallAsleep.FallAsleepJob sets the Sleeping flag. The snapshot reads only the Static flag, so it doesn't need to finish
            // first, although it has.
            using (StepSleepMarker.Auto())
            {
                sleep.Complete();
                new FallAsleep.FallAsleepJob { World = store, Falling = Sleep.fallingAsleep }.Run();
            }
        }

        /// <summary>Finish pending refit work and check whether queries need a new build.</summary>
        internal bool BeginQueryBuild()
        {
            return BuildStage.BeginQueryBuild(store, ref Trees, ref Queries);
        }

        /// <summary>Update the built world from prepared store rows without advancing time.</summary>
        internal void BuildForQueries()
        {
            BuildStage.BuildForQueries(store, globals, query, ref world, ref Partition, ref StaticBodies, ref Trees,
                ref Queries, ref Colliders, ref forcesApplied, ref VesselData.vesselsFilled);
        }

        /// <summary>
        /// Complete the pending dynamic tree refit on the main thread before reading or modifying the built world
        /// or bounds cache.
        /// Call outside jobs; query jobs use HelpPendingRefit instead.
        /// </summary>
        internal void FinishPendingRefit() => Trees.FinishPendingRefit(store);

        /// <summary>
        /// Finish pending refit work without completing a job handle, for callers already inside a job.
        /// FinishPendingRefit later completes the now-empty scheduled jobs.
        /// </summary>
        internal void HelpPendingRefit() => Trees.HelpPendingRefit(store);

        /// <summary>
        /// Whether the published query world waits only for the last step's refit, so that it is current once the refit
        /// has finished.
        /// </summary>
        internal readonly bool OnlyRefitPending => Queries.OnlyRefitPending(store);

    }
}
