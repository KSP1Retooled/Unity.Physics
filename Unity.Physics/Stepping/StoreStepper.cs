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
using Unity.Physics.Stepping.State;

namespace Unity.Physics.Stepping
{
    /// <summary>
    /// Unmanaged simulation state for one store world, used by the Burst step entry points.
    /// </summary>
    internal unsafe partial struct StoreStepper
    {
        private static readonly ProfilerMarker BuildJointsMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.Joints");
        private static readonly ProfilerMarker BuildPartitionMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.Partition");
        private static readonly ProfilerMarker BuildResetMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.Reset");
        private static readonly ProfilerMarker BuildScheduleBodiesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.ScheduleBodies");
        private static readonly ProfilerMarker BuildScheduleBroadphaseMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Build.ScheduleBroadphase");
        private static readonly ProfilerMarker RefreshCollidersMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.RefreshColliders");
        private static readonly ProfilerMarker StepApplyForcesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ApplyForces");
        private static readonly ProfilerMarker StepPublishQueryWorldMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.PublishQueryWorld");
        private static readonly ProfilerMarker StepReadImpulsesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadImpulses");
        private static readonly ProfilerMarker StepReadJointForcesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.ReadJointForces");
        private static readonly ProfilerMarker StepScheduleMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Schedule");
        private static readonly ProfilerMarker StepSleepMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.Sleep");
        private static readonly ProfilerMarker StepWaitForSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WaitForSolve");
        private static readonly ProfilerMarker StepWriteBackMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Step.WriteBack");

        private const int ParallelBatchSize = 64;

        // Building a joint costs much more than building a body, so each batch contains fewer joints.
        private const int JointBatchSize = 8;

        // Copying the joints for one unit requires copying a few hundred bytes.
        private const int CopyBatchSize = 64;

        private readonly WorldData* store;
        private readonly StoreGlobals* globals;
        private readonly QueryWorld* query;

        private PhysicsWorld world;
        internal QuerySnapshotState Queries;
        internal SleepState Sleep;
        internal ContactState Contacts;
        internal VesselState VesselData;
        internal JointBuildState Joints;
        internal TreeState Trees;
        internal StaticRows StaticBodies;
        internal BodyPartition Partition;

        private IslandSolverCache directSolverCache;

        // Prepared simulation, or null since the last FinishStep.
        // Avoid reading the singleton here, which would wait for solver cleanup before scheduling write-back.
        internal Simulation* preparedSimulation;

        // Pre-solve jobs that FinishStepSystem completes before FinishStep reads their lists.
        internal JobHandle preparedInputs;

        private SolverIslandCache solverIslandCache;
        private NativeList<ColliderBuilder.Work> colliderWork;
        private NativeList<long> colliderResults;
        private NativeReference<uint> wholeColliderBuilds;

        private JobHandle collidersBuilt;

        // The job that ApplyForces leaves running. BuildWorld makes the build wait for it.
        private JobHandle forcesApplied;
        private JobHandle warmStartsIndexed;
        private uint builtMaterialVersion;

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
            directSolverCache = IslandSolverCache.Create();
            VesselData.Initialize();
            solverIslandCache = SolverIslandCache.Create();
            Queries.Initialize();
            Contacts.Initialize();
            Sleep.Initialize();
            Contacts.InitializeImpulseMemory();
            colliderWork = new NativeList<ColliderBuilder.Work>(Allocator.Persistent);
            colliderResults = new NativeList<long>(Allocator.Persistent);
            wholeColliderBuilds = new NativeReference<uint>(0, Allocator.Persistent);
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
            collidersBuilt.Complete();

            // A step that threw before FinishStep leaves the fill jobs unfinished, and they write the vessel lists.
            VesselData.vesselsFilled.Complete();
            ReleaseDeadColliders();
            for (uint i = 0; i < store->bodyCount; i++)
            {
                BodyData* body = store->bodies + i;
                ColliderBuilder.Release((*body).Collider()->Value.GetUnsafePtr());
                ColliderBuilder.SetCollider(body, default);
                body->flags |= BodyFlags.ColliderDirty;
            }

            store->queryDirty = 1;
            world.Dispose();
            Trees.DisposeStaticTree();
            Partition.DisposeRows();
            StaticBodies.Dispose();
            Partition.Dispose();
            Joints.Dispose(store);
            directSolverCache.Dispose();
            VesselData.Dispose();
            solverIslandCache.Dispose();
            Queries.Dispose();
            Contacts.Dispose();
            Sleep.Dispose();
            Contacts.DisposeImpulseMemory();
            colliderWork.Dispose();
            colliderResults.Dispose();
            wholeColliderBuilds.Dispose();
            Trees.Dispose();
            Contacts.DisposePairCache();
        }

        /// <summary>The direct solver's cache, including the counts of solves that used the forest.</summary>
        internal readonly IslandSolverCache IslandSolverCache => directSolverCache;

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
            FinishPendingRefit();
            if (s_ThrowBeforeApplyForces.Data != 0)
            {
                ThrowBeforeApplyForces();
            }

            // A step that threw after this stage last time can have left the job running.
            forcesApplied.Complete();
            store->queryWorld = null;
            store->queryDirty = StoreMemory.QueryStale;
        }

        /// <summary>
        /// Wake bodies through joints, then schedule force integration.
        /// BuildWorld waits for the force jobs. The main thread may read flags while they run.
        /// </summary>
        internal void ApplyForces(float deltaTime)
        {
            WheelSolver.RecordStartVelocities(store);

            using (StepApplyForcesMarker.Auto())
            {
                Contacts.WakeRefilteredPairs(store);

                new WakePropagation.WakeJointedBodiesJob { World = store }.Run();

                forcesApplied = new ExternalForces.ApplyForcesJob
                {
                    Bodies = store->bodies,
                    Gravity = store->gravity,
                    DeltaTime = deltaTime,
                }.Schedule((int)store->bodyCount, ParallelBatchSize);
                JobHandle.ScheduleBatchedJobs();
            }
        }

        /// <summary>Builds the bodies, joints and broadphase for the simulation systems to advance.</summary>
        internal JobHandle BuildWorld(float deltaTime, JobHandle dependency)
        {
            using var span = StepScheduleMarker.Auto();

            // Every job below waits for the forces, including the collider jobs, which write the flags that the forces
            // job reads.
            dependency = JobHandle.CombineDependencies(dependency, forcesApplied);
            JobHandle colliders = RefreshColliders(dependency);
            JobHandle broadphase = Build(deltaTime, colliders, out JobHandle joints);

            using (BuildScheduleBroadphaseMarker.Auto())
            {
                // Build joints alongside the broadphase. Only dispatch pair creation needs to wait for them.
                world.DynamicsWorld.JointsBuilt = joints;
                return ScheduleWheels(deltaTime, broadphase, joints);
            }
        }

        /// <summary>
        /// Schedule wheel updates after the broadphase and before consumers of wheel-modified velocities and
        /// joints.
        /// Run only with wheels and a positive timestep.
        /// </summary>
        private JobHandle ScheduleWheels(float deltaTime, JobHandle broadphase, JobHandle joints)
        {
            if (store->vehicleCount == 0 || deltaTime <= 0.0f)
            {
                return broadphase;
            }

            JobHandle wheels = new WheelSolver.WheelJob
            {
                Store = store,
                Globals = globals,
                Collision = world.CollisionWorld,
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                BuiltIndices = Partition.builtIndices.AsArray(),
                Joints = world.DynamicsWorld.InternalJointsList,
                AnchorIndex = Partition.dynamicIndices.Length + Partition.staticIndices.Length,
                DeltaTime = deltaTime,
            }.Schedule(JobHandle.CombineDependencies(broadphase, joints));
            world.DynamicsWorld.JointsBuilt = wheels;
            return wheels;
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
            using var span = StepScheduleMarker.Auto();
            input.VesselSet = VesselData.vesselSet;

            Simulation* simulation = singleton.AsSimulationPtr();
            if (input.World.NumDynamicBodies == 0)
            {
                return dependency;
            }

            NativeList<DispatchPairSequencer.DispatchPair> pairs = simulation->StepContext.PairsToSortByIsland;
            JobHandle filtered = new PairFilter.DropIgnoredPairsJob
            {
                Pairs = pairs.AsDeferredJobArray(),
                Store = store->bodies,
                Globals = globals,
                Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
                Known = Contacts.knownBodyPairs,
                WholeBuilds = wholeColliderBuilds,
                Learned = Contacts.learnedBodyPairs.AsParallelWriter(),
                World = store,
                Units = Joints.jointUnits.AsDeferredJobArray(),
                BodyVessels = VesselData.vesselBuild.BodyVessels.AsArray(),
                VesselActive = VesselData.vesselBuild.Active.AsArray(),
                VesselPairUnits = VesselData.vesselBuild.PairUnits,
            }.Schedule(pairs, ParallelBatchSize, dependency);

            // No job in this step reads the newly learned body pairs. Saving those pairs can therefore run in parallel
            // with finding islands.
            JobHandle remembered = new PairFilter.RememberBodyPairsJob
            {
                Known = Contacts.knownBodyPairs,
                Learned = Contacts.learnedBodyPairs,
                Bodies = (int)store->bodyCount,
            }.Schedule(filtered);

            SimulationStepInput keeping = input;
            keeping.SolverIslandCache = solverIslandCache;
            return JobHandle.CombineDependencies(simulation->ScheduleIslandJobs(keeping, filtered), remembered);
        }

        /// <summary>Reads the contacts produced by the narrow phase before the Jacobians are built from them.</summary>
        internal JobHandle ReadContacts(SimulationSingleton singleton, float deltaTime, JobHandle dependency)
        {
            using var span = StepScheduleMarker.Auto();

            // Submit the batched jobs so that workers can run them while the main thread schedules the jobs below.
            JobHandle.ScheduleBatchedJobs();

            // Drop impulses for removed shapes before new contacts can reuse their addresses.
            if (store->removedShapeCount != 0 && Contacts.lastImpulses.Count != 0)
            {
                new ImpulseMemory.ForgetRemovedImpulsesJob { World = store, LastImpulses = Contacts.lastImpulses }.Run();
            }

            Contacts.manifolds.Clear();
            Contacts.manifoldPoints.Clear();
            JobHandle read = dependency;
            JobHandle merged = dependency;
            NativeStream contacts = singleton.AsSimulationPtr()->StepContext.Contacts;
            if (contacts.IsCreated)
            {
                NativeStream.Reader reader = contacts.AsReader();
                read = new ContactBuckets.PrepareContactBucketsJob
                {
                    Contacts = reader,
                    Buckets = Contacts.contactBuckets,
                    Filled = Contacts.filledContactBuckets,
                }.Schedule(read);
                read = new ContactMaterials.ReadContactsJob
                {
                    Contacts = reader,
                    Buckets = Contacts.contactBuckets,
                    Store = store->bodies,
                    Globals = globals,
                    Bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies),
                    LastImpulses = Contacts.lastImpulses,
                    DeltaTime = deltaTime,
                    BounceThreshold = store->bounceThreshold,
                }.ScheduleUnsafe(contacts, 1, read);

                merged = new ContactBuckets.MergeContactBucketsJob
                {
                    Filled = Contacts.filledContactBuckets,
                    Buckets = Contacts.contactBuckets,
                    Records = Contacts.manifolds,
                    Points = Contacts.manifoldPoints,
                }.Schedule(read);
            }

            // Only warm starting requires the merged manifolds before solving. Merging the manifolds can therefore run
            // in parallel with building the Jacobians.
            warmStartsIndexed = new WarmStart.IndexWarmStartsJob { Records = Contacts.manifolds, WarmStarts = Contacts.warmStarts }.Schedule(merged);

            // If there are no contacts, no more bodies need to be woken. Bodies connected by joints were already woken
            // at the start of the step.
            if (!contacts.IsCreated)
            {
                return read;
            }

            return new WakePropagation.WakeTouchedBodiesJob
            {
                World = store,
                Filled = Contacts.filledContactBuckets,
                Buckets = Contacts.contactBuckets,
                DynamicIndices = Partition.dynamicIndices.AsArray(),
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
            }.Schedule(read);
        }

        /// <summary>Warm starts the Jacobians.</summary>
        internal JobHandle PrepareSolve(SimulationSingleton singleton, ref SimulationStepInput input, JobHandle dependency)
        {
            using var span = StepScheduleMarker.Auto();

            directSolverCache.Forest = new ForestSettings
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
            input.IslandSolverCache = directSolverCache;
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
            FinishPendingRefit();

            // The world that the last step published is current now that its refit has finished.
            if (store->queryDirty == StoreMemory.QueryTreePending && Queries.builtWhole)
            {
                return false;
            }

            store->queryWorld = null;
            return true;
        }

        /// <summary>Update the built world from prepared store rows without advancing time.</summary>
        internal void BuildForQueries()
        {
            // The last build is still valid except for the poses of static bodies. The query snapshot contains no data
            // that depends on a pose.
            if (store->queryDirty == StoreMemory.QueryStaticsMoved && Queries.builtWhole && globals->materialVersion == builtMaterialVersion)
            {
                PoseStatics();
                Queries.PublishQueryWorld(store, query, ref world);
                return;
            }

            // A step that threw between ApplyForces and BuildWorld can have left the forces job running, and the collider
            // jobs write the flags that it reads.
            forcesApplied.Complete();
            JobHandle colliders = RefreshColliders(default);
            JobHandle broadphase = Build(0.0f, colliders, out _);
            Queries.ScheduleQuerySnapshot(store, globals, ref world, ref Partition, broadphase).Complete();
            broadphase.Complete();
            if (Trees.boundsCacheBuilt)
            {
                Trees.boundsCache.MarkCurrent();
            }
            Queries.FinishQuerySnapshot();
            Queries.PublishQueryWorld(store, query, ref world);
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
        internal readonly bool OnlyRefitPending => store->queryDirty == StoreMemory.QueryTreePending && Queries.builtWhole && store->queryWorld != null;

        /// <summary>
        /// Update built statics and their tree from store poses, recording the version to avoid redundant work.
        /// </summary>
        private void PoseStatics()
        {
            StaticBodies.builtStaticsVersion = store->staticsVersion;
            StaticBodies.staticsChanged = true;
            Trees.treeMatchesHeld = true;
            JobHandle posed = new StoreBodies.PoseStaticsJob
            {
                Store = store->bodies,
                StaticIndices = Partition.staticIndices.AsArray(),
                StaticBodies = world.StaticBodies,
            }.Schedule(Partition.staticIndices.Length, ParallelBatchSize, default);

            world.CollisionWorld.ScheduleUpdateStaticTree(ref world, Trees.BuildStaticTree(StaticBodies.staticsChanged), Trees.ScheduleShiftStaticTree(store, ref world, StaticBodies.staticsChanged, colliderWork, posed), true).Complete();
        }

        /// <summary>
        /// Schedule requested collider updates. Consumers must wait for the returned handle before reading
        /// colliders.
        /// </summary>
        private JobHandle RefreshColliders(JobHandle dependency)
        {
            using var span = RefreshCollidersMarker.Auto();

            // Every step and query build completes the previous build, so this handle has already completed.
            collidersBuilt.Complete();
            ColliderBuilder.AskAgainForFailed(store, colliderWork, colliderResults);
            ReleaseDeadColliders();

            // Material changes can replace colliders at the same addresses. Invalidate static rows and the
            // static tree.
            uint materialVersion = globals->materialVersion;
            bool all = materialVersion != builtMaterialVersion;
            if (all)
            {
                store->staticsVersion++;
            }

            collidersBuilt = ColliderBuilder.ScheduleRefresh(store, all, colliderWork, colliderResults,
                wholeColliderBuilds, Trees.colliderChanges, dependency);
            builtMaterialVersion = materialVersion;
            return collidersBuilt;
        }

        private void ReleaseDeadColliders()
        {
            for (uint i = 0; i < store->deadColliderCount; i++)
            {
                ColliderBuilder.Release(store->deadColliders[i]);
            }

            store->deadColliderCount = 0;
        }

        /// <summary>
        /// Schedule body, broadphase and joint builds. Return tree completion and provide joint completion
        /// separately.
        /// </summary>
        private JobHandle Build(float deltaTime, JobHandle dependency, out JobHandle joints)
        {
            // An interrupted step may leave vessel fill jobs running. Finish them before resizing vessel lists.
            VesselData.vesselsFilled.Complete();
            using (BuildPartitionMarker.Auto())
            {
                Partition.Rebuild(store);

                // Indices in the static tree are relative to the static bodies. Queries and pair finding add the number
                // of dynamic bodies, so only changes to static bodies require rebuilding this tree.
                StaticBodies.staticsChanged = !Queries.builtWhole || store->staticsVersion != StaticBodies.builtStaticsVersion;
                StaticBodies.builtStaticsVersion = store->staticsVersion;
                Trees.treeMatchesHeld = Queries.builtWhole;
                Queries.builtWhole = false;
            }

            // Compare cached bounds only when body storage and dynamic count are retained and the cache remains
            // valid.
            void* bodiesBefore = NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies);
            bool compareBodies = Trees.CanCompareBounds(ref world, ref Partition);
            int dynamicBefore = world.NumDynamicBodies;
            bool keepStaticRows = StaticBodies.KeepsStaticRows(world.NumStaticBodies, Partition.staticIndices);

            // Reset the world with no joints; the joint build job will fill them. Unity Physics uses deferred arrays to
            // read the joints, so jobs can be scheduled before the joint count is known.
            using (BuildResetMarker.Auto())
            {
                world.Reset(Partition.staticIndices.Length + 1, Partition.dynamicIndices.Length, 0);
                world.DynamicsWorld.EnableDirectSolver = true;
            }

            bool sameMemory = NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies) == bodiesBefore;
            compareBodies &= sameMemory;
            keepStaticRows = StaticBodies.RetainOrRemember(ref world, ref Partition, dynamicBefore, keepStaticRows, sameMemory);

            DynamicBoundsCache.View boundsView = TreeState.BoundsCacheEnabled ? Trees.boundsCache.Resize(Partition.dynamicIndices.Length) : default;

            // Schedule bodies and trees first: pair creation and island finding depend on them.
            // Joint compilation and vessel work can proceed alongside them.
            JobHandle broadphase;
            using (BuildScheduleBodiesMarker.Auto())
            {
                var bodiesJob = new StoreBodies.BuildBodiesJob
                {
                    Store = store->bodies,
                    DynamicIndices = Partition.dynamicIndices.AsArray(),
                    StaticIndices = Partition.staticIndices.AsArray(),
                    Bodies = world.Bodies,
                    MotionDatas = world.MotionDatas,
                    MotionVelocities = world.MotionVelocities,
                    DeltaTime = deltaTime,
                    Same = boundsView.Same,
                    CompareBodies = compareBodies,
                    KeepStatics = keepStaticRows,
                };

                JobHandle bodiesBuilt = bodiesJob.Schedule(bodiesJob.Count, ParallelBatchSize, dependency);
                JobHandle.ScheduleBatchedJobs();
                broadphase = Trees.ScheduleTrees(store, ref world, StaticBodies.staticsChanged, ref Partition, colliderWork, deltaTime, bodiesBuilt);
                JobHandle.ScheduleBatchedJobs();
            }

            // Queries need no solver joints. During simulation, joint builds wait for collider jobs that write
            // body flags.
            joints = dependency;
            if (deltaTime > 0.0f)
            {
                using var jointsSpan = BuildJointsMarker.Auto();

                // Reuse the groups if the topology and forest setting have not changed since they were found. The joint
                // build refreshes group data that depends on other state.
                bool vessels = VesselSwitches.Enabled;
                bool forest = ForestSwitches.Enabled || vessels;
                bool regrouped = Joints.FindGroups(store, forest, dependency, out JobHandle groupsFound);

                bool compiledThisStep = VesselData.CompileAndChoose(store, ref Joints, ref Partition,
                    vessels, regrouped, ref Contacts.stepCount, groupsFound);

                // The number of joints produced by each unit is not known in advance. JointTranslation.BuildJointsJob therefore writes a
                // separate stream range for each unit.
                JobHandle allocated = NativeStream.ScheduleConstruct(out NativeStream stream, Joints.jointUnits, groupsFound, Allocator.TempJob);
                JobHandle emitted = new JointTranslation.BuildJointsJob
                {
                    World = store,
                    States = Joints.jointStates->Ptr,
                    BuiltIndices = Partition.builtIndices.AsArray(),
                    Units = Joints.jointUnits.AsDeferredJobArray(),
                    Output = stream.AsWriter(),
                    Written = Joints.jointsWritten.AsDeferredJobArray(),
                    Groups = Joints.jointGroups.AsDeferredJobArray(),
                    Members = Joints.jointGroupMembers.AsDeferredJobArray(),
                    DynamicCount = Partition.dynamicIndices.Length,
                    AnchorIndex = Partition.dynamicIndices.Length + Partition.staticIndices.Length,
                    DeltaTime = deltaTime,
                    Roles = Joints.jointRoles.AsDeferredJobArray(),
                    Locking = forest && ForestSwitches.Locking,
                    LockFraction = ForestSwitches.LockFraction,
                    YieldFraction = ForestSwitches.YieldFraction,
                    Vessels = VesselData.vesselSet.IsCreated,
                    UnitVessels = VesselData.vesselBuild.UnitVessels.AsArray(),
                    VesselActive = VesselData.vesselBuild.Active.AsArray(),
                }.Schedule(Joints.jointUnits, JointBatchSize, allocated);

                VesselData.FillConstraints(store, ref Joints, compiledThisStep, groupsFound, emitted, dependency);

                JobHandle laidOut = new JointGrouping.LayOutJointsJob
                {
                    Written = Joints.jointsWritten,
                    Output = world.DynamicsWorld.InternalJointsList,
                }.Schedule(emitted);

                JobHandle copied = new JointGrouping.CopyJointsJob
                {
                    Input = stream.AsReader(),
                    FirstJoints = Joints.jointsWritten.AsDeferredJobArray(),
                    Output = world.DynamicsWorld.InternalJointsList.AsDeferredJobArray(),
                }.Schedule(Joints.jointsWritten, CopyBatchSize, laidOut);

                joints = stream.Dispose(copied);
            }

            return broadphase;
        }
    }
}
