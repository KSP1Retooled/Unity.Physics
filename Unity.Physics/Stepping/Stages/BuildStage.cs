using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Vehicles;
using Unity.Physics.Stepping.State;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Profiling;

namespace Unity.Physics.Stepping.Stages
{
    /// <summary>
    /// Builds the world for simulation or queries and schedules joints, vessels, and wheel updates.
    /// </summary>
    internal static unsafe class BuildStage
    {
        private static readonly ProfilerMarker StepScheduleMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Step.ScheduleBuild");

        private static readonly ProfilerMarker BuildJointsMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Build.Joints");
        private static readonly ProfilerMarker BuildPartitionMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Build.Partition");
        private static readonly ProfilerMarker BuildResetMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Build.Reset");
        private static readonly ProfilerMarker BuildScheduleBodiesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Build.ScheduleBodies");
        private static readonly ProfilerMarker BuildScheduleBroadphaseMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Build.ScheduleBroadphase");

        private const int ParallelBatchSize = 64;

        // Building a joint costs much more than building a body, so each batch contains fewer joints.
        private const int JointBatchSize = 8;

        // Copying the joints for one unit requires copying a few hundred bytes.
        private const int CopyBatchSize = 64;

        private static JobHandle BuildBodies(WorldData* store, ref PhysicsWorld world, ref BodyPartition partition,
            ref StaticRows staticRows, ref TreeState trees, ref ColliderState colliders, ref bool builtWhole,
            float deltaTime, JobHandle dependency)
        {
            using (BuildPartitionMarker.Auto())
            {
                partition.Rebuild(store);

                // Indices in the static tree are relative to the static bodies. Queries and pair finding add the number
                // of dynamic bodies, so only changes to static bodies require rebuilding this tree.
                staticRows.NoteBuild(store, ref builtWhole);
                trees.treeMatchesHeld = builtWhole;
                builtWhole = false;
            }

            // Compare cached bounds only when body storage and dynamic count are retained and the cache remains
            // valid.
            bool compareBodies = trees.CanCompareBounds(ref world, ref partition);
            int dynamicBefore = world.NumDynamicBodies;
            bool keepStaticRows = staticRows.KeepsStaticRows(ref world, ref partition);

            // Reset the world with no joints. The joint build job fills them later.
            // Unity Physics schedules jobs that read the joints through deferred arrays before it knows the count.
            bool keptBodies;
            bool checkMemory = StepCacheSwitches.MemoryChecks;
            using (BuildResetMarker.Auto())
            {
                keptBodies = world.ResetAndKeepBodies(partition.staticIndices.Length + 1, partition.dynamicIndices.Length, 0,
                    checkMemory, out NativeArray<RigidBody> replaced);
                if (checkMemory)
                {
                    staticRows.QuarantineReplacedBodies(replaced, deltaTime > 0.0f);
                }
            }

            compareBodies &= keptBodies;
            keepStaticRows = staticRows.RetainOrRemember(ref world, ref partition, dynamicBefore, keepStaticRows, keptBodies);

            DynamicBoundsCache.View boundsView = trees.PrepareBounds(ref partition);

            // Pair creation and island finding depend on the bodies and trees, so schedule them first.
            // Joint compilation and vessel work can proceed alongside them.
            JobHandle broadphase;
            using (BuildScheduleBodiesMarker.Auto())
            {
                var bodiesJob = new StoreBodies.BuildBodiesJob
                {
                    Store = store->bodies,
                    DynamicIndices = partition.dynamicIndices.AsArray(),
                    StaticIndices = partition.staticIndices.AsArray(),
                    Bodies = world.Bodies,
                    MotionDatas = world.MotionDatas,
                    MotionVelocities = world.MotionVelocities,
                    DeltaTime = deltaTime,
                    Same = boundsView.Same,
                    CompareBodies = compareBodies,
                    KeepStatics = keepStaticRows,
                };

                JobHandle bodiesBuilt = bodiesJob.Schedule(bodiesJob.Count, ParallelBatchSize, dependency);
                if (checkMemory)
                {
                    bodiesBuilt.Complete();
                    StaticRows.Validate(store, ref world, ref partition);
                }
                JobHandle.ScheduleBatchedJobs();
                broadphase = trees.ScheduleTrees(store, ref world, ref staticRows, ref partition, colliders.colliderWork, deltaTime, bodiesBuilt);
                JobHandle.ScheduleBatchedJobs();
            }

            return broadphase;
        }

        private static JobHandle BuildJoints(WorldData* store, ref PhysicsWorld world, ref BodyPartition partition,
            ref JointBuildState jointState, ref VesselState vessels, ref uint stepCount, float deltaTime, JobHandle dependency)
        {
            // Queries need no solver joints. During simulation, joint builds wait for collider jobs that write
            // body flags.
            JobHandle joints = dependency;
            if (deltaTime > 0.0f)
            {
                using var jointsSpan = BuildJointsMarker.Auto();

                // Reuse the groups if the topology and forest setting have not changed since they were found. The joint
                // build refreshes group data that depends on other state.
                bool vesselsEnabled = VesselSwitches.Enabled;
                bool forest = ForestSwitches.Enabled || vesselsEnabled;
                bool regrouped = jointState.FindGroups(store, forest, dependency, out JobHandle groupsFound);

                bool compiledThisStep = vessels.CompileAndChoose(store, ref jointState, ref partition,
                    vesselsEnabled, regrouped, ref stepCount, groupsFound);

                // The number of joints produced by each unit is not known in advance. JointTranslation.BuildJointsJob therefore writes a
                // separate stream range for each unit.
                JobHandle allocated = NativeStream.ScheduleConstruct(out NativeStream stream, jointState.jointUnits, groupsFound, Allocator.TempJob);
                JobHandle emitted = new JointTranslation.BuildJointsJob
                {
                    World = store,
                    States = jointState.jointStates->Ptr,
                    BuiltIndices = partition.builtIndices.AsArray(),
                    Units = jointState.jointUnits.AsDeferredJobArray(),
                    Output = stream.AsWriter(),
                    Written = jointState.jointsWritten.AsDeferredJobArray(),
                    Groups = jointState.jointGroups.AsDeferredJobArray(),
                    Members = jointState.jointGroupMembers.AsDeferredJobArray(),
                    DynamicCount = partition.dynamicIndices.Length,
                    AnchorIndex = partition.dynamicIndices.Length + partition.staticIndices.Length,
                    DeltaTime = deltaTime,
                    Roles = jointState.jointRoles.AsDeferredJobArray(),
                    Locking = forest && ForestSwitches.Locking,
                    LockFraction = ForestSwitches.LockFraction,
                    YieldFraction = ForestSwitches.YieldFraction,
                    Vessels = vessels.vesselSet.IsCreated,
                    UnitVessels = vessels.vesselBuild.UnitVessels.AsArray(),
                    VesselActive = vessels.vesselBuild.Active.AsArray(),
                }.Schedule(jointState.jointUnits, JointBatchSize, allocated);

                vessels.FillConstraints(store, ref jointState, compiledThisStep, groupsFound, emitted, dependency);

                JobHandle laidOut = new JointGrouping.LayOutJointsJob
                {
                    Written = jointState.jointsWritten,
                    Output = world.DynamicsWorld.InternalJointsList,
                }.Schedule(emitted);

                JobHandle copied = new JointGrouping.CopyJointsJob
                {
                    Input = stream.AsReader(),
                    FirstJoints = jointState.jointsWritten.AsDeferredJobArray(),
                    Output = world.DynamicsWorld.InternalJointsList.AsDeferredJobArray(),
                }.Schedule(jointState.jointsWritten, CopyBatchSize, laidOut);

                joints = stream.Dispose(copied);
            }

            return joints;
        }

        /// <summary>
        /// Schedule wheel updates after the broadphase and before consumers of wheel-modified velocities and
        /// joints.
        /// Run only with wheels and a positive timestep.
        /// </summary>
        private static JobHandle ScheduleWheels(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref BodyPartition partition, float deltaTime, JobHandle broadphase, JobHandle joints)
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
                BuiltIndices = partition.builtIndices.AsArray(),
                Joints = world.DynamicsWorld.InternalJointsList,
                AnchorIndex = partition.dynamicIndices.Length + partition.staticIndices.Length,
                DeltaTime = deltaTime,
            }.Schedule(JobHandle.CombineDependencies(broadphase, joints));
            world.DynamicsWorld.JointsBuilt = wheels;
            return wheels;
        }

        /// <summary>Builds the bodies, joints and broadphase for the simulation systems to advance.</summary>
        internal static JobHandle BuildWorld(WorldData* store, StoreGlobals* globals, ref PhysicsWorld world,
            ref BodyPartition partition, ref StaticRows staticRows, ref TreeState trees,
            ref JointBuildState jointState, ref VesselState vessels, ref ColliderState colliders,
            ref JobHandle forcesApplied, ref bool builtWhole, ref uint stepCount, float deltaTime, JobHandle dependency)
        {
            using var span = StepScheduleMarker.Auto();

            // Every job below waits for the forces, including the collider jobs, which write the flags that the forces
            // job reads.
            dependency = JobHandle.CombineDependencies(dependency, forcesApplied);
            JobHandle collidersReady = colliders.RefreshColliders(store, globals, trees.colliderChanges, dependency);
            // An interrupted step may leave vessel fill jobs running. Finish them before resizing vessel lists.
            vessels.vesselsFilled.Complete();
            JobHandle broadphase = BuildBodies(store, ref world, ref partition, ref staticRows, ref trees,
                ref colliders, ref builtWhole, deltaTime, collidersReady);
            JobHandle joints = BuildJoints(store, ref world, ref partition, ref jointState, ref vessels,
                ref stepCount, deltaTime, collidersReady);

            using (BuildScheduleBroadphaseMarker.Auto())
            {
                // Build joints alongside the broadphase. Only dispatch pair creation needs to wait for them.
                world.DynamicsWorld.JointsBuilt = joints;
                return ScheduleWheels(store, globals, ref world, ref partition, deltaTime, broadphase, joints);
            }
        }

        /// <summary>
        /// Update built statics and their tree from store poses, recording the version to avoid redundant work.
        /// </summary>
        private static void PoseStatics(WorldData* store, ref PhysicsWorld world, ref BodyPartition partition,
            ref StaticRows staticRows, ref TreeState trees, ref ColliderState colliders)
        {
            staticRows.NotePoseUpdate(store);
            trees.treeMatchesHeld = true;
            JobHandle posed = new StoreBodies.PoseStaticsJob
            {
                Store = store->bodies,
                StaticIndices = partition.staticIndices.AsArray(),
                StaticBodies = world.StaticBodies,
            }.Schedule(partition.staticIndices.Length, ParallelBatchSize, default);

            world.CollisionWorld.ScheduleUpdateStaticTree(ref world, trees.BuildStaticTree(staticRows.staticsChanged), trees.ScheduleShiftStaticTree(store, ref world, staticRows.staticsChanged, colliders.colliderWork, posed), true).Complete();
        }

        /// <summary>Finish pending refit work and check whether queries need a new build.</summary>
        internal static bool BeginQueryBuild(WorldData* store, ref TreeState trees, ref QuerySnapshotState queries)
        {
            trees.FinishPendingRefit(store);

            return queries.BeginBuild(store);
        }

        /// <summary>Update the built world from prepared store rows without advancing time.</summary>
        internal static void BuildForQueries(WorldData* store, StoreGlobals* globals, QueryWorld* query,
            ref PhysicsWorld world, ref BodyPartition partition, ref StaticRows staticRows, ref TreeState trees,
            ref QuerySnapshotState queries, ref ColliderState colliders, ref JobHandle forcesApplied, ref JobHandle vesselsFilled)
        {
            // The last build is still valid except for the poses of static bodies. The query snapshot contains no data
            // that depends on a pose.
            if (queries.CanUpdateStaticPoses(store, globals, ref colliders))
            {
                PoseStatics(store, ref world, ref partition, ref staticRows, ref trees, ref colliders);
                queries.PublishQueryWorld(store, query, ref world);
                return;
            }

            // A step that threw between ApplyForces and BuildWorld can have left the forces job running, and the collider
            // jobs write the flags that it reads.
            forcesApplied.Complete();
            JobHandle collidersReady = colliders.RefreshColliders(store, globals, trees.colliderChanges, default);
            // The query build also waits for any vessel work left by the prior step.
            vesselsFilled.Complete();
            JobHandle broadphase = BuildBodies(store, ref world, ref partition, ref staticRows, ref trees,
                ref colliders, ref queries.builtWhole, 0.0f, collidersReady);
            queries.ScheduleQuerySnapshot(store, globals, ref world, ref partition, broadphase).Complete();
            broadphase.Complete();
            trees.MarkBoundsCurrent();
            queries.FinishQuerySnapshot();
            queries.PublishQueryWorld(store, query, ref world);
        }
    }
}
