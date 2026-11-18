using System;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Contacts;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Vessels.Detection;
using Unity.Physics.Stepping.State;
using Unity.Physics.Store.World;
using Unity.Profiling;

namespace Unity.Physics.Stepping.Stages
{
    /// <summary>
    /// Sets up the solver, schedules warm starts, and reads vessel loads from solved motions.
    /// </summary>
    internal static unsafe class SolveStage
    {
        private static readonly ProfilerMarker StepScheduleMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Step.ScheduleSolve");

        /// <summary>The settings that the simulation systems use to build their input for a step.</summary>
        internal static PhysicsStep Settings
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

        /// <summary>Warm starts the Jacobians.</summary>
        internal static JobHandle PrepareSolve(ref PhysicsWorld world, ref BodyPartition partition,
            ref ContactState contactState, ref VesselState vessels, ref StepSolverCaches solverCaches,
            JobHandle warmStartsIndexed, out Simulation* preparedSimulation, out JobHandle preparedInputs,
            SimulationSingleton singleton, ref SimulationStepInput input, JobHandle dependency)
        {
            using var span = StepScheduleMarker.Auto();

            solverCaches.directSolverCache.Forest = new ForestSettings
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
            input.IslandSolverCache = solverCaches.directSolverCache;
            input.VesselSet = vessels.vesselSet;
            preparedSimulation = singleton.AsSimulationPtr();
            preparedInputs = JobHandle.CombineDependencies(dependency, warmStartsIndexed, vessels.vesselsFilled);
            return new WarmStart.WarmStartJob
            {
                WarmStarts = contactState.warmStarts,
                Motions = (MotionVelocity*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.MotionVelocities),
                MotionCount = partition.dynamicIndices.Length,
            }.Schedule(singleton, ref world, JobHandle.CombineDependencies(dependency, warmStartsIndexed, vessels.vesselsFilled));
        }

        /// <summary>
        /// Schedule vessel joint load extraction after solving. Read solved motion poses without waiting for
        /// write-back.
        /// </summary>
        internal static JobHandle ScheduleVesselLoads(WorldData* store, ref PhysicsWorld world,
            ref BodyPartition partition, ref JointBuildState jointState, ref VesselState vessels, float deltaTime,
            JobHandle dependency)
        {
            const int vesselLoadsPerBatch = 32;

            // The jobs write the joint loads through arrays that the safety system doesn't track, so the list must not
            // change size after this point in the step.
            jointState.jointLoads.ResizeUninitialized((int)store->jointCount);
            jointState.jointsOverloaded.ResizeUninitialized((int)store->jointCount);
            if (jointState.jointStates->Length != (int)store->jointCount)
            {
                throw new InvalidOperationException("the joints' load states and the store's joints differ in number");
            }

            if (!vessels.vesselSet.IsCreated)
            {
                return default;
            }

            return new VesselLoads.VesselLoadsJob
            {
                World = store,
                States = jointState.jointStates->Ptr,
                Units = jointState.jointUnits.AsArray(),
                Groups = jointState.jointGroups.AsArray(),
                Members = jointState.jointGroupMembers.AsArray(),
                MotionDatas = world.MotionDatas,
                BuiltIndices = partition.builtIndices.AsArray(),
                DeltaTime = deltaTime,
                PartUnits = vessels.vesselBuild.PartUnits.AsArray(),
                ClosureUnits = vessels.vesselBuild.ClosureUnits.AsArray(),
                PartIsBody0 = vessels.vesselBuild.PartIsBody0.AsArray(),
                UnitVessels = vessels.vesselBuild.UnitVessels.AsArray(),
                Active = vessels.vesselBuild.Active.AsArray(),
                EdgeLoads = vessels.vesselBuild.EdgeLoads.AsArray(),
                ClosureLoads = vessels.vesselBuild.ClosureLoads.AsArray(),
                JointLoads = jointState.jointLoads.AsArray(),
                Overloaded = jointState.jointsOverloaded.AsArray(),
            }.Schedule(vessels.vesselBuild.PartUnits.Length + vessels.vesselBuild.ClosureUnits.Length, vesselLoadsPerBatch, dependency);
        }
    }
}
