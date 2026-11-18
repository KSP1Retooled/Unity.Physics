using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Islands;
using Unity.Physics.Dynamics.Vessels.Model;
using UnityEngine;
using UnityEngine.Assertions;
using static Unity.Physics.Math;

namespace Unity.Physics
{
    /// <summary>
    /// <para> Solver types, used for calculating joint and contact forces.</para>
    /// <para> Types appear in order of decreasing solver performance but increasing accuracy.</para>
    /// </summary>
    public enum SolverType : byte
    {
        /// <summary>
        ///     <para> Iterative solver (default). Fast and approximate.</para>
        ///     <para> Ideal for situations involving many colliding rigid bodies.</para>
        /// </summary>
        Iterative = 0,

        /// <summary>
        ///     <para> Direct solver. Accurate but more computationally demanding.</para>
        ///     <para> Ideal for situations involving complex jointed mechanisms with
        ///     high mass or stiffness ratios, and for accurate contact resolution and friction modeling.
        ///     See also <see cref="Solver.DirectSolverSettings"/>.</para>
        /// </summary>
        Direct = 1
    }

    /// <summary>   A static class that exposes Solver configuration structures. </summary>
    public static partial class Solver
    {
        /// <summary> Default solver type </summary>
        public const SolverType kDefaultSolverType = SolverType.Iterative;

        /// <summary>
        /// Takes the time step in ms and returns the inverse of the time step (simulation frequency) in Hz.
        /// </summary>
        /// <param name="timestep"> The time step in ms. </param>
        /// <returns> The inverse time step (simulation frequency) in Hz. </returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CalculateInvTimestep(float timestep)
        {
            return math.select(0.0f, 1.0f / timestep, timestep > 0.0f);
        }

        /// <summary>   Settings for controlling the solver stabilization heuristic. </summary>
        public struct StabilizationHeuristicSettings
        {
            byte m_EnableSolverStabilization;

            /// <summary>   Global switch to enable/disable the whole heuristic (false by default) </summary>
            ///
            /// <value> True if enable solver stabilization, false if not. </value>
            public bool EnableSolverStabilization
            {
                get => m_EnableSolverStabilization > 0;
                set => m_EnableSolverStabilization = (byte)(value ? 1 : 0);
            }

            // Individual features control (only valid when EnableSolverStabilizationHeuristic is true)

            byte m_EnableFrictionVelocities;

            /// <summary>
            /// Switch to enable/disable heuristic when calculating friction velocities. Should be disabled
            /// only if it is causing behavior issues.
            /// </summary>
            ///
            /// <value> True if enable friction velocities, false if not. </value>
            public bool EnableFrictionVelocities
            {
                get => m_EnableFrictionVelocities > 0;
                set => m_EnableFrictionVelocities = (byte)(value ? 1 : 0);
            }

            /// <summary>
            /// Controls the intensity of the velocity clipping. Defaults to 1.0f, while other values will
            /// scale the intensity up/down. Shouldn't go higher than 5.0f, as it will result in bad behavior
            /// (too aggressive velocity clipping). Set it to 0.0f to disable the feature.
            /// </summary>
            public float VelocityClippingFactor;

            /// <summary>
            /// Controls the intensity of inertia scaling. Defaults to 1.0f, while other values will scale
            /// the intensity up/down. Shouldn't go higher than 5.0f, as it will result in bad behavior (too
            /// high inertia of bodies). Set it to 0.0f to disable the feature.
            /// </summary>
            public float InertiaScalingFactor;

            /// <summary>   The defualt stabilization options. </summary>
            public static readonly StabilizationHeuristicSettings Default = new StabilizationHeuristicSettings
            {
                m_EnableSolverStabilization = 0,
                m_EnableFrictionVelocities = 1,
                VelocityClippingFactor = 1.0f,
                InertiaScalingFactor = 1.0f
            };
        }

        /// <summary>   Data used for solver stabilization. </summary>
        public struct StabilizationData
        {
            /// <summary>   Constructor. </summary>
            ///
            /// <param name="stepInput">    The step input. </param>
            /// <param name="context">      The context. </param>
            public StabilizationData(SimulationStepInput stepInput, SimulationContext context)
            {
                StabilizationHeuristicSettings = stepInput.SolverStabilizationHeuristicSettings;
                Gravity = stepInput.Gravity;
                if (stepInput.SolverStabilizationHeuristicSettings.EnableSolverStabilization)
                {
                    InputVelocities = context.InputVelocities;
                    MotionData = context.SolverStabilizationMotionData;
                }
                else
                {
                    InputVelocities = default;
                    MotionData = default;
                }
            }

            // Settings for stabilization heuristics
            internal StabilizationHeuristicSettings StabilizationHeuristicSettings;

            // Disable container safety restriction because it will complain about aliasing
            // with SimulationContext buffers, and it is aliasing, but completely safe.
            // Also, we need the ability to have these not allocated when the feature is not used.

            // Data source: copy of gravity integrated MotionData from output of UpdateInputVelocitiesJob
            [NativeDisableParallelForRestriction]
            [NativeDisableContainerSafetyRestriction]
            internal NativeArray<Velocity> InputVelocities;

            [NativeDisableParallelForRestriction]
            [NativeDisableContainerSafetyRestriction]
            internal NativeArray<StabilizationMotionData> MotionData;

            // Gravity is used to define thresholds for stabilization,
            // and it's not needed in the solver unless stabilization is required.
            internal float3 Gravity;
        }

        // Per motion data for solver stabilization
        internal struct StabilizationMotionData
        {
            public float InverseInertiaScale;
            public byte NumPairs;
        }

        // Internal motion data input for the solver stabilization
        internal struct MotionStabilizationInput
        {
            public Velocity InputVelocity;
            public float InverseInertiaScale;

            public static readonly MotionStabilizationInput Default = new MotionStabilizationInput
            {
                InputVelocity = Velocity.Zero,
                InverseInertiaScale = 1.0f
            };
        }

        internal struct StepInput
        {
            /// <summary> Vector acceleration due to gravity in m/s^2 </summary>
            public float3 Gravity;
            /// <summary> TimeStep in seconds. </summary>
            public float Timestep;
            /// <summary> Enables gyroscopic torque, which will be added to dynamic bodies in every simulation step. </summary>
            public bool EnableGyroscopicTorque;
            /// <summary> The total number of substeps per frame. </summary>
            public int NumSubsteps;
            /// <summary> The total number of solver iterations per substep. </summary>
            public int NumSolverIterations;
            /// <summary> The direct solver settings. </summary>
            public DirectSolverSettings DirectSolverSettings;
            /// <summary> See <see cref="SimulationStepInput.IslandSolverCache"/>. </summary>
            internal IslandSolverCache IslandSolverCache;

            /// <summary> See <see cref="SimulationStepInput.VesselSet"/>. </summary>
            internal VesselSet VesselSet;

            /// <summary>
            /// The world's bodies and their number, so that a vessel island's boundary can know a body outside its
            /// vessels by the body's entity. The pointer is taken without safety checks, and the solver only reads it.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public unsafe RigidBody* Bodies;
            public int BodyCount;
            /// <summary> The current substep. Is -1 when uninitialized. </summary>
            public int CurrentSubstep;
            /// <summary> The current solver iteration. Is -1 when uninitialized. </summary>
            public int CurrentSolverIteration;
            /// <summary> Disable exporting events (TriggerEvents, CollisionEvents or ImpulseEvents). Default is false. </summary>
            public bool DisableExportEvents;

            /// <summary>
            /// The maximum separation velocities for penetrating contacts: first when one body is static, then when
            /// both bodies are dynamic.
            /// </summary>
            public float MaxStaticDepenetrationVelocity;
            public float MaxDynamicDepenetrationVelocity;

            /// <summary>
            /// Remove penetration correction velocity after it has moved bodies apart, preventing continued
            /// separation.
            /// </summary>
            public bool Relaxing;

            public bool IsLastSubstep => CurrentSubstep == NumSubsteps - 1;
            public bool IsFirstSolverIteration => CurrentSolverIteration == 0;
            public bool IsLastSolverIteration => CurrentSolverIteration == NumSolverIterations - 1;

            // Note: events are only exported in the last iteration of the last sub-step, unless events export is disabled.
            public bool ExportEventsInThisIteration => IsLastSubstep && IsLastSolverIteration && !DisableExportEvents;

            // Note: events are only exported in the last sub-step, unless events export is disabled.
            public bool ExportEventsInThisSubstep => IsLastSubstep && !DisableExportEvents;
        }

        #region BuildJacobians

        // Schedule jobs to build Jacobians from the contacts stored in the simulation context for the first substep
        // where timeStep = frame timestep / numSubsteps
        internal static SimulationJobHandles ScheduleBuildJacobiansJobs(in SimulationStepInput input, JobHandle inputDeps,
            ref NativeList<DispatchPairSequencer.DispatchPair> dispatchPairs,
            ref DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo,
            ref NativeStream contacts, ref NativeStream jacobians, bool multiThreaded = true)
        {
            SimulationJobHandles returnHandles = default;

            if (!multiThreaded)
            {
                var buildJob = new BuildJacobiansJob
                {
                    Input = input,
                    ContactsReader = contacts.AsReader(),
                    JacobiansWriter = jacobians.AsWriter(),
                    DispatchPairs = dispatchPairs.AsDeferredJobArray(),
                    SolverSchedulerInfo = solverSchedulerInfo
                }.Schedule(inputDeps);

                buildJob = new BuildDirectSolverJacobiansMapJob
                {
                    JacobiansReader = jacobians.AsReader(),
                    SolverSchedulerInfo = solverSchedulerInfo,
                    PhasedDispatchPairs = dispatchPairs.AsDeferredJobArray()
                }.Schedule(buildJob);

                returnHandles.FinalExecutionHandle = buildJob;
            }
            else
            {
                var gravityMagnitude = math.length(input.Gravity);

                var iterativeJacobiansHandle = new ParallelBuildJacobiansJob
                {
                    Input = input,
                    Timestep = input.TimeStep,
                    NumSubsteps = 1,
                    MotorImpulseScale = 1.0f,
                    GravityMagnitude = gravityMagnitude,
                    ContactsReader = contacts.AsReader(),
                    JacobiansWriter = jacobians.AsWriter(),
                    DispatchPairs = dispatchPairs.AsDeferredJobArray(),
                    IterativeSolverSchedulerInfo = solverSchedulerInfo.IterativePairsIterativeScheduling
                }.ScheduleUnsafeIndex0(solverSchedulerInfo.IterativePairsIterativeScheduling.NumWorkItems, 1, inputDeps);

                // Divide motor limits by the direct island substep count because each substep applies its own
                // limit.
                const int directSubsteps = SolverConstants.DirectIslandSubsteps;
                var directJacobiansHandle = new ParallelBuildJacobiansJob
                {
                    Input = input,
                    Timestep = input.TimeStep / directSubsteps,
                    NumSubsteps = directSubsteps,
                    MotorImpulseScale = 1.0f / directSubsteps,
                    GravityMagnitude = gravityMagnitude,
                    ContactsReader = contacts.AsReader(),
                    JacobiansWriter = jacobians.AsWriter(),
                    DispatchPairs = dispatchPairs.AsDeferredJobArray(),
                    IterativeSolverSchedulerInfo = solverSchedulerInfo.DirectPairsIterativeScheduling
                }.ScheduleUnsafeIndex0(solverSchedulerInfo.DirectPairsIterativeScheduling.NumWorkItems, 1, inputDeps);

                directJacobiansHandle = new ParallelBuildDirectSolverJacobiansMapJob
                {
                    JacobiansReader = jacobians.AsReader(),
                    SolverSchedulerInfo = solverSchedulerInfo,
                    PhasedDispatchPairs = dispatchPairs.AsDeferredJobArray()
                }.ScheduleUnsafeIndex0(solverSchedulerInfo.DirectPairsIterativeScheduling.NumWorkItems, 1, directJacobiansHandle);

                returnHandles.FinalExecutionHandle = JobHandle.CombineDependencies(iterativeJacobiansHandle, directJacobiansHandle);
            }

            returnHandles.FinalDisposeHandle = JobHandle.CombineDependencies(
                dispatchPairs.Dispose(returnHandles.FinalExecutionHandle),
                contacts.Dispose(returnHandles.FinalExecutionHandle));

            return returnHandles;
        }

        [BurstCompile]
        struct BuildJacobiansJob : IJob
        {
            [ReadOnly] public SimulationStepInput Input;

            public NativeStream.Reader ContactsReader;
            public NativeStream.Writer JacobiansWriter;
            public NativeArray<DispatchPairSequencer.DispatchPair> DispatchPairs;

            [ReadOnly] public DispatchPairSequencer.SolverSchedulerInfo SolverSchedulerInfo;

            public void Execute()
            {
                BuildIslandJacobians(Input, DispatchPairs, SolverSchedulerInfo, ContactsReader, JacobiansWriter);
            }
        }

        [BurstCompile]
        struct ParallelBuildJacobiansJob : IJobParallelForDefer
        {
            [ReadOnly]
            public SimulationStepInput Input;
            public float Timestep;
            public int NumSubsteps;
            public float MotorImpulseScale;
            public float GravityMagnitude;

            public NativeStream.Reader ContactsReader;

            [NativeDisableContainerSafetyRestriction]
            public NativeStream.Writer JacobiansWriter;

            [NativeDisableContainerSafetyRestriction]
            public NativeArray<DispatchPairSequencer.DispatchPair> DispatchPairs;

            [ReadOnly]
            public DispatchPairSequencer.IterativeSolverSchedulerInfo IterativeSolverSchedulerInfo;

            public void Execute(int workItemIndexOffset)
            {
                var firstDispatchPairIndex = IterativeSolverSchedulerInfo.FirstDispatchPairIndex.Value
                    + IterativeSolverSchedulerInfo.GetLinkedWorkItemReadOffset(workItemIndexOffset, out int dispatchPairCount);
                var workItemIndex = IterativeSolverSchedulerInfo.FirstWorkItemIndex.Value + workItemIndexOffset;

                BuildJacobians(Input.World, Timestep, GravityMagnitude, NumSubsteps, Input.NumSolverIterations,
                    Input.MaxDynamicDepenetrationVelocity, Input.MaxStaticDepenetrationVelocity, Input.DirectSolverSettings,
                    workItemIndex, DispatchPairs, firstDispatchPairIndex, dispatchPairCount, ref ContactsReader, ref JacobiansWriter,
                    MotorImpulseScale);
            }
        }

        /// <summary>
        /// Build Jacobians from the contacts and joints stored in the simulation context.
        /// </summary>
        internal static void BuildIslandJacobians(in SimulationStepInput input,
            NativeArray<DispatchPairSequencer.DispatchPair> dispatchPairs, in DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo,
            NativeStream.Reader contactsReader, NativeStream.Writer jacobiansWriter)
        {
            var job = new ParallelBuildJacobiansJob
            {
                Input = input,
                Timestep = input.TimeStep,
                NumSubsteps = 1,
                MotorImpulseScale = 1.0f,
                GravityMagnitude = math.length(input.Gravity),
                ContactsReader = contactsReader,
                JacobiansWriter = jacobiansWriter,
                DispatchPairs = dispatchPairs,
                IterativeSolverSchedulerInfo = solverSchedulerInfo.IterativePairsIterativeScheduling,
            };
            for (int i = 0; i < job.IterativeSolverSchedulerInfo.NumWorkItems[0]; i++)
            {
                job.Execute(i);
            }

            const int substeps = SolverConstants.DirectIslandSubsteps;
            job.Timestep = input.TimeStep / substeps;
            job.NumSubsteps = substeps;
            job.MotorImpulseScale = 1.0f / substeps;
            job.IterativeSolverSchedulerInfo = solverSchedulerInfo.DirectPairsIterativeScheduling;
            for (int i = 0; i < job.IterativeSolverSchedulerInfo.NumWorkItems[0]; i++)
            {
                job.Execute(i);
            }
        }

        static unsafe void BuildJacobians(
            in PhysicsWorld world,
            float timestep, // the substep timestep
            float gravityMagnitude,
            int numSubsteps,
            int numSolverIterations,
            float maxDynamicDepenetrationVelocity,
            float maxStaticDepenetrationVelocity,
            in DirectSolverSettings directSolverSettings,
            int workItemIndex,
            NativeArray<DispatchPairSequencer.DispatchPair> dispatchPairs,
            int firstDispatchPairIndex,
            int dispatchPairCount,
            ref NativeStream.Reader contactReader,
            ref NativeStream.Writer jacobianWriter,
            float motorImpulseScale = 1.0f,
            float directTimestep = 0.0f)
        {
            contactReader.BeginForEachIndex(workItemIndex);
            jacobianWriter.BeginForEachIndex(workItemIndex);

            var invTimestep = CalculateInvTimestep(timestep);

            // fall back to defaults if depenetration velocities not set or invalid.
            maxDynamicDepenetrationVelocity = math.select(maxDynamicDepenetrationVelocity,
                SimulationStepInput.DefaultMaxDynamicDepenetrationVelocity, maxDynamicDepenetrationVelocity <= 0);
            maxStaticDepenetrationVelocity = math.select(maxStaticDepenetrationVelocity,
                SimulationStepInput.DefaultMaxStaticDepenetrationVelocity, maxStaticDepenetrationVelocity <= 0);

            // Source: Narrowphase uses predicted velocity to generate dispatchPairs and contactReader data
            for (int i = 0; i < dispatchPairCount; i++)
            {
                var dispatchPairIndex = i + firstDispatchPairIndex;
                var pair = dispatchPairs[dispatchPairIndex];
                if (!pair.IsValid)
                {
                    continue;
                }

                var motionDatas = world.MotionDatas; // Motion data has gravity applied
                var motionVelocities = world.MotionVelocities;
                var bodies = world.Bodies;

                if (pair.IsContact)
                {
                    bool pairHasContacts = false;

                    // At some point during this frame, there MAY be a contact between these bodies
                    while (contactReader.RemainingItemCount > 0)
                    {
                        // Check if this is the matching contact
                        {
                            var header = contactReader.Peek<ContactHeader>();
                            if (pair.BodyIndexA != header.BodyPair.BodyIndexA ||
                                pair.BodyIndexB != header.BodyPair.BodyIndexB)
                            {
                                break;
                            }
                        }

                        ref ContactHeader contactHeader = ref contactReader.Read<ContactHeader>();
                        GetMotions(contactHeader.BodyPair, ref motionDatas, ref motionVelocities,
                            out MotionVelocity velocityA, out MotionVelocity velocityB,
                            out MTransform worldFromA, out MTransform worldFromB);

                        float sumInvMass = velocityA.InverseMass + velocityB.InverseMass;

                        // Note that for the velocity of static bodies, velocity.IsKinematic also returns true since
                        // for static bodies we obtain a MotionVelocity.Zero in GetMotions above. So the check below
                        // is true for pairs of type kinematic / kinematic and kinematic / static.
                        // Note also that static / static pairs are not generated by the collision detection by design
                        // and thus don't appear here.
                        bool bothMotionsAreKinematic = velocityA.IsKinematic && velocityB.IsKinematic;

                        // Skip contact between infinite mass bodies which don't want to raise events. These cannot have any effect during solving.
                        // These should not normally appear, because the collision detector doesn't generate such contacts.
                        if (bothMotionsAreKinematic)
                        {
                            if ((contactHeader.JacobianFlags & (JacobianFlags.IsTrigger | JacobianFlags.EnableCollisionEvents)) == 0)
                            {
                                for (int j = 0; j < contactHeader.NumContacts; j++)
                                {
                                    contactReader.Read<ContactPoint>();
                                }
                                continue;
                            }
                        }
                        else
                        {
                            contactHeader.JacobianFlags |= JacobianFlags.IsContactDynamic;
                        }

                        JacobianType jacType = ((int)(contactHeader.JacobianFlags) & (int)(JacobianFlags.IsTrigger)) != 0 ?
                            JacobianType.Trigger : JacobianType.Contact;

                        int numContacts = contactHeader.NumContacts - math.countbits(contactHeader.RejectedPoints);
                        if (numContacts == 0)
                        {
                            for (int j = 0; j < contactHeader.NumContacts; j++)
                            {
                                contactReader.Read<ContactPoint>();
                            }

                            continue;
                        }

                        JacobianFlags jacFlags = contactHeader.JacobianFlags;

                        if (jacType == JacobianType.Contact && pair.SolverType == SolverType.Direct)
                        {
                            // @todo direct solver: for now we need to set this flag so that the contact manifold is stored in the jacobian,
                            // for later access in the direct solver. Once we build the direct solver's Jacobian matrices in the BuildJacobians phase,
                            // we won't need to do this anymore.
                            jacFlags |= JacobianFlags.EnableCollisionEvents;
                        }

                        // Write size before every jacobian and allocate all necessary data for this jacobian
                        int jacobianSize = JacobianHeader.CalculateSize(jacType, jacFlags, numContacts);
                        jacobianWriter.Write(jacobianSize);
                        byte* jacobianPtr = jacobianWriter.Allocate(jacobianSize);

#if UNITY_ENABLE_CHECKS
                        SafetyChecks.Check4ByteAlignmentAndThrow(jacobianPtr, nameof(jacobianPtr));
#endif
                        ref JacobianHeader jacobianHeader = ref UnsafeUtility.AsRef<JacobianHeader>(jacobianPtr);
                        jacobianHeader.BodyPair = contactHeader.BodyPair;
                        jacobianHeader.Type = jacType;
                        jacobianHeader.Flags = jacFlags;
                        jacobianHeader.ConstraintBlockInfo = new ConstraintBlockInfo { Index = 0, Length = 1};

                        var baseJac = new BaseContactJacobian
                        {
                            NumContacts = numContacts,
                            Normal = contactHeader.Normal
                        };

                        // Body A must be dynamic
                        Assert.IsTrue(contactHeader.BodyPair.BodyIndexA < motionVelocities.Length);
                        bool isDynamicStaticPair = contactHeader.BodyPair.BodyIndexB >= motionVelocities.Length;

                        // If contact distance is negative, use an artificially reduced penetration depth to prevent the
                        // dynamic-dynamic contacts from depenetrating too quickly
                        float maxDepenetrationVelocity = math.select(maxDynamicDepenetrationVelocity, maxStaticDepenetrationVelocity, isDynamicStaticPair);

                        if (jacobianHeader.Type == JacobianType.Contact)
                        {
                            ref ContactJacobian contactJacobian = ref jacobianHeader.AccessBaseJacobian<ContactJacobian>();
                            contactJacobian.BaseJacobian = baseJac;
                            contactJacobian.CoefficientOfFriction = contactHeader.CoefficientOfFriction;
                            contactJacobian.CoefficientOfRestitution = contactHeader.CoefficientOfRestitution;
                            contactJacobian.SumImpulsesOverSubsteps = 0.0f;
                            contactJacobian.SumImpulsesOverSolverIterations = 0.0f;
                            contactJacobian.NormalImpulseOverSubsteps = 0.0f;
                            contactJacobian.FrictionImpulseOverSubsteps = float3.zero;

                            // Write polygons from colliders if they are required for detailed static mesh collision detection.
                            WriteJacobianPolygonData(ref jacobianHeader, bodies, contactHeader, worldFromA, worldFromB, motionVelocities.Length);

                            // Initialize modifier data (in order from JacobianModifierFlags) before angular jacobians
                            InitModifierData(ref jacobianHeader, contactHeader.ColliderKeys, new EntityPair
                            {
                                EntityA = bodies[contactHeader.BodyPair.BodyIndexA].Entity,
                                EntityB = bodies[contactHeader.BodyPair.BodyIndexB].Entity
                            });

                            // Build ContactJacobian for each contact in the dispatch pair
                            var centerA = new float3(0.0f);
                            var centerB = new float3(0.0f);
                            int solveCount = 0; // tracks if a bounce has been solved for the current iteration. Used to update friction
                            int pointIndex = 0;
                            for (int j = 0; j < contactHeader.NumContacts; j++)
                            {
                                ContactPoint point = contactReader.Read<ContactPoint>();
                                if ((contactHeader.RejectedPoints & (1u << j)) != 0)
                                {
                                    continue;
                                }

                                int keptIndex = pointIndex++;
                                ContactJacobian.BuildIndividualContactJacobians(keptIndex, contactJacobian.BaseJacobian.Normal,
                                    worldFromA, worldFromB, invTimestep, numSubsteps, velocityA, velocityB,
                                    sumInvMass, maxDepenetrationVelocity, ref jacobianHeader, ref centerA, ref centerB,
                                    point);

                                if (contactJacobian.CoefficientOfRestitution > 0.0f)
                                {
                                    ref ContactJacAngAndVelToReachCp jacAngular = ref jacobianHeader.AccessAngularJacobian(keptIndex);

                                    float dv = ContactJacobian.CalculateRelativeVelocityAlongNormal(velocityA, velocityB, ref jacAngular,
                                        contactJacobian.BaseJacobian.Normal, out float relativeVelocity);

                                    bool applyRestitution = false;
                                    if (numSubsteps > 1)
                                    {
                                        // Determine if the contact is reached during this substep: if the distance
                                        // travelled by the end of this substep results in a penetration, then we need
                                        // to bounce now.
                                        float distanceTraveled = timestep * relativeVelocity;
                                        float newDistanceToCp = jacAngular.ContactDistance + distanceTraveled;
                                        if (newDistanceToCp < 0.0f)
                                        {
                                            applyRestitution = ContactJacobian.CalculateRestitution(timestep,
                                                gravityMagnitude, contactJacobian.CoefficientOfRestitution,
                                                ref jacAngular, relativeVelocity, jacAngular.ContactDistance, dv);
                                        }

                                        // If VelToReachCp was updated in CalculateRestution, update contact distance to 0.
                                        jacAngular.ContactDistance = applyRestitution ? 0.0f : newDistanceToCp;
                                        if (!applyRestitution) jacAngular.ApplyImpulse = false;
                                    }
                                    else
                                    {
                                        applyRestitution = ContactJacobian.CalculateRestitution(timestep,
                                            gravityMagnitude, contactJacobian.CoefficientOfRestitution,
                                            ref jacAngular, relativeVelocity, jacAngular.ContactDistance, dv);
                                    }

                                    if (applyRestitution) solveCount++;
                                }
                            }

                            contactJacobian.CenterA = centerA;
                            contactJacobian.CenterB = centerB;

                            // Build friction jacobians (skip friction between two infinite-mass objects)
                            if (!bothMotionsAreKinematic)
                            {
                                ContactJacobian.BuildFrictionJacobians(
                                    ref contactJacobian,
                                    ref centerA, ref centerB,
                                    worldFromA, worldFromB,
                                    velocityA, velocityB,
                                    sumInvMass);

                                // Reduce friction by 1/4 if there was restitution applied on any contact point
                                if (solveCount > 0)
                                {
                                    contactJacobian.Friction0.EffectiveMass *= 0.25f;
                                    contactJacobian.Friction1.EffectiveMass *= 0.25f;
                                    contactJacobian.AngularFriction.EffectiveMass *= 0.25f;
                                    contactJacobian.FrictionEffectiveMassOffDiag *= 0.25f;
                                }
                            }
                        }
                        // Much less data needed for triggers
                        else
                        {
                            ref TriggerJacobian triggerJacobian = ref jacobianHeader.AccessBaseJacobian<TriggerJacobian>();

                            triggerJacobian.BaseJacobian = baseJac;
                            triggerJacobian.ColliderKeys = contactHeader.ColliderKeys;
                            triggerJacobian.Entities = new EntityPair
                            {
                                EntityA = bodies[contactHeader.BodyPair.BodyIndexA].Entity,
                                EntityB = bodies[contactHeader.BodyPair.BodyIndexB].Entity
                            };

                            // Build normal jacobians
                            var centerA = new float3(0.0f);
                            var centerB = new float3(0.0f);
                            for (int j = 0; j < contactHeader.NumContacts; j++)
                            {
                                // Build the jacobian
                                ContactJacobian.BuildIndividualContactJacobians(
                                    j, triggerJacobian.BaseJacobian.Normal, worldFromA, worldFromB,
                                    invTimestep, numSubsteps, velocityA, velocityB, sumInvMass, maxDepenetrationVelocity,
                                    ref jacobianHeader, ref centerA, ref centerB, contactReader.Read<ContactPoint>());
                            }
                        }

                        pairHasContacts = true;
                    }

                    if (!pairHasContacts)
                    {
                        // invalidate dispatch pair since it has no contacts and thus shouldn't be processed
                        dispatchPairs[dispatchPairIndex] = DispatchPairSequencer.DispatchPair.Invalid;
                    }
                }
                else
                {
                    // Read the joint in place to avoid copying a large structure that this build does not modify. Scale
                    // motor limits on the local copy of each constraint as it is read.
                    ref Joint joint = ref UnsafeUtility.ArrayElementAsRef<Joint>(world.Joints.GetUnsafeReadOnlyPtr(), pair.JointIndex);

                    // Need to fetch the real body indices from the joint, as the scheduler may have reordered them
                    int bodyIndexA = joint.BodyPair.BodyIndexA;
                    int bodyIndexB = joint.BodyPair.BodyIndexB;

                    GetMotion(world, bodyIndexA, out MotionVelocity velocityA, out MotionData motionA);
                    GetMotion(world, bodyIndexB, out MotionVelocity velocityB, out MotionData motionB);

                    BuildJointJacobian(ref joint, pair.SolverType, velocityA, velocityB, motionA, motionB, timestep,
                        numSolverIterations, directSolverSettings, ref jacobianWriter, directTimestep, motorImpulseScale);
                }
            }

            contactReader.EndForEachIndex();
            jacobianWriter.EndForEachIndex();
        }

        private static unsafe void WriteJacobianPolygonData(ref JacobianHeader jacobianHeader, NativeArray<RigidBody> bodies, in ContactHeader contactHeader, in MTransform worldFromA, in MTransform worldFromB, int dynamicBodiesCount)
        {
            bool isBodyAStatic = contactHeader.BodyPair.BodyIndexA >= dynamicBodiesCount;
            bool isBodyBStatic = contactHeader.BodyPair.BodyIndexB >= dynamicBodiesCount;

            if (jacobianHeader.HasDetailedStaticMeshCollision && (isBodyAStatic || isBodyBStatic))
            {
                ref ContactJacobianPolygonData jacobianPolygonContact = ref jacobianHeader.AccessJacobianContactData();

                // setup conditional variables.
                RigidBody body = default;
                float3 centerOfMass = default; // The center of mass is based on the opposite collider. A over B, and B over A.
                ColliderKey colliderKey = default;

                if (isBodyAStatic)
                {
                    centerOfMass = worldFromB.Translation;
                    body = bodies[contactHeader.BodyPair.BodyIndexA];
                    colliderKey = contactHeader.ColliderKeys.ColliderKeyA;
                }
                else
                {
                    centerOfMass = worldFromA.Translation;
                    body = bodies[contactHeader.BodyPair.BodyIndexB];
                    colliderKey = contactHeader.ColliderKeys.ColliderKeyB;
                }

                // assigning conditional variables
                jacobianPolygonContact.CenterOfMass = centerOfMass;
                jacobianPolygonContact.Transform = new AffineTransform(body.WorldFromBody.pos, body.WorldFromBody.rot, body.Scale);

                Collider* collider = (Collider*)body.Collider.GetUnsafePtr();
                SafetyChecks.CheckAreEqualAndThrow(true, collider != null);
                jacobianPolygonContact.IsValid = collider->GetLeaf(colliderKey, out ChildCollider leafCollider);
                if (jacobianPolygonContact.IsValid)
                {
                    PolygonCollider* polygon = (PolygonCollider*)leafCollider.Collider;
                    jacobianPolygonContact.Vertex0 = polygon->Vertices[0];
                    jacobianPolygonContact.Vertex1 = polygon->Vertices[1];
                    jacobianPolygonContact.Vertex2 = polygon->Vertices[2];
                    jacobianPolygonContact.LeafTransform = new AffineTransform(leafCollider.TransformFromChild);
                    jacobianPolygonContact.IsBodyAStatic = isBodyAStatic;
                    jacobianPolygonContact.IsBodyBStatic = isBodyBStatic;
                }
            }
        }

        #endregion //BuildJacobians

        #region BuildIndividualJacobians

        static void ScaleMotorImpulse(ref Constraint constraint, float scale)
        {
            switch (constraint.Type)
            {
                case ConstraintType.RotationMotor:
                case ConstraintType.AngularVelocityMotor:
                case ConstraintType.PositionMotor:
                case ConstraintType.LinearVelocityMotor:
                    constraint.MaxImpulse *= scale;
                    break;
            }
        }

        internal static unsafe void BuildJointJacobian(ref Joint joint, SolverType solverTypeOverride,
            in MotionVelocity velocityA, in MotionVelocity velocityB, in MotionData motionA, in MotionData motionB,
            float timestep, int numSolverIterations, in DirectSolverSettings directSolverSettings, [NoAlias] ref NativeStream.Writer jacobianWriter,
            float directTimestep = 0.0f, float motorImpulseScale = 1.0f)
        {
            var bodyAFromMotionA = new MTransform(motionA.BodyFromMotion);
            MTransform motionAFromJoint = Mul(Inverse(bodyAFromMotionA), joint.AFromJoint);

            var bodyBFromMotionB = new MTransform(motionB.BodyFromMotion);
            MTransform motionBFromJoint = Mul(Inverse(bodyBFromMotionB), joint.BFromJoint);

            ref var constraintBlock = ref joint.Constraints;
            fixed(void* ptr = &constraintBlock)
            {
                var constraintPtr = (Constraint*)ptr;

                // count effective number of constraints in block
                var numConstraints = 0;
                for (var i = 0; i < constraintBlock.Length; i++)
                {
                    numConstraints += math.select(0, 1, constraintPtr[i].Dimension > 0);
                }

                var constraintIndex = 0;
                bool regularized = false;
                uint regularizedKind = 0;
                Constraint regularizedFrom = default;
                JacobianHeader.DirectSolverRegularizationData regularization = default;
                for (var i = 0; i < constraintBlock.Length; i++)
                {
                    Constraint constraint = constraintPtr[i];
                    if (motorImpulseScale != 1.0f)
                    {
                        ScaleMotorImpulse(ref constraint, motorImpulseScale);
                    }

                    int constraintDimension = constraint.Dimension;
                    if (0 == constraintDimension)
                    {
                        // Unconstrained, so no need to create a header.
                        continue;
                    }

                    JacobianType jacType;
                    switch (constraint.Type)
                    {
                        case ConstraintType.Linear:
                            jacType = JacobianType.LinearLimit;
                            break;
                        case ConstraintType.Angular:
                            switch (constraintDimension)
                            {
                                case 1:
                                    jacType = JacobianType.AngularLimit1D;
                                    break;
                                case 2:
                                    jacType = JacobianType.AngularLimit2D;
                                    break;
                                case 3:
                                    jacType = JacobianType.AngularLimit3D;
                                    break;
                                default:
                                    SafetyChecks.ThrowNotImplementedException();
                                    return;
                            }

                            break;
                        case ConstraintType.RotationMotor:
                            jacType = JacobianType.RotationMotor;
                            break;
                        case ConstraintType.AngularVelocityMotor:
                            jacType = JacobianType.AngularVelocityMotor;
                            break;
                        case ConstraintType.PositionMotor:
                            jacType = JacobianType.PositionMotor;
                            break;
                        case ConstraintType.LinearVelocityMotor:
                            jacType = JacobianType.LinearVelocityMotor;
                            break;
                        default:
                            SafetyChecks.ThrowNotImplementedException();
                            return;
                    }

                    // Write size before every jacobian
                    JacobianFlags jacFlags = constraint.ShouldRaiseImpulseEvents ? JacobianFlags.EnableImpulseEvents : 0;
                    if (joint.ForestRole != JointForestRole.None)
                    {
                        jacFlags |= JacobianFlags.ForestRow | (joint.ForestRole == JointForestRole.Tree ? JacobianFlags.ForestTree : 0)
                            | ((joint.ForestLock & JointForestLock.Lockable) != 0 ? JacobianFlags.ForestLockable : 0)
                            | ((joint.ForestLock & JointForestLock.Holdable) != 0 ? JacobianFlags.ForestHoldable : 0);
                    }
                    int jacobianSize = JacobianHeader.CalculateJointSize(jacType, jacFlags, solverTypeOverride);
                    jacobianWriter.Write(jacobianSize);

                    // Allocate all necessary data for this jacobian
                    byte* jacobianPtr = jacobianWriter.Allocate(jacobianSize);
#if UNITY_ENABLE_CHECKS
                    SafetyChecks.Check4ByteAlignmentAndThrow(jacobianPtr, nameof(jacobianPtr));
#endif
                    ref JacobianHeader header = ref UnsafeUtility.AsRef<JacobianHeader>(jacobianPtr);
                    header.BodyPair = joint.BodyPair;
                    header.Type = jacType;
                    header.Flags = jacFlags;
                    header.ConstraintBlockInfo = new ConstraintBlockInfo { Index = constraintIndex++, Length = numConstraints };

                    // Prepare joint regularization parameters:

                    JacobianUtilities.CalculateConstraintTauAndDamping(constraint.SpringFrequency, constraint.DampingRatio,
                        timestep, numSolverIterations, out float tau, out float damping);

                    if (solverTypeOverride == SolverType.Direct)
                    {
                        // Use the direct solver's substep duration for regularization.
                        // Reuse regularization when the row kind and spring settings match.
                        uint kind = (header.IsAngular ? 1u : 0u) | (header.IsVelocityConstraint ? 2u : 0u);
                        bool same = regularized && kind == regularizedKind
                            && math.asuint(constraint.SpringFrequency) == math.asuint(regularizedFrom.SpringFrequency)
                            && math.asuint(constraint.DampingRatio) == math.asuint(regularizedFrom.DampingRatio)
                            && constraint.HasDriveCoefficients == regularizedFrom.HasDriveCoefficients
                            && math.all(math.asuint(constraint.DriveCoefficients) == math.asuint(regularizedFrom.DriveCoefficients))
                            && math.asuint(constraint.DirectStiffnessScale) == math.asuint(regularizedFrom.DirectStiffnessScale);
                        if (!same)
                        {
                            regularization = JacobianUtilities.CalculateDirectSolverRegularizationData(
                                header, constraint, velocityA, velocityB, directSolverSettings, directTimestep > 0.0f ? directTimestep : timestep);
                            regularized = true;
                            regularizedKind = kind;
                            regularizedFrom = constraint;
                        }

                        header.AccessDirectSolverRegularizationData() = regularization;
                    }

                    // Build the Jacobian:
                    switch (constraint.Type)
                    {
                        case ConstraintType.Linear:
                            header.AccessBaseJacobian<LinearLimitJacobian>().Build(
                                motionAFromJoint, motionBFromJoint, velocityA, velocityB,
                                motionA, motionB, constraint, tau, damping);
                            break;
                        case ConstraintType.Angular:
                            switch (constraintDimension)
                            {
                                case 1:
                                    header.AccessBaseJacobian<AngularLimit1DJacobian>().Build(
                                        motionAFromJoint, motionBFromJoint,
                                        velocityA, velocityB, motionA, motionB, constraint, tau, damping);
                                    break;
                                case 2:
                                    header.AccessBaseJacobian<AngularLimit2DJacobian>().Build(
                                        motionAFromJoint, motionBFromJoint,
                                        velocityA, velocityB, motionA, motionB,
                                        constraint, tau, damping);
                                    break;
                                case 3:
                                    header.AccessBaseJacobian<AngularLimit3DJacobian>().Build(
                                        motionAFromJoint, motionBFromJoint,
                                        velocityA, velocityB, motionA, motionB, constraint, tau, damping);
                                    break;
                                default:
                                    SafetyChecks.ThrowNotImplementedException();
                                    return;
                            }

                            break;
                        case ConstraintType.RotationMotor:
                            header.AccessBaseJacobian<RotationMotorJacobian>().Build(
                                motionAFromJoint, motionBFromJoint,
                                motionA, motionB, constraint, tau, damping);
                            break;
                        case ConstraintType.AngularVelocityMotor:
                            header.AccessBaseJacobian<AngularVelocityMotorJacobian>().Build(
                                motionAFromJoint, motionBFromJoint,
                                motionA, motionB, constraint, tau, damping);
                            break;
                        case ConstraintType.PositionMotor:
                            header.AccessBaseJacobian<PositionMotorJacobian>().Build(
                                motionAFromJoint, motionBFromJoint,
                                motionA, motionB, constraint, tau, damping);
                            break;
                        case ConstraintType.LinearVelocityMotor:
                            header.AccessBaseJacobian<LinearVelocityMotorJacobian>().Build(
                                motionAFromJoint, motionBFromJoint,
                                motionA, motionB, constraint, tau, damping);
                            break;
                        default:
                            SafetyChecks.ThrowNotImplementedException();
                            return;
                    }

                    if ((jacFlags & JacobianFlags.EnableImpulseEvents) != 0)
                    {
                        ref ImpulseEventSolverData impulseEventData = ref header.AccessImpulseEventSolverData();
                        impulseEventData.AccumulatedImpulse = float3.zero;
                        impulseEventData.JointEntity = joint.Entity;
                        impulseEventData.MaxImpulse = math.abs(constraint.MaxImpulse);
                        impulseEventData.IsConfigurableLimit = constraint.IsConfigurableLimit;
                    }
                }
            }
        }

        #endregion //BuildIndividualJacobians

        internal static void UpdateJacobian(ref JacobianHeader header,
            [NoAlias] NativeArray<MotionData> motionDatas,
            [NoAlias] NativeArray<MotionVelocity> motionVelocities,
            [NoAlias] NativeArray<RigidBody> bodies,
            in StepInput stepInput)
        {
            // Static-static pairs should have been filtered during broadphase overlap test
            Assert.IsTrue(header.BodyPair.BodyIndexA < motionDatas.Length || header.BodyPair.BodyIndexB < motionDatas.Length);

            // Update the jacobians
            switch (header.Type)
            {
                case JacobianType.Trigger:
                    break;

                case JacobianType.Contact:
                    GetMotions(header.BodyPair, ref motionDatas, ref motionVelocities,
                        out MotionVelocity velocityA, out MotionVelocity velocityB,
                        out MTransform worldFromA, out MTransform worldFromB);

                    header.UpdateContact(in velocityA, in velocityB, in worldFromA, in worldFromB, stepInput);
                    break;

                default: // for everything else
                    // Get the motion pair
                    MotionData motionDataA = header.BodyPair.BodyIndexA < motionDatas.Length ?
                        motionDatas[header.BodyPair.BodyIndexA] :
                        new MotionData
                    {
                        WorldFromMotion = bodies[header.BodyPair.BodyIndexA].WorldFromBody,
                        BodyFromMotion = RigidTransform.identity
                    };

                    MotionData motionDataB = header.BodyPair.BodyIndexB < motionDatas.Length ?
                        motionDatas[header.BodyPair.BodyIndexB] : new MotionData
                    {
                        WorldFromMotion = bodies[header.BodyPair.BodyIndexB].WorldFromBody,
                        BodyFromMotion = RigidTransform.identity
                    };

                    header.UpdateJoints(in motionDataA, in motionDataB);
                    break;
            }
        }

        [NoAlias]
        internal struct IterativeSolveWork
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [NativeDisableParallelForRestriction]
            public StabilizationData SolverStabilizationData;

            [NoAlias]
            public NativeStream.Reader JacobiansReader;

            [NativeDisableContainerSafetyRestriction]
            [NoAlias]
            public NativeStream.Writer CollisionEventsWriter;

            [NativeDisableContainerSafetyRestriction]
            [NoAlias]
            public NativeStream.Writer TriggerEventsWriter;

            [NativeDisableContainerSafetyRestriction]
            [NoAlias]
            public NativeStream.Writer ImpulseEventsWriter;

            [ReadOnly]
            public DispatchPairSequencer.IterativeSolverSchedulerInfo IterativeSolverSchedulerInfo;

            public int PhaseIndex;
            public StepInput stepInput;

            public void Execute(int phaseWorkItemOffset)
            {
                int firstWorkItemIndex = IterativeSolverSchedulerInfo.FirstWorkItemIndex.Value + IterativeSolverSchedulerInfo.PhaseInfo[PhaseIndex].FirstWorkItemIndexOffset;
                int workItemIndex = firstWorkItemIndex + phaseWorkItemOffset;
                if (stepInput.ExportEventsInThisIteration)
                {
                    CollisionEventsWriter.PatchMinMaxRange(workItemIndex);
                    TriggerEventsWriter.PatchMinMaxRange(workItemIndex);
                    ImpulseEventsWriter.PatchMinMaxRange(workItemIndex);
                }

                IterativeSolverIteration(ref MotionVelocities, JacobiansReader,
                    ref CollisionEventsWriter, ref TriggerEventsWriter, ref ImpulseEventsWriter,
                    workItemIndex, stepInput, SolverStabilizationData);
            }
        }

        static void IterativeSolverIteration(
            ref NativeArray<MotionVelocity> motionVelocities,
            [NoAlias] in NativeStream.Reader jacobianReader,
            [NoAlias] ref NativeStream.Writer collisionEventsWriter,
            [NoAlias] ref NativeStream.Writer triggerEventsWriter,
            [NoAlias] ref NativeStream.Writer impulseEventsWriter,
            int workItemIndex,
            in StepInput stepInput,
            in StabilizationData solverStabilizationData)
        {
            if (stepInput.ExportEventsInThisIteration)
            {
                collisionEventsWriter.BeginForEachIndex(workItemIndex);
                triggerEventsWriter.BeginForEachIndex(workItemIndex);
                impulseEventsWriter.BeginForEachIndex(workItemIndex);
            }

            MotionStabilizationInput motionStabilizationSolverInputA = MotionStabilizationInput.Default;
            MotionStabilizationInput motionStabilizationSolverInputB = MotionStabilizationInput.Default;

            var jacIterator = new JacobianIterator(jacobianReader, workItemIndex);
            while (jacIterator.HasJacobiansLeft())
            {
                ref JacobianHeader header = ref jacIterator.ReadJacobianHeader();

                // Static-static pairs should have been filtered during broadphase overlap test
                SafetyChecks.CheckAreEqualAndThrow(true, header.BodyPair.BodyIndexA < motionVelocities.Length || header.BodyPair.BodyIndexB < motionVelocities.Length);

                // Get the motion pair
                MotionVelocity velocityA = header.BodyPair.BodyIndexA < motionVelocities.Length ?
                    motionVelocities[header.BodyPair.BodyIndexA] : MotionVelocity.Zero;
                MotionVelocity velocityB = header.BodyPair.BodyIndexB < motionVelocities.Length ?
                    motionVelocities[header.BodyPair.BodyIndexB] : MotionVelocity.Zero;

                // Relaxation removes the separation velocity added by contacts. Skip other constraints to preserve the
                // results of the earlier solver iterations.
                if (stepInput.Relaxing && header.Type != JacobianType.Contact)
                {
                    continue;
                }

                // For Contacts Only: Populate the Solver stabilization data
                if (solverStabilizationData.StabilizationHeuristicSettings.EnableSolverStabilization
                    && header.Type == JacobianType.Contact)
                {
                    SolverStabilizationUpdate(ref header, stepInput.IsFirstSolverIteration, velocityA, velocityB,
                        solverStabilizationData, ref motionStabilizationSolverInputA, ref motionStabilizationSolverInputB);
                }

                // Solve the jacobian
                header.Solve(ref velocityA, ref velocityB, stepInput,
                    ref collisionEventsWriter, ref triggerEventsWriter, ref impulseEventsWriter,
                    solverStabilizationData.StabilizationHeuristicSettings.EnableSolverStabilization &&
                    solverStabilizationData.StabilizationHeuristicSettings.EnableFrictionVelocities,
                    motionStabilizationSolverInputA, motionStabilizationSolverInputB);

                // Write back velocity for dynamic bodies
                if (header.BodyPair.BodyIndexA < motionVelocities.Length)
                {
                    motionVelocities[header.BodyPair.BodyIndexA] = velocityA;
                }
                if (header.BodyPair.BodyIndexB < motionVelocities.Length)
                {
                    motionVelocities[header.BodyPair.BodyIndexB] = velocityB;
                }
            }

            if (stepInput.ExportEventsInThisIteration)
            {
                collisionEventsWriter.EndForEachIndex();
                triggerEventsWriter.EndForEachIndex();
                impulseEventsWriter.EndForEachIndex();
            }
        }

        #region Events

        [BurstCompile]
        internal struct ParallelExportEventsJob : IJobParallelForDefer
        {
            public NativeStream.Reader JacobiansReader;

            [NativeDisableContainerSafetyRestriction]
            public NativeArray<MotionVelocity> MotionVelocities;

            [ReadOnly]
            public NativeArray<MotionData> MotionDatas;

            public NativeStream.Writer CollisionEventsWriter;
            public NativeStream.Writer TriggerEventsWriter;
            public NativeStream.Writer ImpulseEventsWriter;

            [ReadOnly]
            public DispatchPairSequencer.IterativeSolverSchedulerInfo IterativeSolverSchedulerInfo;

            public float Timestep;

            public void Execute(int workItemIndexOffset)
            {
                var workItemIndex = IterativeSolverSchedulerInfo.FirstWorkItemIndex.Value + workItemIndexOffset;

                CollisionEventsWriter.PatchMinMaxRange(workItemIndex);
                TriggerEventsWriter.PatchMinMaxRange(workItemIndex);
                ImpulseEventsWriter.PatchMinMaxRange(workItemIndex);

                ExportEvents(workItemIndex, Timestep, JacobiansReader, MotionVelocities, MotionDatas,
                    ref CollisionEventsWriter, ref TriggerEventsWriter, ref ImpulseEventsWriter);
            }
        }

        static void ExportEvents(int workItemIndex, float timestep, in NativeStream.Reader jacobiansReader,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas,
            ref NativeStream.Writer collisionEventsWriter, ref NativeStream.Writer triggerEventsWriter, ref NativeStream.Writer impulseEventsWriter)
        {
            collisionEventsWriter.BeginForEachIndex(workItemIndex);
            triggerEventsWriter.BeginForEachIndex(workItemIndex);
            impulseEventsWriter.BeginForEachIndex(workItemIndex);

            var iterator = new JacobianIterator(jacobiansReader, workItemIndex);
            while (iterator.HasJacobiansLeft())
            {
                ref var header = ref iterator.ReadJacobianHeader();

                switch (header.Type)
                {
                    case JacobianType.Contact:
                        if ((header.Flags & JacobianFlags.EnableCollisionEvents) != 0)
                        {
                            ref var contactJac = ref header.AccessBaseJacobian<ContactJacobian>();

                            if (!header.IsContactDynamic)
                            {
                                MotionVelocity velA = header.BodyPair.BodyIndexA < motionVelocities.Length ?
                                    motionVelocities[header.BodyPair.BodyIndexA] : MotionVelocity.Zero;
                                MotionVelocity velB = header.BodyPair.BodyIndexB < motionVelocities.Length ?
                                    motionVelocities[header.BodyPair.BodyIndexB] : MotionVelocity.Zero;

                                contactJac.ExportInfMassEvent(ref header, velA, velB, timestep, ref collisionEventsWriter);
                            }
                            else
                            {
                                bool forceCollisionEvent = false;
                                for (int i = 0; i < contactJac.BaseJacobian.NumContacts; ++i)
                                {
                                    ref var contactAngularJac = ref header.AccessAngularJacobian(i);

                                    // Force contact event even when no impulse is applied, but there is penetration.
                                    // This includes kinematic/kinematic or kinematic/static pairs.
                                    forceCollisionEvent |= contactAngularJac.VelToReachCp > 0.0f;
                                }

                                if (contactJac.SumImpulsesOverSubsteps > 0.0f || forceCollisionEvent)
                                {
                                    contactJac.ExportCollisionEvent(contactJac.SumImpulsesOverSubsteps, ref header, ref collisionEventsWriter);
                                }
                            }
                        }
                        break;

                    case JacobianType.Trigger:
                        MotionVelocity velocityA = header.BodyPair.BodyIndexA < motionVelocities.Length ?
                            motionVelocities[header.BodyPair.BodyIndexA] : MotionVelocity.Zero;
                        MotionVelocity velocityB = header.BodyPair.BodyIndexB < motionVelocities.Length ?
                            motionVelocities[header.BodyPair.BodyIndexB] : MotionVelocity.Zero;

                        ref var triggerJac = ref header.AccessBaseJacobian<TriggerJacobian>();
                        triggerJac.ExportEvent(ref header, velocityA, velocityB, ref triggerEventsWriter);

                        break;

                    case JacobianType.LinearLimit:
                    case JacobianType.AngularLimit1D:
                    case JacobianType.AngularLimit2D:
                    case JacobianType.AngularLimit3D:
                        if ((header.Flags & JacobianFlags.EnableImpulseEvents) != 0)
                        {
                            ref ImpulseEventSolverData impulseEventData = ref header.AccessImpulseEventSolverData();
                            bool linear = header.Type == JacobianType.LinearLimit;
                            float3 impulse = impulseEventData.AccumulatedImpulse;
                            // As in the iterative kernel, linear thresholds use world axes. Angular thresholds
                            // use the joint axes. Event consumers read linear impulses in frame B and angular
                            // impulses in frame A, using the poses at the end of the step.
                            if (linear && !math.any(math.abs(impulse) > impulseEventData.MaxImpulse))
                            {
                                break;
                            }

                            impulse = math.rotate(math.conjugate(ImpulseEventFrame(ref header, motionDatas)), impulse);
                            if (linear || math.any(math.abs(impulse) > impulseEventData.MaxImpulse))
                            {
                                impulseEventsWriter.Write(new ImpulseEventData
                                {
                                    Type = linear ? ConstraintType.Linear : ConstraintType.Angular,
                                    IsConfigurableLimit = impulseEventData.IsConfigurableLimit,
                                    Impulse = impulse,
                                    JointEntity = impulseEventData.JointEntity,
                                    BodyIndices = header.BodyPair
                                });
                            }
                        }
                        break;
                    default:
                        // Nothing to do for motor joints. These don't support events.
                        break;
                }
            }

            collisionEventsWriter.EndForEachIndex();
            triggerEventsWriter.EndForEachIndex();
            impulseEventsWriter.EndForEachIndex();
        }

        static quaternion ImpulseEventFrame(ref JacobianHeader header, in NativeArray<MotionData> motionDatas)
        {
            int body = header.BodyPair.BodyIndexA;
            quaternion worldFromMotion;
            quaternion motionFromJoint;
            switch (header.Type)
            {
                case JacobianType.LinearLimit:
                    ref var linear = ref header.AccessBaseJacobian<LinearLimitJacobian>();
                    body = header.BodyPair.BodyIndexB;
                    worldFromMotion = linear.WorldFromB.rot;
                    motionFromJoint = new quaternion(linear.BodyFromConstraintB.Rotation);
                    break;
                case JacobianType.AngularLimit1D:
                    ref var twist = ref header.AccessBaseJacobian<AngularLimit1DJacobian>();
                    worldFromMotion = twist.WorldFromMotionA;
                    motionFromJoint = twist.MotionAFromJoint;
                    break;
                case JacobianType.AngularLimit2D:
                    ref var swing = ref header.AccessBaseJacobian<AngularLimit2DJacobian>();
                    worldFromMotion = swing.WorldFromA.rot;
                    motionFromJoint = new quaternion(swing.AnchorFrameInBodyA.Rotation);
                    break;
                default:
                    ref var orientation = ref header.AccessBaseJacobian<AngularLimit3DJacobian>();
                    worldFromMotion = orientation.WorldFromA.rot;
                    motionFromJoint = new quaternion(orientation.AnchorFrameInBodyA.Rotation);
                    break;
            }

            if (body < motionDatas.Length)
            {
                worldFromMotion = motionDatas[body].WorldFromMotion.rot;
            }

            return math.mul(worldFromMotion, motionFromJoint);
        }

        #endregion // Events

        #region ApplyGravityAndUpdateInputVelocities

        /// Schedules jobs that apply gravity to all dynamic bodies and update InputVelocities. This method:
        /// 1) updates MotionVelocities with the gravity integrated linear velocity.
        /// 2) updates InputVelocities with the new MotionVelocities Linear and Angular data.
        internal static JobHandle ScheduleApplyGravityAndUpdateInputVelocitiesJob(ref DynamicsWorld world, NativeArray<Velocity> inputVelocities,
            float3 gravityAcceleration, float timestep, bool enableGyroscopicTorque, JobHandle inputDeps, bool multiThreaded = true)
        {
            if (!multiThreaded)
            {
                var job = new ApplyGravityAndUpdateInputVelocitiesJob
                {
                    MotionVelocities = world.MotionVelocities,
                    InputVelocities = inputVelocities,
                    GravityAcceleration = gravityAcceleration,
                    Timestep = timestep,
                    EnableGyroscopicTorque = enableGyroscopicTorque
                };
                return job.Schedule(inputDeps);
            }
            else
            {
                var job = new ParallelApplyGravityAndUpdateInputVelocitiesJob
                {
                    MotionVelocities = world.MotionVelocities,
                    InputVelocities = inputVelocities,
                    GravityAcceleration = gravityAcceleration,
                    Timestep = timestep,
                    EnableGyroscopicTorque = enableGyroscopicTorque
                };
                return job.Schedule(world.MotionVelocities.Length, 64, inputDeps);
            }
        }

        [BurstCompile]
        private struct ApplyGravityAndUpdateInputVelocitiesJob : IJob
        {
            public NativeArray<MotionVelocity> MotionVelocities;
            public NativeArray<Velocity> InputVelocities;
            public float3 GravityAcceleration;
            public float Timestep;
            public bool EnableGyroscopicTorque;

            public void Execute()
            {
                ApplyGravityAndUpdateInputVelocities(MotionVelocities, InputVelocities, GravityAcceleration, Timestep, EnableGyroscopicTorque);
            }
        }

        /// <summary>   Apply gravity to all dynamic bodies and update InputVelocities. This method:
        /// 1) updates MotionVelocities with the gravity integrated linear velocity.
        /// 2) updates InputVelocities with the new MotionVelocities Linear and Angular data.
        /// </summary>
        internal static void ApplyGravityAndUpdateInputVelocities(NativeArray<MotionVelocity> motionVelocities,
            NativeArray<Velocity> inputVelocities, float3 gravityAcceleration, float timestep, bool enableGyroscopicTorque)
        {
            for (int i = 0; i < motionVelocities.Length; i++)
            {
                ParallelApplyGravityAndUpdateInputVelocitiesJob.ExecuteImpl(i, motionVelocities, inputVelocities,
                    gravityAcceleration, timestep, enableGyroscopicTorque);
            }
        }

        [BurstCompile]
        private struct ParallelApplyGravityAndUpdateInputVelocitiesJob : IJobParallelFor
        {
            public NativeArray<MotionVelocity> MotionVelocities;
            public NativeArray<Velocity> InputVelocities;
            public float3 GravityAcceleration;
            public float Timestep;
            public bool EnableGyroscopicTorque;

            public void Execute(int i)
            {
                ExecuteImpl(i, MotionVelocities, InputVelocities, GravityAcceleration, Timestep, EnableGyroscopicTorque);
            }

            internal static void ExecuteImpl(int i, NativeArray<MotionVelocity> motionVelocities,
                NativeArray<Velocity> inputVelocities, float3 gravityAcceleration, float timestep, bool enableGyroscopicTorque)
            {
                MotionVelocity motionVelocity = motionVelocities[i];

                // Apply gravity
                motionVelocity.LinearVelocity += gravityAcceleration * motionVelocity.GravityFactor * timestep;

                if (enableGyroscopicTorque && !motionVelocity.IsKinematic)
                {
                    // Apply gyroscopic torque to generalized velocity.
                    ApplyGyroscopicTorque(ref motionVelocity, timestep);
                }

                // Write back
                motionVelocities[i] = motionVelocity;

                // Make a copy
                inputVelocities[i] = new Velocity
                {
                    Linear = motionVelocity.LinearVelocity,
                    Angular = motionVelocity.AngularVelocity
                };
            }
        }

        #endregion // ApplyGravityAndUpdateInputVelocities

        #region StabilizationHeuristics

        private static void SolverStabilizationUpdate(
            ref JacobianHeader header, bool isFirstIteration,
            MotionVelocity velocityA, MotionVelocity velocityB,
            StabilizationData solverStabilizationData,
            ref MotionStabilizationInput motionStabilizationSolverInputA,
            ref MotionStabilizationInput motionStabilizationSolverInputB)
        {
            // Solver stabilization heuristic, count pairs and reset inverse inertia scale only in first iteration
            var inputVelocities = solverStabilizationData.InputVelocities;
            var motionData = solverStabilizationData.MotionData;
            if (isFirstIteration)
            {
                // Only count heavier (or up to 2 times lighter) bodies as pairs
                // Also reset inverse inertia scale
                if (header.BodyPair.BodyIndexA < motionData.Length)
                {
                    var data = motionData[header.BodyPair.BodyIndexA];
                    if (0.5f * velocityB.InverseMass <= velocityA.InverseMass)
                    {
                        data.NumPairs++;
                    }
                    data.InverseInertiaScale = 1.0f;
                    motionData[header.BodyPair.BodyIndexA] = data;
                }
                if (header.BodyPair.BodyIndexB < motionData.Length)
                {
                    var data = motionData[header.BodyPair.BodyIndexB];
                    if (0.5f * velocityA.InverseMass <= velocityB.InverseMass)
                    {
                        data.NumPairs++;
                    }
                    data.InverseInertiaScale = 1.0f;
                    motionData[header.BodyPair.BodyIndexB] = data;
                }
            }

            // Motion solver input stabilization data
            {
                if (solverStabilizationData.StabilizationHeuristicSettings.EnableFrictionVelocities)
                {
                    motionStabilizationSolverInputA.InputVelocity = header.BodyPair.BodyIndexA < inputVelocities.Length ?
                        inputVelocities[header.BodyPair.BodyIndexA] : Velocity.Zero;
                    motionStabilizationSolverInputB.InputVelocity = header.BodyPair.BodyIndexB < inputVelocities.Length ?
                        inputVelocities[header.BodyPair.BodyIndexB] : Velocity.Zero;
                }

                motionStabilizationSolverInputA.InverseInertiaScale = header.BodyPair.BodyIndexA < motionData.Length ?
                    motionData[header.BodyPair.BodyIndexA].InverseInertiaScale : 1.0f;
                motionStabilizationSolverInputB.InverseInertiaScale = header.BodyPair.BodyIndexB < motionData.Length ?
                    motionData[header.BodyPair.BodyIndexB].InverseInertiaScale : 1.0f;
            }
        }

        #endregion // StabilizationHeuristics

        #region Implementation

        private static void InitModifierData(ref JacobianHeader jacobianHeader, ColliderKeyPair colliderKeys, EntityPair entities)
        {
            if (jacobianHeader.HasContactManifold)
            {
                jacobianHeader.AccessColliderKeys() = colliderKeys;
                jacobianHeader.AccessEntities() = entities;
            }
            if (jacobianHeader.HasSurfaceVelocity)
            {
                jacobianHeader.AccessSurfaceVelocity() = new SurfaceVelocity();
            }
            if (jacobianHeader.HasMassFactors)
            {
                jacobianHeader.AccessMassFactors() = MassFactors.Default;
            }
        }

        private static void GetMotions(
            BodyIndexPair pair,
            ref NativeArray<MotionData> motionDatas,
            ref NativeArray<MotionVelocity> motionVelocities,
            out MotionVelocity velocityA,
            out MotionVelocity velocityB,
            out MTransform worldFromA,
            out MTransform worldFromB)
        {
            bool bodyAIsStatic = pair.BodyIndexA >= motionVelocities.Length;
            bool bodyBIsStatic = pair.BodyIndexB >= motionVelocities.Length;

            if (bodyAIsStatic)
            {
                if (bodyBIsStatic)
                {
                    Assert.IsTrue(false); // static-static pairs should have been filtered during broadphase overlap test
                    velocityA = MotionVelocity.Zero;
                    velocityB = MotionVelocity.Zero;
                    worldFromA = MTransform.Identity;
                    worldFromB = MTransform.Identity;
                    return;
                }

                velocityA = MotionVelocity.Zero;
                velocityB = motionVelocities[pair.BodyIndexB];

                worldFromA = MTransform.Identity;
                worldFromB = new MTransform(motionDatas[pair.BodyIndexB].WorldFromMotion);
            }
            else if (bodyBIsStatic)
            {
                velocityA = motionVelocities[pair.BodyIndexA];
                velocityB = MotionVelocity.Zero;

                worldFromA = new MTransform(motionDatas[pair.BodyIndexA].WorldFromMotion);
                worldFromB = MTransform.Identity;
            }
            else
            {
                velocityA = motionVelocities[pair.BodyIndexA];
                velocityB = motionVelocities[pair.BodyIndexB];

                worldFromA = new MTransform(motionDatas[pair.BodyIndexA].WorldFromMotion);
                worldFromB = new MTransform(motionDatas[pair.BodyIndexB].WorldFromMotion);
            }
        }

        // Gets a body's motion, even if the body is static
        private static void GetMotion([NoAlias] in PhysicsWorld world, int bodyIndex,
            [NoAlias] out MotionVelocity velocity, [NoAlias] out MotionData motion)
        {
            if (bodyIndex >= world.MotionVelocities.Length)
            {
                // Body is static
                velocity = MotionVelocity.Zero;
                motion = new MotionData
                {
                    WorldFromMotion = world.Bodies[bodyIndex].WorldFromBody,
                    BodyFromMotion = RigidTransform.identity
                        // remaining fields all zero
                };
            }
            else
            {
                // Body is dynamic
                velocity = world.MotionVelocities[bodyIndex];
                motion = world.MotionDatas[bodyIndex];
            }
        }

        #endregion // Implementation
    }
}
