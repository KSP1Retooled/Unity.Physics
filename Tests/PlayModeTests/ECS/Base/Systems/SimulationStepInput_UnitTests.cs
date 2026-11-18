using System.Diagnostics;
using NUnit.Framework;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics.Systems;
using Unity.Collections;
using Unity.Jobs;

namespace Unity.Physics.Tests.Systems
{
    class SimulationStepInput_UnitTests
    {
        [Test]
        public void VerifySetSolverIterations([Values(0, 2)] int numSolverIterations)
        {
            float timeStep = 0.5f;
            using (var world = new World("Test world"))
            {
                // create the build physics world system
                var buildPhysicsWorld = world.GetOrCreateSystem<BuildPhysicsWorld>();
                ref var bpwData = ref world.EntityManager.GetComponentDataRW<BuildPhysicsWorldData>(buildPhysicsWorld)
                    .ValueRW;

                var simulationStepInput = new SimulationStepInput()
                {
                    World = bpwData.PhysicsData.PhysicsWorld,
                    TimeStep = timeStep,
                    Gravity = new float3(0, -9.81f, 0),
                    SynchronizeCollisionWorld = false,
                    NumSolverIterations = numSolverIterations,
                    SolverStabilizationHeuristicSettings = Solver.StabilizationHeuristicSettings.Default,
                    HaveStaticBodiesChanged = new NativeReference<int>(1, Allocator.Temp)
                };

                Assert.AreEqual(numSolverIterations, simulationStepInput.NumSolverIterations);
            }
        }

        [Test]
        public void VerifyDefaultPhysicsStepSolverIterations()
        {
            using (var world = new World("Test world"))
            {
                // create the build physics world system
                var buildPhysicsWorld = world.GetOrCreateSystem<BuildPhysicsWorld>();

                // add physics step component and set collision tolerance
                var physicsStepEntity = world.EntityManager.CreateSingleton<PhysicsStep>();
                world.EntityManager.SetComponentData(physicsStepEntity, PhysicsStep.Default);

                buildPhysicsWorld.Update(world.Unmanaged);
                var jobHandle = world.Unmanaged.ResolveSystemStateRef(buildPhysicsWorld).Dependency;
                jobHandle.Complete();
                Assert.IsTrue(jobHandle.IsCompleted);

                var physicsStep = world.EntityManager.GetComponentData<PhysicsStep>(physicsStepEntity);
                Assert.That(physicsStep.SolverIterationCount, Is.EqualTo(4));
            }
        }

#if (UNITY_EDITOR)
        [Test]
        [Conditional(CompilationSymbols.CollectionsChecksSymbol)]
        public void VerifyStepImmediateSetSolverIterations([Values(0, 2)] int numSolverIterations)
        {
            float timeStep = 0.02f;
            using (var world = new World("Test world"))
            {
                // create the build physics world system
                var buildPhysicsWorld = world.GetOrCreateSystem<BuildPhysicsWorld>();
                ref var bpwData = ref world.EntityManager.GetComponentDataRW<BuildPhysicsWorldData>(buildPhysicsWorld)
                    .ValueRW;
                var simulationStepInput = new SimulationStepInput()
                {
                    World = bpwData.PhysicsData.PhysicsWorld,
                    TimeStep = timeStep,
                    Gravity = new float3(0, -9.81f, 0),
                    SynchronizeCollisionWorld = false,
                    NumSolverIterations = numSolverIterations,
                    SolverStabilizationHeuristicSettings = Solver.StabilizationHeuristicSettings.Default,
                    HaveStaticBodiesChanged = new NativeReference<int>(1, Allocator.Temp)
                };

                var context = new SimulationContext();
                if (numSolverIterations > 0)
                {
                    Simulation.StepImmediate(simulationStepInput, ref context); //should not throw an exception
                }
                else
                {
                    // numSolverIterations will be set to 1 and will throw an error.
                    Assert.That(() => Simulation.StepImmediate(simulationStepInput, ref context),
                        Throws.Exception.TypeOf<System.ArgumentOutOfRangeException>());
                }
            }
        }

        [Test]
        [Conditional(CompilationSymbols.CollectionsChecksSymbol)]
        public void VerifyStepImmediateWithUnspecifiedSolverIterations()
        {
            using (var world = new World("Test world"))
            {
                // create the build physics world system
                var buildPhysicsWorld = world.GetOrCreateSystem<BuildPhysicsWorld>();
                ref var bpwData = ref world.EntityManager.GetComponentDataRW<BuildPhysicsWorldData>(buildPhysicsWorld)
                    .ValueRW;
                var simulationStepInput = new SimulationStepInput()
                {
                    World = bpwData.PhysicsData.PhysicsWorld,
                    TimeStep = 0.02f,
                    Gravity = new float3(0, -9.81f, 0),
                    SynchronizeCollisionWorld = false,
                    //NumSolverIterations = DNE,
                    SolverStabilizationHeuristicSettings = Solver.StabilizationHeuristicSettings.Default,
                    HaveStaticBodiesChanged = new NativeReference<int>(1, Allocator.Temp)
                };

                Assert.That(simulationStepInput.NumSolverIterations, Is.EqualTo(0));
                var context = new SimulationContext();

                // numSolverIterations will be set to 1 and will throw an error.
                Assert.That(() => Simulation.StepImmediate(simulationStepInput, ref context),
                    Throws.Exception.TypeOf<System.ArgumentOutOfRangeException>());
            }
        }

        [Test]
        [Conditional(CompilationSymbols.CollectionsChecksSymbol)]
        public void VerifyScheduleStepJobsWithUnspecifiedSolverIterations()
        {
            using (var world = new World("Test world"))
            {
                // create the build physics world system
                var buildPhysicsWorld = world.GetOrCreateSystem<BuildPhysicsWorld>();
                ref var bpwData = ref world.EntityManager.GetComponentDataRW<BuildPhysicsWorldData>(buildPhysicsWorld)
                    .ValueRW;

                var stepInput = new SimulationStepInput()
                {
                    World = bpwData.PhysicsData.PhysicsWorld,
                    TimeStep = 0.02f,
                    Gravity = new float3(0, -9.81f, 0),
                    SynchronizeCollisionWorld = false,
                    //NumSolverIterations = DNE,
                    SolverStabilizationHeuristicSettings = Solver.StabilizationHeuristicSettings.Default,
                    HaveStaticBodiesChanged = new NativeReference<int>(1, Allocator.Temp)
                };

                var simulation = Simulation.Create();
                var handles = new SimulationJobHandles(new JobHandle());

                // numSolverIterations will be set to 1 and will throw an error.
                Assert.That(() => simulation.ScheduleStepJobs(stepInput, default, false),
                    Throws.Exception.TypeOf<System.ArgumentOutOfRangeException>());
            }
        }

        [Test]
        [Conditional(CompilationSymbols.CollectionsChecksSymbol)]
        public void StepImmediateAppliesGravityAndDamping()
        {
            float timeStep = 0.02f;
            using (var world = new PhysicsWorld(0,1,0))
            {
                var body = new RigidTransform(quaternion.identity, new float3(0,10,0));

                var geometry = new BoxGeometry
                {
                    Center = float3.zero,
                    Orientation = quaternion.identity,
                    Size = new float3(1,1,1),
                    BevelRadius = 0.0f
                };
                using var collider = BoxCollider.Create(geometry);
                var rigidBody = new RigidBody {WorldFromBody = body, Collider = collider, Scale = 1.0f};

                var velocity = new MotionVelocity
                {
                    LinearVelocity = float3.zero,
                    AngularVelocity = float3.zero,
                    InverseInertia = new float3(6f, 6f, 6f),
                    InverseMass = 1.0f,
                    AngularExpansionFactor = 0.692820311f,
                    GravityFactor = 1.0f
                };

                var motionData = new MotionData
                {
                    WorldFromMotion = body,
                    BodyFromMotion = RigidTransform.identity,
                    LinearDamping = 0.5f,
                    AngularDamping = 0.01f
                };

                //----------------------------------------------------------------------
                var bodies = world.Bodies;
                bodies[0] = rigidBody;

                var velocities = world.MotionVelocities;
                velocities[0] = velocity;

                var motionDatas = world.MotionDatas;
                motionDatas[0] = motionData;

                var simulationStepInput = new SimulationStepInput()
                {
                    World = world,
                    TimeStep = timeStep,
                    Gravity = new float3(0, -10.0f, 0),
                    SynchronizeCollisionWorld = false,
                    NumSolverIterations = 3,
                    SolverStabilizationHeuristicSettings = Solver.StabilizationHeuristicSettings.Default,
                    HaveStaticBodiesChanged = new NativeReference<int>(1, Allocator.Temp)
                };
                var context = new SimulationContext();
                context.Reset(simulationStepInput);

                //-----------------------------------------------------------------------
                int frames = 20;
                for (var i = 0; i < frames; i++)
                {
                    simulationStepInput.World.CollisionWorld.BuildBroadphase(ref simulationStepInput.World,
                        simulationStepInput.TimeStep, simulationStepInput.Gravity);

                    Simulation.StepImmediate(simulationStepInput, ref context);
                }

                var vFinal = world.MotionVelocities[0].LinearVelocity.y;

                var mass = math.rcp(velocity.InverseMass);
                var gravity = simulationStepInput.Gravity.y;
                var vNoDamping = mass * gravity * timeStep * frames;
                Assert.That(vFinal, Is.GreaterThan(vNoDamping).And.LessThan(0));

                var dStart = motionData.WorldFromMotion.pos.y;
                var dFinal = world.MotionDatas[0].WorldFromMotion.pos.y;
                Assert.That(dFinal, Is.LessThan(dStart));
                context.Dispose();
            }
        }

#endif
    }
}
