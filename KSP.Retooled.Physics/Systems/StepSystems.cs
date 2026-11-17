using KSP.Retooled.Physics;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Systems;

namespace KSP.Retooled.Systems.Physics;

// Backend step stages placed around Unity Physics systems. Register PhysicsCollider writes to join
// simulation dependencies; access body state through store rows.

/// <summary>Integrates the forces and changes in velocity applied to bodies since the last step.</summary>
[BurstCompile]
[UpdateInGroup(typeof(BeforePhysicsSystemGroup))]
internal partial struct ApplyForcesSystem : ISystem
{
    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        StepKernel.ApplyForces(state.WorldUnmanaged.Time.DeltaTime);
    }
}

/// <summary>
/// Replaces <see cref="BuildPhysicsWorld"/> and builds the world from the store instead of an entity query.
/// </summary>
[BurstCompile]
[UpdateInGroup(typeof(PhysicsInitializeGroup))]
[CreateAfter(typeof(BuildPhysicsWorld))]
[CreateAfter(typeof(ExportPhysicsWorld))]
internal unsafe partial struct BuildBodyWorldSystem : ISystem
{
    private EntityQuery settings;
    private EntityQuery worlds;

    // Cache the flag reference so that reading BuildPhysicsWorldData does not add a build dependency.
    private NativeReference<int> staticsChanged;

    public void OnCreate(ref SystemState state)
    {
        state.EntityManager.CreateSingleton(PhysicsStep.Default);
        state.GetEntityQuery(ComponentType.ReadWrite<PhysicsCollider>());
        settings = state.GetEntityQuery(ComponentType.ReadWrite<PhysicsStep>());
        worlds = state.GetEntityQuery(ComponentType.ReadWrite<PhysicsWorldSingleton>());

        SystemHandle stockBuild = state.WorldUnmanaged.GetExistingUnmanagedSystem<BuildPhysicsWorld>();
        staticsChanged = state.EntityManager.GetComponentData<BuildPhysicsWorldData>(stockBuild).HaveStaticBodiesChanged;
        DisableStockSystem<BuildPhysicsWorld>(ref state);
        DisableStockSystem<ExportPhysicsWorld>(ref state);
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        JobHandle dependency = state.Dependency;
        StepKernel.BuildWorld(state.WorldUnmanaged.Time.DeltaTime, &dependency);
        state.Dependency = dependency;

        settings.SetSingleton(StepKernel.Current->Settings);
        worlds.GetSingletonRW<PhysicsWorldSingleton>().ValueRW.PhysicsWorld = StepKernel.Current->World;
        staticsChanged.Value = StepKernel.Current->StaticsChanged;
    }

    private static void DisableStockSystem<T>(ref SystemState state)
        where T : unmanaged, ISystem
    {
        SystemHandle system = state.WorldUnmanaged.GetExistingUnmanagedSystem<T>();
        state.WorldUnmanaged.ResolveSystemStateRef(system).Enabled = false;
    }
}

/// <summary>
/// Sorts the pairs from the broadphase into islands after pairs that will not be solved have been removed.
/// </summary>
[BurstCompile]
[UpdateInGroup(typeof(PhysicsCreateBodyPairsGroup), OrderLast = true)]
internal unsafe partial struct SortIslandsSystem : ISystem
{
    private EntityQuery simulations;
    private EntityQuery inputs;

    public void OnCreate(ref SystemState state)
    {
        state.GetEntityQuery(ComponentType.ReadWrite<PhysicsCollider>());
        simulations = state.GetEntityQuery(ComponentType.ReadWrite<SimulationSingleton>());
        inputs = new EntityQueryBuilder(Allocator.Temp).WithAllRW<StepInputSingleton>().WithOptions(EntityQueryOptions.IncludeSystems).Build(ref state);
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        SimulationSingleton singleton = simulations.GetSingleton<SimulationSingleton>();
        ref StepInputSingleton input = ref inputs.GetSingletonRW<StepInputSingleton>().ValueRW;
        JobHandle dependency = state.Dependency;
        StepKernel.SortIslands(&singleton, (SimulationStepInput*)UnsafeUtility.AddressOf(ref input.StepInput), &dependency);
        state.Dependency = dependency;
    }
}

/// <summary>
/// Reads contacts after the narrow phase has generated them and before Jacobians are built from them.
/// </summary>
[BurstCompile]
[UpdateInGroup(typeof(PhysicsCreateJacobiansGroup), OrderFirst = true)]
internal unsafe partial struct ReadContactsSystem : ISystem
{
    private EntityQuery simulations;

    public void OnCreate(ref SystemState state)
    {
        state.GetEntityQuery(ComponentType.ReadWrite<PhysicsCollider>());
        simulations = state.GetEntityQuery(ComponentType.ReadWrite<SimulationSingleton>());
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        SimulationSingleton singleton = simulations.GetSingleton<SimulationSingleton>();
        JobHandle dependency = state.Dependency;
        StepKernel.ReadContacts(&singleton, state.WorldUnmanaged.Time.DeltaTime, &dependency);
        state.Dependency = dependency;
    }
}

/// <summary>Initializes the Jacobians with cached impulses to warm start the solver.</summary>
[BurstCompile]
[UpdateInGroup(typeof(PhysicsSolveAndIntegrateGroup), OrderFirst = true)]
internal unsafe partial struct PrepareSolveSystem : ISystem
{
    private EntityQuery simulations;
    private EntityQuery inputs;

    public void OnCreate(ref SystemState state)
    {
        state.GetEntityQuery(ComponentType.ReadWrite<PhysicsCollider>());
        simulations = state.GetEntityQuery(ComponentType.ReadWrite<SimulationSingleton>());
        inputs = new EntityQueryBuilder(Allocator.Temp).WithAllRW<StepInputSingleton>().WithOptions(EntityQueryOptions.IncludeSystems).Build(ref state);
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        SimulationSingleton singleton = simulations.GetSingleton<SimulationSingleton>();
        ref StepInputSingleton input = ref inputs.GetSingletonRW<StepInputSingleton>().ValueRW;
        JobHandle dependency = state.Dependency;
        StepKernel.PrepareSolve(&singleton, (SimulationStepInput*)UnsafeUtility.AddressOf(ref input.StepInput), &dependency);
        state.Dependency = dependency;
    }
}

/// <summary>Replaces <see cref="ExportPhysicsWorld"/> and writes the solved world back to the store.</summary>
[BurstCompile]
[UpdateInGroup(typeof(PhysicsSystemGroup))]
[UpdateAfter(typeof(PhysicsSimulationGroup))]
internal unsafe partial struct FinishStepSystem : ISystem
{
    private EntityQuery simulations;

    public void OnCreate(ref SystemState state)
    {
        state.GetEntityQuery(ComponentType.ReadWrite<PhysicsCollider>());
        simulations = state.GetEntityQuery(ComponentType.ReadWrite<SimulationSingleton>());
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        // Schedule post-solve work before waiting for the solve. Reading the singleton here would also wait for
        // cleanup.
        if (StepKernel.FinishPreparedStep(state.WorldUnmanaged.Time.DeltaTime) != 0)
        {
            return;
        }

        state.CompleteDependency();
        Simulation simulation = simulations.GetSingleton<SimulationSingleton>().AsSimulation();
        StepKernel.FinishStep(&simulation, state.WorldUnmanaged.Time.DeltaTime);
    }
}
