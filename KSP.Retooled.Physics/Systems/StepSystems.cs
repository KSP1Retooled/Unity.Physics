using KSP.Retooled.Physics;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Systems;

namespace KSP.Retooled.Systems.Physics;

// The stages of a step that aren't the simulation's own, each in the group that puts it in the right place around
// Unity.Physics' systems. They write PhysicsCollider so that the simulation's job dependencies pass through them too: that
// is the component those systems chain their jobs on. Bodies live on entities, but a step works from the store's rows, which
// is why none of these query them.

/// <summary>Integrates what was applied to bodies since the last step.</summary>
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

/// <summary>Stands in for <see cref="BuildPhysicsWorld"/>, building the world from the store instead of from a query.</summary>
[BurstCompile]
[UpdateInGroup(typeof(PhysicsInitializeGroup))]
[CreateAfter(typeof(BuildPhysicsWorld))]
[CreateAfter(typeof(ExportPhysicsWorld))]
internal unsafe partial struct BuildBodyWorldSystem : ISystem
{
    private EntityQuery settings;
    private EntityQuery worlds;
    private EntityQuery buildData;

    public void OnCreate(ref SystemState state)
    {
        state.EntityManager.CreateSingleton(PhysicsStep.Default);
        state.GetEntityQuery(ComponentType.ReadWrite<PhysicsCollider>());
        settings = state.GetEntityQuery(ComponentType.ReadWrite<PhysicsStep>());
        worlds = state.GetEntityQuery(ComponentType.ReadWrite<PhysicsWorldSingleton>());
        buildData = new EntityQueryBuilder(Allocator.Temp).WithAllRW<BuildPhysicsWorldData>().WithOptions(EntityQueryOptions.IncludeSystems).Build(ref state);
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
        BuildPhysicsWorldData data = buildData.GetSingleton<BuildPhysicsWorldData>();
        data.PhysicsData.HaveStaticBodiesChanged.Value = StepKernel.Current->StaticsChanged;
    }

    private static void DisableStockSystem<T>(ref SystemState state)
        where T : unmanaged, ISystem
    {
        SystemHandle system = state.WorldUnmanaged.GetExistingUnmanagedSystem<T>();
        state.WorldUnmanaged.ResolveSystemStateRef(system).Enabled = false;
    }
}

/// <summary>Sorts the broadphase's pairs into islands, once it has dropped those that won't be solved.</summary>
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

/// <summary>Reads the contacts between the narrow phase and the Jacobians built from them.</summary>
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

/// <summary>Warm starts the Jacobians.</summary>
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

/// <summary>Stands in for <see cref="ExportPhysicsWorld"/>, writing the solved world back to the store.</summary>
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
        state.CompleteDependency();
        Simulation simulation = simulations.GetSingleton<SimulationSingleton>().AsSimulation();
        StepKernel.FinishStep(&simulation, state.WorldUnmanaged.Time.DeltaTime);
    }
}
