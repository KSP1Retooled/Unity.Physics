using KSP.Retooled.Physics;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Systems;

namespace KSP.Retooled.Systems.Physics;

// These systems run the stages of a step that Unity.Physics does not provide. Each belongs to a group that places it
// before or after the appropriate Unity.Physics systems. They register write access to PhysicsCollider so that their
// jobs participate in the simulation's dependencies. The Unity.Physics systems use that component to chain their jobs.
// Bodies are stored on entities, but these systems use the rows in the store instead of querying those entities.

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

    // Keep this reference instead of reading BuildPhysicsWorldData each step. Querying that component would register
    // this system's jobs as dependencies for it. Stock systems that read it through EntityManager would then have to
    // wait for the entire build.
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
        state.CompleteDependency();
        Simulation simulation = simulations.GetSingleton<SimulationSingleton>().AsSimulation();
        StepKernel.FinishStep(&simulation, state.WorldUnmanaged.Time.DeltaTime);
    }
}
