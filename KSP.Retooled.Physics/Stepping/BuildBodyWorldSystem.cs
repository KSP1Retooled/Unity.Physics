using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Systems;

namespace KSP.Retooled.Physics.Stepping;

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
        StepEntryPoints.BuildWorld(state.WorldUnmanaged.Time.DeltaTime, &dependency);
        state.Dependency = dependency;

        settings.SetSingleton(StepEntryPoints.Current->Settings);
        worlds.GetSingletonRW<PhysicsWorldSingleton>().ValueRW.PhysicsWorld = StepEntryPoints.Current->World;
        staticsChanged.Value = StepEntryPoints.Current->StaticsChanged;
    }

    private static void DisableStockSystem<T>(ref SystemState state)
        where T : unmanaged, ISystem
    {
        SystemHandle system = state.WorldUnmanaged.GetExistingUnmanagedSystem<T>();
        state.WorldUnmanaged.ResolveSystemStateRef(system).Enabled = false;
    }
}
