using Unity.Burst;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Systems;

namespace KSP.Retooled.Physics.Stepping;

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
        StepEntryPoints.ReadContacts(&singleton, state.WorldUnmanaged.Time.DeltaTime, &dependency);
        state.Dependency = dependency;
    }
}
