using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Systems;

namespace KSP.Retooled.Physics.Stepping;

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
        StepEntryPoints.SortIslands(&singleton, (SimulationStepInput*)UnsafeUtility.AddressOf(ref input.StepInput), &dependency);
        state.Dependency = dependency;
    }
}
