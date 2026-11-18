using Unity.Burst;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Systems;

namespace KSP.Retooled.Physics.Stepping;

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
        JobHandle dependency = state.Dependency;
        if (StepEntryPoints.FinishPreparedStep(state.WorldUnmanaged.Time.DeltaTime, &dependency) != 0)
        {
            state.Dependency = default;
            return;
        }

        state.CompleteDependency();
        Simulation simulation = simulations.GetSingleton<SimulationSingleton>().AsSimulation();
        StepEntryPoints.FinishStep(&simulation, state.WorldUnmanaged.Time.DeltaTime);
    }
}
