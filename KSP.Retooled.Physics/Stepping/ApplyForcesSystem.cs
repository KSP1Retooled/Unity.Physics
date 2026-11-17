using Unity.Burst;
using Unity.Entities;
using Unity.Physics.Systems;

namespace KSP.Retooled.Physics.Stepping;

// The backend systems run around the Unity Physics systems. They register PhysicsCollider writes to
// join the simulation dependencies and access body state through the store rows.

/// <summary>Integrates the forces and changes in velocity applied to bodies since the last step.</summary>
[BurstCompile]
[UpdateInGroup(typeof(BeforePhysicsSystemGroup))]
internal partial struct ApplyForcesSystem : ISystem
{
    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        StepEntryPoints.ApplyForces(state.WorldUnmanaged.Time.DeltaTime);
    }
}
