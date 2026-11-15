using KSP.Retooled.Physics;
using Unity.Entities;

namespace KSP.Retooled.Systems.Physics;

/// <summary>
/// Binds rigid bodies to the world that contains this system for the lifetime of that world. The system is created and
/// destroyed with the world, so no separate calls are needed to bind or unbind bodies, regardless of how the world
/// ends.
/// </summary>
internal sealed partial class BodyBindingSystem : SystemBase
{
    protected override void OnCreate()
    {
        Enabled = false;
        BackendHost.BindWorld(World);
    }

    protected override void OnDestroy()
    {
        BackendHost.UnbindWorld(World);
    }

    protected override void OnUpdate() { }
}
