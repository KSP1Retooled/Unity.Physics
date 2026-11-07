using KSP.Retooled.Physics;
using Unity.Entities;

namespace KSP.Retooled.Systems.Physics;

/// <summary>
/// Gives rigid bodies to the world this system is in, for as long as that world lasts. Having it created with the world
/// means the world's start and end need nothing else, whatever ends it.
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
