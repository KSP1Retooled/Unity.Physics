using KSP.Retooled.Physics;
using Unity.Entities;

namespace KSP.Retooled.Systems.Physics;

/// <summary>
/// Bind body components to this system's Entities world until the system is destroyed.
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
