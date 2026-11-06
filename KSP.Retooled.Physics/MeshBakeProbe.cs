using UnityEngine;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>The root asset of the content build in MeshBakeCheck, which must be a ScriptableObject.</summary>
    public sealed class MeshBakeProbe : ScriptableObject
    {
        public GameObject prefab;
    }
}
