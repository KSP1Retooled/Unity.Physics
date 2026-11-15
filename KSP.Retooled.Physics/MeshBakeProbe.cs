using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Provides the root asset for the content build in MeshBakeCheck. That asset must be a ScriptableObject.
    /// </summary>
    public sealed class MeshBakeProbe : ScriptableObject
    {
        public GameObject prefab;
    }
}
