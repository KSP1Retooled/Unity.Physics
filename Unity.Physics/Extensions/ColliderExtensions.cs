namespace Unity.Physics.Extensions
{
    /// <summary>   Utility functions acting on colliders. </summary>
    public static class ColliderExtensions
    {
        /// <summary>
        ///  Converts a <see cref="Collider"/> to a <see cref="UnityEngine.Mesh"/>.
        /// </summary>
        /// <param name="collider"> The collider to convert to a mesh.</param>
        /// <returns> The created mesh. The mesh is created with <see cref="UnityEngine.HideFlags.HideAndDontSave"/>, so it is
        /// not released by <see cref="UnityEngine.Resources.UnloadUnusedAssets"/>; the caller owns it and must destroy it
        /// with <see cref="UnityEngine.Object.Destroy(UnityEngine.Object)"/> (or <see cref="UnityEngine.Object.DestroyImmediate(UnityEngine.Object)"/>
        /// outside Play mode) when it is no longer needed.</returns>
        public static UnityEngine.Mesh ToMesh(in this Collider collider)
        {
            return MeshUtilities.CreateMeshFromCollider(collider);
        }
    }
}
