using System;
using System.Runtime.InteropServices;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Store.Shapes;

namespace Unity.Physics.Store.World
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct StoreGlobals
    {
        public uint materialVersion;
        public uint engineObjectIdOffset;

        public uint jointDataSize;
        public uint bodyDataSize;
        public uint bodySize;

        /// <summary>
        /// Increases whenever the ignored pairs change. Cached results derived from those pairs can be reused while
        /// this version stays the same.
        /// </summary>
        public uint ignoreVersion;

        public IgnoreTable* ignoreTable;

        public MaterialData* defaultMaterial;

        public IntPtr reallocate;
        public IntPtr release;
        public IntPtr retainMesh;
        public IntPtr releaseMesh;
        public IntPtr retainMaterial;
        public IntPtr releaseMaterial;
    }
}
