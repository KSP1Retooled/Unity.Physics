using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Shapes
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct MaterialData
    {
        public float dynamicFriction;
        public float staticFriction;
        public float bounciness;
        public uint frictionCombine;
        public uint bounceCombine;
        public uint references;
        public void* userData;
    }
}
