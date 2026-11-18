using System.Runtime.InteropServices;
using Unity.Physics.Store.Bodies;

namespace Unity.Physics.Store.Joints
{
    /// <summary>
    /// Defines the blittable beginning of Physics::ActorDescriptor. Joint.cpp checks this layout.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct ActorDescriptor
    {
        public Body* actor;
        public void* userData;
        public ushort type;
        private readonly ushort reserved0;
        private readonly uint reserved1;
    }
}
