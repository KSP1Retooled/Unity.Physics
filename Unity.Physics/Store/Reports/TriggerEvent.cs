using System.Runtime.InteropServices;
using Unity.Physics.Store.Shapes;

namespace Unity.Physics.Store.Reports
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct TriggerEvent
    {
        public Shape* triggerShape;
        public Shape* colliderShape;
        public ulong trigger;
        public ulong collider;
        public uint type;
        private readonly uint reserved0;
    }
}
