using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Joints
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct JointBreak
    {
        public ulong engineObjectId;
        public float linearMagnitude;
        public float angularMagnitude;
    }
}
