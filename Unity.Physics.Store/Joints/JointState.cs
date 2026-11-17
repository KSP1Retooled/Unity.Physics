using System;

namespace Unity.Physics.Store.Joints
{
    [Flags]
    internal enum JointState : uint
    {
        Broken = 1 << 0,
        HingeLimitEnabled = 1 << 1,
        HingeMotorFreeSpin = 1 << 2,
        BodyDestroyed = 1 << 3,
    }
}
