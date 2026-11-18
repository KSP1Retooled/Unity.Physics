using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Joints
{
    /// <summary>
    /// Describes one axis of a joint at the index defined by Physics::ConstraintAxis. The min and max values are in
    /// radians for an angular axis and metres for a linear axis. These are the units that Unity supplies.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct JointAxis
    {
        public float min;
        public float max;
        public float bounciness;
        public float bounceMinVelocity;
        public float contactDistance;
        public float limitSpring;
        public float limitDamping;
        public float driveSpring;
        public float driveDamper;
        public float driveMaxForce;
        public uint dofLock; // Physics::JointDofLock
        public uint driveUsesAcceleration;
    }
}
