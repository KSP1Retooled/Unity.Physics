using System.Runtime.InteropServices;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Versions;

namespace Unity.Physics.Store.Joints
{
    /// <summary>
    /// Joint settings and last-step measurements. Handles and the world list point to this record.
    /// A null body represents the world, with that end's pose in world space.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct JointData
    {
        public quaternion rotation0;
        public float3 position0;
        public uint type; // Physics::JointType

        public quaternion rotation1;
        public float3 position1;
        public uint flags; // Physics::JointFlags

        public quaternion targetRotation;
        public float3 targetPosition;
        public float inverseMassScale0;

        public float3 targetLinearVelocity;
        public float inverseMassScale1;
        public float3 targetAngularVelocity;
        public uint index; // In the world's list.

        // Keep per-step joint load fields together to limit cache lines read by SettleAwake.
        public JointState state;
        public float breakForce;
        public float breakTorque;

        // The magnitudes of force and torque used to decide whether the joint breaks. A low-pass filter smooths
        // them over a few steps.
        public float breakingForce;

        // The force and torque applied by locks and limits during the last step. Both are in world space,
        // with force in newtons and torque in newton metres.
        public float3 force;
        public float breakingTorque;
        public float3 torque;

        public float hingeMotorTargetVelocity;
        public float projectionLinearTolerance;
        public float projectionAngularTolerance;
        public float distanceErrorTolerance;
        public float hingeMotorMaxForce;

        public Body* body0;
        public Body* body1;
        public void* userData;

        // JointSettings.Version when this joint last changed. Vessel fill jobs compare it with their cached
        // version.
        public uint settingsStamp;

        // A number that no other joint received. The store reuses the memory of a destroyed joint for the next
        // joint that is made, so the address alone doesn't tell two joints apart.
        public uint serial;

        public float3 fullForce;
        public float3 fullTorque;
        public float smoothedFullForce;
        public float smoothedFullTorque;

        // There are JointAxisCount consecutive axes. AxisOf accesses them by adding an offset to the address of the
        // first axis.
        public JointAxis axes0;
        private readonly JointAxis axes1;
        private readonly JointAxis axes2;
        private readonly JointAxis axes3;
        private readonly JointAxis axes4;
        private readonly JointAxis axes5;
        private readonly JointAxis axes6;
        private readonly JointAxis axes7;
        private readonly JointAxis axes8;

        /// <summary>
        /// Returns the joint axis at a Physics::ConstraintAxis index. The record is allocated on the store's heap, so
        /// its address does not change.
        /// </summary>
        internal static JointAxis* AxisOf(JointData* joint, int axis)
        {
            return &joint->axes0 + axis;
        }
    }
}
