using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>   Integrates world's motions. </summary>
    public static partial class Integrator
    {
        /// <summary>   Integrate the world's motions forward by the given time step. </summary>
        ///
        /// <param name="motionDatas">      The motion datas. </param>
        /// <param name="motionVelocities"> The motion velocities. </param>
        /// <param name="timeStep">         The time step. </param>
        public static void Integrate(NativeArray<MotionData> motionDatas, NativeArray<MotionVelocity> motionVelocities, float timeStep)
        {
            for (int i = 0; i < motionDatas.Length; i++)
            {
                IntegrateMotion(i, motionDatas, motionVelocities, timeStep);
            }
        }

        /// <summary>   Integrate a single transform for the provided velocity and time. </summary>
        ///
        /// <param name="transform">        [in,out] The transform. </param>
        /// <param name="motionVelocity">   The motion velocity. </param>
        /// <param name="timeStep">         The time step. </param>
        public static void Integrate(ref RigidTransform transform, in MotionVelocity motionVelocity, in float timeStep)
        {
            // center of mass
            IntegratePosition(ref transform.pos, motionVelocity.LinearVelocity, timeStep);

            // orientation
            IntegrateOrientation(ref transform.rot, motionVelocity.AngularVelocity, timeStep);
        }

        internal static void IntegrateMotion(int i, NativeArray<MotionData> motionDatas, NativeArray<MotionVelocity> motionVelocities, float timeStep)
        {
            MotionData motionData = motionDatas[i];
            MotionVelocity motionVelocity = motionVelocities[i];

            // Update motion space
            Integrate(ref motionData.WorldFromMotion, motionVelocity, timeStep);

            // Update velocities
            {
                // damping
                motionVelocity.LinearVelocity *= math.clamp(1.0f - motionData.LinearDamping * timeStep, 0.0f, 1.0f);
                motionVelocity.AngularVelocity *= math.clamp(1.0f - motionData.AngularDamping * timeStep, 0.0f, 1.0f);
            }

            // Write back
            motionDatas[i] = motionData;
            motionVelocities[i] = motionVelocity;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void IntegratePosition(ref float3 position, float3 linearVelocity, float timestep)
        {
            position += linearVelocity * timestep;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void IntegrateOrientation(ref quaternion orientation, float3 angularVelocity, float timestep)
        {
            quaternion dq = IntegrateAngularVelocity(angularVelocity, timestep);
            quaternion r = math.mul(orientation, dq);
            orientation = math.normalize(r);
        }

        // Returns a non-normalized quaternion that approximates the change in angle angularVelocity * timestep.
        internal static quaternion IntegrateAngularVelocity(float3 angularVelocity, float timestep)
        {
            float halfDeltaTime = timestep * 0.5f;
            float3 halfDeltaAngle = halfDeltaTime * angularVelocity;
            return new quaternion(new float4(halfDeltaAngle, 1.0f));
        }
    }
}
