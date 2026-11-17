using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Dynamics.Joint.Translation;

namespace Unity.Physics.Stepping
{
    internal static partial class StoreBodies
    {
        /// <summary>
        /// Updates the poses of the built static bodies from the store when nothing else in the world has changed.
        /// </summary>
        [BurstCompile]
        internal unsafe struct PoseStaticsJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public BodyData* Store;

            [ReadOnly]
            public NativeArray<int> StaticIndices;

            /// <summary>The static bodies in the built world, in the order of <see cref="StaticIndices"/> .</summary>
            public NativeArray<RigidBody> StaticBodies;

            public void Execute(int i)
            {
                RigidBody body = StaticBodies[i];
                body.WorldFromBody = BuildBodiesJob.WorldFromBody(ref Store[StaticIndices[i]]);
                StaticBodies[i] = body;
            }
        }

        /// <summary>
        /// Writes the solved body poses and velocities back to the store. Linear velocities are in world space; angular
        /// velocities are in the motion frame.
        /// </summary>
        [BurstCompile]
        internal unsafe struct WriteBackJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public BodyData* Store;

            [ReadOnly]
            public NativeArray<int> DynamicIndices;

            [ReadOnly]
            public NativeArray<MotionData> MotionDatas;

            [ReadOnly]
            public NativeArray<MotionVelocity> MotionVelocities;

            public void Execute(int index)
            {
                ref BodyData body = ref Store[DynamicIndices[index]];
                if ((body.flags & BodyFlags.Sleeping) != 0)
                {
                    return;
                }

                if ((body.flags & BodyFlags.Kinematic) != 0)
                {
                    if ((body.flags & BodyFlags.HasKinematicTarget) != 0)
                    {
                        body.Pose()->Position = body.Target()->Position;
                        BodyKinematics.SetRotation(body, body.Target()->Rotation);
                        body.flags &= ~BodyFlags.HasKinematicTarget;
                    }

                    return;
                }

                MotionVelocity velocity = MotionVelocities[index];
                BodyKinematics.SolvedPose(MotionDatas[index], out body.Pose()->Position, out body.Pose()->Rotation);
                body.Velocity()->Linear = ExternalForces.ApplyForcesJob.ClampLength(velocity.LinearVelocity, body.Limits()->MaxLinearVelocity);
                body.Velocity()->Angular = ExternalForces.ApplyForcesJob.ClampLength(velocity.AngularVelocity, body.Limits()->MaxAngularVelocity);
            }

        }
    }
}
