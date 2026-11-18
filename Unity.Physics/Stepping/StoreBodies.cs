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
using System.Threading;

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

            [NativeDisableUnsafePtrRestriction]
            public StoreStepperCounters* Counters;

            [ReadOnly]
            public NativeArray<int> DynamicIndices;

            public NativeArray<MotionData> MotionDatas;

            public NativeArray<MotionVelocity> MotionVelocities;

            public void Execute(int index)
            {
                ref BodyData body = ref Store[DynamicIndices[index]];
                MotionVelocity velocity = MotionVelocities[index];
                MotionData motion = MotionDatas[index];
                bool sleeping = (body.flags & BodyFlags.Sleeping) != 0;
                bool kinematic = (body.flags & BodyFlags.Kinematic) != 0;
                bool hasTarget = !sleeping && kinematic && (body.flags & BodyFlags.HasKinematicTarget) != 0;
                float3 position = body.pose->Position;
                quaternion rotation = body.pose->Rotation;
                if (hasTarget)
                {
                    position = body.target->Position;
                    rotation = body.target->Rotation;
                    body.flags &= ~BodyFlags.HasKinematicTarget;
                }
                else if (!sleeping && !kinematic)
                {
                    BodyKinematics.SolvedPose(motion, out position, out rotation);
                }

                if (!math.all(math.isfinite(position)) || !math.all(math.isfinite(rotation.value))
                    || !math.all(math.isfinite(motion.WorldFromMotion.pos)) || !math.all(math.isfinite(motion.WorldFromMotion.rot.value))
                    || !math.all(math.isfinite(velocity.LinearVelocity)) || !math.all(math.isfinite(velocity.AngularVelocity)))
                {
                    // The store still holds the pose from before the solve. Restore the built world too, so
                    // the bounds refit and queries never see the invalid motion.
                    motion.WorldFromMotion = math.mul(new RigidTransform(body.pose->Rotation, body.pose->Position), motion.BodyFromMotion);
                    velocity.LinearVelocity = float3.zero;
                    velocity.AngularVelocity = float3.zero;
                    MotionDatas[index] = motion;
                    MotionVelocities[index] = velocity;
                    body.velocity->Linear = float3.zero;
                    body.velocity->Angular = float3.zero;
                    Interlocked.Increment(ref Counters->NonFiniteBodies);
                    return;
                }

                if (sleeping || kinematic)
                {
                    if (hasTarget)
                    {
                        body.pose->Position = position;
                        BodyKinematics.SetRotation(body, rotation);
                    }

                    return;
                }

                body.pose->Position = position;
                body.pose->Rotation = rotation;
                body.velocity->Linear = ExternalForces.ApplyForcesJob.ClampLength(velocity.LinearVelocity, body.limits->MaxLinearVelocity);
                body.velocity->Angular = ExternalForces.ApplyForcesJob.ClampLength(velocity.AngularVelocity, body.limits->MaxAngularVelocity);
            }

        }
    }
}
